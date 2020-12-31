import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, LongformerModel, RobertaModel, RobertaConfig
from transformers.modeling_bert import ACT2FN

from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, mask_tensor
from data import PAD_ID_FOR_COREF


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class EndToEndCoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_span_length = args.max_span_length
        self.top_lambda = args.top_lambda
        self.ffnn_size = args.ffnn_size
        self.do_mlps = self.ffnn_size > 0
        self.ffnn_size = self.ffnn_size if self.do_mlps else config.hidden_size
        self.independent_mention_loss = args.independent_mention_loss
        self.normalise_loss = args.normalise_loss
        self.num_neighboring_antecedents = args.num_neighboring_antecedents
        self.separate_mention_logits = args.separate_mention_logits
        self.separate_mention_reps = args.separate_mention_reps
        self.apply_antecedent_start_end = args.apply_antecedent_start_end
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        if self.separate_mention_reps:
            self.start_mention_mlp_for_coref = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
            self.end_mention_mlp_for_coref = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.entity_mention_start_classifier = nn.Linear(self.ffnn_size, 1)  # In paper w_s
        self.entity_mention_end_classifier = nn.Linear(self.ffnn_size, 1)  # w_e
        self.entity_mention_joint_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # M

        if self.separate_mention_logits:
            self.entity_mention_start_classifier_for_coref = nn.Linear(self.ffnn_size, 1)  # In paper w_s
            self.entity_mention_end_classifier_for_coref = nn.Linear(self.ffnn_size, 1)  # w_e
            self.entity_mention_joint_classifier_for_coref = nn.Linear(self.ffnn_size, self.ffnn_size)  # M

        self.antecedent_start_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # S
        self.antecedent_end_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # E

        if self.apply_antecedent_start_end:
            self.antecedent_start_end_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # S
            self.antecedent_end_start_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # E

        self.init_weights()

    def _get_encoder(self):
        if self.args.model_type == "longformer":
            return self.longformer
        elif self.args.model_type == "roberta":
            return self.roberta

        raise ValueError("Unsupported model type")

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _get_neighboring_antecedent_mask(self, batch_size, max_k):
        """
        :param batch_size: int
        :param max_k: int
        :return: [batch_size, max_k, max_k]
        """
        tmp = torch.ones((batch_size, max_k, max_k), device=self.device)
        return tmp.tril(diagonal=-1).triu(diagonal=-self.num_neighboring_antecedents)

    def _prune_topk_spans(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        topk_span_starts = sorted_topk_1d_indices // seq_length  # [batch_size, max_k]
        topk_span_ends = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_span_starts, topk_span_ends]  # [batch_size, max_k]

        return topk_span_starts, topk_span_ends, span_mask, topk_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask, neighboring_antecedent_mask=None):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        if neighboring_antecedent_mask is not None:
            antecedents_mask = antecedents_mask * neighboring_antecedent_mask
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]

        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)

        return antecedent_logits

    def _compute_pos_neg_loss(self, weights, labels, logits):
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        all_loss = loss_fct(logits, labels)

        pos_weights = weights * labels
        pos_loss = (all_loss * pos_weights).sum() / (pos_weights.sum() + 1e-4)

        neg_weights = weights * (1 - labels)
        neg_loss = (all_loss * neg_weights).sum() / (neg_weights.sum() + 1e-4)

        loss = 0.5 * neg_loss + 0.5 * pos_loss
        return loss, (neg_loss, pos_loss)

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _compute_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits,
                              attention_mask=None):
        """
        :param start_entity_mention_labels: [batch_size, num_mentions]
        :param end_entity_mention_labels: [batch_size, num_mentions]
        :param mention_logits: [batch_size, seq_length, seq_length]
        :return:
        """
        device = start_entity_mention_labels.device
        batch_size, seq_length, _ = mention_logits.size()
        num_mentions = start_entity_mention_labels.size(-1)

        # We now take the index tensors and turn them into sparse tensors
        labels = torch.zeros(size=(batch_size, seq_length, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_mentions)
        labels[
            batch_temp, start_entity_mention_labels, end_entity_mention_labels] = 1.0  # [batch_size, seq_length, seq_length]
        labels[:, PAD_ID_FOR_COREF, PAD_ID_FOR_COREF] = 0.0  # Remove the padded mentions
        weights = (attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2))
        mention_mask = self._get_mention_mask(weights)
        weights = weights * mention_mask

        loss, (neg_loss, pos_loss) = self._compute_pos_neg_loss(weights, labels, mention_logits)
        return loss, {"mention_joint_neg_loss": neg_loss, "mention_joint_pos_loss": pos_loss}

    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = coref_logits + ((1.0 - cluster_labels_after_pruning) * -10000.0)
        gold_coref_logits = torch.clamp(gold_coref_logits, min=-10000.0, max=10000.0)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = -gold_log_probs
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def _calc_mention_logits_for_coref(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.entity_mention_start_classifier_for_coref(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier_for_coref(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier_for_coref(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def forward(self, input_ids, attention_mask=None, start_entity_mention_labels=None, end_entity_mention_labels=None,
                start_antecedent_labels=None, end_antecedent_labels=None, gold_clusters=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, dim]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        if self.separate_mention_reps:
            start_mention_reps_for_coref = self.start_mention_mlp_for_coref(sequence_output) if self.do_mlps else sequence_output
            end_mention_reps_for_coref = self.end_mention_mlp_for_coref(sequence_output) if self.do_mlps else sequence_output
        start_coref_reps = self.start_coref_mlp(sequence_output) if self.do_mlps else sequence_output
        end_coref_reps = self.end_coref_mlp(sequence_output) if self.do_mlps else sequence_output

        # Entity mention scores for pruning
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)
        span_starts, span_ends, span_mask, top_k_mention_logits = self._prune_topk_spans(mention_logits, attention_mask)

        if self.separate_mention_logits:
            if self.separate_mention_reps:
                mention_logits_for_coref = self._calc_mention_logits_for_coref(start_mention_reps_for_coref, end_mention_reps_for_coref)
            else:
                mention_logits_for_coref = self._calc_mention_logits_for_coref(start_mention_reps, end_mention_reps)
            batch_size, max_k = span_ends.size()
            top_k_mention_logits = mention_logits_for_coref[
                torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k), span_starts, span_ends]  # [batch_size, max_k]

        batch_size, _, dim = start_coref_reps.size()
        max_k = span_starts.size(-1)
        size = (batch_size, max_k, dim)

        top_k_start_coref_reps = torch.gather(start_coref_reps, dim=1, index=span_starts.unsqueeze(-1).expand(size))
        top_k_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=span_ends.unsqueeze(-1).expand(size))
        # Antecedent scores
        temp = self.antecedent_start_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_start_coref_logits = torch.matmul(temp,
                                                top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]
        temp = self.antecedent_end_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_end_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        top_k_mention_logits = top_k_mention_logits.unsqueeze(-1) + top_k_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]
        coref_logits = top_k_mention_logits + top_k_start_coref_logits + top_k_end_coref_logits  # [batch_size, max_k, max_k]

        if self.apply_antecedent_start_end:
            temp = self.antecedent_start_end_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
            top_k_start_end_coref_logits = torch.matmul(temp,
                                                        top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]
            temp = self.antecedent_end_start_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
            top_k_end_start_coref_logits = torch.matmul(temp,
                                                        top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]
            coref_logits = coref_logits + top_k_start_end_coref_logits + top_k_end_start_coref_logits

        if self.num_neighboring_antecedents <= 0:
            neighboring_antecedents_mask = None
        else:
            neighboring_antecedents_mask = self._get_neighboring_antecedent_mask(batch_size, max_k)
        coref_logits = self._mask_antecedent_logits(coref_logits, span_mask, neighboring_antecedents_mask)

        # adding zero logits for null span
        coref_logits = torch.cat((coref_logits, torch.zeros((batch_size, max_k, 1), device=self.device)),
                                 dim=-1)  # [batch_size, max_k, max_k + 1]
        if return_all_outputs:
            outputs = (span_starts, span_ends, coref_logits, mention_logits)
        else:
            outputs = tuple()
        if gold_clusters is not None:
            losses = {}
            loss = 0.0
            if self.independent_mention_loss:
                mention_loss, mention_losses = self._compute_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    mention_logits=mention_logits,
                    attention_mask=attention_mask)
                loss += mention_loss
                losses.update(mention_losses)
            labels_after_pruning = self._get_cluster_labels_after_pruning(span_starts, span_ends, gold_clusters)
            end_to_end_loss = self._get_marginal_log_likelihood_loss(coref_logits, labels_after_pruning, span_mask)
            losses.update({"end_to_end_loss": end_to_end_loss})
            loss += end_to_end_loss
            losses.update({"loss": loss})
            outputs = (loss,) + outputs + (losses,)

        return outputs


class BaselineCoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.antecedent_loss = args.antecedent_loss  # can be either allowed loss or bce
        self.max_span_length = args.max_span_length
        self.top_lambda = args.top_lambda
        # self.seperate_mention_loss = seperate_mention_loss
        # self.prune_mention_for_antecedents = prune_mention_for_antecedents
        self.normalize_antecedent_loss = args.normalize_antecedent_loss
        self.only_joint_mention_logits = args.only_joint_mention_logits
        self.no_joint_mention_logits = args.no_joint_mention_logits
        self.pos_coeff = args.pos_coeff
        self.independent_mention_loss = args.independent_mention_loss
        self.normalise_loss = args.normalise_loss
        self.max_c = args.max_c
        self.independent_start_end_loss = args.independent_start_end_loss
        self.coarse_to_fine = args.coarse_to_fine
        self.apply_attended_reps = args.apply_attended_reps
        self.ffnn_size = 3000
        self.separate_mention_logits = args.separate_mention_logits
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.attention_classifier = nn.Linear(config.hidden_size, 1)
        self.span_dim = 3 * config.hidden_size if self.apply_attended_reps else 2 * config.hidden_size

        self.mention_mlp = FullyConnectedLayer(config, self.span_dim, self.ffnn_size, args.dropout_prob)
        self.coref_mlp = FullyConnectedLayer(config, 3 * self.ffnn_size, self.ffnn_size, args.dropout_prob)

        if self.separate_mention_logits:
            self.entity_coref_mention_classifier = nn.Linear(self.ffnn_size, 1)  # In paper w_s

        self.entity_mention_classifier = nn.Linear(self.ffnn_size, 1)  # In paper w_s
        self.fast_antecedent_classifier = nn.Linear(self.ffnn_size, self.ffnn_size)  # M

        self.coref_classifier = nn.Linear(self.ffnn_size, 1)  # S

        self.init_weights()

    def _get_encoder(self):
        if self.args.model_type == "longformer":
            return self.longformer
        elif self.args.model_type == "roberta":
            return self.roberta

        raise ValueError("Unsupported model type")

    def _get_start_end_reps(self, sequence_output, attention_mask):
        """
        gets the start and end embeddings of the candidate mention spans
        :param sequence_output: [batch_size, seq_len, dim]
        :param attention_mask: [batch_size, seq_len]
        :return:
        """
        batch_size, seq_len, dim = sequence_output.size()
        span_starts = torch.arange(seq_len, device=self.device).reshape(1, -1, 1).expand(
            (batch_size, seq_len, self.max_span_length))  # [batch_size, seq_len, max_span_len]

        span_end_offsets = torch.arange(self.max_span_length, device=self.device).reshape(1, 1, -1)  # [1, 1, max_span_len]
        span_ends = span_starts + span_end_offsets  # [batch_size, seq_len, max_span_len]
        actual_seq_lens = torch.sum(attention_mask, dim=-1).reshape(-1, 1, 1)  # [batch_size, 1, 1]
        span_ends[span_ends >= actual_seq_lens] = 0  # end can't be larger than seq_len

        candidate_mask = span_ends.bool()  # [batch_size, seq_len, max_span_len]
        candidate_mask[:, 0, 0] = True  # the (0,0) span is always valid

        size = (batch_size, seq_len * self.max_span_length, dim)
        start_reps = torch.gather(sequence_output, dim=1,
                                  index=span_starts.reshape(batch_size, -1, 1).expand(size))  # [batch_size, seq_len * max_span_length, dim]
        end_reps = torch.gather(sequence_output, dim=1, index=span_ends.reshape(batch_size, -1, 1).expand(size))  # [batch_size, seq_len * max_span_length, dim]
        start_reps = start_reps.reshape(batch_size, seq_len, self.max_span_length, -1)  # [batch_size, seq_len, max_span_length, dim]
        end_reps = end_reps.reshape(batch_size, seq_len, self.max_span_length, -1)  # [batch_size, seq_len, max_span_length, dim]
        return start_reps, end_reps, span_starts, span_ends, candidate_mask

    def _get_attended_reps(self, sequence_output, span_starts, span_ends):
        """
        gets the attended embeddings of the candidate mention spans
        :param sequence_output: [batch_size, seq_len, dim]
        :param span_starts: [batch_size, seq_len, max_span_len, dim]
        :param span_ends: [batch_size, seq_len, max_span_len, dim]
        :return attended_reps: [batch_size, seq_len, max_span_len, dim]
        """
        batch_size, seq_len, dim = sequence_output.size()
        doc_range = torch.arange(seq_len, device=self.device).view(1, 1, seq_len).expand(seq_len, self.max_span_length, seq_len).unsqueeze(
            0)  # [1, seq_len, max_span_length, seq_len]
        mention_mask = (doc_range >= span_starts.unsqueeze(-1)) & (doc_range <= span_ends.unsqueeze(-1))  # [batch_size, seq_len, max_span_length, seq_len]
        token_attn = self.attention_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        mention_token_attn_logits = mask_tensor(token_attn.view(batch_size, seq_len, 1, 1), mention_mask)
        mention_token_attn = torch.nn.functional.softmax(mention_token_attn_logits, dim=-1)  # [batch_size, seq_len, max_span_length, seq_len]
        attended_reps = torch.matmul(mention_token_attn.view(batch_size, -1, seq_len), sequence_output).view(batch_size, seq_len, self.max_span_length, -1)
        return attended_reps

    def get_span_reps(self, sequence_output, attention_mask):
        """
        gets the span embeddings of the candidate mention spans
        :param sequence_output: [batch_size, seq_len, dim]
        :param attention_mask: [batch_size, seq_len]
        :return:
        """
        start_reps, end_reps, span_starts, span_ends, candidate_mask = self._get_start_end_reps(sequence_output, attention_mask)
        if self.apply_attended_reps:
            attended_reps = self._get_attended_reps(sequence_output, span_starts, span_ends)
            span_reps = torch.cat((start_reps, end_reps, attended_reps), dim=-1)  # [batch_size, max_span_length, seq_len, span_dim]
        else:
            span_reps = torch.cat((start_reps, end_reps), dim=-1)  # [batch_size, max_span_length, seq_len, span_dim]
        return span_reps, candidate_mask

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        min_k_per_entry = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        actual_k_per_entry = k.unsqueeze(1).expand(size)
        return (min_k_per_entry < actual_k_per_entry).int()

    def _prune_top_lambda_spans(self, span_logits, attention_mask, candidate_mention_reps):
        """
        :param span_logits: Shape [batch_size, seq_length, max_span_length]
        :param attention_mask: [batch_size, seq_length]
        :param candidate_mention_reps: [batch_size, seq_length, max_span_length, span_dim]
        :return:
        """
        batch_size, seq_length, max_span_length = span_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(span_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        # pad idx is of the last idx (pad or not)
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length * max_span_length) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        span_starts = sorted_topk_1d_indices // max_span_length  # [batch_size, max_k]
        span_end_offsets = sorted_topk_1d_indices % max_span_length  # [batch_size, max_k]

        span_logits = span_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k), span_starts, span_end_offsets]  # [batch_size, max_k]
        top_k_span_reps = candidate_mention_reps[
            torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k), span_starts, span_end_offsets]  # [batch_size, max_k, span_dim]
        assert top_k_span_reps.size() == (batch_size, max_k, candidate_mention_reps.size(-1))

        return span_starts, span_end_offsets, span_mask, span_logits, top_k_span_reps, k, max_k

    def _get_cluster_labels_after_pruning(self, span_starts, span_end_offsets, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        span_ends = span_starts + span_end_offsets
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = coref_logits + ((1.0 - cluster_labels_after_pruning) * -10000.0)
        gold_coref_logits = torch.clamp(gold_coref_logits, min=-10000.0, max=10000.0)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = -gold_log_probs
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _compute_pos_neg_loss(self, weights, labels, logits):
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        all_loss = loss_fct(logits, labels)

        pos_weights = weights * labels
        pos_loss = (all_loss * pos_weights).sum() / (pos_weights.sum() + 1e-4)

        neg_weights = weights * (1 - labels)
        neg_loss = (all_loss * neg_weights).sum() / (neg_weights.sum() + 1e-4)

        loss = 0.5 * neg_loss + 0.5 * pos_loss
        return loss, (neg_loss, pos_loss)

    def _compute_joint_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits, attention_mask, candidate_mask):
        """
        :param start_entity_mention_labels: [batch_size, num_mentions]
        :param end_entity_mention_labels: [batch_size, num_mentions]
        :param mention_logits: [batch_size, seq_length, seq_length]
        :return:
        """
        device = start_entity_mention_labels.device
        batch_size, seq_length, max_span_length = mention_logits.size()
        end_offset_labels = end_entity_mention_labels - start_entity_mention_labels
        span_length_invalid_mention_indices = end_offset_labels >= self.max_span_length  # offset >= max_span_length --> actual_span_length > max_span_length
        start_entity_mention_labels[span_length_invalid_mention_indices] = PAD_ID_FOR_COREF
        end_offset_labels[span_length_invalid_mention_indices] = PAD_ID_FOR_COREF
        # We now take the index tensors and turn them into sparse tensors
        labels = torch.zeros_like(mention_logits)
        num_mentions = start_entity_mention_labels.size(-1)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_mentions)
        labels[batch_temp, start_entity_mention_labels, end_offset_labels] = 1.0  # [batch_size, seq_length, seq_length]
        labels[:, PAD_ID_FOR_COREF, PAD_ID_FOR_COREF] = 0.0  # Remove the padded mentions

        loss, (neg_loss, pos_loss) = self._compute_pos_neg_loss(candidate_mask, labels, mention_logits)
        return loss, {"mention_joint_neg_loss": neg_loss, "mention_joint_pos_loss": pos_loss}

    def _mask_fast_antecedent_logits(self, antecedent_logits, span_mask=None):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        if span_mask is not None:
            antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]

        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)

        return antecedent_logits

    def get_slow_coref_scores(self, top_k_span_reps, top_coref_reps, antecedents_mask):
        """
        :param top_k_span_reps: [batch_size, max_k, span_dim]
        :param top_coref_reps: [batch_size, max_k, max_c, span_dim]
        :param antecedents_mask: [batch_size, max_k, max_c]
        :return:
        """
        batch_size, max_k, max_c, _ = top_coref_reps.size()
        similarity_reps = top_k_span_reps.unsqueeze(2) * top_coref_reps  # [batch_size, max_k, max_c, span_dim]
        target_reps = top_k_span_reps.unsqueeze(2).expand((batch_size, max_k, max_c, -1))  # [batch_size, max_k, max_c, span_dim]
        pair_reps = torch.cat((target_reps, top_coref_reps, similarity_reps), dim=-1)  # [batch_size, max_k, max_c, coref_spans_dim]
        intermediate_pair_reps = self.coref_mlp(pair_reps)  # [batch_size, max_k, max_c, ffnn_size]
        slow_antecedent_scores = self.coref_classifier(intermediate_pair_reps).squeeze(-1)  # [batch_size, max_k, max_c]
        slow_antecedent_scores = mask_tensor(slow_antecedent_scores, antecedents_mask)  # [batch_size, max_k, max_c]
        return slow_antecedent_scores

    def _get_topc_antecedent_mask(self, batch_size, max_k, max_c, mention_mask):
        """
        :param batch_size: []
        :param max_c: []
        :param mention_mask: [batch_size, max_k]
        :return: [batch_size, max_k, max_c]
        """
        antecedent_mask = torch.ones((max_k, max_c), dtype=mention_mask.dtype, device=mention_mask.device)  # [max_k, max_c]
        antecedent_mask = antecedent_mask.tril(diagonal=-1)  # Each mention can only predict antecedent from earlier mentions
        antecedent_mask = antecedent_mask.unsqueeze(0).expand((batch_size, max_k, max_c))  # [batch_size, max_k, max_c]
        antecedent_mask = antecedent_mask * mention_mask.unsqueeze(-1)  # Mask out padded mentions
        return antecedent_mask

    def _compute_fast_coref_scores(self, top_k_span_reps, top_k_span_logits, mention_mask):
        """
        :param top_k_span_reps: [batch_size, max_k, span_dim]
        :param top_k_span_logits: [batch_size, max_k]
        :param mention_mask: [batch_size, max_k]
        :return:
        """
        temp = self.fast_antecedent_classifier(top_k_span_reps)  # [batch_size, max_k, span_dim]
        top_k_coref_antecedent_logits = torch.matmul(temp, top_k_span_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]
        top_k_coref_span_logits = top_k_span_logits.unsqueeze(-1) + top_k_span_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]
        fast_coref_logits = top_k_coref_span_logits + top_k_coref_antecedent_logits  # [batch_size, max_k, max_k]
        fast_coref_logits = self._mask_fast_antecedent_logits(fast_coref_logits, mention_mask)  # [batch_size, max_k, max_k]

        return fast_coref_logits

    def _get_top_c_coref_scores(self, fast_coref_logits, mention_mask):
        batch_size, max_k, _ = fast_coref_logits.size()

        # if c > k
        max_c = min(self.max_c, max_k)
        antecedent_mask = self._get_topc_antecedent_mask(batch_size, max_k, max_c, mention_mask)  # [batch_size, max_k, max_c]

        _, top_c_indices = torch.topk(fast_coref_logits, dim=-1, k=max_c)  # [batch_size, max_k, max_c]
        # pad idx is last idx
        top_c_indices = (top_c_indices * antecedent_mask) + (1 - antecedent_mask) * (max_k - 1)  # [batch_size, max_k, max_c]
        antecedents_ids, _ = torch.sort(top_c_indices, dim=-1)  # [batch_size, max_k, max_c]
        top_fast_coref_logits = torch.gather(fast_coref_logits, dim=-1, index=antecedents_ids)  # [batch_size, max_k, max_c]
        top_fast_coref_logits = mask_tensor(top_fast_coref_logits, antecedent_mask)  # [batch_size, max_k, max_c]
        return top_fast_coref_logits, antecedent_mask, antecedents_ids, max_c

    def forward(self, input_ids, attention_mask=None, start_entity_mention_labels=None, end_entity_mention_labels=None,
                start_antecedent_labels=None, end_antecedent_labels=None, gold_clusters=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, dim]

        candidate_mention_reps, candidate_mask = self.get_span_reps(sequence_output, attention_mask)  # [batch_size, seq_len, max_span_len, span_dim]
        candidate_mention_reps = self.mention_mlp(candidate_mention_reps)

        mention_logits = self.entity_mention_classifier(candidate_mention_reps).squeeze(-1)  # [batch_size, seq_len, max_span_len]
        mention_logits = mask_tensor(mention_logits, candidate_mask)


        span_starts, span_end_offsets, span_mask, top_k_span_logits, top_k_span_reps, k, max_k = self._prune_top_lambda_spans(mention_logits, attention_mask,
                                                                                                                       candidate_mention_reps)
        if self.separate_mention_logits:
            coref_mention_logits = self.entity_coref_mention_classifier(candidate_mention_reps).squeeze(-1)  # [batch_size, seq_len, max_span_len]
            coref_mention_logits = mask_tensor(coref_mention_logits, candidate_mask)
            top_k_span_logits = coref_mention_logits[torch.arange(span_starts.size(0)).unsqueeze(-1).expand(span_starts.size()), span_starts, span_end_offsets]  # [batch_size, max_k]

        batch_size, max_k = top_k_span_logits.size()

        # Antecedent scores
        # get fast antecedents
        antecedents_ids = None
        fast_coref_logits = self._compute_fast_coref_scores(top_k_span_reps, top_k_span_logits, span_mask)  # [batch_size, max_k, max_k]
        if self.coarse_to_fine:
            top_fast_coref_logits, general_antecedents_mask, antecedents_ids, max_c = self._get_top_c_coref_scores(fast_coref_logits, span_mask)
            span_dim = top_k_span_reps.size(-1)

            index = antecedents_ids.view((batch_size, max_k * max_c, 1)).expand((batch_size, max_k * max_c, span_dim))  # [batch_size, max_k*max_c, span_dim]
            top_coref_reps = torch.gather(top_k_span_reps, dim=1, index=index).view(batch_size, max_k, max_c, -1)  # [batch_size, max_k, max_c, span_dim]
            top_slow_coref_logits = self.get_slow_coref_scores(top_k_span_reps, top_coref_reps, general_antecedents_mask)  # [batch_size, max_k, max_c]
            top_antecedent_logits = top_fast_coref_logits + top_slow_coref_logits
            # top_antecedent_weights = torch.nn.functional.softmax(top_antecedent_logits, dim=-1)
        else:
            top_antecedent_logits = fast_coref_logits

        # adding zero logits for null span
        top_antecedent_logits = torch.cat((top_antecedent_logits, torch.zeros((batch_size, max_k, 1), device=self.device)),
                                          dim=-1)  # [batch_size, max_k, (max_k + 1) or (max_c + 1)]

        outputs = (span_starts, span_end_offsets, top_antecedent_logits, mention_logits, antecedents_ids)
        if gold_clusters is not None:
            losses = {}
            loss = 0.0
            if self.independent_mention_loss:
                entity_mention_loss, entity_losses = self._compute_joint_entity_mention_loss(start_entity_mention_labels=start_entity_mention_labels,
                                                                                             end_entity_mention_labels=end_entity_mention_labels,
                                                                                             mention_logits=mention_logits,
                                                                                             attention_mask=attention_mask,
                                                                                             candidate_mask=candidate_mask)
                loss += entity_mention_loss
                losses.update(entity_losses)

            labels_after_pruning = self._get_cluster_labels_after_pruning(span_starts, span_end_offsets, gold_clusters)
            if self.coarse_to_fine:
                labels_after_pruning = torch.gather(labels_after_pruning, dim=-1, index=antecedents_ids)  # [batch_size, max_k, max_c]
                no_antecedents_indicator = (torch.sum(labels_after_pruning, dim=-1) == 0).float() * span_mask  # [batch_size, max_k, max_c]
                labels_after_pruning = torch.cat((labels_after_pruning, no_antecedents_indicator.unsqueeze(-1)), dim=-1)  # [batch_size, max_k, max_c + 1]
            end_to_end_loss = self._get_marginal_log_likelihood_loss(top_antecedent_logits, labels_after_pruning, span_mask)
            losses.update({"end_to_end_loss": end_to_end_loss})
            loss += end_to_end_loss
            losses.update({"loss": loss})
            outputs = (loss,) + outputs + (losses,)

        return outputs
