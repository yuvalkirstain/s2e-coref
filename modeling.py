import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, LongformerModel, RobertaModel, RobertaConfig
from transformers.modeling_bert import ACT2FN

from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters
from data import PAD_ID_FOR_COREF


class FullyConnectedLayer(Module):
    # TODO: many layers
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


class CoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args, antecedent_loss, max_span_length, seperate_mention_loss,
                 prune_mention_for_antecedents, normalize_antecedent_loss, only_joint_mention_logits, no_joint_mention_logits, pos_coeff, zero_null_logits):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.antecedent_loss = antecedent_loss  # can be either allowed loss or bce
        self.max_span_length = max_span_length
        self.seperate_mention_loss = seperate_mention_loss
        self.prune_mention_for_antecedents = prune_mention_for_antecedents
        self.normalize_antecedent_loss = normalize_antecedent_loss
        self.only_joint_mention_logits = only_joint_mention_logits
        self.no_joint_mention_logits = no_joint_mention_logits
        self.pos_coeff = pos_coeff
        self.zero_null_logits = zero_null_logits
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)

        self.entity_mention_start_classifier = nn.Linear(config.hidden_size, 1)  # In paper w_s
        self.entity_mention_end_classifier = nn.Linear(config.hidden_size, 1)  # w_e
        self.entity_mention_joint_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # M

        self.antecedent_start_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # S
        self.antecedent_end_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # E

        self.init_weights()

    def _compute_joint_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits,
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
        labels[batch_temp, start_entity_mention_labels, end_entity_mention_labels] = 1.0  # [batch_size, seq_length, seq_length]
        labels[:, PAD_ID_FOR_COREF, PAD_ID_FOR_COREF] = 0.0  # Remove the padded mentions

        weights = (attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2))
        mention_mask = self._get_mention_mask(weights)
        weights = weights * mention_mask

        loss, (neg_loss, pos_loss) = self._compute_pos_neg_loss(weights, labels, mention_logits)
        return loss, {"mention_joint_neg_loss": neg_loss, "mention_joint_pos_loss": pos_loss}

    def _calc_boundary_loss(self, boundary_entity_mention_labels, boundary_mention_logits, attention_mask):
        device = boundary_entity_mention_labels.device
        batch_size, seq_length = boundary_mention_logits.size()
        num_mentions = boundary_entity_mention_labels.size(-1)
        # We now take the index tensors and turn them into sparse tensors
        boundary_labels = torch.zeros(size=(batch_size, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_mentions)
        boundary_labels[
            batch_temp, boundary_entity_mention_labels] = 1.0  # [batch_size, seq_length]
        boundary_labels[:, 0] = 0.0  # Remove the padded mentions

        boundary_weights = attention_mask
        loss, (neg_loss, pos_loss) = self._compute_pos_neg_loss(boundary_weights, boundary_labels,
                                                                boundary_mention_logits)
        return loss, {"neg_loss": neg_loss, "pos_loss": pos_loss}

    def _compute_seperate_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels,
                                              start_mention_logits, end_mention_logits, joint_mention_logits,
                                              attention_mask=None):
        joint_loss, (joint_neg_loss, joint_pos_loss) = self._compute_joint_entity_mention_loss(
            start_entity_mention_labels, end_entity_mention_labels,
            joint_mention_logits,
            attention_mask)
        start_loss, (start_neg_loss, start_pos_loss) = self._calc_boundary_loss(start_entity_mention_labels,
                                                                                start_mention_logits, attention_mask)
        end_loss, (end_neg_loss, end_pos_loss) = self._calc_boundary_loss(end_entity_mention_labels, end_mention_logits,
                                                                          attention_mask)
        loss = (joint_loss + start_loss + end_loss) / 3
        return loss, {"mention_start_neg_loss": start_neg_loss,
                      "mention_start_pos_loss": start_pos_loss,
                      "mention_end_neg_loss": end_neg_loss,
                      "mention_end_pos_loss": end_pos_loss,
                      "mention_joint_neg_loss": joint_neg_loss,
                      "mention_joint_pos_loss": joint_pos_loss}

    def _prepare_antecedent_matrix(self, antecedent_labels):
        """
        :param antecedent_labels: [batch_size, seq_length, cluster_size]
        :return: [batch_size, seq_length, seq_length]
        """
        device = antecedent_labels.device
        batch_size, seq_length, cluster_size = antecedent_labels.size()

        # We now prepare a tensor with the gold antecedents for each span
        labels = torch.zeros(size=(batch_size, seq_length, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, seq_length,
                                                                                                cluster_size)
        seq_length_temp = torch.arange(seq_length, device=device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1,
                                                                                                    cluster_size)

        labels[batch_temp, seq_length_temp, antecedent_labels] = 1.0
        labels[:, :, -1] = 0.0  # Fix all pad-antecedents

        return labels

    def _compute_pos_neg_loss(self, weights, labels, logits):
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        all_loss = loss_fct(logits, labels)

        pos_weights = weights * labels
        pos_loss = (all_loss * pos_weights).sum() / (pos_weights.sum() + 1e-4)

        neg_weights = weights * (1 - labels)
        neg_loss = (all_loss * neg_weights).sum() / (neg_weights.sum() + 1e-4)

        loss = (1 - self.pos_coeff) * neg_loss + self.pos_coeff * pos_loss
        return loss, (neg_loss, pos_loss)

    def _calc_pruned_mention_masks(self, mention_logits):
        batch_size, seq_len, _ = mention_logits.size()
        device = mention_logits.device
        mention_logits = mention_logits.clone()
        mention_logits[mention_logits <= 0] = 0
        mention_logits[mention_logits > 0] = 1
        mask_indices = torch.nonzero(mention_logits)  # should have only indices that passed pruning ( > 0)

        start_indices = mask_indices[:, [0, 1]]
        start_mention_mask = torch.zeros((batch_size, seq_len), device=device)
        start_mention_mask[start_indices[:, 0], start_indices[:, 1]] = 1

        end_indices = mask_indices[:, [0, 2]]
        end_mention_mask = torch.zeros((batch_size, seq_len), device=device)
        end_mention_mask[end_indices[:, 0], end_indices[:, 1]] = 1

        return start_mention_mask, end_mention_mask

    def _compute_antecedent_loss(self, antecedent_labels, antecedent_logits, attention_mask, mention_mask):
        """
        :param antecedent_labels: [batch_size, seq_length, cluster_size]
        :param antecedent_logits: [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        """
        batch_size, seq_length = attention_mask.size()
        labels = self._prepare_antecedent_matrix(antecedent_labels)  # [batch_size, seq_length, seq_length]
        labels_mask = labels.clone().to(dtype=self.dtype)  # fp16 compatibility
        gold_antecedent_logits = antecedent_logits + ((1.0 - labels_mask) * -10000.0)
        gold_antecedent_logits = torch.clamp(gold_antecedent_logits, min=-10000.0, max=10000.0)

        if self.antecedent_loss == "allowed":
            only_non_null_labels = labels.clone()
            only_non_null_labels[:, :, 0] = 0

            non_null_loss_weights = torch.sum(only_non_null_labels, dim=-1)
            non_null_loss_weights[non_null_loss_weights != 0] = 1
            num_non_null_labels = torch.sum(non_null_loss_weights)

            only_null_labels = torch.zeros_like(labels)
            only_null_labels[:, :, 0] = 1
            only_null_labels = only_null_labels * labels
            null_loss_weights = torch.sum(only_null_labels, dim=-1)
            null_loss_weights[null_loss_weights != 0] = 1
            if self.prune_mention_for_antecedents:
                null_loss_weights = null_loss_weights * mention_mask
            num_null_labels = torch.sum(null_loss_weights)

            gold_log_sum_exp = torch.logsumexp(gold_antecedent_logits, dim=-1)  # [batch_size, seq_length]
            all_log_sum_exp = torch.logsumexp(antecedent_logits, dim=-1)  # [batch_size, seq_length]

            gold_log_probs = gold_log_sum_exp - all_log_sum_exp
            losses = -gold_log_probs
            losses = losses * attention_mask

            non_null_losses = losses * non_null_loss_weights
            denom_non_null = num_non_null_labels if self.normalize_antecedent_loss else batch_size
            non_null_loss = torch.sum(non_null_losses) / (denom_non_null + 1e-4)

            null_losses = losses * null_loss_weights
            denom_null = num_null_labels if self.normalize_antecedent_loss else batch_size
            null_loss = torch.sum(null_losses) / (denom_null + 1e-4)

            loss = self.pos_coeff * non_null_loss + (1 - self.pos_coeff) * null_loss
        else:  # == bce
            weights = torch.ones_like(labels).tril()
            weights[:, 0, 0] = 1
            attention_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)
            weights = weights * attention_mask

            # Compute pos-neg loss for all non-null antecedents
            non_null_weights = weights.clone()
            non_null_weights[:, :, 0] = 0
            non_null_loss = self._compute_pos_neg_loss(non_null_weights, labels, antecedent_logits)

            # Compute pos-neg loss for all null antecedents
            null_weights = weights.clone()
            null_weights[:, :, 1:] = 0
            null_loss = self._compute_pos_neg_loss(null_weights, labels, antecedent_logits)

            loss = self.pos_coeff * non_null_loss + (1 - self.pos_coeff) * null_loss

        return loss, {"antecedent_non_null_loss": non_null_loss, "antecedent_null_loss": null_loss}

    def mask_antecedent_logits(self, antecedent_logits):
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).triu(diagonal=1) * -10000.0  # [batch_size, seq_length, seq_length]
        antecedents_mask[:, 0, 0] = 0
        antecedent_logits = antecedent_logits + antecedents_mask  # [batch_size, seq_length, seq_length]
        antecedent_logits = torch.clamp(antecedent_logits, min=-10000.0, max=10000.0)
        return antecedent_logits

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

    def _get_encoder(self):
        if self.args.model_type == "longformer":
            return self.longformer
        elif self.args.model_type == "roberta":
            return self.roberta

        raise ValueError("Unsupported model type")

    def forward(self, input_ids, attention_mask=None, start_entity_mention_labels=None, end_entity_mention_labels=None,
                start_antecedent_labels=None, end_antecedent_labels=None, gold_clusters=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)
        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        # Entity mention scores
        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp, end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        if self.only_joint_mention_logits:
            mention_logits = joint_mention_logits
        elif self.no_joint_mention_logits:
            mention_logits = start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        else:
            mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)

        mention_mask = self._get_mention_mask(mention_logits)
        mention_mask = (1.0 - mention_mask) * -10000.0
        mention_logits = mention_logits + mention_mask
        mention_logits = torch.clamp(mention_logits, min=-10000.0, max=10000.0)

        # Antecedent scores
        temp = self.antecedent_start_classifier(start_coref_reps)  # [batch_size, seq_length, dim]
        start_coref_logits = torch.matmul(temp, start_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        start_coref_logits = self.mask_antecedent_logits(start_coref_logits)
        temp = self.antecedent_end_classifier(end_coref_reps)  # [batch_size, seq_length, dim]
        end_coref_logits = torch.matmul(temp, end_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        end_coref_logits = self.mask_antecedent_logits(end_coref_logits)
        if self.zero_null_logits:
            start_coref_logits[:, :, 0] = 0
            end_coref_logits[:, :, 0] = 0
        if return_all_outputs:
            outputs = (mention_logits, start_coref_logits, end_coref_logits)

        if start_entity_mention_labels is not None and end_entity_mention_labels is not None \
                and start_antecedent_labels is not None and end_antecedent_labels is not None:
            loss_dict = {}
            if self.seperate_mention_loss:
                entity_mention_loss, entity_losses = self._compute_seperate_entity_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    start_mention_logits=start_mention_logits,
                    end_mention_logits=end_mention_logits,
                    joint_mention_logits=joint_mention_logits,
                    attention_mask=attention_mask
                )
            else:
                entity_mention_loss, entity_losses = self._compute_joint_entity_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    mention_logits=mention_logits,
                    attention_mask=attention_mask)
            loss_dict.update(entity_losses)
            if self.prune_mention_for_antecedents:
                start_mention_mask, end_mention_mask = self._calc_pruned_mention_masks(mention_logits)
            else:
                start_mention_mask, end_mention_mask = None, None
            start_coref_loss, start_antecedent_losses = self._compute_antecedent_loss(antecedent_labels=start_antecedent_labels,
                                                             antecedent_logits=start_coref_logits,
                                                             attention_mask=attention_mask,
                                                             mention_mask=start_mention_mask)
            start_antecedent_losses = {"start_" + key: val for key, val in start_antecedent_losses.items()}
            loss_dict.update(start_antecedent_losses)
            end_coref_loss, end_antecedent_losses = self._compute_antecedent_loss(antecedent_labels=end_antecedent_labels,
                                                           antecedent_logits=end_coref_logits,
                                                           attention_mask=attention_mask,
                                                           mention_mask=end_mention_mask)
            end_antecedent_losses = {"end_" + key: val for key, val in end_antecedent_losses.items()}
            loss_dict.update(end_antecedent_losses)
            loss = entity_mention_loss + start_coref_loss + end_coref_loss

            loss_dict["entity_mention_loss"] = entity_mention_loss
            loss_dict["start_coref_loss"] = start_coref_loss
            loss_dict["end_coref_loss"] = end_coref_loss
            loss_dict["loss"] = loss

            outputs = (loss,) + outputs + (loss_dict,)

        return outputs


class EndToEndCoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args, antecedent_loss, max_span_length, seperate_mention_loss,
                 prune_mention_for_antecedents, normalize_antecedent_loss, only_joint_mention_logits,
                 no_joint_mention_logits, pos_coeff, independent_mention_loss, normalise_loss,
                 num_neighboring_antecedents):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.antecedent_loss = antecedent_loss  # can be either allowed loss or bce
        self.max_span_length = max_span_length
        self.top_lambda = args.top_lambda
        # self.seperate_mention_loss = seperate_mention_loss
        # self.prune_mention_for_antecedents = prune_mention_for_antecedents
        # self.normalize_antecedent_loss = normalize_antecedent_loss
        self.only_joint_mention_logits = only_joint_mention_logits
        self.no_joint_mention_logits = no_joint_mention_logits
        # self.pos_coeff = pos_coeff
        self.independent_mention_loss = independent_mention_loss
        self.normalise_loss = normalise_loss
        self.num_neighboring_antecedents = num_neighboring_antecedents
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size, args.dropout_prob)

        self.entity_mention_start_classifier = nn.Linear(config.hidden_size, 1)  # In paper w_s
        self.entity_mention_end_classifier = nn.Linear(config.hidden_size, 1)  # w_e
        self.entity_mention_joint_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # M

        self.antecedent_start_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # S
        self.antecedent_end_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # E

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
        if self.num_neighboring_antecedents <= 0:
            return None

        tmp = torch.ones((batch_size, max_k, max_k), device=self.device)
        return tmp.tril(diagonal=-1).triu(diagonal=-self.num_neighboring_antecedents)

    def _prune_top_lambda_spans(self, mention_logits, attention_mask):
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

        topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k).indices  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        span_starts = sorted_topk_1d_indices // seq_length  # [batch_size, max_k]
        span_ends = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        new_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                            span_starts, span_ends]  # [batch_size, max_k]

        return span_starts, span_ends, span_mask, new_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask, neighboring_antecedent_mask=None):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        if neighboring_antecedent_mask is not None:
            antecedents_mask = antecedents_mask * neighboring_antecedent_mask
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]

        antecedent_logits = antecedent_logits + (1 - antecedents_mask) * -1e4  # [batch_size, seq_length, seq_length]
        antecedent_logits = torch.clamp(antecedent_logits, min=-10000.0, max=10000.0)

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
        # TODO yuval
        # needs to have a null place and to add 0 to logits for null and to add to labels.
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


    def _compute_joint_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits,
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


    def forward(self, input_ids, attention_mask=None,  start_entity_mention_labels=None, end_entity_mention_labels=None,
                start_antecedent_labels=None, end_antecedent_labels=None, gold_clusters=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)
        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        # Entity mention scores
        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        if self.only_joint_mention_logits:
            mention_logits = joint_mention_logits
        elif self.no_joint_mention_logits:
            mention_logits = start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        else:
            mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)

        mention_mask = self._get_mention_mask(mention_logits)
        mention_mask = (1.0 - mention_mask) * -10000.0
        mention_logits = mention_logits + mention_mask
        mention_logits = torch.clamp(mention_logits, min=-10000.0, max=10000.0)

        span_starts, span_ends, span_mask, top_k_mention_logits = self._prune_top_lambda_spans(mention_logits, attention_mask)

        batch_size,  _, dim = start_coref_reps.size()
        max_k = span_starts.size(-1)
        size = (batch_size, max_k, dim)

        top_k_start_coref_reps = torch.gather(start_coref_reps, dim=1, index=span_starts.unsqueeze(-1).expand(size))
        top_k_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=span_ends.unsqueeze(-1).expand(size))
        # Antecedent scores
        temp = self.antecedent_start_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_start_coref_logits = torch.matmul(temp,
                                          top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]
        temp = self.antecedent_end_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_end_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        top_k_mention_logits = top_k_mention_logits.unsqueeze(-1) + top_k_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]
        coref_logits = top_k_mention_logits + top_k_start_coref_logits + top_k_end_coref_logits  # [batch_size, max_k, max_k]

        neighboring_antecedents_mask = self._get_neighboring_antecedent_mask(batch_size, max_k)
        coref_logits = self._mask_antecedent_logits(coref_logits, span_mask, neighboring_antecedents_mask)

        # adding zero logits for null span
        coref_logits = torch.cat((coref_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1) # [batch_size, max_k, max_k + 1]
        outputs = (span_starts, span_ends, coref_logits, mention_logits)
        if gold_clusters is not None:
            losses = {}
            loss = 0.0
            if self.independent_mention_loss:
                entity_mention_loss, entity_losses = self._compute_joint_entity_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    mention_logits=mention_logits,
                    attention_mask=attention_mask)
                loss += entity_mention_loss
                losses.update(entity_losses)
            labels_after_pruning = self._get_cluster_labels_after_pruning(span_starts, span_ends, gold_clusters)
            end_to_end_loss = self._get_marginal_log_likelihood_loss(coref_logits, labels_after_pruning, span_mask)
            loss += end_to_end_loss
            losses.update({"loss": loss})
            outputs = (loss,) + outputs + (losses,)

        return outputs