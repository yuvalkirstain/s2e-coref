import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm
from transformers import BertPreTrainedModel, LongformerModel, RobertaModel, RobertaConfig
from transformers.modeling_bert import ACT2FN


class FullyConnectedLayer(Module):
    # TODO: many layers
    def __init__(self, config, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        return temp


class CoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args, antecedent_loss, max_span_length, seperate_mention_loss,
                 prune_mention_for_antecedents, normalize_antecedent_loss, only_joint_mention_logits, no_joint_mention_logits, pos_coeff):
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
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)

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
        labels[
            batch_temp, start_entity_mention_labels, end_entity_mention_labels] = 1.0  # [batch_size, seq_length, seq_length]
        labels[:, 0, 0] = 0.0  # Remove the padded mentions

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
        pos_weights = weights * labels
        per_example_pos_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        per_example_pos_loss = per_example_pos_loss_fct(logits, labels)
        pos_loss = (per_example_pos_loss * pos_weights).sum() / (pos_weights.sum() + 1e-4)

        neg_weights = weights * (1 - labels)
        per_example_neg_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        per_example_neg_loss = per_example_neg_loss_fct(logits, labels)
        neg_loss = (per_example_neg_loss * neg_weights).sum() / (neg_weights.sum() + 1e-4)

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
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).triu(
            diagonal=1) * -10000.0  # [batch_size, seq_length, seq_length]
        antecedents_mask = torch.clamp(antecedents_mask, min=-10000.0, max=10000.0)
        antecedents_mask[:, 0, 0] = 0
        antecedent_logits = antecedent_logits + antecedents_mask  # [batch_size, seq_length, seq_length]
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
                start_antecedent_labels=None, end_antecedent_labels=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)
        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        # Entity mention scores
        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(
            -1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(
            -1)  # [batch_size, seq_length]

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
        mention_mask = torch.clamp(mention_mask, min=-10000.0, max=10000.0)
        mention_logits = mention_logits + mention_mask

        # Antecedent scores
        temp = self.antecedent_start_classifier(start_coref_reps)  # [batch_size, seq_length, dim]
        start_coref_logits = torch.matmul(temp,
                                          start_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        start_coref_logits = self.mask_antecedent_logits(start_coref_logits)
        temp = self.antecedent_end_classifier(end_coref_reps)  # [batch_size, seq_length, dim]
        end_coref_logits = torch.matmul(temp, end_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        end_coref_logits = self.mask_antecedent_logits(end_coref_logits)

        outputs = outputs[2:]
        if return_all_outputs:
            outputs = (mention_logits, start_coref_logits, end_coref_logits) + outputs

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
