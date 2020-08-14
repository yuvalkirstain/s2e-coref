import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm
from transformers import BertPreTrainedModel, LongformerModel, LongformerConfig
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


class LongformerForCoreferenceResolution(BertPreTrainedModel):
    config_class = LongformerConfig
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config)

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

    def _compute_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits, attention_mask=None):
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
        full_entity_mentions = torch.zeros(size=(batch_size, seq_length, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(num_mentions)
        full_entity_mentions[batch_temp, start_entity_mention_labels, end_entity_mention_labels] = 1.0  # [batch_size, seq_length, seq_length]
        full_entity_mentions[:, 0, 0] = 0.0  # Remove the padded mentions

        weights = (attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2))

        loss_fct = nn.BCEWithLogitsLoss(weight=weights)
        loss = loss_fct(mention_logits, full_entity_mentions)

        return loss

    def forward(self, input_ids, attention_mask=None, entity_mentions=None,
                start_antecedents=None, end_antecedents=None, return_all_outputs=False):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)
        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp, end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)

        temp = self.antecedent_start_classifier(start_coref_reps)  # [batch_size, seq_length, dim]
        start_coref_logits = torch.matmul(temp, start_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        temp = self.antecedent_end_classifier(end_coref_reps)  # [batch_size, seq_length, dim]
        end_coref_logits = torch.matmul(temp, end_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        if entity_mentions is not None and start_antecedents is not None and end_antecedents is not None:


        pass
