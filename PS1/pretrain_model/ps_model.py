import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import math

from transformers import ElectraModel



def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class ElectraForParagraphClassification(ElectraModel):
    def __init__(self, config):
        super(ElectraForParagraphClassification, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        cls_output = outputs[0]
        cls_output = torch.index_select(cls_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits

        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits

