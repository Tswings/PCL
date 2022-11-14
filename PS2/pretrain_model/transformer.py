import torch.nn as nn

from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from position_ffn import PositionWiseFeedForward


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, head_num, dropout, feedforward_size):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, head_num, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = LayerNormalization(hidden_size)
        self.ffw = PositionWiseFeedForward(hidden_size, feedforward_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = LayerNormalization(hidden_size)

    def forward(self, x, mask):
        """
        forward
        :param x: sequence input [batch_size * seq_length * hidden_size]
        :param mask: mask: [batch_size * 1 * seq_length * seq_length]
        :return: [batch_size * seq_length * hidden_size]
        """
        inner = self.dropout_1(self.self_attention(x, x, x, mask))
        inner = self.layer_norm_1(inner + x)
        # output = self.dropout_2(self.ffw(inner))
        # output = self.layer_norm_2(output + inner)
        return inner
