import torch.nn as nn
from transformers.activations import gelu


class PositionWiseFeedForward(nn.Module):
    """ Feed forward layer"""
    def __init__(self, hidden_size, feedforward_size):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)

    def forward(self, x):
        inner = gelu(self.linear_1(x))
        output = self.linear_2(inner)
        return output
