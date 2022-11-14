import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.per_head_size = hidden_size // head_num
        self.ws = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        :param key: [batch_size * seq_length * hidden_size]
        :param value: [batch_size * seq_length * hidden_size]
        :param query: [batch_size * seq_length * hidden_size]
        :param mask: [batch_size * 1 * seq_length * seq_length]
        :return: [batch_size * seq_length * hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        head_num = self.head_num
        per_head_size = self.per_head_size

        query, key, value = [w(x).view(batch_size, -1, head_num, per_head_size).transpose(1, 2)
                             for w, x in zip(self.ws, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=1)(scores)
        probs = self.dropout(probs)
        output = torch.matmul(probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.output(output)
        return output