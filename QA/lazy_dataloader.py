import torch
import numpy as np
from torch.utils.data import Dataset


class LazyLoadTensorDataset(Dataset):

    def __init__(self, features, is_training):
        """
        Args:
            features: feature list
        """
        self.features = features
        self.is_training = is_training

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        sent_mask = torch.tensor(feature.sent_mask, dtype=torch.long)
        content_len = torch.tensor(feature.content_len, dtype=torch.long)
        pq_end_pos = torch.tensor(feature.pq_end_pos, dtype=torch.long)

        tensors = [input_ids,
                   input_mask,
                   segment_ids,
                   sent_mask,
                   content_len,
                   # word_sim_matrix
                   pq_end_pos
                   ]

        if self.is_training:
            start_positions = torch.tensor(feature.start_position, dtype=torch.long)
            end_positions = torch.tensor(feature.end_position, dtype=torch.long)
            sent_lbs = torch.tensor(feature.sent_lbs, dtype=torch.long)
            sent_weight = torch.tensor(feature.sent_weight, dtype=torch.float)
            tensors.append(start_positions)
            tensors.append(end_positions)
            tensors.append(sent_lbs)
            tensors.append(sent_weight)
        else:
            example_index = torch.tensor(index, dtype=torch.long)
            tensors.append(example_index)
        return tensors

    def __len__(self):
        return len(self.features)
