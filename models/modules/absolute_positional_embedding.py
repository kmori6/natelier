import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        l, d = hs.size()[1:]
        pe = hs.new_zeros(l, d)  # (L, D)
        pos = torch.arange(0, l).unsqueeze(1)  # (1, L)
        pe[:, 0::2] = torch.sin(pos / torch.pow(10000, torch.arange(0, d, 2) / d))
        pe[:, 1::2] = torch.cos(pos / torch.pow(10000, torch.arange(0, d, 2) / d))
        return self.dropout(hs + pe.unsqueeze(0))
