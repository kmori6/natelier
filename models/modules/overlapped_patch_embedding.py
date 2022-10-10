import torch
import torch.nn as nn


class OverlappedPatchMerging(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        padding_size: int,
    ):
        super().__init__()
        self.feats_embedding = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding_size,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, hs: torch.Tensor, hight: int, width: int) -> torch.Tensor:
        b = hs.size(0)
        hs = self.feats_embedding(hs)  # (B, C', H', W')
        hs = hs.view(b, -1, hight * width).transpose(1, 2)
        hs = self.norm(hs)  # (B, L', C')
        return hs
