from typing import List, Tuple
import torch
import torch.nn as nn


class LightweightAllMlpDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        encoder_d_model: List[int],
        height: int,
        width: int,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.linear1 = nn.ModuleList(
            [nn.Linear(size, hidden_size) for size in encoder_d_model]
        )
        self.upsample = nn.Upsample((int(height / 4), int(width / 4)), mode="bilinear")
        self.linear2 = nn.Sequential(
            nn.Conv2d(
                hidden_size * len(encoder_d_model),
                hidden_size,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.linear3 = nn.Conv2d(hidden_size, num_labels, kernel_size=1)

    def forward(self, hte_hs: Tuple[torch.Tensor]) -> torch.Tensor:
        hs_list = []
        for hs, linear in zip(hte_hs, self.linear1):
            # step1: linear
            b, d, h, w = hs.size()
            hs = hs.view(b, d, -1).transpose(1, 2)  # (B, D, H/*, W/*) -> (B, H/*W/*, D)
            hs = linear(hs)
            hs = hs.transpose(1, 2).view(b, -1, h, w)
            # step2: upsample
            hs = self.upsample(hs)
            hs_list.append(hs)
        # step3: linear
        hs = torch.concat(hs_list, dim=1)  # (B, 4D, H/4, W/4)
        hs = self.linear2(hs)
        # step4: linear
        logits = self.linear3(hs)  # (B, C, H/4, W/4)
        return logits
