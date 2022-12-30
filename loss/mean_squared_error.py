import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """mean squared error loss

        Args:
            logits (torch.Tensor): Logits (B)
            labels (torch.Tensor): Labels (B)

        """
        return torch.mean((logits - labels) ** 2)
