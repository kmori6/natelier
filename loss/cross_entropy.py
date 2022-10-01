import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """cross entropy loss

        Args:
            logits (torch.Tensor): Logits (B, C)
            labels (torch.Tensor): Labels (B)

        """
        num_classes = logits.size(1)
        logits = logits[labels != self.ignore_index]
        labels = labels[labels != self.ignore_index]
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, C)
        onehot_labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(-1), 1.0)
        if self.label_smoothing > 0:
            onehot_labels = (
                onehot_labels * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        loss = -(log_probs * onehot_labels).sum(-1)  # (B)
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
