from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from metrics import single_label_accuracy


class ICCnn(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.num_labels = args.num_labels
        self.encoder = nn.Sequential(
            nn.Conv2d(args.idim, args.hidden_size, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
        )
        self.h_out = (args.h_size - (3 - 1) - 1) // 2 + 1
        self.w_out = (args.w_size - (3 - 1) - 1) // 2 + 1
        self.classifier = nn.Linear(
            args.hidden_size * self.h_out * self.w_out, args.num_labels
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:

        hs = self.encoder(images)
        logits = self.classifier(hs.view(hs.size(0), -1))

        loss = self.loss_fn(logits, labels.view(-1))
        stats = {
            "loss": loss.item(),
            "acc": single_label_accuracy(logits.argmax(-1), labels),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
