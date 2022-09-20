from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from models.albert import AlbertModel
from metrics import single_label_accuracy


class MCAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained()
        self.classifier = nn.Linear(self.encoder.d_model, 1)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        segments: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        batch_size, num_choices, length = tokens.size()
        hs_cls = self.encoder(
            tokens.view(batch_size * num_choices, length),
            masks.view(batch_size * num_choices, length),
            segments.view(batch_size * num_choices, length),
        )[:, 0, :]
        logits = self.classifier(hs_cls).view(batch_size, num_choices)

        loss = self.loss_fn(logits, labels.view(-1))
        stats = {
            "loss": loss.item(),
            "acc": single_label_accuracy(logits.argmax(-1), labels),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
