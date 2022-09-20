from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from models.albert import AlbertModel
from metrics import tokens_accuracy


class NERAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.num_labels = args.num_labels
        self.encoder = AlbertModel.from_pretrained()
        self.classifier = nn.Linear(self.encoder.d_model, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        segments: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs = self.encoder(tokens, masks, segments)
        logits = self.classifier(hs)

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}

        return {"loss": loss, "logits": logits, "stats": stats}
