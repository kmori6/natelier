from argparse import Namespace
import torch
import torch.nn as nn
from typing import Dict, Any
from models.albert import AlbertModel
from metrics import single_label_accuracy, spearman_correlation


class TCAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.num_labels = args.num_labels
        self.task = args.task
        self.encoder = AlbertModel.from_pretrained()
        self.classifier = nn.Linear(self.encoder.d_model, self.num_labels)
        if args.task == "single_label_classification":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        segments: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs_cls = self.encoder(tokens, masks, segments)[:, 0, :]
        logits = self.classifier(hs_cls)

        stats = dict()
        if self.task == "single_label_classification":
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels)
            stats["acc"] = single_label_accuracy(logits.argmax(-1), labels)
        else:
            loss = self.loss_fn(logits.view(-1), labels)
            stats["sc"] = spearman_correlation(
                logits.view(-1).detach().cpu(), labels.cpu()
            )
        stats["loss"] = loss.item()

        return {"loss": loss, "logits": logits, "stats": stats}
