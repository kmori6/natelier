from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
from metrics import single_label_accuracy


class MCAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.config = AlbertConfig.from_pretrained(args.model_name)
        self.encoder = AlbertModel.from_pretrained(
            args.model_name, add_pooling_layer=False
        )
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        bs, nc, lmax = input_ids.size()
        hs_cls = self.encoder(
            input_ids.view(bs * nc, lmax),
            attention_mask.view(bs * nc, lmax),
            token_type_ids.view(bs * nc, lmax),
        )[0][:, 0, :]
        logits = self.classifier(hs_cls).view(bs, nc)

        loss = self.loss_fn(logits, labels.view(-1))
        stats = {
            "loss": loss.item(),
            "acc": single_label_accuracy(logits.argmax(-1), labels),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
