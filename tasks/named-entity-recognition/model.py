from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
from metrics import tokens_accuracy


class NERAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.num_labels = args.num_labels
        self.config = AlbertConfig.from_pretrained(args.model_name)
        self.encoder = AlbertModel.from_pretrained(
            args.model_name, add_pooling_layer=False
        )
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs_pad = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]
        logits = self.classifier(hs_pad)

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}

        return {"loss": loss, "logits": logits, "stats": stats}
