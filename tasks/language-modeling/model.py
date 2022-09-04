from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig
from metrics import ppl


class LMAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.config = AlbertConfig.from_pretrained(args.model_name)
        self.encoder = AlbertModel.from_pretrained(
            args.model_name, add_pooling_layer=False
        )
        self.classifier = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs_pad = self.encoder(input_ids, attention_mask, token_type_ids)[0]
        logits = self.classifier(hs_pad)

        loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "ppl": ppl(loss.item())}

        return {"loss": loss, "logits": logits, "stats": stats}
