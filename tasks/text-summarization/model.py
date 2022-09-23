from argparse import Namespace
import torch
import torch.nn as nn
from models.bart import BartModel
from typing import Any, Dict
from metrics import tokens_accuracy


class TSBart(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.model = BartModel.from_pretrained()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        encoder_masks: torch.Tensor,
        decoder_tokens: torch.Tensor,
        decoder_masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        logits = self.model(
            encoder_tokens, encoder_masks, decoder_tokens, decoder_masks
        )
        loss = self.loss_fn(logits.view(-1, self.model.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}

        return {"loss": loss, "logits": logits, "stats": stats}

    def summarize(
        self, tokens: torch.Tensor, beam_size: int, max_length: int
    ) -> Dict[str, Any]:
        return self.model.decode(tokens, beam_size, max_length)
