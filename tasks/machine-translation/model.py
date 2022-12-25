from argparse import Namespace
from typing import Any, Dict

import torch
import torch.nn as nn

from loss.cross_entropy import CrossEntropyLoss
from metrics import tokens_accuracy
from models.mbart import MbartModel
from outputs import TrainOutputs


class NMTBart(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.model = MbartModel.from_pretrained()
        self.model.freeze_embeddings()
        self.loss_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}
        return TrainOutputs(loss, stats)

    def translate(
        self, tokens: torch.Tensor, beam_size: int, max_length: int
    ) -> Dict[str, Any]:
        return self.model.decode(tokens, beam_size, max_length)
