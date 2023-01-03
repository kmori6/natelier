from argparse import Namespace
from typing import Any, Dict

import torch

from loss.cross_entropy import CrossEntropyLoss
from metrics import tokens_accuracy
from models.mbart import Mbart
from outputs import ModelOutputs


class NMTBart(Mbart):
    def __init__(self, args: Namespace):
        super().__init__(load_pretrained_weight=True)
        self.freeze_embeddings()
        self.loss_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        encoder_masks: torch.Tensor,
        decoder_tokens: torch.Tensor,
        decoder_masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        logits = super().forward(
            encoder_tokens, encoder_masks, decoder_tokens, decoder_masks
        )
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}
        return ModelOutputs(loss, stats)
