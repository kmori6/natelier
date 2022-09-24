from argparse import Namespace
import torch
import torch.nn as nn
from transformers import BartModel as HuggingFaceBartModel
from models.bart import BartModel
from typing import Any, Dict
from metrics import tokens_accuracy


class DBart(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.model = BartModel.from_pretrained()
        self.initialize_embeddings(args.vocab_size)
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
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}

        return {"loss": loss, "logits": logits, "stats": stats}

    def initialize_embeddings(self, vocab_size: int):
        hf_model = HuggingFaceBartModel.from_pretrained("facebook/bart-base")
        common_token_embedding = hf_model.shared
        self.model.encoder.embed_tokens = common_token_embedding
        self.model.decoder.embed_tokens = common_token_embedding
        self.model.decoder.classifier = nn.Linear(
            self.model.d_model, vocab_size, bias=False
        )
        self.model.decoder.classifier.weight = common_token_embedding.weight

    def response(
        self, tokens: torch.Tensor, beam_size: int, max_length: int
    ) -> Dict[str, Any]:
        return self.model.decode(tokens, beam_size, max_length)
