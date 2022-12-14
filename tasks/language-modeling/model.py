from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from models.albert import AlbertModel
from metrics import ppl


class LMAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained()
        self.vocab_size = self.encoder.vocab_size
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.d_model, self.encoder.embedding_size),
            nn.Linear(self.encoder.embedding_size, self.vocab_size, bias=False),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.initialize_classifier()

    def initialize_classifier(self):
        self.classifier[1].weight = self.encoder.embedding.token_embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        segments: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs = self.encoder(tokens, masks, segments)
        logits = self.classifier(hs)

        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "ppl": ppl(loss.item())}

        return {"loss": loss, "logits": logits, "stats": stats}
