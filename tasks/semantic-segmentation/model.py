from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from models.modules.lightweight_all_mlp_decoder import LightweightAllMlpDecoder
from models.segformer import SegformerModel
from metrics import tokens_accuracy


class SSSegformer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained()
        self.decoder = LightweightAllMlpDecoder(
            hidden_size=self.encoder.d_model[-1],
            num_labels=1000,
            encoder_d_model=self.encoder.d_model,
            hight=args.hight,
            width=args.width,
        )
        self.upsampler = nn.Upsample((args.height, args.width), mode="bilinear")
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.semantic_loss_ignore_index
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        hte_hs = self.encoder(images)
        logits = self.decoder(hte_hs)  # (B, C, H/4, W/4)
        logits = self.upsampler(logits)  # (B, C, H, W)

        loss = self.loss_fn(logits, labels)
        stats = {
            "loss": loss.item(),
            "acc": tokens_accuracy(logits.argmax(1), labels, ignore_id=255),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
