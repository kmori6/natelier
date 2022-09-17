from argparse import Namespace
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from metrics import tokens_accuracy


class Decoder(nn.Module):
    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        encoder_hidden_sizes: List[int],
        height: int,
        width: int,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.linear1 = nn.ModuleList(
            [nn.Linear(size, hidden_size) for size in encoder_hidden_sizes]
        )
        self.upsample = nn.Upsample((int(height / 4), int(width / 4)), mode="bilinear")
        self.linear2 = nn.Sequential(
            nn.Conv2d(
                hidden_size * len(encoder_hidden_sizes),
                hidden_size,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.linear3 = nn.Conv2d(hidden_size, num_labels, kernel_size=1)

    def forward(self, hac_hs: Tuple[torch.Tensor]) -> torch.Tensor:
        hs_list = []
        for hs, linear in zip(hac_hs, self.linear1):
            # step1: linear
            b, d, h, w = hs.size()
            hs = hs.view(b, d, -1).transpose(1, 2)  # (B, D, H/*, W/*) -> (B, H/*W/*, D)
            hs = linear(hs)
            hs = hs.transpose(1, 2).view(b, -1, h, w)
            # step2: upsample
            hs = self.upsample(hs)
            hs_list.append(hs)
        # step3: linear
        hs = torch.concat(hs_list, dim=1)  # (B, 4D, H/4, W/4)
        hs = self.linear2(hs)
        # step4: linear
        logits = self.linear3(hs)  # (B, C, H/4, W/4)
        return logits


class SSSegformer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.config = SegformerConfig.from_pretrained(args.model_name)
        self.encoder = SegformerModel.from_pretrained(args.model_name)
        self.decoder = Decoder(
            self.config.num_labels,
            self.config.decoder_hidden_size,
            self.config.hidden_sizes,
            args.height,
            args.width,
        )
        self.upsampler = nn.Upsample((args.height, args.width), mode="bilinear")
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.semantic_loss_ignore_index
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        hs = self.encoder(images, output_hidden_states=True)[1]
        logits = self.decoder(hs)  # (B, C, H/4, W/4)
        logits = self.upsampler(logits)  # (B, C, H, W)

        loss = self.loss_fn(logits, labels)
        stats = {
            "loss": loss.item(),
            "acc": tokens_accuracy(logits.argmax(1), labels, ignore_id=255),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
