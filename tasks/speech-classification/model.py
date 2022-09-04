from argparse import Namespace
from typing import Dict, Any
import torch
import torch.nn as nn
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from metrics import single_label_accuracy


class SCTransformer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.num_labels = args.num_labels
        self.feature_extractor = DefaultFrontend()
        self.encoder = TransformerEncoder(
            input_size=self.feature_extractor.output_size(),
            output_size=args.hidden_size,
            attention_heads=args.encoder_attention_heads,
            linear_units=args.encoder_linear_units,
            num_blocks=args.encoder_num_blocks,
            dropout_rate=args.encoder_dropout_rate,
            positional_dropout_rate=args.encoder_positional_dropout_rate,
            attention_dropout_rate=args.encoder_attention_dropout_rate,
            input_layer=args.encoder_input_layer,
        )
        self.classifier = nn.Linear(args.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        input_ws: torch.Tensor,
        input_lens: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        fs_pad, fs_lens = self.feature_extractor(input_ws, input_lens)
        hs_pad = self.encoder(fs_pad, fs_lens)[0]
        logits = self.classifier(hs_pad.mean(dim=1))

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels)
        stats = {
            "loss": loss.item(),
            "acc": single_label_accuracy(logits.argmax(-1), labels),
        }

        return {"loss": loss, "logits": logits, "stats": stats}
