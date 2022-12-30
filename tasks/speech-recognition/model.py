from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from espnet.nets.ctc_prefix_score import CTCPrefixScore

from loss.cross_entropy import CrossEntropyLoss
from metrics import tokens_accuracy
from models.distilhubert import DistilHubert
from models.transformer import Transformer
from outputs import ModelOutputs


class DistilHubertTransformerDecoderModel(Transformer):
    def __init__(self, args: Namespace):
        super().__init__(
            vocab_size=args.vocab_size,
            d_model=768,
            d_ff=3072,
            num_attention_heads=12,
            num_encoder_layers=2,
            num_decoder_layers=args.decoder_layers,
            dropout_rate=0.1,
            padding_id=3,
            bos_id=1,
            eos_id=2,
        )
        self.ctc_blank_id = 0
        self.ctc_loss_weight = args.ctc_loss_weight
        self.encoder = DistilHubert.from_pretrained()
        self.ctc_classifier = nn.Linear(self.d_model, args.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(reduction="sum", zero_infinity=True)
        self.att_loss_fn = CrossEntropyLoss("sum", args.label_smoothing)
        self.encoder.freeze_convolution()

    def forward(
        self,
        encoder_waves: torch.Tensor,
        encoder_wave_lens: torch.Tensor,
        encoder_labels: torch.Tensor,
        encoder_label_lens: torch.Tensor,
        decoder_tokens: torch.Tensor,
        decoder_masks: torch.Tensor,
        decoder_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        # encoder
        encoder_hs, encoder_hs_lens, encoder_masks = self.encoder(
            encoder_waves, encoder_wave_lens
        )
        stats = dict()
        # decoder
        if self.ctc_loss_weight < 1.0:
            att_logits = self.decoder(
                decoder_tokens, decoder_masks, encoder_hs, encoder_masks
            )
            loss_att = self.att_loss_fn(
                att_logits.view(-1, self.vocab_size), decoder_labels.view(-1)
            ) / encoder_waves.size(0)
            stats["loss_att"] = loss_att.item()
            stats["att_acc"] = tokens_accuracy(att_logits.argmax(-1), decoder_labels)
        else:
            loss_att = 0
        # ctc
        if self.ctc_loss_weight > 0.0:
            ctc_logits = self.ctc_classifier(encoder_hs).transpose(0, 1)  # (T, B, C)
            ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
            encoder_labels = encoder_labels[encoder_labels >= 0]  # ignore -100 tokens
            loss_ctc = self.ctc_loss_fn(
                ctc_log_probs, encoder_labels, encoder_hs_lens, encoder_label_lens
            ) / encoder_waves.size(0)
            stats["loss_ctc"] = loss_ctc.item()
        else:
            loss_ctc = 0
        loss = self.ctc_loss_weight * loss_ctc + (1 - self.ctc_loss_weight) * loss_att
        stats["loss"] = loss.item()
        return ModelOutputs(loss, stats, att_logits)

    def recognize(
        self,
        encoder_input_waves: torch.Tensor,
        encoder_input_wave_lens: torch.Tensor,
        beam_size: int = 10,
        ctc_weight: float = 0.3,
    ) -> Dict[str, Any]:

        # encoder
        fs_pad, fs_lens = self.feature_extractor(
            encoder_input_waves, encoder_input_wave_lens
        )
        fs_pad, fs_lens = self.normalizer(fs_pad, fs_lens)
        hs = self.encoder(fs_pad, fs_lens)[0]
        self.ctc_prefix_score = CTCPrefixScore(
            torch.log_softmax(self.ctc_classifier(hs), dim=-1)[0].cpu().numpy(),
            blank=0,
            eos=self.eos_token_id,
            xp=np,
        )

        # initial stats
        running_stats = [
            {
                "score": 0.0,
                "tokens": [self.bos_token_id],
                "ctc_state": self.ctc_prefix_score.initial_state(),
                "ctc_score": 0.0,
            }
        ]
        final_stats = []

        # beam search
        max_length = hs.size(1)
        for i in range(1, max_length):
            # decoder
            decoder_input_tokens = fs_lens.new_zeros(beam_size, i)
            for beam_idx, stat in enumerate(running_stats):
                decoder_input_tokens[beam_idx, :] = fs_lens.new_tensor(stat["tokens"])
            next_token_scores = self.decoder.forward_one_step(
                decoder_input_tokens,
                fs_lens.new_ones(beam_size, i, i).tril(0),
                hs.repeat_interleave(beam_size, dim=0),
            )[0]
            # scoring
            aggregator = []
            for beam_idx, stat in enumerate(running_stats):
                att_scores, att_next_tokens = torch.topk(
                    next_token_scores[beam_idx], beam_size, dim=0
                )
                ctc_scores, ctc_states = self.ctc_prefix_score(
                    stat["tokens"], att_next_tokens.cpu(), stat["ctc_state"]
                )
                scores = (
                    ctc_weight * torch.from_numpy(ctc_scores - stat["ctc_score"])
                    + (1.0 - ctc_weight) * att_scores.cpu()
                )
                next_scores, joint_next_tokens = torch.topk(scores, beam_size, dim=0)
                next_tokens = att_next_tokens[joint_next_tokens]
                for topk_idx in range(beam_size):
                    aggregator.append(
                        {
                            "score": stat["score"] + next_scores[topk_idx].item(),
                            "tokens": stat["tokens"] + [next_tokens[topk_idx].item()],
                            "ctc_state": ctc_states[joint_next_tokens[topk_idx]],
                            "ctc_score": ctc_scores[joint_next_tokens[topk_idx]],
                        }
                    )
            running_stats = sorted(aggregator, key=lambda x: x["score"], reverse=True)[
                :beam_size
            ]

            # add eos_token_id
            if i == max_length - 1:
                for stat in running_stats:
                    stat["tokens"].append(self.eos_token_id)

            # sort stats
            keep_stats = []
            for stat in running_stats:
                if stat["tokens"][-1] == self.eos_token_id:
                    final_stats.append(stat)
                else:
                    keep_stats.append(stat)
            running_stats = keep_stats

            # stop search
            if len(running_stats) < 1 or len(final_stats) >= beam_size:
                break

        return sorted(final_stats, key=lambda x: x["score"], reverse=True)[0]
