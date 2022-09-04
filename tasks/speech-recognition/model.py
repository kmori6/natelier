from argparse import Namespace
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from metrics import tokens_accuracy


class ASRTransformer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.bos_token_id = args.vocab_size - 1
        self.eos_token_id = args.vocab_size - 1
        self.vocab_size = args.vocab_size
        self.ctc_loss_weight = args.ctc_loss_weight

        self.feature_extractor = DefaultFrontend()
        self.spec_augmentor = SpecAug(
            time_warp_window=args.specaug_time_warp_window,
            time_warp_mode=args.specaug_time_warp_mode,
            freq_mask_width_range=args.specaug_freq_mask_width_range,
            num_freq_mask=args.specaug_num_freq_mask,
            time_mask_width_range=args.specaug_time_mask_width_range,
            num_time_mask=args.specaug_num_time_mask,
        )
        self.normalizer = UtteranceMVN(norm_means=True, norm_vars=True)
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
        self.decoder = TransformerDecoder(
            vocab_size=args.vocab_size,
            encoder_output_size=args.hidden_size,
            attention_heads=args.decoder_attention_heads,
            linear_units=args.decoder_linear_units,
            num_blocks=args.decoder_num_blocks,
            dropout_rate=args.decoder_dropout_rate,
            positional_dropout_rate=args.decoder_positional_dropout_rate,
            self_attention_dropout_rate=args.decoder_self_attention_dropout_rate,
            src_attention_dropout_rate=args.decoder_src_attention_dropout_rate,
        )
        self.ctc_classifier = nn.Linear(args.hidden_size, args.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(reduction="sum", zero_infinity=True)
        self.att_loss_fn = nn.CrossEntropyLoss(
            reduction="sum", label_smoothing=args.label_smoothing
        )

    def forward(
        self,
        input_wavs: torch.Tensor,
        input_wav_lens: torch.Tensor,
        encoder_labels: torch.Tensor,
        encoder_label_lens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor,
    ) -> Dict[str, Any]:

        # encoder
        fs_pad, fs_lens = self.feature_extractor(input_wavs, input_wav_lens)
        fs_pad, fs_lens = self.spec_augmentor(fs_pad, fs_lens)
        fs_pad, fs_lens = self.normalizer(fs_pad, fs_lens)
        hs_pad, hs_lens, _ = self.encoder(fs_pad, fs_lens)

        # ctc
        stats = dict()
        if self.ctc_loss_weight != 0.0:
            ctc_logits = self.ctc_classifier(hs_pad).transpose(0, 1)  # (T, B, C)
            ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
            encoder_labels = encoder_labels[encoder_labels >= 0]
            loss_ctc = self.ctc_loss_fn(
                ctc_log_probs, encoder_labels, hs_lens, encoder_label_lens
            ) / input_wavs.size(0)
            stats["loss_ctc"] = loss_ctc.item()
        else:
            loss_ctc = 0

        # decoder
        if self.ctc_loss_weight != 1.0:
            att_logits = self.decoder(
                hs_pad, hs_lens, decoder_input_ids, encoder_label_lens + 1
            )[0]
            loss_att = self.att_loss_fn(
                att_logits.view(-1, self.vocab_size), decoder_labels.view(-1)
            ) / input_wavs.size(0)
            stats["loss_att"] = loss_att.item()
            stats["att_acc"] = tokens_accuracy(att_logits.argmax(-1), decoder_labels)
        else:
            loss_att = 0

        loss = self.ctc_loss_weight * loss_ctc + (1 - self.ctc_loss_weight) * loss_att
        stats["loss"] = loss.item()

        return {"loss": loss, "stats": stats}

    def recognize(
        self,
        input_wavs: torch.Tensor,
        input_wav_lens: torch.Tensor,
        beam_size: int = 10,
        ctc_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:

        # encoder
        fs_pad, fs_lens = self.feature_extractor(input_wavs, input_wav_lens)
        fs_pad, fs_lens = self.normalizer(fs_pad, fs_lens)
        hs_pad = self.encoder(fs_pad, fs_lens)[0]
        self.ctc_prefix_score = CTCPrefixScore(
            torch.log_softmax(self.ctc_classifier(hs_pad), dim=-1)[0].cpu().numpy(),
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
        max_length = hs_pad.size(1)
        for i in range(1, max_length):
            # decoder
            decoder_input_ids = fs_lens.new_zeros(beam_size, i)
            for beam_idx, stat in enumerate(running_stats):
                decoder_input_ids[beam_idx, :] = fs_lens.new_tensor(stat["tokens"])
            next_token_scores = self.decoder.forward_one_step(
                decoder_input_ids,
                fs_lens.new_ones(beam_size, i, i).tril(0),
                hs_pad.repeat_interleave(beam_size, dim=0),
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
