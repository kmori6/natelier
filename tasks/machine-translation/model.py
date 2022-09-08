from argparse import Namespace
import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
from typing import Any, Dict
from metrics import tokens_accuracy


class NMTBart(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.config = BartConfig.from_pretrained(args.model_name)
        self.model = BartModel.from_pretrained(args.model_name)
        self.initialize_embeddings(args.vocab_size)

        self.classifier = nn.Linear(self.config.d_model, args.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def initialize_embeddings(self, vocab_size: int):
        self.model.shared = nn.Embedding(
            vocab_size, self.config.d_model, self.config.pad_token_id
        )
        self.model.encoder.embed_tokens = self.model.shared
        self.model.decoder.embed_tokens = self.model.shared

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs_pad = self.model(input_ids, attention_mask, decoder_input_ids)[0]
        logits = self.classifier(hs_pad)

        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        stats = {"loss": loss.item(), "acc": tokens_accuracy(logits.argmax(-1), labels)}

        return {"loss": loss, "logits": logits, "stats": stats}

    def translate(
        self,
        src_input_ids: torch.Tensor,
        beam_size: int = 5,
        length_magnification: int = 5,
    ) -> Dict[str, Any]:

        # initial stats
        running_stats = [{"score": 0.0, "tokens": [self.config.bos_token_id]}]
        final_stats = []

        # encoder forward
        encoder_outputs = self.model.encoder(
            src_input_ids.repeat_interleave(beam_size, dim=0), return_dict=True
        )

        # beam search
        max_length = src_input_ids.size(1) * length_magnification
        for i in range(1, max_length):

            # decoder forward
            decoder_input_ids = src_input_ids.new_zeros(beam_size, i)
            for beam_idx, stat in enumerate(running_stats):
                decoder_input_ids[beam_idx, :] = src_input_ids.new_tensor(
                    stat["tokens"]
                )
            decoder_outputs = self.model.decoder(
                decoder_input_ids, encoder_hidden_states=encoder_outputs[0]
            )[0]
            next_token_scores = torch.log_softmax(
                self.classifier(decoder_outputs)[:, -1, :], dim=-1
            )

            # scoring
            aggregator = []
            for beam_idx, stat in enumerate(running_stats):
                next_scores, next_tokens = torch.topk(
                    next_token_scores[beam_idx], beam_size, dim=0
                )
                for topk_idx in range(beam_size):
                    aggregator.append(
                        {
                            "score": stat["score"] + next_scores[topk_idx].item(),
                            "tokens": stat["tokens"] + [next_tokens[topk_idx].item()],
                        }
                    )
            running_stats = sorted(aggregator, key=lambda x: x["score"], reverse=True)[
                :beam_size
            ]

            # add eos_token_id
            if i == max_length - 1:
                for stat in running_stats:
                    stat["tokens"].append(self.config.eos_token_id)

            # sort stats
            keep_stats = []
            for stat in running_stats:
                if stat["tokens"][-1] == self.config.eos_token_id:
                    final_stats.append(stat)
                else:
                    keep_stats.append(stat)
            running_stats = keep_stats

            # stop search
            if len(running_stats) < 1 or len(final_stats) >= beam_size:
                break

        return sorted(final_stats, key=lambda x: x["score"], reverse=True)[0]
