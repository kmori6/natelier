from argparse import Namespace
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast


class NMTBatchCollator:
    def __init__(
        self,
        args: Namespace,
        tokenizer: PreTrainedTokenizerFast,
        return_test_encodings: bool = False,
        ignore_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.return_test_encodings = return_test_encodings
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_token_id = ignore_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        encoder_tokens = pad_sequence(
            [sample["encoder_tokens"] for sample in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )  # (B, L1)
        labels = pad_sequence(
            [sample["labels"] for sample in batch],
            batch_first=True,
            padding_value=self.ignore_token_id,
        )  # (B, L2)
        if self.return_test_encodings:
            return {"encoder_tokens": encoder_tokens, "labels": labels}
        else:
            encoder_masks = torch.ones_like(encoder_tokens).masked_fill(
                encoder_tokens == self.pad_token_id, 0
            )  # (B, L1)
            decoder_tokens = pad_sequence(
                [sample["decoder_tokens"] for sample in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            )  # (B, L2)
            batch_size, tgt_length = decoder_tokens.size()
            decoder_masks = torch.ones_like(decoder_tokens).masked_fill(
                decoder_tokens == self.pad_token_id, 0
            )  # (B, L2)
            decoder_masks = (
                decoder_masks.repeat_interleave(tgt_length, dim=0)
                .view(batch_size, tgt_length, tgt_length)
                .tril()
            )  # (B, L2, L2)
            return {
                "encoder_tokens": encoder_tokens,
                "encoder_masks": encoder_masks,
                "decoder_tokens": decoder_tokens,
                "decoder_masks": decoder_masks,
                "labels": labels,
            }
