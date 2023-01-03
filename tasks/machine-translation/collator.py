from argparse import Namespace
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

from utils import fill_tokens


class NMTBatchCollator:
    def __init__(
        self,
        args: Namespace,
        tokenizer: PreTrainedTokenizerFast,
        return_test_encodings: bool = False,
        ignore_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_length = min(args.max_length, tokenizer.model_max_length)
        self.return_test_encodings = return_test_encodings
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_token_id = ignore_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        encoder_tokens = pad_sequence(
            [sample["encoder_tokens"] for sample in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        decoder_tokens = pad_sequence(
            [sample["decoder_tokens"] for sample in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [sample["labels"] for sample in batch],
            batch_first=True,
            padding_value=self.ignore_token_id,
        )
        encoder_masks = torch.ones_like(encoder_tokens).masked_fill(
            encoder_tokens == self.pad_token_id, 0
        )
        decoder_masks = torch.ones_like(decoder_tokens).masked_fill(
            decoder_tokens == self.pad_token_id, 0
        )
        if self.return_test_encodings:
            return {
                "tokens": encodings["input_ids"],
                "labels": fill_tokens(
                    encodings["labels"], self.pad_token_id, self.ignore_token_id
                ),
            }
        else:
            return {
                "encoder_tokens": encoder_tokens,
                "encoder_masks": encoder_masks,
                "decoder_tokens": decoder_tokens,
                "decoder_masks": decoder_masks,
                "labels": labels,
            }
