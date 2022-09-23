from argparse import Namespace
from typing import Dict, List, Any
import torch
from tokenizer import NMTTokenizer


class NMTBatchCollator:
    def __init__(
        self,
        args: Namespace,
        tokenizer: NMTTokenizer,
        return_test_encodings: bool = False,
    ):
        self.tokenizer = tokenizer
        self.return_test_encodings = return_test_encodings

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        src_encodings = self.tokenizer([data["src_text"] for data in batch])
        tgt_encodings = self.tokenizer([data["tgt_text"] for data in batch])
        if self.return_test_encodings:
            return {
                "tokens": src_encodings["tokens"],
                "labels": tgt_encodings["tokens"].masked_fill(
                    tgt_encodings["masks"] == 0, -100
                ),
            }
        else:
            masks = tgt_encodings["masks"][:, :-1]
            batch_size, length = masks.size()
            masks = torch.tril(
                masks.repeat_interleave(length, 0).view(batch_size, length, length)
            )
            return {
                "encoder_tokens": src_encodings["tokens"],
                "encoder_masks": src_encodings["masks"],
                "decoder_tokens": tgt_encodings["tokens"][:, :-1],
                "decoder_masks": masks,
                "labels": tgt_encodings["tokens"].masked_fill(
                    tgt_encodings["masks"] == 0, -100
                )[:, 1:],
            }
