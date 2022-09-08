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
                "input_ids": src_encodings["input_ids"],
                "labels": tgt_encodings["input_ids"].masked_fill(
                    tgt_encodings["attention_mask"] == 0, -100
                ),
            }
        else:
            return {
                "input_ids": src_encodings["input_ids"],
                "attention_mask": src_encodings["attention_mask"],
                "decoder_input_ids": tgt_encodings["input_ids"][:, :-1],
                "labels": tgt_encodings["input_ids"].masked_fill(
                    tgt_encodings["attention_mask"] == 0, -100
                )[:, 1:],
            }
