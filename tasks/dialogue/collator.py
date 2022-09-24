from argparse import Namespace
from typing import Dict, List, Any
from itertools import chain
import torch
from transformers import AutoTokenizer


class DBatchCollator:
    def __init__(self, args: Namespace, return_test_encodings: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)
        self.return_test_encodings = return_test_encodings

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        src_encodings = self.tokenizer(
            list(chain(*[data["dialog"][0::2] for data in batch])),
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tgt_encodings = self.tokenizer(
            list(chain(*[data["dialog"][1::2] for data in batch])),
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        num_inputs = tgt_encodings["input_ids"].size(0)  # for odd turns
        if self.return_test_encodings:
            return {
                "tokens": src_encodings["input_ids"][:num_inputs, :],
                "labels": tgt_encodings["input_ids"].masked_fill(
                    tgt_encodings["attention_mask"] == 0, -100
                ),
            }
        else:
            masks = tgt_encodings["attention_mask"][:, :-1]
            batch_size, length = masks.size()
            masks = torch.tril(
                masks.repeat_interleave(length, 0).view(batch_size, length, length)
            )
            return {
                "encoder_tokens": src_encodings["input_ids"][:num_inputs, :],
                "encoder_masks": src_encodings["attention_mask"][:num_inputs, :],
                "decoder_tokens": tgt_encodings["input_ids"][:, :-1],
                "decoder_masks": masks,
                "labels": tgt_encodings["input_ids"].masked_fill(
                    tgt_encodings["attention_mask"] == 0, -100
                )[:, 1:],
            }
