from argparse import Namespace
import torch
from typing import List, Dict
from transformers import AutoTokenizer


class LMBatchCollator:
    def __init__(self, args: Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            [data["sent1"] for data in batch],
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        # shift model inputs and labels
        # labels: input_ids[2:t]
        # inputs: input_ids[1:t-1]
        encodings["labels"] = encodings["input_ids"].masked_fill(
            encodings["attention_mask"] == 0, -100
        )[:, 1:]
        encodings["input_ids"] = encodings["input_ids"][:, :-1]
        encodings["attention_mask"] = encodings["attention_mask"][:, :-1]
        encodings["token_type_ids"] = encodings["token_type_ids"][:, :-1]
        return {k: v for k, v in encodings.items()}
