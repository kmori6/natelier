from argparse import Namespace
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer


class NERBatchCollator:
    def __init__(self, args: Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            [data["words"] for data in batch],
            padding=True,
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [
                [
                    batch[i]["labels"][word_id] if word_id is not None else -100
                    for word_id in encodings.word_ids(i)
                ]
                for i in range(len(batch))
            ],
            dtype=torch.long,
        )
        batch = {
            "tokens": encodings["input_ids"],
            "masks": encodings["attention_mask"],
            "segments": encodings["token_type_ids"],
            "labels": labels
        }
        return batch
