from argparse import Namespace
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer


class TCBatchCollator:
    def __init__(self, args: Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            [
                [data["sent1"], data["sent2"]] if "sent2" in data else data["sent1"]
                for data in batch
            ],
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch = {
            "tokens": encodings["input_ids"],
            "masks": encodings["attention_mask"],
            "segments": encodings["token_type_ids"],
            "labels": torch.tensor(
                [data["label"] for data in batch],
                dtype=torch.float32
                if isinstance(batch[0]["label"], float)
                else torch.long,
            ),
        }
        return batch
