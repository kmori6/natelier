from argparse import Namespace
from typing import Dict, List, Any
from itertools import chain
import torch
from transformers import AutoTokenizer


class MCBatchCollator:
    def __init__(self, args: Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)
        self.num_choices = args.num_choices

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        all_sentences = [
            [[data["sentence"], data[f"choice{i}"]] for i in range(self.num_choices)]
            for data in batch
        ]
        # flatten: [[[s1a], [s1b]], [[s2a], [s2b]]] -> [[s1a], [s1b], [s2a], [s2b]]
        all_sentences = list(chain(*all_sentences))
        encodings = self.tokenizer(
            all_sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            "tokens": encodings["input_ids"].view(batch_size, self.num_choices, -1),
            "masks": encodings["attention_mask"].view(batch_size, self.num_choices, -1),
            "segments": encodings["token_type_ids"].view(
                batch_size, self.num_choices, -1
            ),
            "labels": torch.tensor([data["label"] for data in batch], dtype=torch.long),
        }
        return inputs
