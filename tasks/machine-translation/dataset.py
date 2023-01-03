from argparse import Namespace
from typing import Callable, Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class Iwslt2017Dataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        self.src_lang = args.src_lang.split("_")[0]
        self.tgt_lang = args.tgt_lang.split("_")[0]
        setattr(args, "dataset", f"iwslt2017-{self.src_lang}-{self.tgt_lang}")
        if split in ["train", "validation"]:
            dataset = load_dataset("iwslt2017", args.dataset, split=split)
        else:
            dataset = load_dataset(
                "iwslt2017", args.dataset, split=f"test[:{args.test_percent}%]"
            )
        self.dataset = [
            {
                "src_text": data["translation"][self.src_lang],
                "tgt_text": data["translation"][self.tgt_lang],
            }
            for data in dataset
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.dataset[idx]

    def tokenize(self, tokenizer: Callable, max_length: int):
        for i, sample in tqdm(
            enumerate(self.dataset), total=len(self.dataset), desc="Tokenizing"
        ):
            results = tokenizer(
                sample["src_text"],
                text_target=sample["tgt_text"],
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.dataset[i] = {
                "encoder_tokens": results["input_ids"][0],
                "decoder_tokens": torch.roll(results["labels"][0], shifts=1),
                "labels": results["labels"][0],
            }
