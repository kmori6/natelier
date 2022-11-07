from argparse import Namespace
from typing import Dict
from torch.utils.data import Dataset
from datasets import load_dataset


class Iwslt2017Dataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        self.src_lang = args.src_lang.split("_")[0]
        self.tgt_lang = args.tgt_lang.split("_")[0]
        datasets = load_dataset("iwslt2017", args.dataset)
        if split == "train":
            self.dataset = datasets["train"]
        elif split == "validation":
            self.dataset = datasets["validation"]
        else:
            self.dataset = (
                datasets["test"]
                if args.test_ratio is None
                else load_dataset(
                    "iwslt2017", args.dataset, split=f"test[:{args.test_ratio}%]"
                )
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "src_text": self.dataset[idx]["translation"][self.src_lang],
            "tgt_text": self.dataset[idx]["translation"][self.tgt_lang],
        }
