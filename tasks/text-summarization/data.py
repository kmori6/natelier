from argparse import Namespace
from typing import Dict
from torch.utils.data import Dataset
from datasets import load_dataset


class CnnDailyMailDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        datasets = load_dataset("cnn_dailymail", "3.0.0")
        if split == "train":
            self.dataset = (
                datasets["train"]
                if args.train_ratio is None
                else load_dataset(
                    "cnn_dailymail", "3.0.0", split=f"train[:{args.train_ratio}%]"
                )
            )
        elif split == "validation":
            self.dataset = datasets["validation"]
        else:
            self.dataset = (
                datasets["test"]
                if args.test_ratio is None
                else load_dataset(
                    "cnn_dailymail", "3.0.0", split=f"test[:{args.test_ratio}%]"
                )
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "src_text": self.dataset[idx]["article"],
            "tgt_text": self.dataset[idx]["highlights"],
        }
