from argparse import Namespace
from typing import Dict
from torch.utils.data import Dataset
from datasets import load_dataset


class WikitextDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        datasets = load_dataset("wikitext", args.dataset)
        if split == "train":
            self.dataset = datasets["train"]
        elif split == "validation":
            self.dataset = datasets["validation"]
        else:
            self.dataset = datasets["test"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"sent1": self.dataset[idx]["text"]}
