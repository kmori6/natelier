from argparse import Namespace
from typing import Dict
from torch.utils.data import Dataset
from datasets import load_dataset


class SquadDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        datasets = load_dataset(args.dataset)
        if split == "train":
            self.dataset = datasets["train"]
        else:
            self.dataset = datasets["validation"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "id": self.dataset[idx]["id"],
            "context": self.dataset[idx]["context"],
            "question": self.dataset[idx]["question"],
            'answers': self.dataset[idx]["answers"],
        }
