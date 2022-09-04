from argparse import Namespace
from typing import Dict
from torch.utils.data import Dataset
from datasets import load_dataset


class SwagDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        datasets = load_dataset("swag", args.dataset)
        if split == "train":
            self.dataset = datasets["train"]
        else:
            self.dataset = datasets["validation"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "sentence": self.dataset[idx]["sent1"],
            "choice0": f"{self.dataset[idx]['sent2']} {self.dataset[idx]['ending0']}",
            "choice1": f"{self.dataset[idx]['sent2']} {self.dataset[idx]['ending1']}",
            "choice2": f"{self.dataset[idx]['sent2']} {self.dataset[idx]['ending2']}",
            "choice3": f"{self.dataset[idx]['sent2']} {self.dataset[idx]['ending3']}",
            "label": self.dataset[idx]["label"],
        }
