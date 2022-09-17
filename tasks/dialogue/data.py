from argparse import Namespace
from typing import Dict, List
from torch.utils.data import Dataset
from datasets import load_dataset


class DailyDialogDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        super().__init__()
        datasets = load_dataset("daily_dialog")
        if split == "train":
            self.dataset = datasets["train"]
        elif split == "validation":
            self.dataset = datasets["validation"]
        else:
            self.dataset = (
                datasets["test"]
                if args.test_ratio is None
                else load_dataset("daily_dialog", split=f"test[:{args.test_ratio}%]")
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        return {"dialog": self.dataset[idx]["dialog"]}
