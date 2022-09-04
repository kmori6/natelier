from argparse import Namespace
from typing import Dict, Any
from torch.utils.data import Dataset
from datasets import load_dataset

GLUE_CONFIG = {
    "cola": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "mcc",
        "sent1_key": "sentence",
        "sent2_key": None,
    },
    "mnli-m": {
        "task": "single_label_classification",
        "num_labels": 3,
        "metrics": "acc",
        "sent1_key": "premise",
        "sent2_key": "hypothesis",
    },
    "mnli-mm": {
        "task": "single_label_classification",
        "num_labels": 3,
        "metrics": "acc",
        "sent1_key": "premise",
        "sent2_key": "hypothesis",
    },
    "mrpc": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": ["acc", "f1_score"],
        "sent1_key": "sentence1",
        "sent2_key": "sentence2",
    },
    "sst2": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "acc",
        "sent1_key": "sentence",
        "sent2_key": None,
    },
    "stsb": {
        "task": "regression",
        "num_labels": 1,
        "metrics": ["pearson_correlation", "spearman_correlation"],
        "sent1_key": "sentence1",
        "sent2_key": "sentence2",
    },
    "qqp": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "acc",
        "sent1_key": "question1",
        "sent2_key": "question2",
    },
    "qnli": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "acc",
        "sent1_key": "question",
        "sent2_key": "sentence",
    },
    "rte": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "acc",
        "sent1_key": "sentence1",
        "sent2_key": "sentence2",
    },
    "wnli": {
        "task": "single_label_classification",
        "num_labels": 2,
        "metrics": "acc",
        "sent1_key": "sentence1",
        "sent2_key": "sentence2",
    },
}


class GlueDataset(Dataset):
    def __init__(self, args: Namespace, split: str):
        datasets = load_dataset(
            "glue",
            args.dataset if args.dataset not in ["mnli-m", "mnli-mm"] else "mnli",
        )
        if split == "train":
            self.dataset = datasets["train"]
        else:
            if args.dataset == "mnli-m":
                self.dataset = datasets["validation_matched"]
            elif args.dataset == "mnli-mm":
                self.dataset = datasets["validation_mismatched"]
            else:
                self.dataset = datasets["validation"]
        self.sent1_key = GLUE_CONFIG[args.dataset]["sent1_key"]
        self.sent2_key = GLUE_CONFIG[args.dataset]["sent2_key"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {
            "sent1": self.dataset[idx][self.sent1_key],
            "label": self.dataset[idx]["label"],
        }
        if self.sent2_key:
            data["sent2"] = self.dataset[idx][self.sent2_key]
        return data

    @staticmethod
    def set_data_specific_arguments(args: Namespace):
        setattr(args, "task", GLUE_CONFIG[args.dataset]["task"])
        setattr(args, "metrics", GLUE_CONFIG[args.dataset]["metrics"])
        setattr(args, "num_labels", GLUE_CONFIG[args.dataset]["num_labels"])
