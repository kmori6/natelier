import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from data import WikitextDataset
from model import LMAlbert
from collator import LMBatchCollator
from train import train
from test import test
from metrics import ppl
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.Namespace):
    parser.add_argument("--dataset", default="wikitext-2-raw-v1", type=str)
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--max_length", default=256, type=int)


def test_steps(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    test_stats = {"loss": 0, "ppl": 0}
    for batch in tqdm(test_dataloader, desc="test"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        test_stats["loss"] += outputs["stats"]["loss"] * batch["input_ids"].size(0)
    test_stats["loss"] /= len(test_dataloader.dataset)
    test_stats["ppl"] = ppl(test_stats["loss"])
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = WikitextDataset(args, "train")
    dev_dataset = WikitextDataset(args, "validation")
    test_dataset = WikitextDataset(args, "test")

    if args.train:
        train(args, LMAlbert, train_dataset, dev_dataset, LMBatchCollator(args))

    if args.test:
        test(args, LMAlbert, test_dataset, LMBatchCollator(args), test_steps)


if __name__ == "__main__":
    main()
