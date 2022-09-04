import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from data import SwagDataset
from model import MCAlbert
from collator import MCBatchCollator
from train import train
from test import test
from metrics import single_label_accuracy
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser):
    parser.add_argument("--dataset", default="regular", type=str)
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_choices", default=4, type=int)


def test_steps(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_preds.append(outputs["logits"].argmax(-1).cpu())
        all_labels.append(batch["labels"].cpu())
    test_stats["acc"] = single_label_accuracy(
        torch.concat(all_preds), torch.concat(all_labels)
    )
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = SwagDataset(args, "train")
    dev_dataset = SwagDataset(args, "validation")
    test_dataset = SwagDataset(args, "test")

    if args.train:
        train(args, MCAlbert, train_dataset, dev_dataset, MCBatchCollator(args))

    if args.test:
        test(args, MCAlbert, test_dataset, MCBatchCollator(args), test_steps)


if __name__ == "__main__":
    main()
