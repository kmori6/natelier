import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from data import Conll2003Dataset
from model import NERAlbert
from collator import NERBatchCollator
from train import train
from test import test
from metrics import f1_score
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser):
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--num_labels", default=9, type=int)
    parser.add_argument("--max_length", default=256, type=int)


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
        all_preds.append(outputs["logits"].argmax(-1)[batch["labels"] != -100].cpu())
        all_labels.append(batch["labels"][batch["labels"] != -100].cpu())
    test_stats["f1_score"] = f1_score(
        torch.concat(all_preds), torch.concat(all_labels), average="micro"
    )
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = Conll2003Dataset(args, "train")
    dev_dataset = Conll2003Dataset(args, "validation")
    test_dataset = Conll2003Dataset(args, "test")

    if args.train:
        train(args, NERAlbert, train_dataset, dev_dataset, NERBatchCollator(args))

    if args.test:
        test(args, NERAlbert, test_dataset, NERBatchCollator(args), test_steps)


if __name__ == "__main__":
    main()
