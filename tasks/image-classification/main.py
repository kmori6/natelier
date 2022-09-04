import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from data import MnistDataset
from model import ICCnn
from collator import mnist_collate_fn
from train import train
from test import test
from metrics import single_label_accuracy
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.Namespace):
    parser.add_argument("--download_dir", default=None, type=str, required=True)
    parser.add_argument("--num_labels", default=10, type=int)
    parser.add_argument("--h_size", default=28, type=int)
    parser.add_argument("--w_size", default=28, type=int)
    parser.add_argument("--idim", default=1, type=int)
    parser.add_argument("--hidden_size", default=8, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)


def test_steps(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader, desc="test"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs["logits"].argmax(-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())
    test_stats = {
        "acc": single_label_accuracy(torch.concat(all_preds), torch.concat(all_labels))
    }
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = MnistDataset(args.download_dir, "train")
    dev_dataset = MnistDataset(args.download_dir, "validation")
    test_dataset = MnistDataset(args.download_dir, "test")

    if args.train:
        train(args, ICCnn, train_dataset, dev_dataset, mnist_collate_fn)

    if args.test:
        test(args, ICCnn, test_dataset, mnist_collate_fn, test_steps)


if __name__ == "__main__":
    main()
