import argparse
from tqdm import tqdm
from typing import Dict, Union, Tuple
import torch
from torch.utils.data import DataLoader
from data import GlueDataset
from model import TCAlbert
from collator import TCBatchCollator
from train import Trainer
from test import test
from metrics import (
    single_label_accuracy,
    f1_score,
    matthews_correlation_coefficient,
    pearson_correlation,
    spearman_correlation,
)
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default="cola", type=str)
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--max_length", default=256, type=int)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    metrics: Union[str, Tuple[str]],
) -> Dict[str, float]:
    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader, desc="test"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs["logits"].argmax(-1) if num_labels > 1 else outputs["logits"]
        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())
    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)
    if "acc" in metrics:
        test_stats["acc"] = single_label_accuracy(all_preds, all_labels)
    if "f1_score" in metrics:
        test_stats["f1_score"] = f1_score(all_preds, all_labels, average="binary")
    if "mcc" in metrics:
        test_stats["mcc"] = matthews_correlation_coefficient(all_preds, all_labels)
    if "pearson_correlation" in metrics:
        test_stats["pearson_correlation"] = pearson_correlation(all_preds, all_labels)
    if "spearman_correlation" in metrics:
        test_stats["spearman_correlation"] = spearman_correlation(all_preds, all_labels)
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = GlueDataset(args, "train")
    dev_dataset = GlueDataset(args, "validation")
    test_dataset = GlueDataset(args, "test")
    GlueDataset.set_data_specific_arguments(args)

    if args.train:
        trainer = Trainer(args, TCAlbert)
        trainer.run(train_dataset, dev_dataset, TCBatchCollator(args))

    if args.test:
        test(
            args,
            TCAlbert,
            test_dataset,
            TCBatchCollator(args),
            test_steps,
            num_labels=args.num_labels,
            metrics=args.metrics,
        )


if __name__ == "__main__":
    main()
