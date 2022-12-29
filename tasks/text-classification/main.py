from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from collator import TCBatchCollator
from data import GlueDataset
from model import TCAlbert
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import Tester as BaseTester
from metrics import (
    f1_score,
    matthews_correlation_coefficient,
    pearson_correlation,
    single_label_accuracy,
    spearman_correlation,
)
from train import Trainer


def add_specific_arguments(parser: ArgumentParser):
    parser.add_argument("--dataset", default="cola", type=str)
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--max_length", default=256, type=int)


class Tester(BaseTester):
    def __init__(self, args: Namespace, model_class: nn.Module):
        super().__init__(args, model_class)

    def test_epoch(
        self,
        test_dataloader: DataLoader,
        num_labels: int,
        metrics: Union[str, Tuple[str]],
    ) -> Dict[str, float]:
        self.model.eval()
        test_stats = dict()
        all_preds, all_labels = [], []
        for batch in tqdm(test_dataloader, desc="test"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            preds = outputs.logits.argmax(-1) if num_labels > 1 else outputs.logits
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
            test_stats["pearson_correlation"] = pearson_correlation(
                all_preds, all_labels
            )
        if "spearman_correlation" in metrics:
            test_stats["spearman_correlation"] = spearman_correlation(
                all_preds, all_labels
            )
        return test_stats


def main():
    parser = ArgumentParser()
    Trainer.add_train_args(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    train_dataset = GlueDataset(args, "train")
    dev_dataset = GlueDataset(args, "validation")
    test_dataset = GlueDataset(args, "test")
    GlueDataset.set_data_specific_arguments(args)

    if args.train:
        trainer = Trainer(args, TCAlbert)
        trainer.run(train_dataset, dev_dataset, TCBatchCollator(args))

    if args.test:
        tester = Tester(args, TCAlbert)
        tester.run(
            test_dataset,
            TCBatchCollator(args),
            num_labels=args.num_labels,
            metrics=args.metrics,
        )


if __name__ == "__main__":
    main()
