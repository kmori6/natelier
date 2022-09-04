import argparse
from itertools import chain
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from data import SquadDataset
from model import QAAlbert
from collator import QABatchCollator
from train import train
from test import test
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.Namespace):
    parser.add_argument(
        "--dataset", default="squad", type=str, choices=["squad", "squad_v2"]
    )
    parser.add_argument("--model_name", default="albert-base-v2", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--threshold", default=0.0, type=float)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    threshold: float,
    dataset: str,
) -> Dict[str, float]:

    model.eval()
    test_stats = dict()
    metric = load_metric(dataset)
    all_preds, all_refs = [], []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            outputs = model.answer(
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                offset_mapping=batch["offset_mapping"],
                contexts=batch["contexts"],
                ids=batch["ids"],
                beam_size=beam_size,
                threshold=threshold,
            )
        all_preds.append(outputs)
        all_refs.append(batch["references"])
    all_refs = list(chain(*all_refs))
    all_preds = [
        {
            "prediction_text": data["prediction_text"],
            "id": data["id"],
            "no_answer_probability": 0.0,
        }
        if dataset == "squad_v2"
        else {"prediction_text": data["prediction_text"], "id": data["id"]}
        for data in list(chain(*all_preds))
    ]
    test_stats = metric.compute(predictions=all_preds, references=all_refs)

    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = SquadDataset(args, "train")
    dev_dataset = SquadDataset(args, "validation")
    test_dataset = SquadDataset(args, "test")

    if args.train:
        train(args, QAAlbert, train_dataset, dev_dataset, QABatchCollator(args))

    if args.test:
        test(
            args,
            QAAlbert,
            test_dataset,
            QABatchCollator(args, return_references=True),
            test_steps,
            beam_size=args.beam_size,
            threshold=args.threshold,
            dataset=args.dataset,
        )


if __name__ == "__main__":
    main()
