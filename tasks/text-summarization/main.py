import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from data import CnnDailyMailDataset
from collator import TSBatchCollator
from model import TSBart
from train import train
from test import test
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--train_ratio", default=None, type=int)
    parser.add_argument("--test_ratio", default=None, type=int)
    parser.add_argument("--model_name", default="facebook/bart-base", type=str)
    parser.add_argument("--beam_size", default=5, type=int)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, float]:

    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bootstrap_aggregator = scoring.BootstrapAggregator()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            preds_token = model.summarize(
                src_input_ids=batch["input_ids"].to(device),
                beam_size=beam_size,
            )["tokens"]
        preds_text = tokenizer.decode(preds_token, skip_special_tokens=True)
        labels_text = tokenizer.decode(batch["labels"][0], skip_special_tokens=True)
        all_preds.append(preds_text.strip())
        all_labels.append(labels_text.strip())
    for label, pred in zip(all_labels, all_preds):
        scores = scorer.score(label, pred)
        bootstrap_aggregator.add_scores(scores)
    results = bootstrap_aggregator.aggregate()
    test_stats = {k: v.mid.fmeasure * 100 for k, v in results.items()}

    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = CnnDailyMailDataset(args, "train")
    dev_dataset = CnnDailyMailDataset(args, "validation")
    test_dataset = CnnDailyMailDataset(args, "test")

    if args.train:
        train(args, TSBart, train_dataset, dev_dataset, TSBatchCollator(args))

    if args.test:
        if args.batch_size > 1:
            setattr(args, "batch_size", 1)
        test(
            args,
            TSBart,
            test_dataset,
            TSBatchCollator(args, return_test_encodings=True),
            test_steps,
            beam_size=args.beam_size,
            tokenizer=AutoTokenizer.from_pretrained(args.model_name),
        )


if __name__ == "__main__":
    main()
