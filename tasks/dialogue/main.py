import argparse
from tqdm import tqdm
import sacrebleu
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from data import DailyDialogDataset
from collator import DBatchCollator
from model import DBart
from train import train
from test import test
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--test_ratio", default=None, type=int)
    parser.add_argument("--model_name", default="facebook/bart-base", type=str)
    parser.add_argument("--vocab_size", default=50265, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_length", default=256, type=int)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, float]:

    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader):
        for utt_idx in range(batch["tokens"].size(0)):
            with torch.no_grad():
                preds_token = model.response(
                    src_input_ids=batch["tokens"][utt_idx].unsqueeze(0).to(device),
                    beam_size=beam_size,
                    max_length=max_length,
                )["tokens"]
            preds_text = tokenizer.decode(preds_token, skip_special_tokens=True)
            labels_text = tokenizer.decode(
                batch["labels"][0][batch["labels"][0] != -100], skip_special_tokens=True
            )
            all_preds.append(preds_text.strip())
            all_labels.append([labels_text.strip()])
    metrics = sacrebleu.corpus_bleu(
        all_preds,
        all_labels,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        use_effective_order=False,
    )
    test_stats["sacrebleu"] = metrics.score

    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = DailyDialogDataset(args, "train")
    dev_dataset = DailyDialogDataset(args, "validation")
    test_dataset = DailyDialogDataset(args, "test")

    if args.batch_size > 1:
        setattr(args, "batch_size", 1)

    if args.train:
        train(args, DBart, train_dataset, dev_dataset, DBatchCollator(args))

    if args.test:
        test(
            args,
            DBart,
            test_dataset,
            DBatchCollator(args, return_test_encodings=True),
            test_steps,
            beam_size=args.beam_size,
            max_length=args.max_length,
            tokenizer=AutoTokenizer.from_pretrained(args.model_name),
        )


if __name__ == "__main__":
    main()
