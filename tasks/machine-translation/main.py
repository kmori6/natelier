import os
import argparse
from tqdm import tqdm
from itertools import chain
import sacrebleu
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tokenizer import NMTTokenizer
from data import Iwslt2017Dataset
from collator import NMTBatchCollator
from model import NMTBart
from train import train
from test import test
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default="iwslt2017-en-de", type=str)
    parser.add_argument("--test_ratio", default=None, type=int)
    parser.add_argument("--model_name", default="facebook/bart-base", type=str)
    parser.add_argument("--train_tokenizer", default=False, action="store_true")
    parser.add_argument("--src_lang", default="en", type=str)
    parser.add_argument("--tgt_lang", default="de", type=str)
    parser.add_argument("--vocab_size", default=1000, type=int)
    parser.add_argument("--beam_size", default=5, type=int)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    tokenizer: NMTTokenizer,
) -> Dict[str, float]:

    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            preds_token = model.translate(
                src_input_ids=batch["input_ids"].to(device),
                beam_size=beam_size,
            )["tokens"]
        preds_text = tokenizer.decode(preds_token)
        labels_text = tokenizer.decode(batch["labels"][0].tolist())
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

    train_dataset = Iwslt2017Dataset(args, "train")
    dev_dataset = Iwslt2017Dataset(args, "validation")
    test_dataset = Iwslt2017Dataset(args, "test")

    if args.train:

        if args.train_tokenizer:
            os.makedirs(args.out_dir + "/tokenizer", exist_ok=True)
            text_file = args.out_dir + "/tokenizer/train_text.txt"
            with open(text_file, "w") as f:
                for data in train_dataset:
                    f.write(data["src_text"] + "\n" + data["tgt_text"] + "\n")
            NMTTokenizer.train_tokenizer(
                text_file, args.vocab_size, args.out_dir + "/tokenizer"
            )
        tokenizer = NMTTokenizer(args.out_dir + "/tokenizer/tokenizer.model")
        train(
            args,
            NMTBart,
            train_dataset,
            dev_dataset,
            NMTBatchCollator(args, tokenizer),
        )

    if args.test:
        if args.batch_size > 1:
            setattr(args, "batch_size", 1)
        tokenizer = NMTTokenizer(args.out_dir + "/tokenizer/tokenizer.model")
        test(
            args,
            NMTBart,
            test_dataset,
            NMTBatchCollator(args, tokenizer, return_test_encodings=True),
            test_steps,
            beam_size=args.beam_size,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
