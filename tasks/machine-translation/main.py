from argparse import ArgumentParser, Namespace
from evaluate import test
from typing import Dict

import sacrebleu
import torch
from dataset import Iwslt2017Dataset
from model import NMTBart
from tokenizer import NMTTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import NMTBatchCollator
from sampler import LengthGroupBatchSampler
from train import Trainer
from utils import get_parser


def get_args(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--test_percent", default=100, type=int)
    parser.add_argument("--model_name", default="facebook/mbart-large-cc25", type=str)
    parser.add_argument("--train_tokenizer", default=False, action="store_true")
    parser.add_argument("--src_lang", default="en_XX", type=str)
    parser.add_argument("--tgt_lang", default="ja_XX", type=str)
    parser.add_argument("--vocab_size", default=250027, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    args = parser.parse_args()
    return args


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    max_length: int,
    tokenizer: NMTTokenizer,
) -> Dict[str, float]:

    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            preds_token = model.translate(
                tokens=batch["tokens"].to(device),
                beam_size=beam_size,
                max_length=max_length,
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
    parser = get_parser()
    args = get_args(parser)

    if args.train:
        train_dataset = Iwslt2017Dataset(args, "train")
        dev_dataset = Iwslt2017Dataset(args, "validation")
        tokenizer = NMTBart.get_pretrained_tokenizer(args.src_lang, args.tgt_lang)
        train_dataset.tokenize(
            tokenizer, min(args.max_length, tokenizer.model_max_length)
        )
        dev_dataset.tokenize(
            tokenizer, min(args.max_length, tokenizer.model_max_length)
        )
        trainer = Trainer(args, NMTBart)
        trainer.run(train_dataset, dev_dataset, NMTBatchCollator(args, tokenizer))

    if args.test:
        test_dataset = Iwslt2017Dataset(args, "test")
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
            max_length=args.max_length,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
