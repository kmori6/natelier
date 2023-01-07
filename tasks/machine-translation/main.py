import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import sacrebleu
import torch
import torch.nn as nn
from dataset import Iwslt2017Dataset
from model import NMTBart, NMTTransformer
from tokenizer import NMTTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from collator import NMTBatchCollator
from evaluate import Tester as BaseTester
from sampler import LengthGroupBatchSampler
from train import Trainer
from utils import get_parser


def get_args(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--test_percent", default=100, type=int)
    parser.add_argument("--model_name", default="facebook/mbart-large-cc25", type=str)
    parser.add_argument("--train_tokenizer", default=False, action="store_true")
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument("--src_lang", default="en_XX", type=str)
    parser.add_argument("--tgt_lang", default="ja_XX", type=str)
    parser.add_argument("--vocab_size", default=8000, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    args = parser.parse_args()
    return args


class Tester(BaseTester):
    def __init__(self, args: Namespace, model_class: nn.Module):
        super().__init__(args, model_class)

    def test_epoch(
        self,
        test_dataloader: DataLoader,
        beam_size: int,
        max_length: int,
        tokenizer: NMTTokenizer,
    ) -> Dict[str, float]:
        self.model.eval()
        test_stats = dict()
        hypotheses, references = [], []
        for batch in tqdm(test_dataloader, desc="Testing"):
            with torch.no_grad():
                preds_token = self.model.decode(
                    tokens=batch["encoder_tokens"].to(self.device),
                    beam_size=beam_size,
                    max_length=max_length,
                )[0]["tokens"]
            preds_text = tokenizer.decode(preds_token)
            labels_text = tokenizer.decode(batch["labels"][0].tolist())
            hypotheses.append(preds_text)
            references.append(labels_text)
        metrics = sacrebleu.corpus_bleu(hypotheses=hypotheses, references=[references])
        test_stats["sacrebleu"] = metrics.score
        return test_stats


def main():
    parser = get_parser()
    args = get_args(parser)

    if args.train:
        train_dataset = Iwslt2017Dataset(args, "train")
        dev_dataset = Iwslt2017Dataset(args, "validation")

        if args.finetune:
            tokenizer = NMTBart.get_pretrained_tokenizer(args.src_lang, args.tgt_lang)
            max_length = min(args.max_length, tokenizer.model_max_length)
        else:
            if args.train_tokenizer:
                NMTTokenizer.train(
                    train_dataset,
                    vocab_size=args.vocab_size,
                    out_dir=f"{args.out_dir}/tokenizer",
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang,
                )
            tokenizer_model = f"{args.out_dir}/tokenizer/tokenizer.model"
            assert os.path.exists(tokenizer_model)
            tokenizer = NMTTokenizer(
                tokenizer_model, f"<{args.src_lang}>", f"<{args.tgt_lang}>"
            )
            assert tokenizer.tokenizer.GetPieceSize() == args.vocab_size
            max_length = args.max_length

        train_dataset.tokenize(tokenizer, max_length, args.finetune)
        dev_dataset.tokenize(tokenizer, max_length, args.finetune)

        trainer = Trainer(args, NMTBart if args.finetune else NMTTransformer)
        trainer.run(
            train_dataset,
            dev_dataset,
            NMTBatchCollator(args, tokenizer),
            LengthGroupBatchSampler,
        )

    if args.test:
        if args.batch_size > 1:
            setattr(args, "batch_size", 1)
        test_dataset = Iwslt2017Dataset(args, "test")

        if args.finetune:
            tokenizer = NMTBart.get_pretrained_tokenizer(args.src_lang, args.tgt_lang)
            max_length = min(args.max_length, tokenizer.model_max_length)
        else:
            tokenizer_model = f"{args.out_dir}/tokenizer/tokenizer.model"
            assert os.path.exists(tokenizer_model)
            tokenizer = NMTTokenizer(
                tokenizer_model, f"<{args.src_lang}>", f"<{args.tgt_lang}>"
            )
            assert tokenizer.tokenizer.GetPieceSize() == args.vocab_size
            max_length = args.max_length

        test_dataset.tokenize(tokenizer, max_length, args.finetune)

        tester = Tester(args, NMTBart if args.finetune else NMTTransformer)
        tester.run(
            test_dataset,
            NMTBatchCollator(args, tokenizer),
            beam_size=args.beam_size,
            max_length=args.max_length,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
