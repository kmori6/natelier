import logging
import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader, Subset
from jiwer import compute_measures
from data import LibriDataset
from model import ASRTransformer
from train import train
from test import test
from collator import ASRBatchCollator
from tokenizer import ASRTokenizer
from utils import add_base_arguments, set_logging, set_reproducibility

logger = logging.getLogger(__name__)


def add_specific_arguments(parser):
    parser.add_argument("--download_dir", default=None, type=str, required=True)
    parser.add_argument("--test_ratio", default=None, type=int)
    parser.add_argument("--train_subset", default="train-10h", type=str)
    parser.add_argument("--dev_subset", default="dev-clean", type=str)
    parser.add_argument("--test_subset", default="test-clean", type=str)
    # specaug
    parser.add_argument("--specaug_time_warp_window", default=5, type=int)
    parser.add_argument("--specaug_time_warp_mode", default="bicubic", type=str)
    parser.add_argument("--specaug_num_time_mask", default=2, type=int)
    parser.add_argument("--specaug_num_freq_mask", default=2, type=int)
    parser.add_argument("--specaug_time_mask_width_range", default=40, type=int)
    parser.add_argument("--specaug_freq_mask_width_range", default=30, type=int)
    # model
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--encoder_attention_heads", default=4, type=int)
    parser.add_argument("--encoder_linear_units", default=2048, type=int)
    parser.add_argument("--encoder_num_blocks", default=12, type=int)
    parser.add_argument("--encoder_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_positional_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_attention_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_input_layer", default="conv2d", type=str)
    parser.add_argument("--decoder_attention_heads", default=4, type=int)
    parser.add_argument("--decoder_linear_units", default=2048, type=int)
    parser.add_argument("--decoder_num_blocks", default=6, type=int)
    parser.add_argument("--decoder_dropout_rate", default=0.1, type=float)
    parser.add_argument("--decoder_positional_dropout_rate", default=0.1, type=float)
    parser.add_argument(
        "--decoder_self_attention_dropout_rate", default=0.1, type=float
    )
    parser.add_argument("--decoder_src_attention_dropout_rate", default=0.1, type=float)
    parser.add_argument("--ctc_loss_weight", default=0.3, type=float)
    # decode
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--ctc_decode_weight", default=0.3, type=float)


def test_steps(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    ctc_decode_weight: float,
    label2token: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    all_preds, all_truth = [], []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            best_pred = model.recognize(
                batch["input_wavs"].to(device),
                batch["input_wav_lens"].to(device),
                beam_size=beam_size,
                ctc_weight=ctc_decode_weight,
            )["tokens"]
        # remove sos/eos
        pred_labels = best_pred[1:-1]
        # remove ctc blanks
        pred_labels = [label for label in pred_labels if label != 0]
        pred_tokens = [label2token[label] for label in pred_labels]
        pred_text = "".join(pred_tokens)
        all_preds.append(pred_text.replace("|", " "))
        all_truth.append(batch["transcript"][0].replace("|", " "))
    return {"wer": compute_measures(all_truth, all_preds)["wer"]}


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = LibriDataset(args.download_dir, args.train_subset)
    dev_dataset = LibriDataset(args.download_dir, args.dev_subset)
    test_dataset = LibriDataset(args.download_dir, args.test_subset)
    if args.test_ratio:
        test_dataset = Subset(
            test_dataset, list(range(int(len(test_dataset) * args.test_ratio / 100)))
        )

    if args.train:
        assert not args.enable_amp
        tokenizer = ASRTokenizer.build_from_dataset(args, train_dataset)
        train(
            args,
            ASRTransformer,
            train_dataset,
            dev_dataset,
            ASRBatchCollator(tokenizer),
        )

    if args.test:
        if args.batch_size > 1:
            setattr(args, "batch_size", 1)
        tokenizer = ASRTokenizer.load_from_args(args)
        test(
            args,
            ASRTransformer,
            test_dataset,
            ASRBatchCollator(tokenizer, return_transcript=True),
            test_steps,
            beam_size=args.beam_size,
            ctc_decode_weight=args.ctc_decode_weight,
            label2token=tokenizer.label2token,
        )


if __name__ == "__main__":
    main()
