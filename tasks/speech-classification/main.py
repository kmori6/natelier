import argparse
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.data import DataLoader
from data import SpeechcommandsDataset
from model import SCTransformer
from collator import SCBatchCollator
from train import train
from test import test
from metrics import single_label_accuracy
from utils import add_base_arguments, set_logging, set_reproducibility


def add_specific_arguments(parser):
    parser.add_argument("--download_dir", default=None, type=str, required=True)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--encoder_attention_heads", default=4, type=int)
    parser.add_argument("--encoder_linear_units", default=512, type=int)
    parser.add_argument("--encoder_num_blocks", default=3, type=int)
    parser.add_argument("--encoder_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_positional_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_attention_dropout_rate", default=0.1, type=float)
    parser.add_argument("--encoder_input_layer", default="conv2d", type=str)


def test_steps(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    test_stats = dict()
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader, desc="test"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_preds.append(outputs["logits"].argmax(-1).cpu())
        all_labels.append(batch["labels"].cpu())
    test_stats["acc"] = single_label_accuracy(
        torch.concat(all_preds), torch.concat(all_labels)
    )
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    add_specific_arguments(parser)
    args = parser.parse_args()

    set_logging()
    set_reproducibility()

    train_dataset = SpeechcommandsDataset(
        download_dir=args.download_dir, subset="training"
    )
    dev_dataset = SpeechcommandsDataset(
        download_dir=args.download_dir, subset="validation"
    )
    test_dataset = SpeechcommandsDataset(
        download_dir=args.download_dir, subset="testing"
    )

    if args.train:
        command_list = set()
        for data in train_dataset:
            command_list.add(data["label"])
        command_list = sorted(list(command_list))
        setattr(args, "command_list", command_list)
        setattr(args, "num_labels", len(command_list))
        train(args, SCTransformer, train_dataset, dev_dataset, SCBatchCollator(args))

    if args.test:
        test(args, SCTransformer, test_dataset, SCBatchCollator(args), test_steps)


if __name__ == "__main__":
    main()
