import argparse
from typing import Dict, Any
import logging
import random
import numpy as np
import torch


def set_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
    )


def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--out_dir", default="./results", type=str)
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--enable_amp", default=False, action="store_true")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--accum_grad_steps", default=4, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_norm", default=5.0, type=float)
    parser.add_argument("--train_monitor_steps", default=50, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--patience", default=3, type=int)


def set_reproducibility(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def aggregate_step_stats(
    batch: Dict[str, torch.Tensor],
    epoch_stats: Dict[str, Any],
    step_stats: Dict[str, float],
) -> Dict[str, Any]:
    epoch_stats["loss"] += step_stats["loss"] * batch[next(iter(batch))].size(0)
    epoch_stats["metrics"].append(step_stats)
    return epoch_stats


def aggregate_epoch_stats(
    epoch_stats: Dict[str, Any], num_samples: int
) -> Dict[str, float]:
    epoch_stats["loss"] /= num_samples
    stats_keys = [k for k in epoch_stats["metrics"][0].keys() if k != "loss"]
    if len(stats_keys) > 0:
        for k in stats_keys:
            epoch_stats[k] = np.mean([stats[k] for stats in epoch_stats["metrics"]])
    epoch_stats.pop("metrics", None)
    return epoch_stats
