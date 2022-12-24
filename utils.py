import argparse
import random
import sys
from logging import INFO, Formatter, StreamHandler, getLogger

import numpy as np
import torch


def get_logger(name: str, level: int = INFO):
    logger = getLogger("trainer")
    logger.setLevel(INFO)
    logger.propagate = False
    hdlr = StreamHandler(sys.stdout)
    hdlr.setLevel(level)
    fmt = Formatter("%(asctime)s (%(name)s) %(levelname)s: %(message)s")
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    return logger


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
