import random
from logging import INFO, Formatter, StreamHandler, getLogger

import numpy as np
import torch


def get_logger(name: str, level: int = INFO):
    logger = getLogger(name)
    logger.setLevel(INFO)
    logger.propagate = False
    hdlr = StreamHandler()
    hdlr.setLevel(level)
    fmt = Formatter("%(asctime)s (%(name)s) %(levelname)s: %(message)s")
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    return logger


def set_reproducibility(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
