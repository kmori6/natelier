from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ModelOutputs:
    loss: torch.Tensor = None
    stats: Dict[str, float] = None
    logits: torch.Tensor = None
