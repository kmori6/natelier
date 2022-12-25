from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TrainOutputs:
    loss: torch.Tensor
    stats: Dict[str, float]
