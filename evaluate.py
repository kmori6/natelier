import json
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_logger

logger = get_logger("tester")


class Tester(ABC):
    def __init__(self, args: Namespace, model_class: nn.Module):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(args.out_dir + "/train_args.json", "r") as f:
            self.train_args = Namespace(**json.load(f))
        self.model = model_class(self.train_args)
        self.model.load_state_dict(
            torch.load(args.out_dir + "/model_best_loss.pt", map_location=self.device)
        )
        self.model.to(self.device)

    def run(
        self,
        test_dataset: DataLoader,
        collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
        **kwargs,
    ):
        test_dataloader = DataLoader(
            test_dataset, self.args.batch_size, collate_fn=collate_fn
        )
        test_stats = self.test_epoch(test_dataloader, **kwargs)
        self.save_results(test_stats)

    @abstractmethod
    def test_epoch(self) -> Dict[str, float]:
        raise NotImplementedError

    def save_results(self, test_stats: Dict[str, float]):
        for k, v in test_stats.items():
            logger.info(f"test_{k}: {v:.3f}")
        with open(self.args.out_dir + "/results.json", "w", encoding="utf-8") as f:
            json.dump(test_stats, f, indent=4, ensure_ascii=False)


def test(
    args: Namespace,
    model_class: torch.nn.Module,
    test_dataset: DataLoader,
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
    test_steps: Callable[[Any], Dict[str, float]],
    **kwargs,
):

    with open(args.out_dir + "/train_args.json", "r") as f:
        train_args = Namespace(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(train_args).to(device)
    model.load_state_dict(
        torch.load(
            args.out_dir + "/model_best_loss.pt",
            map_location=lambda storage, loc: storage,
        )
    )

    test_stats = test_steps(
        model,
        DataLoader(test_dataset, args.batch_size, collate_fn=collate_fn),
        device,
        **kwargs,
    )

    # save results
    if test_stats:
        for k, v in test_stats.items():
            logger.info(f"test_{k}: {v:.3f}")
        with open(args.out_dir + "/results.json", "w", encoding="utf-8") as f:
            json.dump(test_stats, f, indent=4)
