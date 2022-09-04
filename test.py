import logging
import json
from typing import Dict, List, Any, Callable
from argparse import Namespace
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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
