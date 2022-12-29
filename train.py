import json
from argparse import ArgumentParser
import os
from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import get_logger, set_reproducibility

logger = get_logger("trainer")


class Trainer:
    def __init__(self, args: Namespace, model_class: nn.Module):
        self.args = args
        set_reproducibility()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(args).to(self.device)
        self.early_stop_counter = 0

    def run(
        self,
        train_dataset: DataLoader,
        dev_dataset: DataLoader,
        collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
    ):
        os.makedirs(self.args.out_dir, exist_ok=True)
        train_dataloader, dev_dataloader = self.build_dataloaders(
            train_dataset, dev_dataset, collate_fn
        )
        optimizer = self.build_optimizer()
        scaler = self.build_scaler()
        self.load_checkpoint(optimizer, scaler)
        self.display_model_stats()
        log = []
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            # train
            train_stats = self.train_epoch(
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scaler=scaler,
                current_epoch=epoch,
            )
            # validate
            dev_stats = self.validate_epoch(dev_dataloader)
            # aggregate
            log.append({"epoch": epoch, "train": train_stats, "validate": dev_stats})
            statement = f"epoch: {epoch}/{self.args.epochs}"
            for k, v in train_stats.items():
                statement += f" - train_{k}: {v:.3f}"
            for k, v in dev_stats.items():
                statement += f" - dev_{k}: {v:.3f}"
            logger.info(statement)
            # save
            self.save_best_model(dev_stats["loss"])
            self.save_checkpoint(epoch, optimizer, scaler)
            # early stop
            if self.early_stop(self.args.patience):
                logger.info(f"early stop training at epoch {epoch}")
                break
        self.save_log(log)
        self.save_train_args()

    def build_dataloaders(
        self, train_dataset: Dataset, dev_dataset: Dataset, collate_fn: Callable
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            train_dataset,
            self.args.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
        )
        dev_dataloader = DataLoader(
            dev_dataset, self.args.batch_size, collate_fn=collate_fn
        )
        logger.info(f"# train samples: {len(train_dataloader.dataset):,}")
        logger.info(f"# validate samples: {len(dev_dataloader.dataset):,}")
        return train_dataloader, dev_dataloader

    def build_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay
        )

    def build_scaler(self) -> GradScaler:
        return GradScaler(enabled=self.args.enable_amp)

    def load_checkpoint(self, optimizer: optim.Optimizer, scaler: GradScaler):
        if self.args.checkpoint_path:
            checkpoint = torch.load(self.args.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler"])
            self.best_loss = checkpoint["best_loss"]
            self.start_epoch = checkpoint["epoch"] + 1
            if self.start_epoch >= self.args.epochs:
                raise ValueError(
                    f"'epochs' shoule be more than checkpoint ({self.start_epoch})"
                )
        else:
            self.best_loss = float("inf")
            self.start_epoch = 1

    def display_model_stats(self):
        params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(self.model)
        logger.info(f"# model parameters: {params:,}")
        logger.info(f"# trainable parameters: {trainable:,}")

    def save_train_args(self):
        with open(self.args.out_dir + "/train_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        current_epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        train_stats = {"loss": 0, "metrics": []}
        for step, batch in enumerate(train_dataloader, 1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
                enabled=self.args.enable_amp,
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / self.args.accum_grad_steps
            scaler.scale(loss).backward()
            if (step % self.args.accum_grad_steps == 0) or (
                step == len(train_dataloader)
            ):
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if step % self.args.train_monitor_steps == 0:
                message = (
                    f"epoch: {current_epoch}/{self.args.epochs}"
                    f" - step: {step}/{len(train_dataloader)}"
                )
                for k, v in outputs.stats.items():
                    message += f" - {k}: {v:.3f}"
                logger.info(message)
            train_stats = self.aggregate_step_stats(batch, train_stats, outputs.stats)
        return self.aggregate_epoch_stats(train_stats, len(train_dataloader.dataset))

    def validate_epoch(self, dev_dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        dev_stats = {"loss": 0, "metrics": []}
        for batch in tqdm(dev_dataloader, desc="validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            dev_stats = self.aggregate_step_stats(batch, dev_stats, outputs.stats)
        return self.aggregate_epoch_stats(dev_stats, len(dev_dataloader.dataset))

    @staticmethod
    def aggregate_step_stats(
        batch: Dict[str, torch.Tensor],
        epoch_stats: Dict[str, Any],
        step_stats: Dict[str, float],
    ) -> Dict[str, Any]:
        batch_size = batch[next(iter(batch))].size(0)
        epoch_stats["loss"] += step_stats["loss"] * batch_size
        epoch_stats["metrics"].append(step_stats)
        return epoch_stats

    @staticmethod
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

    def save_best_model(self, dev_loss: float):
        if dev_loss < self.best_loss:
            file_path = self.args.out_dir + "/model_best_loss.pt"
            torch.save(self.model.state_dict(), file_path)
            self.best_loss = dev_loss
            self.early_stop_counter = 0
            logger.info(f"saved the best loss model at {file_path}")
        else:
            self.early_stop_counter += 1

    def save_checkpoint(
        self, epoch: int, optimizer: optim.Optimizer, scaler: GradScaler
    ):
        file_path = self.args.out_dir + "/model_checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": self.best_loss,
            },
            file_path,
        )
        logger.info(f"saved the model checkpoint at {file_path}")

    def early_stop(self, early_stop_patience: int) -> bool:
        return (
            early_stop_patience > 0 and self.early_stop_counter >= early_stop_patience
        )

    def save_log(self, train_log: Dict[str, Any]):
        file_path = self.args.out_dir + "/train_log.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=4, sort_keys=True)
        logger.info(f"saved the training log at {file_path}")

    @staticmethod
    def add_train_args(parser: ArgumentParser):
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
