import os
import json
import logging
from argparse import Namespace
from tqdm import tqdm
from typing import Dict, List, Any, Callable
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from utils import aggregate_step_stats, aggregate_epoch_stats

logger = logging.getLogger(__name__)


def train(
    args: Namespace,
    model_class: torch.nn.Module,
    train_dataset: DataLoader,
    dev_dataset: DataLoader,
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
):

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir + "/train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # data loaders
    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, collate_fn=collate_fn)
    logger.info(f"# train samples: {len(train_dataloader.dataset):,}")
    logger.info(f"# validate samples: {len(dev_dataloader.dataset):,}")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(args).to(device)
    logger.info(model)
    logger.info(f"# model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # scaler
    scaler = GradScaler(enabled=args.enable_amp)

    # load checkpoint
    if args.checkpoint_path:
        checkpoint_states = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint_states["scaler"])
        best_loss = checkpoint_states["best_loss"]
        start_epoch = checkpoint_states["epoch"] + 1
        if start_epoch >= args.epochs:
            raise ValueError(f"'epochs' shoule be more than checkpoint ({start_epoch})")
    else:
        best_loss = float("inf")
        start_epoch = 1

    stop_counter = 0
    train_log = []
    for epoch in range(start_epoch, args.epochs + 1):

        # train
        train_stats = train_steps(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            enable_amp=args.enable_amp,
            device=device,
            accum_grad_steps=args.accum_grad_steps,
            max_norm=args.max_norm,
            monitor_steps=args.train_monitor_steps,
            current_epoch=epoch,
            total_epoch=args.epochs,
        )

        # validate
        dev_stats = validate_steps(
            model=model, dev_dataloader=dev_dataloader, device=device
        )

        # aggregate
        train_log.append({"epoch": epoch, "train": train_stats, "validate": dev_stats})
        statement = f"epoch: {epoch}/{args.epochs}"
        for k, v in train_stats.items():
            statement += f" - train_{k}: {v:.3f}"
        for k, v in dev_stats.items():
            statement += f" - dev_{k}: {v:.3f}"
        logger.info(statement)

        # save
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
            },
            args.out_dir + "/model_checkpoint.pt",
        )
        if dev_stats["loss"] < best_loss:
            torch.save(model.state_dict(), args.out_dir + "/model_best_loss.pt")
            best_loss = dev_stats["loss"]
            stop_counter = 0
        else:
            stop_counter += 1

        # early stop
        if args.patience > 0 and stop_counter >= args.patience:
            logging.info("early stop")
            break

    with open(args.out_dir + "/train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=4, sort_keys=True)


def train_steps(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    enable_amp: bool,
    device: torch.device,
    accum_grad_steps: int,
    max_norm: float,
    monitor_steps: int,
    current_epoch: int,
    total_epoch: int,
) -> Dict[str, float]:

    model.train()
    train_stats = {"loss": 0, "metrics": []}
    for step, batch in enumerate(train_dataloader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
            enabled=enable_amp,
        ):
            outputs = model(**batch)
            loss = outputs["loss"] / accum_grad_steps

        scaler.scale(loss).backward()

        if (step % accum_grad_steps == 0) or (step == len(train_dataloader)):

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        if step % monitor_steps == 0:
            message = (
                f"epoch: {current_epoch}/{total_epoch}"
                f" - step: {step}/{len(train_dataloader)}"
            )
            for k, v in outputs["stats"].items():
                message += f" - {k}: {v:.3f}"
            logger.info(message)

        train_stats = aggregate_step_stats(batch, train_stats, outputs["stats"])

    return aggregate_epoch_stats(train_stats, len(train_dataloader.dataset))


def validate_steps(
    model: torch.nn.Module, dev_dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:

    model.eval()
    dev_stats = {"loss": 0, "metrics": []}
    for batch in tqdm(dev_dataloader, desc="validation"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        dev_stats = aggregate_step_stats(batch, dev_stats, outputs["stats"])

    return aggregate_epoch_stats(dev_stats, len(dev_dataloader.dataset))
