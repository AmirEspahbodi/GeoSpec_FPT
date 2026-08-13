from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import TrainConfig


def save_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    best_metric: float,
    cfg: TrainConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "best_metric": float(best_metric),
        "config": cfg.to_dict(),
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> tuple[int, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain 'model'.")

    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and scaler.is_enabled() and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", float("-inf")))

    return epoch, best_metric
