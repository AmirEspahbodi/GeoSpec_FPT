from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainConfig


def _should_no_decay(name: str, param: torch.Tensor) -> bool:
    """
    Conservative weight-decay grouping.

    No decay for:
      - biases
      - 1D parameters
      - normalization affine parameters
      - small scalar/geometric parameters
      - coarse prior logits
      - local bias-like convolution
    """
    if param.ndim <= 1:
        return True

    lower_name = name.lower()

    if lower_name.endswith(".bias"):
        return True

    no_decay_keywords = [
        "norm",
        "gamma",
        "beta",
        "theta",
        "tau",
        "phi",
        "alpha_skip",
        "einstein",
        "coarse_prior",
        "local_bias",
        "class_prototypes",
    ]

    return any(keyword in lower_name for keyword in no_decay_keywords)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay_params = []
    no_decay_params = []

    trainable_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        trainable_count += 1

        if _should_no_decay(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if trainable_count == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check Side-ViT builder and model freezing behavior."
        )

    return [
        {
            "params": decay_params,
            "weight_decay": float(weight_decay),
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> AdamW:
    param_groups = build_param_groups(
        model=model,
        weight_decay=cfg.weight_decay,
    )

    optimizer = AdamW(
        param_groups,
        lr=float(cfg.learning_rate),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_eps),
    )

    return optimizer


def build_scheduler(
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Linear warmup + cosine decay without restarts.
    Scheduler is stepped per iteration.
    """
    warmup_steps = int(float(cfg.warmup_epochs) * float(steps_per_epoch))
    total_steps = int(float(cfg.epochs) * float(steps_per_epoch))

    if total_steps <= 0:
        raise ValueError("Total training steps must be positive.")

    base_lr = float(cfg.learning_rate)
    min_lr = float(cfg.min_lr)

    if base_lr <= 0:
        raise ValueError("learning_rate must be positive.")

    min_factor = max(0.0, min_lr / base_lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            if warmup_steps <= 0:
                return 1.0
            return max(1e-6, float(step + 1) / float(warmup_steps))

        if step >= total_steps:
            return min_factor

        denominator = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denominator)
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_factor + (1.0 - min_factor) * cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
