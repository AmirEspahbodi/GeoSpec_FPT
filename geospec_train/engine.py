from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .losses import LossOrchestrator
from .metrics import ClassificationMetrics
from .optim import build_optimizer, build_scheduler
from .utils import JsonlLogger, get_logger


def _prefix_dict(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg

        self.device = self._resolve_device(cfg.device)
        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")

        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(
            cfg=cfg,
            optimizer=self.optimizer,
            steps_per_epoch=len(self.train_loader),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.loss_orchestrator = LossOrchestrator(
            cfg=cfg,
            num_classes=cfg.num_classes,
            device=self.device,
        )

        self.train_metrics = ClassificationMetrics(cfg.num_classes)
        self.val_metrics = ClassificationMetrics(cfg.num_classes)
        self.test_metrics = ClassificationMetrics(cfg.num_classes)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.json_logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.console_logger = get_logger("geospec_train", self.output_dir)

        self.best_metric = float("-inf")
        self.start_epoch = 0

        if cfg.resume:
            self.start_epoch, self.best_metric = load_checkpoint(
                path=cfg.resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            self.start_epoch += 1
            self.console_logger.info(
                f"Resumed from checkpoint {cfg.resume} at epoch {self.start_epoch}."
            )

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if len(self.trainable_params) == 0:
            raise RuntimeError("No trainable parameters found.")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device_obj

    def _move(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)

        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(x) for x in obj)

        return obj

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.train_metrics.reset()

        nonfinite_steps = 0

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)

            images, key_states, value_states, labels, domains = self._move(batch)

            # Symmetry pass is expensive; only compute it when it will be used.
            compute_equivariance = (
                self.cfg.sym_weight > 0 and epoch >= self.cfg.sym_start_epoch
            )

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    images,
                    key_states,
                    value_states,
                    use_causal_mask=True,
                    compute_equivariance=compute_equivariance,
                    suppress_side_effects=False,
                )

            total_loss, loss_logs, is_valid = self.loss_orchestrator(
                outputs=outputs,
                labels=labels,
                domains=domains,
                epoch=epoch,
            )

            if not is_valid:
                nonfinite_steps += 1
                self.json_logger.write(
                    {
                        "event": "nonfinite_loss",
                        "epoch": epoch,
                        "step": step,
                        **loss_logs,
                    }
                )

                if nonfinite_steps >= self.cfg.max_nonfinite_steps:
                    raise RuntimeError(
                        f"Encountered {nonfinite_steps} non-finite loss steps. "
                        "Aborting training for safety."
                    )

                continue

            probs = outputs["probs"].detach().float()
            preds = probs.argmax(dim=-1)

            self.train_metrics.update(
                preds=preds,
                labels=labels,
                probs=probs,
                vacuity=outputs.get("vacuity", None),
            )

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    max_norm=self.cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    max_norm=self.cfg.grad_clip,
                )
                self.optimizer.step()

            self.scheduler.step()

            if step % self.cfg.log_interval == 0:
                current_lr = float(self.scheduler.get_last_lr()[0])

                grad_norm_value = (
                    float(grad_norm.item())
                    if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm)
                    else 0.0
                )

                payload = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "lr": current_lr,
                    "grad_norm": grad_norm_value,
                    **loss_logs,
                }

                self.json_logger.write(payload)
                self.console_logger.info(
                    "epoch={} step={} loss={:.6f} ce={:.6f} evid={:.6f} lr={:.2e}".format(
                        epoch,
                        step,
                        loss_logs.get("loss_total", 0.0),
                        loss_logs.get("loss_ce", 0.0),
                        loss_logs.get("loss_evid_total", 0.0),
                        current_lr,
                    )
                )

        return self.train_metrics.compute()

    def _validate(
        self,
        data_loader: DataLoader,
        metrics: ClassificationMetrics,
    ) -> Dict[str, float]:
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in data_loader:
                images, key_states, value_states, labels, domains = self._move(batch)

                # Conservative evaluation:
                # - no equivariance side pass
                # - no side effects
                # - causal mask active
                # - FP32 forward for stable metrics
                outputs = self.model(
                    images,
                    key_states,
                    value_states,
                    use_causal_mask=True,
                    compute_equivariance=False,
                    suppress_side_effects=True,
                )

                probs = outputs["probs"].float()
                preds = probs.argmax(dim=-1)

                metrics.update(
                    preds=preds,
                    labels=labels,
                    probs=probs,
                    vacuity=outputs.get("vacuity", None),
                )

        return metrics.compute()

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_metric: float,
    ) -> None:
        save_checkpoint(
            path=self.output_dir / filename,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            best_metric=best_metric,
            cfg=self.cfg,
        )

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_metrics = self._train_epoch(epoch)

            payload: Dict[str, Any] = {
                "event": "train_epoch",
                "epoch": epoch,
                **_prefix_dict("train", train_metrics),
            }

            indicator = float(train_metrics.get("balanced_accuracy", 0.0))

            if (
                self.val_loader is not None
                and (epoch + 1) % self.cfg.eval_interval == 0
            ):
                val_metrics = self._validate(self.val_loader, self.val_metrics)
                indicator = float(val_metrics.get("balanced_accuracy", 0.0))

                payload.update(
                    {
                        "event": "eval_epoch",
                        "epoch": epoch,
                        **_prefix_dict("val", val_metrics),
                        "best_metric": self.best_metric,
                    }
                )

            is_best = indicator > self.best_metric

            if is_best:
                self.best_metric = indicator
                self._save_checkpoint(
                    "best.pt", epoch=epoch, best_metric=self.best_metric
                )

            if (epoch + 1) % self.cfg.save_interval == 0:
                self._save_checkpoint(
                    "last.pt", epoch=epoch, best_metric=self.best_metric
                )

            payload["best_metric"] = self.best_metric
            payload["indicator"] = indicator
            self.json_logger.write(payload)

            self.console_logger.info(
                "epoch={} train_acc={:.4f} train_bacc={:.4f} indicator={:.4f} best={:.4f}".format(
                    epoch,
                    train_metrics.get("accuracy", 0.0),
                    train_metrics.get("balanced_accuracy", 0.0),
                    indicator,
                    self.best_metric,
                )
            )

        # Final last checkpoint.
        self._save_checkpoint(
            "final.pt", epoch=self.cfg.epochs - 1, best_metric=self.best_metric
        )

    def evaluate_best_on_test(self) -> Dict[str, float]:
        if self.test_loader is None:
            self.console_logger.warning(
                "No test loader provided. Skipping test evaluation."
            )
            return {}

        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(
                path=best_path,
                model=self.model,
                optimizer=None,
                scheduler=None,
                scaler=None,
            )
            self.console_logger.info(f"Loaded best checkpoint for test: {best_path}")
        else:
            self.console_logger.warning(
                "No best.pt found. Evaluating current in-memory model on test set."
            )

        test_metrics = self._validate(self.test_loader, self.test_metrics)

        payload = {
            "event": "test",
            **_prefix_dict("test", test_metrics),
        }

        self.json_logger.write(payload)

        self.console_logger.info("Test results:")
        for key, value in test_metrics.items():
            self.console_logger.info(f"  {key}: {value}")

        return test_metrics
