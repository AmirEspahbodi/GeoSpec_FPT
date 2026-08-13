from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import TrainConfig


def _load_evidential_loss():
    """
    Import evidential_loss from GeoSpec_FPT.py.
    """
    candidate_modules = [
        "GeoSpec_FPT",
        "geospec_fpt",
        "model.GeoSpec_FPT",
        "models.GeoSpec_FPT",
    ]

    last_exc: Exception | None = None

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "evidential_loss")
        except Exception as exc:
            last_exc = exc

    raise ImportError(
        "Could not import evidential_loss. "
        "Place GeoSpec_FPT.py in the project root or make it importable."
    ) from last_exc


def finite_scalar(value: Any, device: torch.device) -> torch.Tensor:
    """
    Convert a loss-like object to a finite scalar tensor.
    Non-finite values are replaced by zero.
    """
    if value is None:
        return torch.zeros((), device=device, dtype=torch.float32)

    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=device, dtype=torch.float32)

    value = value.detach().float().mean()

    if not torch.isfinite(value):
        return torch.zeros((), device=device, dtype=torch.float32)

    return value


class LossOrchestrator:
    """
    Conservative loss orchestration for GeoSpecClassifier.

    Primary loss:
      CrossEntropy on outputs["logits"]

    Secondary loss:
      evidential_loss on outputs["alpha"]

    Auxiliary losses:
      outputs["aux_losses"]["curvature_reg"]
      outputs["aux_losses"]["disentangle_norm"]
      outputs["aux_losses"]["bayesian_prior"]
      outputs["aux_losses"]["branch_diversity"]
      outputs["sym_loss"]
      domain CE on outputs["domain_logits"] if domain labels exist
    """

    def __init__(
        self,
        cfg: TrainConfig,
        num_classes: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.num_classes = int(num_classes)
        self.device = device

        class_weight: Optional[torch.Tensor] = None
        if cfg.class_weights is not None:
            if len(cfg.class_weights) != num_classes:
                raise ValueError(
                    f"class_weights length {len(cfg.class_weights)} does not match "
                    f"num_classes {num_classes}."
                )
            class_weight = torch.tensor(
                cfg.class_weights,
                dtype=torch.float32,
                device=device,
            )

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weight)
        self.domain_ce_loss = nn.CrossEntropyLoss()

        self._evidential_loss_fn = None

    @property
    def evidential_loss_fn(self):
        if self._evidential_loss_fn is None:
            self._evidential_loss_fn = _load_evidential_loss()
        return self._evidential_loss_fn

    @staticmethod
    def _ramp(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
        """
        Returns 0 before start_epoch.
        Returns linear ramp in [0, 1] from start_epoch.
        """
        if epoch < start_epoch:
            return 0.0

        if ramp_epochs <= 0:
            return 1.0

        progress = float(epoch - start_epoch + 1) / float(ramp_epochs)
        return min(1.0, max(0.0, progress))

    def _evidential_weight(self, epoch: int) -> float:
        if self.cfg.evid_weight <= 0:
            return 0.0

        if self.cfg.evid_ramp_epochs <= 0:
            return float(self.cfg.evid_weight)

        progress = float(epoch + 1) / float(self.cfg.evid_ramp_epochs)
        progress = min(1.0, max(0.0, progress))
        return float(self.cfg.evid_weight) * progress

    def __call__(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
        domains: Optional[torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float], bool]:
        """
        Returns:
          total_loss, log_dict, is_valid
        """
        device = labels.device
        labels = labels.detach().long()

        logs: Dict[str, float] = {}
        total = torch.zeros((), device=device, dtype=torch.float32)

        # --------------------------------------------------------------
        # Primary CE loss
        # --------------------------------------------------------------
        ce_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.cfg.ce_weight > 0 and outputs.get("logits") is not None:
            logits = outputs["logits"].float()
            ce_loss = finite_scalar(self.ce_loss(logits, labels), device)

        total = total + float(self.cfg.ce_weight) * ce_loss
        logs["loss_ce"] = float(ce_loss.item())

        # --------------------------------------------------------------
        # Evidential loss
        # --------------------------------------------------------------
        evid_weight = self._evidential_weight(epoch)
        evid_loss = torch.zeros((), device=device, dtype=torch.float32)
        evid_err = torch.zeros((), device=device, dtype=torch.float32)
        evid_kl = torch.zeros((), device=device, dtype=torch.float32)

        if evid_weight > 0 and outputs.get("alpha") is not None:
            alpha = outputs["alpha"].float()

            evid_total, evid_err_detached, evid_kl_detached = self.evidential_loss_fn(
                alpha,
                labels,
                self.num_classes,
                self.cfg.evid_kl_lambda,
                epoch,
                self.cfg.evid_kl_anneal_epochs,
            )

            evid_loss = finite_scalar(evid_total, device)
            evid_err = finite_scalar(evid_err_detached, device)
            evid_kl = finite_scalar(evid_kl_detached, device)

        total = total + evid_weight * evid_loss
        logs["loss_evid_total"] = float(evid_loss.item())
        logs["loss_evid_err"] = float(evid_err.item())
        logs["loss_evid_kl"] = float(evid_kl.item())
        logs["weight_evid"] = float(evid_weight)

        # --------------------------------------------------------------
        # Fusion auxiliary losses
        # --------------------------------------------------------------
        aux_losses = outputs.get("aux_losses", {}) or {}

        auxiliary_terms = [
            ("curvature_reg", float(self.cfg.curvature_weight)),
            ("disentangle_norm", float(self.cfg.disentangle_weight)),
            ("bayesian_prior", float(self.cfg.bayesian_weight)),
        ]

        for key, weight in auxiliary_terms:
            value = torch.zeros((), device=device, dtype=torch.float32)
            if weight > 0:
                value = finite_scalar(aux_losses.get(key, None), device)

            total = total + weight * value
            logs[f"loss_{key}"] = float(value.item())
            logs[f"weight_{key}"] = float(weight)

        # --------------------------------------------------------------
        # Branch diversity loss
        # --------------------------------------------------------------
        branch_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.cfg.branch_weight > 0:
            branch_source = aux_losses.get(
                "branch_diversity",
                outputs.get("branch_diversity_loss", None),
            )
            branch_loss = finite_scalar(branch_source, device)

        total = total + float(self.cfg.branch_weight) * branch_loss
        logs["loss_branch_diversity"] = float(branch_loss.item())
        logs["weight_branch_diversity"] = float(self.cfg.branch_weight)

        # --------------------------------------------------------------
        # Symmetry/equivariance loss
        # --------------------------------------------------------------
        sym_weight = 0.0
        if self.cfg.sym_weight > 0:
            sym_weight = float(self.cfg.sym_weight) * self._ramp(
                epoch=epoch,
                start_epoch=self.cfg.sym_start_epoch,
                ramp_epochs=self.cfg.sym_ramp_epochs,
            )

        sym_loss = torch.zeros((), device=device, dtype=torch.float32)
        if sym_weight > 0:
            sym_loss = finite_scalar(outputs.get("sym_loss", None), device)

        total = total + sym_weight * sym_loss
        logs["loss_sym"] = float(sym_loss.item())
        logs["weight_sym"] = float(sym_weight)

        # --------------------------------------------------------------
        # Domain-adversarial loss
        # --------------------------------------------------------------
        domain_weight = 0.0
        if self.cfg.domain_weight > 0:
            domain_weight = float(self.cfg.domain_weight) * self._ramp(
                epoch=epoch,
                start_epoch=self.cfg.domain_start_epoch,
                ramp_epochs=self.cfg.domain_ramp_epochs,
            )

        domain_loss = torch.zeros((), device=device, dtype=torch.float32)

        if (
            domain_weight > 0
            and outputs.get("domain_logits") is not None
            and domains is not None
        ):
            domains = domains.detach().long()
            valid_domain_mask = domains >= 0

            if bool(valid_domain_mask.any()):
                domain_logits = outputs["domain_logits"].float()[valid_domain_mask]
                domain_labels = domains[valid_domain_mask]
                domain_loss = finite_scalar(
                    self.domain_ce_loss(domain_logits, domain_labels),
                    device,
                )

        total = total + domain_weight * domain_loss
        logs["loss_domain"] = float(domain_loss.item())
        logs["weight_domain"] = float(domain_weight)

        # --------------------------------------------------------------
        # Final finite check
        # --------------------------------------------------------------
        logs["loss_total"] = float(total.item())

        if not torch.isfinite(total):
            zero = torch.zeros((), device=device, dtype=torch.float32)
            logs["loss_total"] = 0.0
            return zero, logs, False

        return total, logs, True
