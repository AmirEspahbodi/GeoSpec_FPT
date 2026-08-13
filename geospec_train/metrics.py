from __future__ import annotations

from typing import Dict, Optional

import torch


class ClassificationMetrics:
    """
    Confusion-matrix based metrics.

    Computes:
      - accuracy
      - balanced accuracy
      - macro F1
      - mean vacuity
    """

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.float32,
            device="cpu",
        )
        self.total_samples = 0
        self.vacuity_sum = 0.0

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        vacuity: Optional[torch.Tensor] = None,
    ) -> None:
        preds = preds.detach().cpu().long().reshape(-1)
        labels = labels.detach().cpu().long().reshape(-1)

        if preds.numel() != labels.numel():
            raise ValueError("preds and labels must have the same number of elements.")

        preds = preds.clamp(0, self.num_classes - 1)
        labels = labels.clamp(0, self.num_classes - 1)

        idx = labels * self.num_classes + preds
        counts = torch.bincount(
            idx,
            minlength=self.num_classes * self.num_classes,
        )

        self.confusion += counts.reshape(self.num_classes, self.num_classes).float()
        self.total_samples += int(labels.numel())

        if vacuity is not None:
            vacuity = torch.nan_to_num(
                vacuity.detach().float().cpu(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            self.vacuity_sum += float(vacuity.mean().item()) * int(labels.numel())

    def compute(self) -> Dict[str, float]:
        if self.total_samples <= 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "vacuity": 0.0,
                "num_samples": 0,
            }

        confusion = self.confusion

        total = float(confusion.sum().item())
        trace = float(confusion.trace().item())
        accuracy = trace / max(1.0, total)

        tp = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)

        recall = torch.zeros(self.num_classes, dtype=torch.float32)
        precision = torch.zeros(self.num_classes, dtype=torch.float32)
        f1 = torch.zeros(self.num_classes, dtype=torch.float32)

        support_mask = support > 0
        predicted_mask = predicted > 0

        recall[support_mask] = tp[support_mask] / support[support_mask]
        precision[predicted_mask] = tp[predicted_mask] / predicted[predicted_mask]

        f1_mask = (precision + recall) > 0
        f1[f1_mask] = (
            2.0
            * precision[f1_mask]
            * recall[f1_mask]
            / (precision[f1_mask] + recall[f1_mask])
        )

        if bool(support_mask.any()):
            balanced_accuracy = float(recall[support_mask].mean().item())
            macro_f1 = float(f1[support_mask].mean().item())
        else:
            balanced_accuracy = 0.0
            macro_f1 = 0.0

        vacuity = float(self.vacuity_sum / max(1, self.total_samples))

        return {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
            "vacuity": float(vacuity),
            "num_samples": int(self.total_samples),
        }
