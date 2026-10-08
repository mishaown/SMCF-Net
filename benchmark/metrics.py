"""Globally accumulated binary flood-segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @torch.no_grad()
    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        predicted = probabilities >= threshold
        expected = targets >= 0.5
        valid = torch.ones_like(expected, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        self.true_positive += int((predicted & expected & valid).sum().item())
        self.false_positive += int((predicted & ~expected & valid).sum().item())
        self.true_negative += int((~predicted & ~expected & valid).sum().item())
        self.false_negative += int((~predicted & expected & valid).sum().item())

    def compute(self) -> dict[str, float]:
        tp, fp = self.true_positive, self.false_positive
        tn, fn = self.true_negative, self.false_negative
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        total = tp + tn + fp + fn
        expected_agreement = (
            (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)
        ) / (total * total + eps)
        kappa = (accuracy - expected_agreement) / (1.0 - expected_agreement + eps)
        return {
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "kappa": kappa,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
