"""Segmentation losses used by the controlled and reproduction protocols."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TverskyLoss(nn.Module):
    """Binary Tversky loss.

    ``alpha`` weights false positives and ``beta`` weights false negatives.
    The controlled protocol uses alpha=0.3 and beta=0.7 to give missed flood
    pixels the larger penalty.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("alpha and beta must be non-negative with a positive sum")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if probabilities.shape != targets.shape:
            raise ValueError(
                f"probabilities and targets must have the same shape, got "
                f"{tuple(probabilities.shape)} and {tuple(targets.shape)}"
            )
        probabilities = probabilities.float().clamp(0.0, 1.0)
        targets = targets.float()
        if valid_mask is not None:
            valid_mask = valid_mask.float()
            probabilities = probabilities * valid_mask
            targets = targets * valid_mask
        dims = tuple(range(1, probabilities.ndim))
        true_positive = (probabilities * targets).sum(dim=dims)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=dims)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=dims)
        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return (1.0 - score).mean()


class BCEDiceLoss(nn.Module):
    """The probability-space BCE + Dice objective used by upstream CFB-Net."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = probabilities.float().clamp(1e-7, 1.0 - 1e-7)
        targets = targets.float()
        bce_map = F.binary_cross_entropy(probabilities, targets, reduction="none")
        if valid_mask is None:
            valid_mask = torch.ones_like(targets)
        valid_mask = valid_mask.float()
        bce = (bce_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        probabilities = probabilities * valid_mask
        targets = targets * valid_mask
        intersection = (probabilities * targets).sum()
        dice = (2.0 * intersection + self.eps) / (
            probabilities.sum() + targets.sum() + self.eps
        )
        return bce + 1.0 - dice


def deep_supervision_loss(
    predictions: list[torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    valid_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Apply one objective equally to every prediction head and sum the losses."""
    if not predictions:
        raise ValueError("at least one prediction is required")
    losses = torch.stack([criterion(pred, targets, valid_mask) for pred in predictions])
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("deep-supervision reduction must be 'mean' or 'sum'")
