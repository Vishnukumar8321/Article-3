"""Objectives for imbalanced edge classification.

Default is class-weighted cross entropy (section 20).  Focal loss is provided
for the loss ablation.  Graph-aware resampling is implemented as a *sampler*
over training edges rather than as synthetic minority generation: SMOTE-style
interpolation between flow records would fabricate communication events that
never occurred, which is not defensible on a graph whose edges are supposed to
be observed traffic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(y: np.ndarray, n_classes: int, scheme: str = "balanced",
                  clip: Optional[float] = 50.0) -> torch.Tensor:
    """Inverse-frequency weights computed from the **training** labels only."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.ones(n_classes, dtype=np.float64)
    if scheme == "balanced":
        w[present] = counts[present].sum() / (present.sum() * counts[present])
    elif scheme == "sqrt_balanced":
        inv = counts[present].sum() / (present.sum() * counts[present])
        w[present] = np.sqrt(inv)
    elif scheme == "none":
        w[:] = 1.0
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    if clip is not None:
        w = np.clip(w, 1.0 / clip, clip)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), optionally class-weighted."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).gather(0, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, y_train: np.ndarray, n_classes: int,
               weighting: str = "balanced", gamma: float = 2.0,
               label_smoothing: float = 0.0) -> nn.Module:
    name = name.lower()
    w = class_weights(y_train, n_classes, scheme=weighting)
    if name in {"ce", "cross_entropy", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=w)
    if name == "unweighted_ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss {name!r}")


def balanced_edge_sampler_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-edge sampling weights for graph-aware resampling.

    Resamples *existing* edges rather than synthesising new ones, so every
    training example remains a real observed flow.
    """
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    per_class = 1.0 / counts
    w = per_class[np.asarray(y)]
    return w / w.sum()
