"""Deep-learning tabular baseline (model A in the ablation ladder)."""
from __future__ import annotations
from .base import FlowMLP


def build(edge_dim: int, n_classes: int, hidden_dim: int = 128,
          n_layers: int = 3, dropout: float = 0.2, **_) -> FlowMLP:
    return FlowMLP(in_dim=edge_dim, n_classes=n_classes, hidden=hidden_dim,
                   n_layers=n_layers, dropout=dropout)
