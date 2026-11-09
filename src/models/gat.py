"""Standard GAT baselines.

``build``            -- model D: attention over topology only.
``build_edge_gat``   -- model E: flow attributes enter the *message*, but the
                        attention logit is still computed from node states
                        alone. The gap between E and E-GAT-IDS isolates
                        edge-aware attention specifically.
"""
from __future__ import annotations
from .base import EdgeGNN, EdgeGNNConfig


def _cfg(node_dim, edge_dim, n_classes, *, edge_in_message, **kw) -> EdgeGNNConfig:
    return EdgeGNNConfig(
        node_dim=node_dim, edge_dim=edge_dim, n_classes=n_classes,
        layer="gat",
        edge_in_attention=False,
        edge_in_message=edge_in_message,
        edge_in_classifier=kw.pop("edge_in_classifier", True),
        **{k: v for k, v in kw.items() if k in EdgeGNNConfig.__dataclass_fields__},
    )


def build(node_dim: int, edge_dim: int, n_classes: int, **kw) -> EdgeGNN:
    return EdgeGNN(_cfg(node_dim, edge_dim, n_classes, edge_in_message=False, **kw))


def build_edge_gat(node_dim: int, edge_dim: int, n_classes: int, **kw) -> EdgeGNN:
    return EdgeGNN(_cfg(node_dim, edge_dim, n_classes, edge_in_message=True, **kw))
