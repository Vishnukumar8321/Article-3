"""GCN baseline (model B): topology-only message passing.

Edge attributes reach the classifier but never the aggregation, which is what
distinguishes it from the edge-aware variants.
"""
from __future__ import annotations
from .base import EdgeGNN, EdgeGNNConfig


def build(node_dim: int, edge_dim: int, n_classes: int, **kw) -> EdgeGNN:
    cfg = EdgeGNNConfig(
        node_dim=node_dim, edge_dim=edge_dim, n_classes=n_classes,
        layer="gcn",
        edge_in_attention=False, edge_in_message=False,
        edge_in_classifier=kw.pop("edge_in_classifier", True),
        **{k: v for k, v in kw.items() if k in EdgeGNNConfig.__dataclass_fields__},
    )
    return EdgeGNN(cfg)
