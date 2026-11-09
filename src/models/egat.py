"""E-GAT-IDS -- the proposed edge-aware graph attention model (model F).

Difference from the GAT baseline, and nothing else:

    e_ij = LeakyReLU( a^T [ W h_i || W h_j || U z_ij ] )
    a_ij = softmax_j( e_ij )

that is, the flow attribute vector z_ij participates in the attention logit as
well as in the aggregated message and in the final edge representation.
"""
from __future__ import annotations
from .base import EdgeGNN, EdgeGNNConfig


def build(node_dim: int, edge_dim: int, n_classes: int, **kw) -> EdgeGNN:
    cfg = EdgeGNNConfig(
        node_dim=node_dim, edge_dim=edge_dim, n_classes=n_classes,
        layer="gat",
        edge_in_attention=kw.pop("edge_in_attention", True),
        edge_in_message=kw.pop("edge_in_message", True),
        edge_in_classifier=kw.pop("edge_in_classifier", True),
        **{k: v for k, v in kw.items() if k in EdgeGNNConfig.__dataclass_fields__},
    )
    return EdgeGNN(cfg)
