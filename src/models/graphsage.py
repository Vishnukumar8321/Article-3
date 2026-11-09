"""GraphSAGE baselines.

``build``              -- topology-only mean aggregation (model C).
``build_e_graphsage``  -- GraphSAGE with the flow attributes concatenated into
                          the message, i.e. the configuration published as
                          E-GraphSAGE (Lo et al., NOMS 2022,
                          doi:10.1109/NOMS54207.2022.9789878). This is the
                          closest published competitor to the proposed model and
                          is reported as a first-class baseline, not a variant.
"""
from __future__ import annotations
from .base import EdgeGNN, EdgeGNNConfig


def build(node_dim: int, edge_dim: int, n_classes: int, **kw) -> EdgeGNN:
    cfg = EdgeGNNConfig(
        node_dim=node_dim, edge_dim=edge_dim, n_classes=n_classes,
        layer="sage",
        edge_in_attention=False,
        edge_in_message=kw.pop("edge_in_message", False),
        edge_in_classifier=kw.pop("edge_in_classifier", True),
        **{k: v for k, v in kw.items() if k in EdgeGNNConfig.__dataclass_fields__},
    )
    return EdgeGNN(cfg)


def build_e_graphsage(node_dim: int, edge_dim: int, n_classes: int,
                      **kw) -> EdgeGNN:
    """GraphSAGE with edge features in the message (E-GraphSAGE)."""
    kw.pop("edge_in_message", None)
    return build(node_dim, edge_dim, n_classes, edge_in_message=True, **kw)
