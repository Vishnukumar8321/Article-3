"""Edge-classification backbone shared by every graph model.

Pipeline
--------
::

    X (node behaviour features)
        -> node projection
        -> L message-passing layers over the *training* edges
        -> h_i , h_j
    r_ij = [ h_i || h_j || z_ij ]
        -> edge fusion
        -> MLP
        -> y_ij

Every graph baseline and the proposed model instantiate this same backbone and
differ only in (a) which layer class is used and (b) three boolean edge
pathways.  Keeping the scaffolding identical is what makes the ablation in
section 25 an ablation rather than a comparison of unrelated codebases.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GATLayer, LAYER_REGISTRY


@dataclass
class EdgeGNNConfig:
    node_dim: int
    edge_dim: int
    n_classes: int
    layer: str = "gat"                 # gcn | sage | gat
    hidden_dim: int = 64
    n_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    edge_in_attention: bool = False
    edge_in_message: bool = False
    edge_in_classifier: bool = True
    use_node_features: bool = True
    residual: bool = True
    layer_norm: bool = True
    classifier_hidden: int = 128
    edge_proj_dim: int = 0             # 0 = feed raw edge features to the classifier

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EdgeGNN(nn.Module):
    """Graph encoder + edge-level classifier."""

    def __init__(self, cfg: EdgeGNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.layer not in LAYER_REGISTRY:
            raise ValueError(f"unknown layer {cfg.layer!r}")
        LayerCls = LAYER_REGISTRY[cfg.layer]

        node_in = cfg.node_dim if cfg.use_node_features else 1
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        if not cfg.use_node_features:
            # Constant input: the node state then carries only what message
            # passing puts there. Used by the "no node behavioural features"
            # ablation.
            self.register_buffer("_const_node", torch.ones(1, 1))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.convs.append(
                LayerCls(
                    in_dim=cfg.hidden_dim,
                    out_dim=cfg.hidden_dim,
                    heads=cfg.heads,
                    edge_dim=cfg.edge_dim,
                    edge_in_attention=cfg.edge_in_attention,
                    edge_in_message=cfg.edge_in_message,
                    dropout=cfg.dropout,
                )
            )
            self.norms.append(
                nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
            )

        self.edge_proj: Optional[nn.Module] = None
        edge_clf_dim = 0
        if cfg.edge_in_classifier and cfg.edge_dim > 0:
            if cfg.edge_proj_dim > 0:
                self.edge_proj = nn.Sequential(
                    nn.Linear(cfg.edge_dim, cfg.edge_proj_dim),
                    nn.ReLU(),
                )
                edge_clf_dim = cfg.edge_proj_dim
            else:
                edge_clf_dim = cfg.edge_dim

        fusion_in = 2 * cfg.hidden_dim + edge_clf_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, cfg.classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.classifier_hidden // 2, cfg.n_classes)

    # ------------------------------------------------------------------ #

    def encode(
        self,
        x: torch.Tensor,
        msg_edge_index: torch.Tensor,
        msg_edge_attr: Optional[torch.Tensor],
        collect_attention: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run message passing and return final node states (+ attention)."""
        if not self.cfg.use_node_features:
            x = self._const_node.expand(x.size(0), 1)
        h = self.node_proj(x)
        attentions: List[torch.Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            if isinstance(conv, GATLayer):
                out = conv(h, msg_edge_index, msg_edge_attr,
                           return_attention=collect_attention)
                if collect_attention and conv.last_attention is not None:
                    attentions.append(conv.last_attention)
            else:
                out = conv(h, msg_edge_index, msg_edge_attr)
            out = norm(out)
            out = F.elu(out)
            out = F.dropout(out, p=self.cfg.dropout, training=self.training)
            h = h + out if self.cfg.residual else out
        return h, attentions

    def classify_edges(
        self,
        h: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        src, dst = target_edge_index[0], target_edge_index[1]
        parts = [h.index_select(0, src), h.index_select(0, dst)]
        if self.cfg.edge_in_classifier and self.cfg.edge_dim > 0:
            if target_edge_attr is None:
                raise ValueError("edge_in_classifier=True but no edge attributes given")
            z = target_edge_attr
            if self.edge_proj is not None:
                z = self.edge_proj(z)
            parts.append(z)
        r = torch.cat(parts, dim=-1)
        return self.head(self.fusion(r))

    def forward(
        self,
        x: torch.Tensor,
        msg_edge_index: torch.Tensor,
        msg_edge_attr: Optional[torch.Tensor],
        target_edge_index: torch.Tensor,
        target_edge_attr: Optional[torch.Tensor],
        collect_attention: bool = False,
    ) -> torch.Tensor:
        h, _ = self.encode(x, msg_edge_index, msg_edge_attr,
                           collect_attention=collect_attention)
        return self.classify_edges(h, target_edge_index, target_edge_attr)


class FlowMLP(nn.Module):
    """Tabular deep-learning baseline: the flow vector alone, no graph at all.

    This is model **A** in the ablation ladder and the control for RQ1: if the
    graph contributes nothing, this model matches the GNNs.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128,
                 n_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for i in range(n_layers):
            out = hidden if i < n_layers - 1 else hidden // 2
            layers += [nn.Linear(d, out), nn.BatchNorm1d(out), nn.ReLU(),
                       nn.Dropout(dropout)]
            d = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(z))
