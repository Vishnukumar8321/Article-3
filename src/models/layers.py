"""Message-passing layers implemented directly on ``torch.scatter`` primitives.

Written from scratch rather than pulled from a library for two reasons:

1.  The edge-aware attention of section 15 needs the *flow* attribute inside the
    attention logit, which the stock GAT implementations do not expose.
2.  Keeping every layer in one file makes the ablation ladder (topology-only ->
    edge-in-message -> edge-in-attention) a matter of three boolean flags on one
    code path, so the variants differ only in what they are meant to differ in.

All layers take ``(x, edge_index, edge_attr)`` and return updated node states.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Scatter helpers
# --------------------------------------------------------------------------- #


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    shape = (n,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    summed = scatter_sum(src, index, n)
    count = torch.zeros(n, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))
    return summed / count


def scatter_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    shape = (n,) + src.shape[1:]
    out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Numerically-stable softmax over the edges incident to each target node."""
    m = scatter_max(logits.detach(), index, n)
    logits = logits - m.index_select(0, index)
    ex = logits.exp()
    denom = scatter_sum(ex, index, n).index_select(0, index)
    return ex / (denom + 1e-16)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class GCNLayer(nn.Module):
    """Symmetric-normalised graph convolution (Kipf & Welling, 2017).

    Topology only: ``edge_attr`` is accepted and ignored, by design, so that the
    ablation can attribute any difference to edge information rather than to an
    unrelated architectural change.
    """

    def __init__(self, in_dim: int, out_dim: int, **_: object) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x)
        deg = torch.zeros(n, dtype=h.dtype, device=h.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg_src = torch.zeros(n, dtype=h.dtype, device=h.device)
        deg_src.scatter_add_(0, src, torch.ones_like(src, dtype=h.dtype))
        norm = (deg_src.clamp(min=1).pow(-0.5).index_select(0, src)
                * deg.clamp(min=1).pow(-0.5).index_select(0, dst)).unsqueeze(-1)
        msg = h.index_select(0, src) * norm
        return scatter_sum(msg, dst, n) + h * deg.clamp(min=1).pow(-1).unsqueeze(-1) \
            + self.bias


class SAGELayer(nn.Module):
    """GraphSAGE mean aggregator with an explicit self transform."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0,
                 edge_in_message: bool = False, **_: object) -> None:
        super().__init__()
        self.edge_in_message = edge_in_message and edge_dim > 0
        msg_in = in_dim + (edge_dim if self.edge_in_message else 0)
        self.lin_msg = nn.Linear(msg_in, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        m = x.index_select(0, src)
        if self.edge_in_message:
            if edge_attr is None:
                raise ValueError("edge_in_message=True but edge_attr is None")
            m = torch.cat([m, edge_attr], dim=-1)
        m = self.lin_msg(m)
        agg = scatter_mean(m, dst, n)
        return agg + self.lin_self(x)


class GATLayer(nn.Module):
    """Multi-head graph attention with three switchable edge pathways.

    ``edge_in_attention``
        Adds ``U z_ij`` to the attention logit, i.e.

        ``e_ij = LeakyReLU( a^T [ W h_i || W h_j || U z_ij ] )``

        This is the edge-aware attention of section 15 and the single
        methodological difference between the GAT baseline and E-GAT-IDS.

    ``edge_in_message``
        Adds the projected flow attributes to the message being aggregated, so
        edge information reaches the node state as well as the attention weight.

    With both flags ``False`` the layer is a standard GAT (Velickovic et al.,
    2018) and is used unchanged as the attention baseline.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        edge_dim: int = 0,
        edge_in_attention: bool = False,
        edge_in_message: bool = False,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        concat: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if concat and out_dim % heads != 0:
            raise ValueError(f"out_dim {out_dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = out_dim // heads if concat else out_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_in_attention = edge_in_attention and edge_dim > 0
        self.edge_in_message = edge_in_message and edge_dim > 0

        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim if concat else self.head_dim))

        if self.edge_in_attention or self.edge_in_message:
            self.lin_edge = nn.Linear(edge_dim, heads * self.head_dim, bias=False)
            nn.init.xavier_uniform_(self.lin_edge.weight)
        if self.edge_in_attention:
            self.att_edge = nn.Parameter(torch.empty(1, heads, self.head_dim))
            nn.init.xavier_uniform_(self.att_edge)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        self._last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x).view(n, self.heads, self.head_dim)

        h_src = h.index_select(0, src)      # [E, heads, d]
        h_dst = h.index_select(0, dst)

        logit = (h_src * self.att_src).sum(-1) + (h_dst * self.att_dst).sum(-1)

        z = None
        if self.edge_in_attention or self.edge_in_message:
            if edge_attr is None:
                raise ValueError("edge pathway enabled but edge_attr is None")
            z = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
        if self.edge_in_attention:
            logit = logit + (z * self.att_edge).sum(-1)

        logit = F.leaky_relu(logit, self.negative_slope)
        alpha = scatter_softmax(logit, dst, n)                      # [E, heads]
        if return_attention:
            self._last_attention = alpha.detach()
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h_src
        if self.edge_in_message:
            msg = msg + z
        out = scatter_sum(msg * alpha.unsqueeze(-1), dst, n)

        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attention


LAYER_REGISTRY = {
    "gcn": GCNLayer,
    "sage": SAGELayer,
    "gat": GATLayer,
}
