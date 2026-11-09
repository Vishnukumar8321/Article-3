"""Behaviour-aware node features for the communication graph.

A node is an IP endpoint.  Its feature vector summarises the *traffic incident
to it* -- degree, peer diversity, volume, timing and port/protocol diversity --
so that a node carries meaningful information before any message passing
happens.  The alternative (a learned embedding indexed by IP) would let the
model memorise endpoint identity, which is exactly the shortcut a reviewer will
ask about on a simulated dataset with a fixed attacker address.

Leakage control
---------------
Node statistics are aggregated over a caller-supplied *subset* of flows.  The
experiment runners always pass the **training** edges only.  Test-set flow
attributes therefore never influence any node's representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger(__name__)


NODE_FEATURE_NAMES: List[str] = [
    "in_degree",
    "out_degree",
    "total_degree",
    "n_unique_in_peers",
    "n_unique_out_peers",
    "n_unique_peers",
    "in_flow_count",
    "out_flow_count",
    "in_pkt_count",
    "out_pkt_count",
    "in_bytes",
    "out_bytes",
    "mean_pkt_len_in",
    "mean_pkt_len_out",
    "mean_iat_in",
    "mean_iat_out",
    "std_iat_in",
    "std_iat_out",
    "proto_diversity",
    "src_port_diversity",
    "dst_port_diversity",
    "dst_port_entropy",
    "bytes_ratio_out_in",
    "pkts_per_flow_out",
    "pkts_per_flow_in",
]


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def build_node_features(
    df: pd.DataFrame,
    node_index: Dict[str, int],
    edge_subset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Aggregate incident-flow statistics into a ``[n_nodes, n_features]`` matrix.

    Parameters
    ----------
    df:
        The full flow table (one row per flow) with ``ip_src``/``ip_dst``.
    node_index:
        Mapping from IP string to node id.  Built over *all* rows so that node
        ids are stable across splits; the *features* still come only from
        ``edge_subset``.
    edge_subset:
        Positional row indices of the flows allowed to contribute statistics.
        Pass the training indices.  ``None`` uses every row -- only appropriate
        for the transductive ablation, and it must be labelled as such.
    """
    work = df if edge_subset is None else df.iloc[np.asarray(edge_subset)]
    if edge_subset is None:
        log.warning(
            "build_node_features called with edge_subset=None: node features "
            "will see every flow, including test flows. Use only for the "
            "explicitly-labelled transductive ablation."
        )

    n_nodes = len(node_index)
    src = work["ip_src"].map(node_index).to_numpy()
    dst = work["ip_dst"].map(node_index).to_numpy()
    if np.isnan(src.astype(float)).any() or np.isnan(dst.astype(float)).any():
        raise ValueError("Unmapped IP encountered; node_index must cover all rows.")
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)

    num_pkts = _safe_col(work, "num_pkts").to_numpy()
    num_bytes = _safe_col(work, "num_bytes").to_numpy()
    mean_pkt_len = _safe_col(work, "mean_pkt_len").to_numpy()
    mean_iat = _safe_col(work, "mean_iat").to_numpy()
    std_iat = _safe_col(work, "std_iat").to_numpy()
    proto = _safe_col(work, "proto").to_numpy()
    prt_src = _safe_col(work, "prt_src").to_numpy()
    prt_dst = _safe_col(work, "prt_dst").to_numpy()

    def agg_sum(idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
        out = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(out, idx, vals)
        return out

    out_flow = agg_sum(src, np.ones(len(work)))
    in_flow = agg_sum(dst, np.ones(len(work)))
    out_pkts = agg_sum(src, num_pkts)
    in_pkts = agg_sum(dst, num_pkts)
    out_bytes = agg_sum(src, num_bytes)
    in_bytes = agg_sum(dst, num_bytes)
    out_plen_sum = agg_sum(src, mean_pkt_len)
    in_plen_sum = agg_sum(dst, mean_pkt_len)
    out_iat_sum = agg_sum(src, mean_iat)
    in_iat_sum = agg_sum(dst, mean_iat)
    out_siat_sum = agg_sum(src, std_iat)
    in_siat_sum = agg_sum(dst, std_iat)

    # Distinct-set statistics. Built with dict-of-sets: the node count here is
    # small (tens of endpoints) even though the edge count is large.
    out_peers = [set() for _ in range(n_nodes)]
    in_peers = [set() for _ in range(n_nodes)]
    protos = [dict() for _ in range(n_nodes)]
    sports = [set() for _ in range(n_nodes)]
    dports = [dict() for _ in range(n_nodes)]

    for s, d, pr, ps, pd_ in zip(src, dst, proto, prt_src, prt_dst):
        out_peers[s].add(d)
        in_peers[d].add(s)
        protos[s][pr] = protos[s].get(pr, 0) + 1
        protos[d][pr] = protos[d].get(pr, 0) + 1
        sports[s].add(ps)
        dports[d][pd_] = dports[d].get(pd_, 0) + 1

    n_out_peers = np.array([len(x) for x in out_peers], dtype=np.float64)
    n_in_peers = np.array([len(x) for x in in_peers], dtype=np.float64)
    n_peers = np.array(
        [len(out_peers[i] | in_peers[i]) for i in range(n_nodes)], dtype=np.float64
    )
    proto_div = np.array([len(x) for x in protos], dtype=np.float64)
    sport_div = np.array([len(x) for x in sports], dtype=np.float64)
    dport_div = np.array([len(x) for x in dports], dtype=np.float64)
    dport_ent = np.array(
        [_entropy(np.array(list(x.values()), dtype=np.float64)) if x else 0.0
         for x in dports],
        dtype=np.float64,
    )

    eps = 1e-9
    feats = np.column_stack([
        in_flow,                                   # in_degree (parallel edges)
        out_flow,                                  # out_degree
        in_flow + out_flow,                        # total_degree
        n_in_peers,
        n_out_peers,
        n_peers,
        in_flow,                                   # in_flow_count
        out_flow,                                  # out_flow_count
        in_pkts,
        out_pkts,
        in_bytes,
        out_bytes,
        in_plen_sum / (in_flow + eps),
        out_plen_sum / (out_flow + eps),
        in_iat_sum / (in_flow + eps),
        out_iat_sum / (out_flow + eps),
        in_siat_sum / (in_flow + eps),
        out_siat_sum / (out_flow + eps),
        proto_div,
        sport_div,
        dport_div,
        dport_ent,
        out_bytes / (in_bytes + eps),
        out_pkts / (out_flow + eps),
        in_pkts / (in_flow + eps),
    ]).astype(np.float64)

    assert feats.shape[1] == len(NODE_FEATURE_NAMES), (
        f"{feats.shape[1]} columns vs {len(NODE_FEATURE_NAMES)} names"
    )
    return feats, list(NODE_FEATURE_NAMES)


@dataclass
class NodeFeatureScaler:
    """log1p + standardise, with statistics taken from the supplied matrix only."""

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    log_cols_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NodeFeatureScaler":
        X = np.asarray(X, dtype=np.float64)
        # log1p any column whose dynamic range spans orders of magnitude
        self.log_cols_ = (np.nanmax(np.abs(X), axis=0) > 1e3)
        Z = X.copy()
        Z[:, self.log_cols_] = np.log1p(np.clip(Z[:, self.log_cols_], 0, None))
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("NodeFeatureScaler used before fit")
        Z = np.asarray(X, dtype=np.float64).copy()
        Z[:, self.log_cols_] = np.log1p(np.clip(Z[:, self.log_cols_], 0, None))
        Z = (Z - self.mean_) / self.std_
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
