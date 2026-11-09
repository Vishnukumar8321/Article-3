"""Flow table -> directed communication multigraph.

Representation (section 10 of the specification)
------------------------------------------------
* **Nodes**  ``V`` -- one per unique IP endpoint.  The IP string is used for
  *identity only*; it never enters a feature vector.
* **Edges**  ``E`` -- one directed edge per flow record, ``e_ij = v_i -> v_j``.
  Multiple flows between the same ordered pair are kept as **parallel edges**,
  because the classification target belongs to the individual flow record.  A
  line-graph variant is provided for the representation ablation.
* **Edge attributes** ``Z`` -- the preprocessed flow feature vector.
* **Node attributes** ``X`` -- behaviour aggregates from
  :mod:`src.feature_engineering`, computed over training flows only.

Message passing
---------------
``message_edge_mask`` marks which edges may be used to propagate information.
The runners set it to the training edges, so a test flow is scored using node
representations that were never informed by test traffic.  This is the
inductive-flavoured protocol; the fully transductive variant (all edges in the
adjacency) is available as an explicit ablation and is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_engineering import (
    NodeFeatureScaler, build_node_features,
)
from .utils import get_logger, save_json

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


@dataclass
class FlowGraph:
    """A directed multigraph whose edges are individual flow records."""

    node_ids: List[str]                    # node index -> IP string
    node_features: np.ndarray              # [N, Dn] float32
    node_feature_names: List[str]
    edge_index: np.ndarray                 # [2, E] int64  (row 0 = src, 1 = dst)
    edge_attr: np.ndarray                  # [E, De] float32
    edge_feature_names: List[str]
    binary_label: np.ndarray               # [E] int64
    multiclass_label: np.ndarray           # [E] int64
    message_edge_mask: np.ndarray          # [E] bool
    scenario: np.ndarray                   # [E] object (provenance only)
    meta: Dict[str, object] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def labels(self, task: str) -> np.ndarray:
        if task == "binary":
            return self.binary_label
        if task == "multiclass":
            return self.multiclass_label
        raise ValueError(f"unknown task {task!r}")

    def statistics(self) -> Dict[str, object]:
        src, dst = self.edge_index
        pair_keys = src.astype(np.int64) * self.n_nodes + dst.astype(np.int64)
        uniq_pairs, counts = np.unique(pair_keys, return_counts=True)
        out_deg = np.bincount(src, minlength=self.n_nodes)
        in_deg = np.bincount(dst, minlength=self.n_nodes)
        self_loops = int((src == dst).sum())
        undirected_pairs = {
            tuple(sorted((int(a), int(b)))) for a, b in zip(src, dst)
        }
        return {
            "n_nodes": self.n_nodes,
            "n_edges_flow_instances": self.n_edges,
            "n_unique_ordered_pairs": int(len(uniq_pairs)),
            "n_unique_unordered_pairs": int(len(undirected_pairs)),
            "parallel_edge_multiplicity_mean": float(counts.mean()),
            "parallel_edge_multiplicity_max": int(counts.max()),
            "parallel_edge_multiplicity_median": float(np.median(counts)),
            "graph_density_ordered": float(
                len(uniq_pairs) / max(1, self.n_nodes * (self.n_nodes - 1))
            ),
            "self_loops": self_loops,
            "mean_out_degree": float(out_deg.mean()),
            "max_out_degree": int(out_deg.max()),
            "mean_in_degree": float(in_deg.mean()),
            "max_in_degree": int(in_deg.max()),
            "isolated_nodes": int(((out_deg + in_deg) == 0).sum()),
            "n_message_edges": int(self.message_edge_mask.sum()),
            "node_feature_dim": int(self.node_features.shape[1]),
            "edge_feature_dim": int(self.edge_attr.shape[1]),
            "binary_class_counts": {
                str(k): int(v) for k, v in
                zip(*np.unique(self.binary_label, return_counts=True))
            },
            "multiclass_class_counts": {
                str(k): int(v) for k, v in
                zip(*np.unique(self.multiclass_label, return_counts=True))
            },
        }


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_node_index(df: pd.DataFrame) -> Dict[str, int]:
    """Stable IP -> node id mapping over every row of the table.

    Node *identity* spans all splits (the physical network does not change
    between train and test); node *features* are still train-only.
    """
    for col in ("ip_src", "ip_dst"):
        if col not in df.columns:
            raise KeyError(f"{col} required to build the communication graph")
    ips = pd.unique(pd.concat([df["ip_src"], df["ip_dst"]], ignore_index=True))
    ips = sorted(map(str, ips))
    return {ip: i for i, ip in enumerate(ips)}


def build_flow_graph(
    df: pd.DataFrame,
    edge_attr: np.ndarray,
    edge_feature_names: Sequence[str],
    train_idx: np.ndarray,
    node_feature_source: str = "train",
    node_scaler: Optional[NodeFeatureScaler] = None,
) -> Tuple[FlowGraph, NodeFeatureScaler]:
    """Assemble the graph.

    Parameters
    ----------
    df:
        Flow table, one row per flow, already carrying ``ip_src``/``ip_dst``,
        ``binary_label``, ``multiclass_label`` and ``scenario``.
    edge_attr:
        ``[n_rows, De]`` preprocessed flow features (scaler fitted on train).
    train_idx:
        Positional indices of the training flows.  Used for both the node
        feature aggregation and the message-passing mask.
    node_feature_source:
        ``"train"`` (default, leakage-free) or ``"all"`` (transductive
        ablation -- must be reported as such).
    """
    if len(df) != len(edge_attr):
        raise ValueError(
            f"edge_attr has {len(edge_attr)} rows but df has {len(df)}"
        )
    if node_feature_source not in {"train", "all"}:
        raise ValueError("node_feature_source must be 'train' or 'all'")

    node_index = build_node_index(df)
    node_ids = [ip for ip, _ in sorted(node_index.items(), key=lambda kv: kv[1])]

    subset = np.asarray(train_idx) if node_feature_source == "train" else None
    raw_node_feats, node_names = build_node_features(df, node_index, edge_subset=subset)

    if node_scaler is None:
        node_scaler = NodeFeatureScaler().fit(raw_node_feats)
    node_feats = node_scaler.transform(raw_node_feats)

    src = df["ip_src"].astype(str).map(node_index).to_numpy(dtype=np.int64)
    dst = df["ip_dst"].astype(str).map(node_index).to_numpy(dtype=np.int64)
    edge_index = np.vstack([src, dst])

    mask = np.zeros(len(df), dtype=bool)
    mask[np.asarray(train_idx)] = True

    graph = FlowGraph(
        node_ids=node_ids,
        node_features=np.asarray(node_feats, dtype=np.float32),
        node_feature_names=node_names,
        edge_index=edge_index,
        edge_attr=np.asarray(edge_attr, dtype=np.float32),
        edge_feature_names=list(edge_feature_names),
        binary_label=df["binary_label"].to_numpy(dtype=np.int64),
        multiclass_label=df["multiclass_label"].to_numpy(dtype=np.int64),
        message_edge_mask=mask,
        scenario=df["scenario"].to_numpy(),
        meta={
            "node_feature_source": node_feature_source,
            "representation": df.attrs.get("representation", "unknown"),
            "n_train_edges": int(mask.sum()),
        },
    )
    log.info(
        "graph built: %d nodes, %d flow-edges, %d message edges, "
        "node_dim=%d edge_dim=%d (node features from %s)",
        graph.n_nodes, graph.n_edges, int(mask.sum()),
        graph.node_features.shape[1], graph.edge_attr.shape[1], node_feature_source,
    )
    return graph, node_scaler


# --------------------------------------------------------------------------- #
# Line-graph variant (Option B in the specification, kept for the ablation)
# --------------------------------------------------------------------------- #


def build_line_graph_adjacency(
    edge_index: np.ndarray, max_neighbours: int = 64, seed: int = 42,
) -> np.ndarray:
    """Connect flow-edges that share an endpoint.

    The exact line graph of a multigraph with hundreds of thousands of parallel
    edges is quadratic in the multiplicity and does not fit in memory, so
    neighbours are uniformly subsampled to ``max_neighbours`` per flow.  The
    subsampling is seeded and reported; this variant exists only to support the
    representation ablation, not the headline model.
    """
    rng = np.random.default_rng(seed)
    src, dst = edge_index
    n_edges = src.shape[0]

    by_node: Dict[int, List[int]] = {}
    for e, (s, d) in enumerate(zip(src, dst)):
        by_node.setdefault(int(s), []).append(e)
        by_node.setdefault(int(d), []).append(e)

    rows: List[int] = []
    cols: List[int] = []
    for e in range(n_edges):
        cand: List[int] = []
        for node in (int(src[e]), int(dst[e])):
            cand.extend(by_node.get(node, ()))
        cand = [c for c in cand if c != e]
        if not cand:
            continue
        if len(cand) > max_neighbours:
            cand = rng.choice(cand, size=max_neighbours, replace=False).tolist()
        rows.extend(cand)
        cols.extend([e] * len(cand))
    return np.vstack([np.asarray(rows, dtype=np.int64),
                      np.asarray(cols, dtype=np.int64)])


def save_graph_statistics(graph: FlowGraph, path) -> None:
    save_json(graph.statistics(), path)
