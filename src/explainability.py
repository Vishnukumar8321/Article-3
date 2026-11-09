"""Attention-based attribution.

Deliberate wording: everything here is **attribution**, not explanation, and
certainly not causation.  Attention weights say where a trained model routed
information; they do not establish that the highlighted relationship caused the
decision.  The manuscript must use "attention-based attribution" throughout, and
the limitations section must say that attention weights are not a faithful
explanation in general (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .graph_builder import FlowGraph
from .models.base import EdgeGNN
from .models.layers import GATLayer
from .train import GraphTensors, TrainConfig, sample_message_edges
from .utils import get_logger, save_json

log = get_logger(__name__)


@torch.no_grad()
def extract_attention(
    model: EdgeGNN,
    gt: GraphTensors,
    graph: FlowGraph,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    seed: int = 0,
) -> Dict[str, object]:
    """Attention coefficients on a fixed, seeded message-passing subgraph.

    Returns per-layer ``[n_message_edges, n_heads]`` arrays together with the
    message edges they refer to, so every coefficient can be traced back to a
    concrete source/destination pair and flow record.
    """
    if not any(isinstance(m, GATLayer) for m in model.convs):
        raise TypeError("attention extraction requires a GAT-family model")

    model.eval()
    rng = np.random.default_rng(seed)
    msg = sample_message_edges(graph.edge_index, train_idx, graph.n_nodes,
                               cfg.eval_msg_edges_per_node, rng)
    msg_t = torch.tensor(msg, dtype=torch.long, device=gt.x.device)
    h, attentions = model.encode(gt.x, gt.edge_index[:, msg_t],
                                 gt.edge_attr[msg_t], collect_attention=True)
    return {
        "message_edge_idx": msg,
        "layers": [a.cpu().numpy() for a in attentions],
        "edge_index": graph.edge_index[:, msg],
        "node_ids": list(graph.node_ids),
        "seed": seed,
        "n_message_edges": int(len(msg)),
    }


def attention_by_relationship(
    att: Dict[str, object],
    graph: FlowGraph,
    task: str = "multiclass",
    class_names: Optional[Sequence[str]] = None,
    layer: int = -1,
):
    """Aggregate attention onto (source IP, destination IP) relationships.

    Returns a DataFrame with, per ordered pair and per class, the mean and total
    attention mass that the model placed on flows of that relationship.
    """
    import pandas as pd

    alphas = att["layers"][layer]                 # [E_msg, heads]
    mean_alpha = alphas.mean(axis=1)
    msg_idx = np.asarray(att["message_edge_idx"])
    src, dst = att["edge_index"]
    node_ids = att["node_ids"]
    labels = graph.labels(task)[msg_idx]
    names = list(class_names) if class_names else [str(i) for i in range(int(labels.max()) + 1)]

    frame = pd.DataFrame({
        "src_ip": [node_ids[i] for i in src],
        "dst_ip": [node_ids[i] for i in dst],
        "class": [names[i] for i in labels],
        "attention": mean_alpha,
    })
    agg = (frame.groupby(["class", "src_ip", "dst_ip"], as_index=False)
                .agg(mean_attention=("attention", "mean"),
                     total_attention=("attention", "sum"),
                     n_flows=("attention", "size")))
    agg = agg.sort_values(["class", "total_attention"], ascending=[True, False])
    return agg.reset_index(drop=True)


def influential_nodes(att: Dict[str, object], layer: int = -1, top_k: int = 15):
    """Rank endpoints by incoming and outgoing attention mass."""
    import pandas as pd

    alphas = att["layers"][layer].mean(axis=1)
    src, dst = att["edge_index"]
    node_ids = att["node_ids"]
    n = len(node_ids)
    out_mass = np.zeros(n)
    in_mass = np.zeros(n)
    np.add.at(out_mass, src, alphas)
    np.add.at(in_mass, dst, alphas)
    frame = pd.DataFrame({
        "ip": node_ids,
        "outgoing_attention": out_mass,
        "incoming_attention": in_mass,
        "total_attention": out_mass + in_mass,
    }).sort_values("total_attention", ascending=False)
    return frame.head(top_k).reset_index(drop=True)


def permutation_edge_feature_importance(
    model: EdgeGNN,
    gt: GraphTensors,
    graph: FlowGraph,
    eval_idx: np.ndarray,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    task: str,
    n_repeats: int = 5,
    seed: int = 42,
    metric: str = "f1_macro",
) -> "object":
    """Model-agnostic feature attribution by permuting one edge feature.

    Reported alongside attention because permutation importance measures the
    effect on the *decision*, which attention does not.  Computed on the
    validation split so the test set stays untouched.
    """
    import pandas as pd

    from .evaluate import compute_metrics
    from .train import predict_graph_model

    y_true = graph.labels(task)[eval_idx]
    n_classes = int(graph.labels(task).max()) + 1

    base_prob = predict_graph_model(model, gt, graph, eval_idx, train_idx, cfg, seed=seed)
    base = compute_metrics(y_true, base_prob.argmax(1), base_prob,
                           n_classes=n_classes)[metric]

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    original = gt.edge_attr.clone()
    for j, fname in enumerate(graph.edge_feature_names):
        drops: List[float] = []
        for r in range(n_repeats):
            perm = rng.permutation(len(eval_idx))
            col = gt.edge_attr[torch.tensor(eval_idx), j].clone()
            gt.edge_attr[torch.tensor(eval_idx), j] = col[torch.tensor(perm)]
            prob = predict_graph_model(model, gt, graph, eval_idx, train_idx,
                                       cfg, seed=seed)
            score = compute_metrics(y_true, prob.argmax(1), prob,
                                    n_classes=n_classes)[metric]
            drops.append(base - score)
            gt.edge_attr.copy_(original)
        rows.append({
            "feature": fname,
            f"{metric}_drop_mean": float(np.mean(drops)),
            f"{metric}_drop_std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            "n_repeats": n_repeats,
        })
    gt.edge_attr.copy_(original)
    frame = pd.DataFrame(rows).sort_values(f"{metric}_drop_mean", ascending=False)
    frame.attrs["baseline_score"] = float(base)
    frame.attrs["metric"] = metric
    frame.attrs["split"] = "validation"
    return frame.reset_index(drop=True)


ATTRIBUTION_DISCLAIMER = (
    "Attention coefficients are reported as attention-based attribution. They "
    "indicate how the trained model weighted neighbouring flows during "
    "aggregation and are not evidence of a causal relationship between a "
    "communication pattern and an attack."
)
