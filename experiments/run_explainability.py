"""Phase 18: attention-based attribution and permutation importance.

Loads the best E-GAT-IDS checkpoint for a configuration, extracts attention
coefficients on a fixed seeded message subgraph, aggregates them onto
communication relationships, and computes permutation feature importance on the
**validation** split so the test split stays untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import PreparedData  # noqa: E402
from src.explainability import (  # noqa: E402
    ATTRIBUTION_DISCLAIMER, attention_by_relationship, extract_attention,
    influential_nodes, permutation_edge_feature_importance,
)
from src.models import GRAPH_MODELS  # noqa: E402
from src.train import GraphTensors, TrainConfig  # noqa: E402
from src.utils import RESULTS_CKPT, RESULTS_RAW, get_logger, save_json  # noqa: E402

log = get_logger("explainability")


def load_checkpoint(model_name: str, representation: str, task: str,
                    protocol: str, seed: int, data: PreparedData,
                    experiment: str = "main"):
    stem = (f"{experiment}__{model_name}__{representation}__{task}"
            f"__protocol{protocol}__seed{seed}.pt")
    path = RESULTS_CKPT / stem
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Run the main phase for this configuration "
            "before requesting attribution."
        )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GRAPH_MODELS[model_name](
        node_dim=data.graph.node_features.shape[1],
        edge_dim=data.graph.edge_attr.shape[1],
        n_classes=data.n_classes, **ck.get("model_kwargs", {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def run(representation: str = "uniflow", task: str = "multiclass",
        protocol: str = "A", seed: int = 42, model_name: str = "E-GAT-IDS",
        cfg: Optional[TrainConfig] = None, permutation: bool = True,
        nrows: Optional[int] = None, raw_dir: Optional[Path] = None,
        experiment: str = "main", out_dir: Optional[Path] = None) -> None:
    cfg = cfg or TrainConfig()
    data = PreparedData(representation, task, protocol, seed, nrows=nrows,
                        raw_dir=raw_dir)
    if not data.applicable:
        log.warning("configuration not applicable; nothing to attribute")
        return
    model = load_checkpoint(model_name, representation, task, protocol, seed, data,
                            experiment=experiment)
    gt = GraphTensors.from_graph(data.graph, task, device=cfg.device)

    att = extract_attention(model, gt, data.graph, data.splits["train"], cfg,
                            seed=seed)
    out = Path(out_dir) if out_dir else RESULTS_RAW / "attention"
    out.mkdir(parents=True, exist_ok=True)

    rel = attention_by_relationship(att, data.graph, task=task,
                                    class_names=data.class_names)
    rel.to_csv(out / "attention_by_relationship.csv", index=False)

    nodes = influential_nodes(att)
    nodes.to_csv(out / "influential_nodes.csv", index=False)

    summary = {
        "model": model_name,
        "representation": representation,
        "task": task,
        "protocol": protocol,
        "seed": seed,
        "n_message_edges": att["n_message_edges"],
        "n_attention_layers": len(att["layers"]),
        "n_heads": int(att["layers"][0].shape[1]) if att["layers"] else 0,
        "top_relationships_per_class": {
            cls: sub.head(5)[["src_ip", "dst_ip", "total_attention", "n_flows"]]
                    .to_dict("records")
            for cls, sub in rel.groupby("class")
        },
        "top_nodes": nodes.head(10).to_dict("records"),
        "disclaimer": ATTRIBUTION_DISCLAIMER,
    }

    if permutation:
        imp = permutation_edge_feature_importance(
            model, gt, data.graph, data.splits["val"], data.splits["train"],
            cfg, task, seed=seed)
        imp.to_csv(out / "permutation_importance.csv", index=False)
        summary["permutation_importance"] = {
            "baseline_val_f1_macro": imp.attrs.get("baseline_score"),
            "split": "validation",
            "top_features": imp.head(10).to_dict("records"),
        }

    save_json(summary, out / "attention_summary.json")
    log.info("attribution written to %s", out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", default="uniflow")
    p.add_argument("--task", default="multiclass")
    p.add_argument("--protocol", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="E-GAT-IDS")
    p.add_argument("--no-permutation", action="store_true")
    p.add_argument("--nrows", type=int, default=None)
    a = p.parse_args()
    run(a.representation, a.task, a.protocol, a.seed, a.model,
        permutation=not a.no_permutation, nrows=a.nrows)
