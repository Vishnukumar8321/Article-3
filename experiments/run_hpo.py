"""Phase 13: hyperparameter search on the validation split only.

The grid of section 21 is 3x3x3x4x3x3 = 972 configurations per model, which is
not affordable on CPU at this data scale.  Two budgets are provided and the one
used is recorded with every downstream run:

``grid``     the full specified grid (available, honest about its cost)
``random``   uniform random draws from the same space, ``n_trials`` of them
``halving``  successive halving over random draws: cheap epochs for everyone,
             full budget for the survivors

Selection is on validation macro-F1.  The test split is never loaded here --
:class:`PreparedData` builds it, but this module does not touch it, and the
saved report contains validation scores only.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import PreparedData  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.models import GRAPH_MODELS  # noqa: E402
from src.train import (  # noqa: E402
    GraphTensors, TrainConfig, predict_graph_model, train_graph_model,
)
from src.utils import DATA_METADATA, get_logger, save_json, set_seed  # noqa: E402

log = get_logger("hpo")

SEARCH_SPACE: Dict[str, List] = {
    "hidden_dim": [32, 64, 128],
    "heads": [2, 4, 8],
    "n_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3, 0.5],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4],
}

MODEL_KEYS = {"hidden_dim", "heads", "n_layers", "dropout"}
TRAIN_KEYS = {"lr", "weight_decay"}


def _sample(rng: np.random.Generator) -> Dict[str, object]:
    return {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}


def _valid(cfg: Dict[str, object]) -> bool:
    # GAT requires hidden_dim divisible by heads.
    return int(cfg["hidden_dim"]) % int(cfg["heads"]) == 0


def _evaluate(data: PreparedData, model_name: str, params: Dict[str, object],
              epochs: int, seed: int, device: str) -> Dict[str, object]:
    set_seed(seed)
    mk = {k: params[k] for k in MODEL_KEYS if k in params}
    tk = {k: params[k] for k in TRAIN_KEYS if k in params}
    cfg = TrainConfig(epochs=epochs, patience=max(5, epochs // 4), device=device, **tk)
    model = GRAPH_MODELS[model_name](
        node_dim=data.graph.node_features.shape[1],
        edge_dim=data.graph.edge_attr.shape[1],
        n_classes=data.n_classes, **mk)
    loss_fn = build_loss("ce", data.y[data.splits["train"]], data.n_classes)
    t0 = time.perf_counter()
    model, history = train_graph_model(model, data.graph, data.splits, data.task,
                                       loss_fn, cfg, seed=seed)
    gt = GraphTensors.from_graph(data.graph, data.task, device=device)
    prob = predict_graph_model(model, gt, data.graph, data.splits["val"],
                               data.splits["train"], cfg, seed=seed)
    m = compute_metrics(data.y[data.splits["val"]], prob.argmax(1), prob,
                        n_classes=data.n_classes)
    return {
        "params": params,
        "val_f1_macro": float(m["f1_macro"]),
        "val_mcc": float(m["mcc"]),
        "val_accuracy": float(m["accuracy"]),
        "best_epoch": history["best_epoch"],
        "seconds": time.perf_counter() - t0,
        "split_used_for_selection": "validation",
    }


def search(representation: str = "uniflow", task: str = "multiclass",
           protocol: str = "A", model_name: str = "E-GAT-IDS",
           strategy: str = "halving", n_trials: int = 24, seed: int = 42,
           epochs_full: int = 60, epochs_cheap: int = 12, device: str = "cpu",
           nrows: Optional[int] = None) -> Dict[str, object]:
    data = PreparedData(representation, task, protocol, seed, nrows=nrows)
    if not data.applicable:
        raise RuntimeError("configuration not applicable")
    rng = np.random.default_rng(seed)

    if strategy == "grid":
        candidates = [dict(zip(SEARCH_SPACE, combo))
                      for combo in itertools.product(*SEARCH_SPACE.values())]
        candidates = [c for c in candidates if _valid(c)]
        log.warning("full grid: %d configurations -- this is the expensive path",
                    len(candidates))
        trials = [_evaluate(data, model_name, c, epochs_full, seed, device)
                  for c in candidates]
    elif strategy == "random":
        candidates = []
        while len(candidates) < n_trials:
            c = _sample(rng)
            if _valid(c):
                candidates.append(c)
        trials = [_evaluate(data, model_name, c, epochs_full, seed, device)
                  for c in candidates]
    elif strategy == "halving":
        candidates = []
        while len(candidates) < n_trials:
            c = _sample(rng)
            if _valid(c):
                candidates.append(c)
        cheap = [_evaluate(data, model_name, c, epochs_cheap, seed, device)
                 for c in candidates]
        cheap.sort(key=lambda t: -t["val_f1_macro"])
        survivors = [t["params"] for t in cheap[:max(2, n_trials // 4)]]
        full = [_evaluate(data, model_name, c, epochs_full, seed, device)
                for c in survivors]
        trials = [{**t, "stage": "cheap"} for t in cheap] + \
                 [{**t, "stage": "full"} for t in full]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    final = [t for t in trials if t.get("stage", "full") == "full"]
    best = max(final, key=lambda t: t["val_f1_macro"])
    report = {
        "model": model_name,
        "representation": representation,
        "task": task,
        "protocol": protocol,
        "seed": seed,
        "strategy": strategy,
        "n_trials": len(trials),
        "search_space": SEARCH_SPACE,
        "full_grid_size": int(np.prod([len(v) for v in SEARCH_SPACE.values()])),
        "budget_note": (
            "Selection used validation macro-F1 only. The test split was not "
            "evaluated during the search."
        ),
        "best": best,
        "trials": trials,
    }
    out = (DATA_METADATA / "hpo" /
           f"hpo_{model_name}_{representation}_{task}_protocol{protocol}.json")
    save_json(report, out)
    log.info("best val macro-F1 %.4f with %s -> %s",
             best["val_f1_macro"], best["params"], out)
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", default="uniflow")
    p.add_argument("--task", default="multiclass")
    p.add_argument("--protocol", default="A")
    p.add_argument("--model", default="E-GAT-IDS")
    p.add_argument("--strategy", default="halving",
                   choices=["grid", "random", "halving"])
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--epochs-full", type=int, default=60)
    p.add_argument("--epochs-cheap", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--nrows", type=int, default=None)
    a = p.parse_args()
    search(a.representation, a.task, a.protocol, a.model, a.strategy, a.n_trials,
           a.seed, a.epochs_full, a.epochs_cheap, a.device, a.nrows)
