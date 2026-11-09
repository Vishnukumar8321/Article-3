"""Persist test-set probability scores for the PR/ROC figures.

Scores are written once, for a single reference seed, so that figures 9 and 10
are reproducible without retraining.  Writing them for every seed would add
hundreds of megabytes for no scientific gain -- the curves are illustrative and
the numbers that carry the argument are the seed-averaged PR-AUC/ROC-AUC values
already in the run records.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import CLASSICAL, PreparedData  # noqa: E402
from experiments.run_explainability import load_checkpoint  # noqa: E402
from src.models import GRAPH_MODELS, TABULAR_NEURAL_MODELS  # noqa: E402
from src.train import (  # noqa: E402
    GraphTensors, TrainConfig, predict_graph_model, predict_mlp,
)
from src.utils import RESULTS_RAW, get_logger  # noqa: E402

log = get_logger("dump_scores")


def main(representation: str = "uniflow", task: str = "binary", protocol: str = "A",
         seed: int = 42, models: Sequence[str] = ("GAT", "EdgeGAT", "E-GAT-IDS"),
         nrows: Optional[int] = None, raw_dir: Optional[Path] = None,
         experiment: str = "main") -> None:
    cfg = TrainConfig()
    data = PreparedData(representation, task, protocol, seed, nrows=nrows,
                        raw_dir=raw_dir)
    if not data.applicable:
        log.warning("configuration not applicable")
        return
    te, tr = data.splits["test"], data.splits["train"]
    bundle = {"y_true": data.y[te]}

    for name in models:
        try:
            if name in GRAPH_MODELS:
                model = load_checkpoint(name, representation, task, protocol,
                                        seed, data, experiment=experiment)
                gt = GraphTensors.from_graph(data.graph, task, device=cfg.device)
                bundle[name] = predict_graph_model(model, gt, data.graph, te, tr,
                                                   cfg, seed=seed)
            else:
                log.warning("no stored checkpoint path for %s; skipping", name)
        except FileNotFoundError as exc:
            log.warning("%s", exc)

    out = RESULTS_RAW / "scores"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{representation}_{task}_protocol{protocol}.npz"
    np.savez_compressed(path, **bundle)
    log.info("wrote %s with %d score matrices", path, len(bundle) - 1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", default="uniflow")
    p.add_argument("--task", default="binary")
    p.add_argument("--protocol", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default="GAT,EdgeGAT,E-GAT-IDS")
    p.add_argument("--nrows", type=int, default=None)
    a = p.parse_args()
    main(a.representation, a.task, a.protocol, a.seed,
         [m for m in a.models.split(",") if m], a.nrows)
