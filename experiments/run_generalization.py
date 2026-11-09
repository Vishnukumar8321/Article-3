"""Generalisation protocols (sections 18 and 26).

Protocol B  duplicate-aware grouped split -- removes the memorisation route that
            a random split leaves open on scan traffic.
Protocol C  source-grouped split -- entire endpoints held out.  Reported as
            *not applicable* rather than degraded when the endpoint count is too
            small to produce a valid draw, which is a real possibility on a
            twelve-sensor testbed and must be stated honestly.

Also runs the transductive node-feature contrast: node behaviour aggregated over
training edges only (the default) versus over every edge.  The gap between them
quantifies how much a transductive setup would have flattered the results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import GRAPH, NEURAL_TABULAR, run_configuration  # noqa: E402
from src.train import TrainConfig  # noqa: E402
from src.utils import RESULTS_RAW, SEEDS, get_logger, save_json  # noqa: E402

log = get_logger("generalization")

DEFAULT_MODELS = ("XGBoost", "MLP", "GAT", "EdgeGAT", "E-GAT-IDS")


def main(args) -> None:
    cfg = TrainConfig(epochs=getattr(args, "epochs", 100),
                      patience=getattr(args, "patience", 15),
                      batch_size=getattr(args, "batch_size", 8192),
                      device=getattr(args, "device", "cpu"))
    reps = getattr(args, "representations", ["uniflow"])
    tasks = getattr(args, "tasks", ["binary", "multiclass"])
    seeds = getattr(args, "seeds", SEEDS)
    models = getattr(args, "gen_models", DEFAULT_MODELS)
    nrows = getattr(args, "nrows", None)

    notes: List[str] = []
    for rep in reps:
        for task in tasks:
            for protocol in ("B", "C"):
                for seed in seeds:
                    recs = run_configuration(
                        rep, task, protocol, seed, models=models,
                        experiment="generalization", train_cfg=cfg, nrows=nrows)
                    if not recs:
                        notes.append(
                            f"protocol {protocol} produced no runs for "
                            f"{rep}/{task}/seed {seed} (see SKIPPED record)")

    # Transductive contrast, protocol A only.
    for rep in reps:
        for task in tasks:
            for seed in seeds:
                run_configuration(
                    rep, task, "A", seed,
                    models=[m for m in models if m in GRAPH],
                    experiment="transductive_contrast", train_cfg=cfg,
                    nrows=nrows, node_feature_source="all")

    save_json({
        "protocols_attempted": ["B", "C"],
        "transductive_contrast": (
            "experiment='transductive_contrast' repeats protocol A with node "
            "behavioural features aggregated over ALL flows instead of training "
            "flows only. Any gain there is leakage, not capability, and is "
            "reported as such."
        ),
        "notes": notes,
    }, RESULTS_RAW / "_generalization_manifest.json")


if __name__ == "__main__":
    from experiments.run_all import build_parser, normalise

    main(normalise(build_parser().parse_args()))
