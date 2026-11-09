"""Main experimental phase driver.

Ordered so that the protocols that can actually discriminate between methods run
first: if the machine is interrupted, the scientifically load-bearing cells are
already on disk.

    C   unseen victim endpoints   -- discriminating
    CS  unseen source endpoints   -- discriminating
    B   duplicate-aware grouped   -- saturated, but it is the evidence for the
    A   random stratified            saturation claim itself

Within a cell the models run cheap-and-decisive first, expensive-and-slow last,
for the same reason.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_configuration  # noqa: E402
from src.train import TrainConfig  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_RAW, SEEDS, get_logger, save_json,
)

log = get_logger("main_phase")

MODELS = ["EndpointLookup", "MLP", "GCN", "GraphSAGE", "E-GraphSAGE", "GAT",
          "EdgeGAT", "E-GAT-IDS", "XGBoost", "RandomForest", "SVM", "KNN",
          "GA-EBT"]
PROTOCOLS = ["C", "CS", "B", "A"]
REPRESENTATIONS = ["uniflow", "biflow"]
TASKS = ["multiclass", "binary"]

CFG = TrainConfig(epochs=60, patience=12, batch_size=8192,
                  max_msg_edges_per_node=2000, eval_msg_edges_per_node=4000,
                  log_every=20)

# Reduced from the published Algorithm 1 (population 40, 100 generations,
# 10-fold CV), and the deviation is recorded in every GA-EBT run record through
# baselines.ga_ebt.FIDELITY.
#
# The budget was set empirically rather than guessed: a pilot search with 15
# generations reached fitness 0.99832 at generation 0 and 0.99881 by generation
# 5, after which it did not improve for the remaining 10 generations. The GA's
# fitness is cross-validated F1 on the training split, which is itself saturated
# on this dataset, so a longer search has nothing left to optimise. Eight
# generations with patience 3 reproduce the same plateau at a fifth of the cost.
GA_BUDGET = {"population_size": 16, "max_generations": 8,
             "fitness_subsample": 15_000, "fitness_cv_folds": 3,
             "no_improve_patience": 3}


def cell_done(proto: str, rep: str, task: str, seed: int) -> bool:
    """True when every model in this cell already has a record on disk.

    Splits are a deterministic function of (protocol, representation, task,
    seed), so resuming an interrupted run reproduces exactly the same splits and
    a completed cell never needs recomputing. A cell the protocol declared not
    applicable leaves a SKIPPED marker, which also counts as done.
    """
    stem = f"__{rep}__{task}__protocol{proto}__seed{seed}.json"
    if (RESULTS_RAW / f"SKIPPED__main{stem}").exists():
        return True
    return all((RESULTS_RAW / f"main__{m}{stem}").exists() for m in MODELS)


def main() -> None:
    cells = [(p, r, t, s) for p in PROTOCOLS for r in REPRESENTATIONS
             for t in TASKS for s in SEEDS]
    log.info("main phase: %d cells x up to %d models", len(cells), len(MODELS))
    progress = []
    t_all = time.perf_counter()
    for i, (proto, rep, task, seed) in enumerate(cells, 1):
        if cell_done(proto, rep, task, seed):
            log.info("cell %d/%d already complete, skipping: %s %s %s seed=%d",
                     i, len(cells), proto, rep, task, seed)
            continue
        t0 = time.perf_counter()
        recs = run_configuration(rep, task, proto, seed, models=MODELS,
                                 experiment="main", train_cfg=CFG,
                                 ga_budget=GA_BUDGET)
        dt = time.perf_counter() - t0
        progress.append({"cell": i, "protocol": proto, "representation": rep,
                         "task": task, "seed": seed, "n_records": len(recs),
                         "seconds": dt})
        save_json({"completed_cells": len(progress),
                   "total_cells": len(cells),
                   "elapsed_seconds": time.perf_counter() - t_all,
                   "cells": progress},
                  DATA_METADATA / "main_phase_progress.json")
        log.info("cell %d/%d done: %s %s %s seed=%d -> %d records in %.0fs "
                 "(elapsed %.2f h)", i, len(cells), proto, rep, task, seed,
                 len(recs), dt, (time.perf_counter() - t_all) / 3600)
    log.info("main phase complete in %.2f hours", (time.perf_counter() - t_all) / 3600)


if __name__ == "__main__":
    main()
