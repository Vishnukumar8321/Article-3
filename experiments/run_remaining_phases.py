"""Phases 15-20, chained to start when the main phase finishes.

Launched detached alongside the main phase.  It waits for the main-phase
process to exit, then runs, in order:

    14  leave-one-endpoint-out over all 20 candidate endpoints, uni-flow,
        both tasks -- the comparison the paper's ranking rests on
    15  component ablation of E-GAT-IDS (protocols A and C, all seeds)
    16  the transductive node-feature contrast (the only part of the
        generalisation study the main phase does not already cover, since the
        main phase runs protocols B, C and CS across every seed)
    18  attention attribution and permutation importance
    19  stored scores for the PR/ROC curves
    20  tables, figures, quality control

Each step writes to disk before the next begins and logs a completion block, so
an interruption loses at most one step.  Nothing here touches the test split
except through the already-completed evaluations.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_RAW, SEEDS, get_logger, save_json,
)

log = get_logger("remaining")

MAIN_PROCESS_PATTERN = "experiments.run_main_phase"


def main_phase_running() -> bool:
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return False
    me = str(os.getpid())
    for line in out.splitlines():
        if MAIN_PROCESS_PATTERN in line and "run_remaining" not in line:
            pid = line.split(None, 1)[0]
            if pid.isdigit() and pid != me:
                return True
    return False


def wait_for_main(poll_seconds: int = 120, max_hours: float = 30.0) -> bool:
    t0 = time.time()
    seen_running = False
    while True:
        running = main_phase_running()
        seen_running = seen_running or running
        if not running:
            # Give it a moment in case of a transient ps miss.
            time.sleep(15)
            if not main_phase_running():
                log.info("main phase not running after %.2f h of waiting; "
                         "proceeding", (time.time() - t0) / 3600)
                return seen_running
        if (time.time() - t0) / 3600 > max_hours:
            log.warning("waited %.1f h for the main phase; proceeding anyway",
                        max_hours)
            return seen_running
        time.sleep(poll_seconds)


def step(name: str, fn) -> dict:
    t0 = time.perf_counter()
    log.info("=== starting %s ===", name)
    try:
        fn()
        status = "ok"
        err = None
    except Exception as exc:  # a failed step must not stop the rest
        log.exception("step %s failed: %s", name, exc)
        status = "failed"
        err = repr(exc)
    dt = time.perf_counter() - t0
    log.info("=== %s %s in %.0fs ===", name, status, dt)
    return {"step": name, "status": status, "seconds": dt, "error": err}


# --------------------------------------------------------------------------- #


def phase_loeo() -> None:
    """Leave-one-endpoint-out: the decisive comparison, so it runs first."""
    from experiments.run_loeo import run as loeo_run

    loeo_run("uniflow", "multiclass")
    loeo_run("uniflow", "binary")


def phase_ablation() -> None:
    from experiments.run_ablation import run as ablation_run
    from src.train import TrainConfig

    cfg = TrainConfig(epochs=60, patience=12, batch_size=8192, log_every=30)
    # Protocol C first: it is the only one with headroom, so it is where the
    # ablation can actually attribute anything. A is included because the
    # saturation claim needs the ablation to show that no component matters
    # there either.
    for protocol in ("C", "A"):
        ablation_run("uniflow", "multiclass", protocol, SEEDS, train_cfg=cfg)


def phase_transductive_contrast() -> None:
    from experiments.common import run_configuration
    from src.train import TrainConfig

    cfg = TrainConfig(epochs=60, patience=12, batch_size=8192, log_every=30)
    models = ["GCN", "GraphSAGE", "E-GraphSAGE", "GAT", "EdgeGAT", "E-GAT-IDS"]
    for protocol in ("C", "A"):
        for seed in SEEDS:
            run_configuration("uniflow", "multiclass", protocol, seed,
                              models=models,
                              experiment="transductive_contrast",
                              train_cfg=cfg, node_feature_source="all")
    save_json({
        "what": ("Protocols C and A repeated with node behavioural features "
                 "aggregated over ALL flows instead of training flows only."),
        "why": ("Quantifies how much a transductive setup would have flattered "
                "the graph models. Any gain here is leakage, not capability, "
                "and is reported as such."),
        "models": models,
        "seeds": SEEDS,
    }, RESULTS_RAW / "_transductive_contrast_manifest.json")


def phase_explainability() -> None:
    from experiments.run_explainability import run as explain_run
    from src.train import TrainConfig

    cfg = TrainConfig(epochs=60, patience=12, batch_size=8192)
    for protocol in ("A", "C"):
        for model in ("E-GAT-IDS", "EdgeGAT", "GAT"):
            try:
                explain_run("uniflow", "multiclass", protocol, 42, model,
                            cfg=cfg, permutation=(model == "E-GAT-IDS"),
                            out_dir=RESULTS_RAW / "attention" /
                            f"{model}_protocol{protocol}")
            except FileNotFoundError as exc:
                log.warning("attribution skipped: %s", exc)


def phase_scores() -> None:
    from experiments.dump_scores import main as dump_main

    for task in ("binary", "multiclass"):
        for protocol in ("A", "C"):
            dump_main("uniflow", task, protocol, 42,
                      ("GAT", "EdgeGAT", "E-GAT-IDS", "GCN", "E-GraphSAGE",
                       "GraphSAGE"))


def phase_analysis() -> None:
    from experiments.make_figures import main as figures_main
    from experiments.make_tables import main as tables_main

    tables_main()
    figures_main()


def phase_qc() -> None:
    from experiments.quality_control import main as qc_main

    qc_main()


STEPS = [
    ("14_leave_one_endpoint_out", phase_loeo),
    ("15_ablation", phase_ablation),
    ("16_transductive_contrast", phase_transductive_contrast),
    ("18_explainability", phase_explainability),
    ("19_scores", phase_scores),
    ("20_tables_and_figures", phase_analysis),
    ("20_quality_control", phase_qc),
]


def main() -> None:
    log.info("waiting for the main phase to finish before starting phases 15-20")
    saw = wait_for_main()
    log.info("main phase %s; starting remaining phases",
             "completed" if saw else "was not running")
    results = []
    t0 = time.perf_counter()
    for name, fn in STEPS:
        results.append(step(name, fn))
        save_json({"elapsed_hours": (time.perf_counter() - t0) / 3600,
                   "steps": results},
                  DATA_METADATA / "remaining_phases_progress.json")
    ok = sum(1 for r in results if r["status"] == "ok")
    log.info("remaining phases done: %d/%d steps ok in %.2f h",
             ok, len(results), (time.perf_counter() - t0) / 3600)


if __name__ == "__main__":
    main()
