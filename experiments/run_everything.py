"""Single master driver, ordered by scientific value rather than by phase number.

The container this runs in is reclaimed after a period of inactivity, so the
order matters: if the machine disappears at hour ten, we want the decisive
experiment finished and the saturated one unfinished, not the other way round.

Order, with the reason for each position:

  1. leave-one-endpoint-out, uni-flow, multi-class -- the comparison the
     paper's ranking rests on, and the only protocol with enough matched
     observations to support a paired test.
  2. the main multi-seed grid -- protocols C and CS (the variance
     demonstration) run before B and A (the saturation evidence), and
     uni-flow before bi-flow.
  3. component ablation on protocol C, then A.
  4. leave-one-endpoint-out, binary -- lower value, since the binary task is
     more saturated than the five-class one.
  5. the transductive node-feature contrast.
  6. attention attribution, stored scores.
  7. tables, figures, quality control.

Every step is resumable: completed cells and folds are detected on disk and
skipped, so re-launching after an interruption continues rather than restarts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import DATA_METADATA, RESULTS_RAW, SEEDS, get_logger, save_json  # noqa: E402

log = get_logger("everything")


def _cfg():
    from src.train import TrainConfig
    return TrainConfig(epochs=60, patience=12, batch_size=8192, log_every=30)


# --------------------------------------------------------------------------- #


def step_loeo_multiclass() -> None:
    from experiments.run_loeo import run as loeo_run
    loeo_run("uniflow", "multiclass")


def step_main_grid() -> None:
    from experiments.run_main_phase import main as main_phase
    main_phase()


def step_ablation() -> None:
    from experiments.run_ablation import run as ablation_run
    for protocol in ("C", "A"):
        ablation_run("uniflow", "multiclass", protocol, SEEDS, train_cfg=_cfg())


def step_loeo_binary() -> None:
    from experiments.run_loeo import run as loeo_run
    loeo_run("uniflow", "binary")


def step_transductive_contrast() -> None:
    from experiments.common import run_configuration

    models = ["GCN", "GraphSAGE", "E-GraphSAGE", "GAT", "EdgeGAT", "E-GAT-IDS"]
    for protocol in ("C", "A"):
        for seed in SEEDS:
            run_configuration("uniflow", "multiclass", protocol, seed,
                              models=models,
                              experiment="transductive_contrast",
                              train_cfg=_cfg(), node_feature_source="all")
    save_json({
        "what": ("Protocols C and A repeated with node behavioural features "
                 "aggregated over ALL flows instead of training flows only."),
        "why": ("Quantifies how much a transductive setup would have flattered "
                "the graph models. Any gain here is leakage, not capability, "
                "and is reported as such."),
        "models": models, "seeds": SEEDS,
    }, RESULTS_RAW / "_transductive_contrast_manifest.json")


def step_explainability() -> None:
    from experiments.run_explainability import run as explain_run

    for protocol in ("A", "C"):
        for model in ("E-GAT-IDS", "EdgeGAT", "GAT"):
            try:
                explain_run("uniflow", "multiclass", protocol, 42, model,
                            cfg=_cfg(), permutation=(model == "E-GAT-IDS"),
                            out_dir=RESULTS_RAW / "attention" /
                            f"{model}_protocol{protocol}")
            except FileNotFoundError as exc:
                log.warning("attribution skipped: %s", exc)


def step_scores() -> None:
    from experiments.dump_scores import main as dump_main
    for task in ("binary", "multiclass"):
        for protocol in ("A", "C"):
            dump_main("uniflow", task, protocol, 42,
                      ("GAT", "EdgeGAT", "E-GAT-IDS", "GCN", "E-GraphSAGE",
                       "GraphSAGE"))


def step_analysis() -> None:
    from experiments.make_figures import main as figures_main
    from experiments.make_schematics import main as schematics_main
    from experiments.make_tables import main as tables_main
    schematics_main()
    tables_main()
    figures_main()


def step_qc() -> None:
    from experiments.quality_control import main as qc_main
    qc_main()


STEPS = [
    ("1_loeo_multiclass", step_loeo_multiclass),
    ("2_main_grid", step_main_grid),
    ("3_ablation", step_ablation),
    ("4_loeo_binary", step_loeo_binary),
    ("5_transductive_contrast", step_transductive_contrast),
    ("6_explainability", step_explainability),
    ("7_scores", step_scores),
    ("8_analysis", step_analysis),
    ("9_quality_control", step_qc),
]


def main() -> None:
    results = []
    t0 = time.perf_counter()
    log.info("master driver: %d steps", len(STEPS))
    for name, fn in STEPS:
        t = time.perf_counter()
        log.info("=== START %s ===", name)
        try:
            fn()
            status, err = "ok", None
        except Exception as exc:
            log.exception("step %s failed: %s", name, exc)
            status, err = "failed", repr(exc)
        dt = time.perf_counter() - t
        results.append({"step": name, "status": status, "seconds": dt,
                        "error": err})
        log.info("=== END %s: %s in %.0fs (total %.2f h) ===",
                 name, status, dt, (time.perf_counter() - t0) / 3600)
        save_json({"elapsed_hours": (time.perf_counter() - t0) / 3600,
                   "steps": results},
                  DATA_METADATA / "master_progress.json")
    ok = sum(1 for r in results if r["status"] == "ok")
    log.info("master driver finished: %d/%d steps ok in %.2f h",
             ok, len(results), (time.perf_counter() - t0) / 3600)


if __name__ == "__main__":
    main()
