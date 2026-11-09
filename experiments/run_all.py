"""Phase driver: runs the whole protocol in the order given in the specification.

    python -m experiments.run_all --phases audit,main,ablation,generalization,analysis

Each phase prints its own completion block (what was done, files created, metrics
generated, unresolved issues, next phase) and every phase writes to disk before
the next begins, so an interrupted run resumes at phase granularity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    ALL_MODELS, CLASSICAL, GRAPH, NEURAL_TABULAR, run_configuration,
)
from src.train import TrainConfig  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_FIGURES, RESULTS_RAW, RESULTS_TABLES, SEEDS,
    collect_runs, ensure_dirs, get_logger, save_json,
)

log = get_logger("run_all")


def phase_banner(name: str, files: Sequence[str], metrics: Sequence[str],
                 issues: Sequence[str], next_phase: str) -> None:
    line = "=" * 72
    print(f"\n{line}\nPHASE COMPLETE: {name}\n{line}")
    print("Files created:")
    for f in files or ["(none)"]:
        print(f"  - {f}")
    print("Metrics generated:")
    for m in metrics or ["(none)"]:
        print(f"  - {m}")
    print("Unresolved issues:")
    for i in issues or ["(none)"]:
        print(f"  - {i}")
    print(f"Next phase: {next_phase}\n{line}\n", flush=True)


# --------------------------------------------------------------------------- #


def phase_audit(args) -> None:
    from src.audit import run_audit

    audit = run_audit(raw_dir=args.raw_dir, make_figures=not args.no_figures)
    issues: List[str] = []
    for rep, info in audit.get("representations", {}).items():
        if info.get("attack_flagged_in_normal_capture"):
            issues.append(
                f"{rep}: {info['attack_flagged_in_normal_capture']} rows in the "
                "normal capture carry is_attack=1")
        if info.get("benign_background_in_attack_captures"):
            issues.append(
                f"{rep}: {info['benign_background_in_attack_captures']} benign "
                "background flows sit inside attack captures and are labelled "
                "Normal for the multi-class task (expected; recorded in the "
                "manuscript's labelling subsection)")
        if info.get("duplicate_feature_vector_ratio", 0) > 0.05:
            issues.append(
                f"{rep}: {info['duplicate_feature_vector_ratio']:.1%} of rows "
                "share an exact predictive feature vector -- protocol B exists "
                "for this reason")
        if info.get("timestamp_columns_present"):
            issues.append(f"{rep}: unexpected timestamp-like columns "
                          f"{info['timestamp_columns_present']} -- revisit the "
                          "no-temporal-claim rule before using them")
    phase_banner(
        "1-4  dataset audit, graph statistics and audit figures",
        [str(DATA_METADATA / "dataset_audit.json"),
         str(DATA_METADATA / "dataset_audit.csv"),
         str(RESULTS_FIGURES)],
        ["row/column counts", "class and scenario distributions",
         "duplicate and near-zero-variance reports", "graph endpoint counts"],
        issues,
        "5-12  baselines and proposed model",
    )


def phase_main(args) -> None:
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience,
                      batch_size=args.batch_size, device=args.device,
                      max_msg_edges_per_node=args.msg_edges)
    n = 0
    for rep in args.representations:
        for task in args.tasks:
            for protocol in args.protocols:
                for seed in args.seeds:
                    recs = run_configuration(
                        rep, task, protocol, seed, models=args.models,
                        experiment="main", train_cfg=cfg, nrows=args.nrows,
                        ga_budget=args.ga_budget)
                    n += len(recs)
    phase_banner(
        "5-14  classical baselines, GA-EBT, MLP, GCN, GraphSAGE, GAT, "
        "Edge-GAT, E-GAT-IDS, multi-seed",
        [str(RESULTS_RAW)],
        [f"{n} run records, each with full metric battery, confusion matrix, "
         "per-class report, timings and parameter counts"],
        [],
        "15  ablation",
    )


def phase_ablation(args) -> None:
    from experiments.run_ablation import main as ablation_main

    ablation_main(args)
    phase_banner("15  ablation", [str(RESULTS_RAW)],
                 ["component-wise ablation of E-GAT-IDS"], [],
                 "17  source-grouped generalisation")


def phase_generalization(args) -> None:
    from experiments.run_generalization import main as gen_main

    gen_main(args)
    phase_banner("17  generalisation", [str(RESULTS_RAW)],
                 ["protocol B and protocol C results"], [],
                 "18-20  explainability, statistics, tables and figures")


def phase_analysis(args) -> None:
    from experiments.make_figures import main as figures_main
    from experiments.make_tables import main as tables_main

    df = collect_runs()
    if df.empty:
        phase_banner("18-20 analysis", [], [],
                     ["no run records found -- run the main phase first"], "none")
        return
    df.to_csv(RESULTS_TABLES / "all_runs.csv", index=False)
    tables_main(args)
    figures_main(args)
    phase_banner(
        "18-20  attention attribution, statistics, final tables and figures",
        [str(RESULTS_TABLES), str(RESULTS_FIGURES)],
        ["aggregate tables 1-12", "paired significance tests", "figures 5-14"],
        [],
        "21  manuscript (written only once these outputs exist)",
    )


PHASES = {
    "audit": phase_audit,
    "main": phase_main,
    "ablation": phase_ablation,
    "generalization": phase_generalization,
    "analysis": phase_analysis,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phases", default="audit,main,ablation,generalization,analysis")
    p.add_argument("--representations", default="uniflow,biflow")
    p.add_argument("--tasks", default="binary,multiclass")
    p.add_argument("--protocols", default="A,B")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--msg-edges", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--nrows", type=int, default=None,
                   help="subsample rows; for smoke tests only, never for results")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--ga-population", type=int, default=20)
    p.add_argument("--ga-generations", type=int, default=25)
    p.add_argument("--ga-subsample", type=int, default=20_000)
    p.add_argument("--ga-folds", type=int, default=3)
    return p


def normalise(args):
    args.representations = [s for s in args.representations.split(",") if s]
    args.tasks = [s for s in args.tasks.split(",") if s]
    args.protocols = [s for s in args.protocols.split(",") if s]
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s]
    args.models = [s for s in args.models.split(",") if s]
    args.raw_dir = Path(args.raw_dir) if args.raw_dir else None
    args.ga_budget = {
        "population_size": args.ga_population,
        "max_generations": args.ga_generations,
        "fitness_subsample": args.ga_subsample,
        "fitness_cv_folds": args.ga_folds,
    }
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = normalise(build_parser().parse_args(argv))
    ensure_dirs()
    save_json(vars(args) | {"raw_dir": str(args.raw_dir)},
              RESULTS_RAW / "_invocation.json")
    for phase in args.phases.split(","):
        phase = phase.strip()
        if phase not in PHASES:
            raise SystemExit(f"unknown phase {phase!r}; choose from {list(PHASES)}")
        log.info("starting phase %s", phase)
        PHASES[phase](args)


if __name__ == "__main__":
    main()
