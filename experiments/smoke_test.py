"""End-to-end pipeline check on synthetic data.

This exists to prove the plumbing works: every module imports, every model
trains, every metric computes, every artefact lands on disk.  It is **not** an
experiment and its outputs are **not** results.

Guards against confusion:

* the synthetic tables are written to ``data/raw_smoke/``, never ``data/raw/``;
* every artefact is tagged ``experiment="SMOKE"`` so its filename starts with
  ``SMOKE__`` and it can never be picked up by the manuscript tables, which
  filter on ``experiment == "main"``;
* the run deletes its own artefacts unless ``--keep`` is passed.

The synthetic generator deliberately builds a *learnable* signal, so a pass here
says "the code runs", not "the method works".
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import PreparedData, run_configuration  # noqa: E402
from src.train import TrainConfig  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_METADATA, PROJECT_ROOT, RESULTS_CKPT, RESULTS_FIGURES, RESULTS_LOGS,
    RESULTS_RAW, RESULTS_TABLES, get_logger,
)

log = get_logger("smoke")

SMOKE_RAW = PROJECT_ROOT / "data" / "raw_smoke"

#: Files that existed before the smoke run, so cleanup removes only what the
#: smoke run itself created. Tables and figures are named by their manuscript
#: number and cannot be distinguished by name alone.
_PREEXISTING: set = set()

SCENARIOS = {
    "normal": dict(n=2400, attack=0),
    "mqtt_bruteforce": dict(n=900, attack=1),
    "scan_A": dict(n=1100, attack=1),
    "scan_sU": dict(n=800, attack=1),
    "sparta": dict(n=400, attack=1),     # deliberately the minority class
}

UNIFLOW_COLS = [
    "ip_src", "ip_dst", "prt_src", "prt_dst", "proto", "num_pkts",
    "mean_iat", "std_iat", "min_iat", "max_iat", "mean_pkt_len", "num_bytes",
    "num_psh_flags", "num_rst_flags", "num_urg_flags",
    "std_pkt_len", "min_pkt_len", "max_pkt_len", "is_attack",
]


def _endpoints(rng: np.random.Generator):
    sensors = [f"192.168.2.{i}" for i in range(10, 22)]
    broker = "192.168.2.5"
    camera = "192.168.2.6"
    attacker = "192.168.2.100"
    return sensors, broker, camera, attacker


def _synth_scenario(name: str, spec: dict, rng: np.random.Generator,
                    biflow: bool = False) -> pd.DataFrame:
    sensors, broker, camera, attacker = _endpoints(rng)
    n = spec["n"]
    attack = spec["attack"]

    if name == "normal":
        src = rng.choice(sensors + [camera], size=n)
        dst = np.full(n, broker)
        prt_dst = np.full(n, 1883)
        pkts = rng.poisson(8, n) + 2
        iat = rng.gamma(2.0, 0.35, n)
    elif name == "mqtt_bruteforce":
        src = np.full(n, attacker)
        dst = np.full(n, broker)
        prt_dst = np.full(n, 1883)
        pkts = rng.poisson(5, n) + 2
        iat = rng.gamma(1.1, 0.05, n)
    elif name == "scan_A":
        src = np.full(n, attacker)
        dst = rng.choice(sensors + [broker, camera], size=n)
        prt_dst = rng.integers(1, 65535, n)
        pkts = rng.poisson(1.2, n) + 1
        iat = rng.gamma(0.6, 0.01, n)
    elif name == "scan_sU":
        src = np.full(n, attacker)
        dst = rng.choice(sensors, size=n)
        prt_dst = rng.integers(1, 65535, n)
        pkts = np.ones(n, dtype=int)
        iat = rng.gamma(0.5, 0.008, n)
    else:  # sparta
        src = np.full(n, attacker)
        dst = rng.choice(sensors[:4], size=n)
        prt_dst = np.full(n, 22)
        pkts = rng.poisson(14, n) + 4
        iat = rng.gamma(1.6, 0.12, n)

    proto = np.where(prt_dst == 0, 17, 6)
    if name == "scan_sU":
        proto = np.full(n, 17)
    plen = rng.normal(140 if attack else 90, 30, n).clip(40, 1500)
    df = pd.DataFrame({
        "ip_src": src,
        "ip_dst": dst,
        "prt_src": rng.integers(32768, 60999, n),
        "prt_dst": prt_dst,
        "proto": proto,
        "num_pkts": pkts,
        "mean_iat": iat,
        "std_iat": iat * rng.uniform(0.1, 0.6, n),
        "min_iat": iat * rng.uniform(0.01, 0.2, n),
        "max_iat": iat * rng.uniform(1.2, 6.0, n),
        "mean_pkt_len": plen,
        "num_bytes": (plen * pkts).astype(int),
        "num_psh_flags": rng.binomial(pkts, 0.3 if attack else 0.55),
        "num_rst_flags": rng.binomial(pkts, 0.25 if name.startswith("scan") else 0.02),
        "num_urg_flags": rng.binomial(pkts, 0.01),
        "std_pkt_len": rng.gamma(2.0, 12.0, n),
        "min_pkt_len": (plen * rng.uniform(0.3, 0.8, n)).astype(int),
        "max_pkt_len": (plen * rng.uniform(1.1, 2.5, n)).astype(int),
        "is_attack": np.full(n, attack, dtype=int),
    })[UNIFLOW_COLS]

    if biflow:
        for side, scale in (("fwd", 1.0), ("bwd", 0.6)):
            df[f"{side}_num_pkts"] = (df["num_pkts"] * scale).astype(int)
            df[f"{side}_num_bytes"] = (df["num_bytes"] * scale).astype(int)
            df[f"{side}_mean_iat"] = df["mean_iat"] * rng.uniform(0.8, 1.3, n)
            df[f"{side}_std_iat"] = df["std_iat"] * rng.uniform(0.8, 1.3, n)
            df[f"{side}_mean_pkt_len"] = df["mean_pkt_len"] * rng.uniform(0.8, 1.2, n)
        cols = [c for c in df.columns if c != "is_attack"] + ["is_attack"]
        df = df[cols]
    return df


def generate(seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    if SMOKE_RAW.exists():
        shutil.rmtree(SMOKE_RAW)
    for rep, biflow in (("uniflow", False), ("biflow", True)):
        d = SMOKE_RAW / f"{rep}_features"
        d.mkdir(parents=True, exist_ok=True)
        for name, spec in SCENARIOS.items():
            df = _synth_scenario(name, spec, rng, biflow=biflow)
            df.to_csv(d / f"{rep}_{name}.csv", index=False)
    log.info("synthetic tables written to %s", SMOKE_RAW)
    return SMOKE_RAW


def snapshot() -> None:
    global _PREEXISTING
    _PREEXISTING = {p for root in (RESULTS_FIGURES, RESULTS_TABLES)
                    for p in root.rglob("*") if p.is_file()}


def cleanup() -> None:
    n = 0
    for root in (RESULTS_FIGURES, RESULTS_TABLES):
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_file() and p not in _PREEXISTING:
                p.unlink()
                n += 1
    for root in (RESULTS_RAW, RESULTS_CKPT, RESULTS_LOGS, RESULTS_FIGURES,
                 DATA_METADATA):
        for p in sorted(root.rglob("*SMOKE*"), reverse=True):
            if p.is_file():
                p.unlink()
                n += 1
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    for stray in (RESULTS_RAW / "_ablation_variants.json",):
        if stray.exists():
            stray.unlink()
    if SMOKE_RAW.exists():
        shutil.rmtree(SMOKE_RAW)
    log.info("removed %d smoke artefacts and %s", n, SMOKE_RAW)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", action="store_true",
                   help="keep the synthetic data and SMOKE__ artefacts")
    p.add_argument("--models", default="RandomForest,XGBoost,KNN,SVM,MLP,GCN,"
                                       "GraphSAGE,GAT,EdgeGAT,E-GAT-IDS,GA-EBT")
    p.add_argument("--epochs", type=int, default=6)
    a = p.parse_args(argv)

    snapshot()
    raw = generate()
    cfg = TrainConfig(epochs=a.epochs, patience=3, batch_size=1024,
                      max_msg_edges_per_node=200, eval_msg_edges_per_node=400,
                      log_every=2)
    ga_budget = {"population_size": 6, "max_generations": 2,
                 "fitness_subsample": 1200, "fitness_cv_folds": 2,
                 "no_improve_patience": 2}

    models = [m for m in a.models.split(",") if m]
    failures: List[str] = []
    produced = 0

    # Phase 1: the audit must run on a schema it has never seen before.
    try:
        from src.audit import run_audit
        audit = run_audit(raw_dir=raw,
                          out_dir=DATA_METADATA / "SMOKE_audit",
                          figures_dir=RESULTS_FIGURES / "SMOKE_audit")
        assert audit["representations"]["uniflow"]["n_rows"] == sum(
            s["n"] for s in SCENARIOS.values())
        if audit["representations"]["uniflow"]["timestamp_columns_present"]:
            failures.append("audit found timestamp columns in synthetic uni-flow")
    except Exception as exc:
        failures.append(f"audit raised: {exc!r}")

    for rep in ("uniflow", "biflow"):
        for task in ("binary", "multiclass"):
            for protocol in ("A", "B", "C"):
                recs = run_configuration(
                    rep, task, protocol, seed=42, models=models,
                    experiment="SMOKE", train_cfg=cfg, raw_dir=raw,
                    ga_budget=ga_budget)
                produced += len(recs)
                if protocol != "C" and len(recs) < len(models):
                    got = {r.model for r in recs}
                    failures.append(
                        f"{rep}/{task}/protocol{protocol}: missing "
                        f"{sorted(set(models) - got)}")

    # Ablation and explainability paths.
    try:
        from experiments.run_ablation import run as ablation_run
        ablation_run("uniflow", "multiclass", "A", [42], train_cfg=cfg,
                     raw_dir=raw, experiment="SMOKE_ablation")
    except Exception as exc:
        failures.append(f"ablation raised: {exc!r}")

    # Attribution path: needs a checkpoint, which the main loop above saved.
    try:
        from experiments.run_explainability import run as explain_run
        explain_run("uniflow", "multiclass", "A", 42, "E-GAT-IDS", cfg=cfg,
                    permutation=True, raw_dir=raw, experiment="SMOKE",
                    out_dir=RESULTS_RAW / "SMOKE_attention")
    except Exception as exc:
        failures.append(f"explainability raised: {exc!r}")

    # Score dump path (figures 9 and 10).
    try:
        from experiments.dump_scores import main as dump_main
        dump_main("uniflow", "binary", "A", 42, ("E-GAT-IDS", "GAT"),
                  raw_dir=raw, experiment="SMOKE")
    except Exception as exc:
        failures.append(f"dump_scores raised: {exc!r}")

    # Analysis path: tables, statistics and figures, over the SMOKE records.
    try:
        from experiments.make_figures import main as figures_main
        from experiments.make_tables import main as tables_main
        tables_main(experiment="SMOKE", ablation_experiment="SMOKE_ablation")
        figures_main(experiment="SMOKE", ablation_experiment="SMOKE_ablation",
                     attention_dir=RESULTS_RAW / "SMOKE_attention")
        expected = ["fig07_confusion_E-GAT-IDS_multiclass.png",
                    "fig08_per_class_f1_multiclass.png",
                    "fig11a_architecture_ladder.png",
                    "fig11b_component_ablation.png",
                    "fig12_uniflow_vs_biflow.png",
                    "fig13a_attention_graph.png",
                    "fig14_cost_vs_performance.png",
                    "fig09_pr_binary.png", "fig10_roc_binary.png"]
        missing = [f for f in expected if not (RESULTS_FIGURES / f).exists()]
        if missing:
            failures.append(f"figures not produced: {missing}")
        tables = ["table05_binary_results.csv", "table06_multiclass_results.csv",
                  "table07_per_class_results.csv", "table09_ablation.csv",
                  "table10_uniflow_vs_biflow.csv", "table11_computational_cost.csv"]
        missing_t = [f for f in tables if not (RESULTS_TABLES / f).exists()]
        if missing_t:
            failures.append(f"tables not produced: {missing_t}")
    except Exception as exc:
        failures.append(f"analysis phase raised: {exc!r}")

    # Quality control must execute and report, not crash.
    try:
        from experiments.quality_control import main as qc_main
        qc_main()
    except Exception as exc:
        failures.append(f"quality_control raised: {exc!r}")

    line = "=" * 72
    print(f"\n{line}\nSMOKE TEST\n{line}")
    print(f"run records produced : {produced}")
    print(f"failures             : {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    print("\nThese artefacts are SYNTHETIC. They are tagged experiment='SMOKE' "
          "and are not results.")
    if not a.keep:
        cleanup()
        print("Synthetic artefacts removed.")
    else:
        print(f"Kept at {SMOKE_RAW} and results/**/SMOKE__*")
    print(line + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
