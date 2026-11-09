"""Regenerate every data-driven manuscript figure from the result files.

Figures 1-4 are schematics and live in ``experiments/make_schematics.py``.
Figures 5-14 are built here and are pure functions of ``results/raw``, so
deleting ``results/figures`` and rerunning reproduces them byte-for-byte given
the same runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics import aggregate_over_seeds  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_FIGURES, RESULTS_RAW, collect_runs, get_logger,
    load_json,
)
from src.visualization import (  # noqa: E402
    LIGHT, fig_ablation, fig_confusion_matrix, fig_per_class_f1,
    fig_representation_comparison, fig_runtime_vs_performance,
)

log = get_logger("figures")

ABLATION_ORDER = ["MLP", "GCN", "GraphSAGE", "GAT", "EdgeGAT", "E-GAT-IDS"]


def _records(experiment: str = "main") -> List[dict]:
    return [load_json(p) for p in sorted(RESULTS_RAW.glob(f"{experiment}__*.json"))]


def figure7_confusion(task: str = "multiclass", model: str = "E-GAT-IDS",
                      representation: str = "uniflow", protocol: str = "A",
                      experiment: str = "main") -> None:
    recs = [r for r in _records(experiment)
            if r["model"] == model and r["task"] == task
            and r["representation"] == representation and r["protocol"] == protocol]
    if not recs:
        return
    cms = np.array([r["confusion_matrix"] for r in recs], dtype=float)
    cm = cms.sum(axis=0)          # pooled over seeds
    names = recs[0]["config"]["class_names"]
    fig_confusion_matrix(
        cm, names, RESULTS_FIGURES / f"fig07_confusion_{model}_{task}.png",
        title=(f"{model}, {task}, pooled over {len(recs)} "
               f"seed{'s' if len(recs) != 1 else ''}"))


def figure8_per_class(task: str = "multiclass", representation: str = "uniflow",
                      protocol: str = "A",
                      models=("MLP", "GAT", "EdgeGAT", "E-GAT-IDS"),
                      experiment: str = "main") -> None:
    recs = [r for r in _records(experiment)
            if r["task"] == task and r["representation"] == representation
            and r["protocol"] == protocol and r["model"] in models]
    if not recs:
        return
    per_class: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, Dict[str, float]] = {}
    for m in models:
        sub = [r for r in recs if r["model"] == m]
        if not sub:
            continue
        classes = list(sub[0]["per_class"].keys())
        per_class[m] = {c: float(np.mean([s["per_class"][c]["f1"] for s in sub]))
                        for c in classes}
        errors[m] = {c: float(np.std([s["per_class"][c]["f1"] for s in sub], ddof=1))
                     if len(sub) > 1 else 0.0 for c in classes}
    if per_class:
        fig_per_class_f1(
            {m: {c: {"f1": v} for c, v in d.items()} for m, d in per_class.items()},
            RESULTS_FIGURES / f"fig08_per_class_f1_{task}.png",
            errors=errors,
            title=f"Per-class F1, {task}, mean ± s.d. over seeds")


def figures_9_10_curves(task: str = "binary", representation: str = "uniflow",
                        protocol: str = "A", experiment: str = "main") -> None:
    """PR and ROC curves need stored scores; regenerate them from checkpoints.

    Scores are not persisted by default (they are one float per test flow per
    model per seed).  ``experiments/dump_scores.py`` writes them for the
    single reference seed; if that file is absent the curves are skipped and
    the manuscript simply does not claim them.
    """
    score_file = RESULTS_RAW / "scores" / f"{representation}_{task}_protocol{protocol}.npz"
    if not score_file.exists():
        log.warning("no stored scores at %s; skipping figures 9 and 10", score_file)
        return
    from src.visualization import fig_pr_curves, fig_roc_curves

    bundle = np.load(score_file, allow_pickle=True)
    y_true = bundle["y_true"]
    scores = {k: bundle[k] for k in bundle.files if k != "y_true"}
    fig_roc_curves(y_true, scores, RESULTS_FIGURES / f"fig10_roc_{task}.png")
    fig_pr_curves(y_true, scores, RESULTS_FIGURES / f"fig09_pr_{task}.png")


def figure11_ablation(runs: pd.DataFrame, experiment: str = "main",
                      ablation_experiment: str = "ablation") -> None:
    arch = runs[(runs["experiment"] == experiment)
                & (runs["model"].isin(ABLATION_ORDER))
                & (runs["task"] == "multiclass")
                & (runs["representation"] == "uniflow")
                & (runs["protocol"] == "A")]
    if not arch.empty:
        fig_ablation(aggregate_over_seeds(arch),
                     RESULTS_FIGURES / "fig11a_architecture_ladder.png",
                     order=ABLATION_ORDER,
                     title="Architecture ladder, macro-F1 (uni-flow, 5 classes)")
    comp = runs[runs["experiment"] == ablation_experiment]
    if not comp.empty:
        fig_ablation(aggregate_over_seeds(comp),
                     RESULTS_FIGURES / "fig11b_component_ablation.png",
                     title="Component ablation of E-GAT-IDS, macro-F1")


def figure12_representation(runs: pd.DataFrame, experiment: str = "main") -> None:
    sub = runs[(runs["experiment"] == experiment) & (runs["task"] == "multiclass")
               & (runs["protocol"] == "A")]
    if sub.empty or sub["representation"].nunique() < 2:
        log.warning("need both representations for figure 12")
        return
    fig_representation_comparison(
        aggregate_over_seeds(sub),
        RESULTS_FIGURES / "fig12_uniflow_vs_biflow.png")


def figure13_attention(attention_dir: Optional[Path] = None) -> None:
    path = (Path(attention_dir) if attention_dir
            else RESULTS_RAW / "attention") / "attention_by_relationship.csv"
    if not path.exists():
        log.warning("no attention export; run experiments/run_explainability.py")
        return
    from src.visualization import fig_attention_heatmap, fig_communication_graph

    df = pd.read_csv(path)
    top = (df.sort_values("total_attention", ascending=False)
             .drop_duplicates(subset=["src_ip", "dst_ip"]))
    fig_communication_graph(
        top, RESULTS_FIGURES / "fig13a_attention_graph.png",
        weight_col="total_attention",
        title="Communication graph weighted by attention mass")

    pivot = (df.pivot_table(index=["src_ip", "dst_ip"], columns="class",
                            values="mean_attention", aggfunc="mean")
               .fillna(0.0))
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:25]]
    fig_attention_heatmap(
        pivot.to_numpy(),
        [f"{a} → {b}" for a, b in pivot.index],
        list(pivot.columns),
        RESULTS_FIGURES / "fig13b_attention_heatmap.png",
        title="Mean attention per relationship and class")


def figure14_cost(runs: pd.DataFrame, experiment: str = "main") -> None:
    sub = runs[(runs["experiment"] == experiment) & (runs["task"] == "multiclass")
               & (runs["representation"] == "uniflow") & (runs["protocol"] == "A")]
    if sub.empty:
        return
    agg = aggregate_over_seeds(sub)
    times = sub.groupby("model", as_index=False)["time_train_seconds"].mean() \
        if "time_train_seconds" in sub.columns else None
    if times is None:
        return
    merged = agg.merge(times, on="model", how="left")
    fig_runtime_vs_performance(merged, RESULTS_FIGURES / "fig14_cost_vs_performance.png")


def main(args=None, experiment: str = "main",
         ablation_experiment: str = "ablation",
         attention_dir: Optional[Path] = None) -> None:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    if runs.empty:
        log.warning("no runs found; only audit figures are available")
        return
    figure7_confusion(experiment=experiment)
    figure7_confusion(task="binary", experiment=experiment)
    figure8_per_class(experiment=experiment)
    figures_9_10_curves(experiment=experiment)
    figure11_ablation(runs, experiment=experiment,
                      ablation_experiment=ablation_experiment)
    figure12_representation(runs, experiment=experiment)
    figure13_attention(attention_dir=attention_dir)
    figure14_cost(runs, experiment=experiment)
    log.info("figures written to %s", RESULTS_FIGURES)


if __name__ == "__main__":
    main()
