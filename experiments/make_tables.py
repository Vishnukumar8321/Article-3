"""Build every manuscript table directly from the result files.

No number is ever typed by hand.  Each table is written twice: a CSV for
inspection and a LaTeX fragment that ``manuscript/main.tex`` ``\\input``s, so a
figure in the paper and a figure in ``results/`` cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics import (  # noqa: E402
    aggregate_over_seeds, bootstrap_ci, compare_against_reference,
)
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_RAW, RESULTS_TABLES, collect_runs, get_logger,
    load_json, save_json,
)

log = get_logger("tables")

HEADLINE_METRICS = ["f1_macro", "mcc", "pr_auc", "roc_auc", "accuracy", "fpr",
                    "minority_class_recall", "balanced_accuracy"]


def _write(df: pd.DataFrame, stem: str, caption: str, label: str,
           float_fmt: str = "%.4f", index: bool = False) -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_TABLES / f"{stem}.csv", index=index)
    tex = df.to_latex(index=index, float_format=lambda v: float_fmt % v,
                      escape=True, longtable=False, caption=caption,
                      label=label, column_format="l" + "r" * (df.shape[1] - 1))
    (RESULTS_TABLES / f"{stem}.tex").write_text(tex)
    log.info("wrote %s (%d rows)", stem, len(df))


def _fmt_mean_std(mean, std) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    return f"{mean:.4f} ± {std:.4f}" if std is not None and not np.isnan(std) \
        else f"{mean:.4f}"


# --------------------------------------------------------------------------- #
# Tables 1-4: dataset, features, graph, hyperparameters
# --------------------------------------------------------------------------- #


def table1_dataset() -> Optional[pd.DataFrame]:
    path = DATA_METADATA / "dataset_audit.json"
    if not path.exists():
        log.warning("no dataset audit found; run the audit phase first")
        return None
    audit = load_json(path)
    rows = []
    for rep, info in audit.get("representations", {}).items():
        rows.append({
            "Representation": rep,
            "Files": info["n_files"],
            "Flow records": info["n_rows"],
            "Columns": info["n_columns"],
            "Predictive features": info["n_predictive_features"],
            "Endpoints": info["n_unique_endpoints"],
            "Ordered pairs": info["n_unique_src_dst_pairs"],
            "Attack ratio": round(info["attack_ratio"], 4),
            "Benign flows in attack captures": info.get(
                "benign_background_in_attack_captures"),
            "Duplicate feature vectors": info["n_duplicate_predictive_feature_vectors"],
        })
    df = pd.DataFrame(rows)
    _write(df, "table01_dataset_characteristics",
           "Characteristics of the MQTT-IoT-IDS2020 flow representations used in "
           "this study.", "tab:dataset")
    return df


def table2_features() -> Optional[pd.DataFrame]:
    path = DATA_METADATA / "dataset_audit.json"
    if not path.exists():
        return None
    audit = load_json(path)
    rows = []
    for rep, info in audit.get("representations", {}).items():
        for f in info["predictive_feature_columns"]:
            rows.append({"Representation": rep, "Feature": f,
                         "Role": "edge (flow) attribute"})
    from src.feature_engineering import NODE_FEATURE_NAMES
    for f in NODE_FEATURE_NAMES:
        rows.append({"Representation": "both", "Feature": f,
                     "Role": "node (endpoint behaviour) attribute"})
    df = pd.DataFrame(rows)
    _write(df, "table02_feature_description",
           "Edge and node attributes. Edge attributes are the dataset's own flow "
           "features; node attributes are aggregated from training flows only.",
           "tab:features")
    return df


def table3_graph() -> Optional[pd.DataFrame]:
    rows = []
    for p in sorted(DATA_METADATA.glob("context__*.json")):
        ctx = load_json(p)
        gs = ctx.get("graph_statistics", {})
        if not gs:
            continue
        rows.append({
            "Representation": ctx["representation"],
            "Task": ctx["task"],
            "Protocol": ctx["protocol"],
            "Seed": ctx["seed"],
            "Nodes": gs["n_nodes"],
            "Flow edges": gs["n_edges_flow_instances"],
            "Ordered pairs": gs["n_unique_ordered_pairs"],
            "Mean parallel multiplicity": round(gs["parallel_edge_multiplicity_mean"], 2),
            "Max parallel multiplicity": gs["parallel_edge_multiplicity_max"],
            "Density (ordered)": round(gs["graph_density_ordered"], 4),
            "Mean in-degree": round(gs["mean_in_degree"], 1),
            "Message edges": gs["n_message_edges"],
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["Representation", "Task", "Protocol"]).reset_index(drop=True)
    _write(df, "table03_graph_statistics",
           "Communication-graph statistics. Each flow record is a parallel edge, "
           "so the edge count equals the record count.", "tab:graph")
    return df


def table4_hyperparameters(runs: pd.DataFrame,
                           experiment: str = "main") -> Optional[pd.DataFrame]:
    rows = []
    for p in sorted(RESULTS_RAW.glob(f"{experiment}__*.json")):
        rec = load_json(p)
        cfg = rec.get("config", {})
        mc = cfg.get("model_config")
        tc = cfg.get("train_config")
        if not mc and not tc:
            continue
        rows.append({
            "Model": rec["model"],
            "Layer": (mc or {}).get("layer", "--"),
            "Hidden": (mc or {}).get("hidden_dim", "--"),
            "Layers": (mc or {}).get("n_layers", "--"),
            "Heads": (mc or {}).get("heads", "--"),
            "Dropout": (mc or {}).get("dropout", "--"),
            "Edge in attention": (mc or {}).get("edge_in_attention", "--"),
            "Edge in message": (mc or {}).get("edge_in_message", "--"),
            "LR": (tc or {}).get("lr", "--"),
            "Weight decay": (tc or {}).get("weight_decay", "--"),
            "Batch": (tc or {}).get("batch_size", "--"),
            "Loss": cfg.get("loss", "--"),
            "Parameters": rec.get("n_parameters"),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["Model"]).reset_index(drop=True)
    _write(df, "table04_hyperparameters",
           "Selected hyperparameters. All choices were made on validation "
           "macro-F1; the test split was not consulted.", "tab:hyper")
    return df


# --------------------------------------------------------------------------- #
# Tables 5-7: results
# --------------------------------------------------------------------------- #


def _results_table(runs: pd.DataFrame, task: str, stem: str, caption: str,
                   label: str, protocol: str = "A",
                   representation: str = "uniflow",
                   experiment: str = "main") -> Optional[pd.DataFrame]:
    sub = runs[(runs["task"] == task) & (runs["protocol"] == protocol)
               & (runs["representation"] == representation)
               & (runs["experiment"] == experiment)]
    if sub.empty:
        return None
    agg = aggregate_over_seeds(sub)
    rows = []
    for _, r in agg.iterrows():
        row = {"Model": r["model"], "Seeds": int(r["n_seeds"])}
        for m in HEADLINE_METRICS:
            row[m] = _fmt_mean_std(r.get(f"{m}_mean"), r.get(f"{m}_std"))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
    _write(df, stem, caption, label, float_fmt="%s")
    return df


def table5_binary(runs, experiment: str = "main"):
    return _results_table(
        runs, "binary", "table05_binary_results",
        "Binary intrusion detection, uni-flow, protocol A. Mean ± standard "
        "deviation over five seeds.", "tab:binary", experiment=experiment)


def table6_multiclass(runs, experiment: str = "main"):
    return _results_table(
        runs, "multiclass", "table06_multiclass_results",
        "Five-class scenario classification, uni-flow, protocol A. Mean ± "
        "standard deviation over five seeds.", "tab:multiclass", experiment=experiment)


def table7_per_class(task: str = "multiclass", protocol: str = "A",
                     representation: str = "uniflow",
                     experiment: str = "main") -> Optional[pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for p in sorted(RESULTS_RAW.glob(f"{experiment}__*.json")):
        rec = load_json(p)
        if (rec["task"] != task or rec["protocol"] != protocol
                or rec["representation"] != representation):
            continue
        for cls, m in (rec.get("per_class") or {}).items():
            rows.append({"Model": rec["model"], "Seed": rec["seed"], "Class": cls,
                         "Precision": m["precision"], "Recall": m["recall"],
                         "F1": m["f1"], "Support": m["support"], "FPR": m["fpr"]})
    if not rows:
        return None
    raw = pd.DataFrame(rows)
    agg = (raw.groupby(["Model", "Class"], as_index=False)
              .agg(Precision=("Precision", "mean"), Recall=("Recall", "mean"),
                   F1=("F1", "mean"), F1_std=("F1", "std"),
                   FPR=("FPR", "mean"), Support=("Support", "mean"),
                   Seeds=("Seed", "nunique")))
    _write(agg, "table07_per_class_results",
           "Per-class precision, recall and F1 averaged over seeds "
           "(five-class task, uni-flow, protocol A).", "tab:perclass")
    return agg


# --------------------------------------------------------------------------- #
# Table 8: comparison with the base paper
# --------------------------------------------------------------------------- #


BASE_PAPER_REPORTED = {
    "study": "Norouzi & Gurkas-Aydin (2026), Cluster Computing 29:110",
    "doi": "10.1007/s10586-025-05929-2",
    "model": "GA-EBT",
    "dataset": "MQTT-IoT-IDS2020",
    "reported_accuracy": 0.99997,
    "protocol_as_described": (
        "10-fold cross-validation in WEKA; min-max scaling; SMOTE within each "
        "training fold; 33,079 instances; feature/hyperparameter search by GA"
    ),
    "comparability_caveat": (
        "The original does not publish its train/test partition, and the stated "
        "instance count (33,079) is far smaller than the released uni-flow "
        "feature set, so the evaluated population differs. The figure is quoted "
        "as a literature value under its own protocol and is NOT a measurement "
        "produced by this project."
    ),
}


def table8_vs_base_paper(runs: pd.DataFrame,
                         experiment: str = "main") -> Optional[pd.DataFrame]:
    sub = runs[(runs["experiment"] == experiment) & (runs["representation"] == "uniflow")
               & (runs["task"] == "binary") & (runs["protocol"] == "A")]
    if sub.empty:
        return None
    agg = aggregate_over_seeds(sub)
    keep = agg[agg["model"].isin(["GA-EBT", "XGBoost", "GAT", "EdgeGAT", "E-GAT-IDS"])]
    rows = [{
        "Source": "reported in the base paper",
        "Model": "GA-EBT (original)",
        "Protocol": BASE_PAPER_REPORTED["protocol_as_described"],
        "Accuracy": f"{BASE_PAPER_REPORTED['reported_accuracy']:.5f}",
        "Macro-F1": "not reported",
        "MCC": "not reported",
        "PR-AUC": "not reported",
        "FPR": "not reported",
    }]
    for _, r in keep.iterrows():
        rows.append({
            "Source": "this study",
            "Model": ("GA-EBT (reproduced)" if r["model"] == "GA-EBT"
                      else r["model"]),
            "Protocol": "protocol A, 70/15/15 stratified, 5 seeds",
            "Accuracy": _fmt_mean_std(r.get("accuracy_mean"), r.get("accuracy_std")),
            "Macro-F1": _fmt_mean_std(r.get("f1_macro_mean"), r.get("f1_macro_std")),
            "MCC": _fmt_mean_std(r.get("mcc_mean"), r.get("mcc_std")),
            "PR-AUC": _fmt_mean_std(r.get("pr_auc_mean"), r.get("pr_auc_std")),
            "FPR": _fmt_mean_std(r.get("fpr_mean"), r.get("fpr_std")),
        })
    df = pd.DataFrame(rows)
    save_json(BASE_PAPER_REPORTED, RESULTS_TABLES / "table08_base_paper_provenance.json")
    _write(df, "table08_comparison_with_ga_ebt",
           "Comparison with the base paper. The first row is a literature value "
           "obtained under a different protocol and a different evaluated "
           "population; it is not directly comparable with the rows below and is "
           "shown for context only.", "tab:gaebt", float_fmt="%s")
    return df


# --------------------------------------------------------------------------- #
# Tables 9-12
# --------------------------------------------------------------------------- #


def table9_ablation(runs: pd.DataFrame,
                    experiment: str = "ablation") -> Optional[pd.DataFrame]:
    sub = runs[runs["experiment"] == experiment]
    if sub.empty:
        return None
    agg = aggregate_over_seeds(sub)
    rows = []
    for _, r in agg.iterrows():
        row = {"Variant": r["model"], "Seeds": int(r["n_seeds"])}
        for m in ["f1_macro", "mcc", "pr_auc", "minority_class_recall", "fpr"]:
            row[m] = _fmt_mean_std(r.get(f"{m}_mean"), r.get(f"{m}_std"))
        rows.append(row)
    df = pd.DataFrame(rows)
    _write(df, "table09_ablation", "Component ablation of E-GAT-IDS.",
           "tab:ablation", float_fmt="%s")
    return df


def table10_representation(runs: pd.DataFrame,
                           experiment: str = "main") -> Optional[pd.DataFrame]:
    sub = runs[runs["experiment"] == experiment]
    if sub.empty:
        return None
    agg = aggregate_over_seeds(sub)
    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Model": r["model"], "Representation": r["representation"],
            "Task": r["task"], "Protocol": r["protocol"],
            "Macro-F1": _fmt_mean_std(r.get("f1_macro_mean"), r.get("f1_macro_std")),
            "MCC": _fmt_mean_std(r.get("mcc_mean"), r.get("mcc_std")),
            "PR-AUC": _fmt_mean_std(r.get("pr_auc_mean"), r.get("pr_auc_std")),
            "Minority recall": _fmt_mean_std(r.get("minority_class_recall_mean"),
                                             r.get("minority_class_recall_std")),
            "FPR": _fmt_mean_std(r.get("fpr_mean"), r.get("fpr_std")),
        })
    df = pd.DataFrame(rows).sort_values(["Task", "Protocol", "Model",
                                         "Representation"]).reset_index(drop=True)
    _write(df, "table10_uniflow_vs_biflow",
           "Uni-flow against bi-flow. The two are different feature spaces and "
           "are never merged; only matched configurations are compared.",
           "tab:representation", float_fmt="%s")
    return df


def table11_cost(runs: pd.DataFrame,
                 experiment: str = "main") -> Optional[pd.DataFrame]:
    sub = runs[runs["experiment"] == experiment]
    if sub.empty:
        return None
    cols = {"time_train_seconds": "Train (s)",
            "time_inference_seconds": "Inference (s)",
            "time_flows_per_second": "Flows/s",
            "time_latency_ms_per_flow": "Latency (ms/flow)",
            "n_parameters": "Parameters",
            "peak_memory_mb": "Peak RSS (MiB)"}
    present = {k: v for k, v in cols.items() if k in sub.columns}
    agg = sub.groupby("model", as_index=False)[list(present)].mean()
    agg = agg.rename(columns={"model": "Model", **present})
    _write(agg, "table11_computational_cost",
           "Computational cost, measured on the CPU described in the "
           "reproducibility section. Throughput figures are measured, not "
           "extrapolated, and do not by themselves establish real-time "
           "suitability.", "tab:cost")
    return agg


def table13_loeo(runs: pd.DataFrame, task: str = "multiclass",
                 representation: str = "uniflow") -> Optional[pd.DataFrame]:
    """Leave-one-endpoint-out: mean over folds, spread, and the paired test.

    The spread columns matter as much as the mean. A model whose macro-F1
    ranges over half the scale depending on which endpoint is withheld has not
    been measured by any single split, and neither has any published result
    obtained that way.
    """
    sub = runs[(runs["experiment"] == "loeo") & (runs["task"] == task)
               & (runs["representation"] == representation)]
    if sub.empty:
        return None
    rows = []
    for model, m in sub.groupby("model"):
        vals = pd.to_numeric(m["f1_macro"], errors="coerce").dropna()
        mcc = pd.to_numeric(m["mcc"], errors="coerce").dropna()
        bal = pd.to_numeric(m["balanced_accuracy"], errors="coerce").dropna()
        if vals.empty:
            continue
        lo, hi = bootstrap_ci(vals.to_numpy())
        rows.append({
            "Model": model,
            "Folds": int(len(vals)),
            "Macro-F1 mean": round(float(vals.mean()), 4),
            "s.d.": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0,
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "95% CI": f"[{lo:.4f}, {hi:.4f}]",
            "MCC mean": round(float(mcc.mean()), 4) if len(mcc) else None,
            "Bal. acc mean": round(float(bal.mean()), 4) if len(bal) else None,
        })
    df = pd.DataFrame(rows).sort_values("Macro-F1 mean",
                                        ascending=False).reset_index(drop=True)
    _write(df, f"table13_loeo_{task}",
           "Leave-one-endpoint-out over every candidate endpoint (uni-flow, "
           f"{task}). Each fold withholds one endpoint with the model seed "
           "fixed, so the spread reflects holdout composition. The min and max "
           "columns bound what a single endpoint-grouped split could have "
           "reported for the same model.", f"tab:loeo{task}", float_fmt="%s")
    return df


def table14_loeo_significance(runs: pd.DataFrame, task: str = "multiclass",
                              representation: str = "uniflow",
                              reference: str = "E-GAT-IDS",
                              metric: str = "f1_macro") -> Optional[pd.DataFrame]:
    """Paired comparison over folds -- the test the 5-seed protocol cannot run."""
    sub = runs[(runs["experiment"] == "loeo") & (runs["task"] == task)
               & (runs["representation"] == representation)]
    if sub.empty or metric not in sub.columns:
        return None
    ref = sub[sub["model"] == reference].set_index("seed")[metric]
    if ref.empty:
        return None
    from src.statistics import holm_bonferroni, paired_comparison

    rows, family = [], {}
    for model, m in sub.groupby("model"):
        if model == reference:
            continue
        other = m.set_index("seed")[metric]
        common = sorted(set(ref.index) & set(other.index))
        if len(common) < 2:
            continue
        cmp = paired_comparison(ref.loc[common].to_numpy(),
                                other.loc[common].to_numpy(),
                                name_a=reference, name_b=model)
        rows.append(cmp)
        if cmp.get("wilcoxon_p") is not None:
            family[model] = cmp["wilcoxon_p"]
    corrected = holm_bonferroni(family) if family else {}
    out = []
    for r in rows:
        c = corrected.get(r["model_b"], {})
        out.append({
            "Baseline": r["model_b"],
            "Folds": r["n_pairs"],
            f"{reference}": round(r["mean_a"], 4),
            "Baseline mean": round(r["mean_b"], 4),
            "Difference": round(r["mean_difference"], 4),
            "95% CI": f"[{r['difference_ci95'][0]:.4f}, {r['difference_ci95'][1]:.4f}]",
            "Cohen dz": round(r["cohens_dz"], 3) if r.get("cohens_dz") == r.get("cohens_dz") else None,
            "Wilcoxon p": r.get("wilcoxon_p"),
            "Holm p": c.get("p_holm") if isinstance(c, dict) else None,
            "Significant": c.get("significant_at_alpha") if isinstance(c, dict) else None,
        })
    df = pd.DataFrame(out).sort_values("Difference").reset_index(drop=True)
    save_json(rows, RESULTS_TABLES / f"table14_loeo_significance_{task}_full.json")
    _write(df, f"table14_loeo_significance_{task}",
           f"Paired comparison of {reference} against each baseline on "
           f"{metric} over leave-one-endpoint-out folds, Holm-Bonferroni "
           "corrected across the baseline family.", f"tab:loeosig{task}",
           float_fmt="%s")
    return df


def table12_significance(runs: pd.DataFrame, metric: str = "f1_macro",
                         experiment: str = "main") -> Optional[pd.DataFrame]:
    sub = runs[runs["experiment"].isin([experiment])]
    if sub.empty or metric not in sub.columns:
        return None
    results = compare_against_reference(sub, reference_model="E-GAT-IDS",
                                        metric=metric)
    if not results:
        return None
    rows = []
    for r in results:
        rows.append({
            "Representation": r.get("representation"),
            "Task": r.get("task"),
            "Protocol": r.get("protocol"),
            "Baseline": r.get("model_b"),
            "n pairs": r.get("n_pairs"),
            "E-GAT-IDS": r.get("mean_a"),
            "Baseline mean": r.get("mean_b"),
            "Difference": r.get("mean_difference"),
            "95% CI": (f"[{r['difference_ci95'][0]:.4f}, "
                       f"{r['difference_ci95'][1]:.4f}]"
                       if r.get("difference_ci95") else "--"),
            "Cohen dz": r.get("cohens_dz"),
            "t-test p": r.get("paired_t_p"),
            "Wilcoxon p": r.get("wilcoxon_p"),
            "Holm p": r.get("wilcoxon_p_holm"),
            "Significant": r.get("significant_holm"),
        })
    df = pd.DataFrame(rows)
    save_json(results, RESULTS_TABLES / "table12_significance_full.json")
    _write(df, "table12_statistical_significance",
           f"Paired comparison of E-GAT-IDS against each baseline on {metric}, "
           "matched by seed, with Holm-Bonferroni correction across the baseline "
           "family.", "tab:significance", float_fmt="%s")
    return df


# --------------------------------------------------------------------------- #


def main(args=None, experiment: str = "main",
         ablation_experiment: str = "ablation") -> None:
    runs = collect_runs()
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    if runs.empty:
        log.warning("no run records in %s", RESULTS_RAW)
    else:
        runs.to_csv(RESULTS_TABLES / "all_runs.csv", index=False)

    table1_dataset()
    table2_features()
    table3_graph()
    if runs.empty:
        return
    table4_hyperparameters(runs, experiment=experiment)
    table5_binary(runs, experiment=experiment)
    table6_multiclass(runs, experiment=experiment)
    table7_per_class(experiment=experiment)
    table8_vs_base_paper(runs, experiment=experiment)
    table9_ablation(runs, experiment=ablation_experiment)
    table10_representation(runs, experiment=experiment)
    table11_cost(runs, experiment=experiment)
    table12_significance(runs, experiment=experiment)
    for tsk in ("multiclass", "binary"):
        table13_loeo(runs, task=tsk)
        table14_loeo_significance(runs, task=tsk)
    log.info("tables written to %s", RESULTS_TABLES)


if __name__ == "__main__":
    main()
