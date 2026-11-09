"""Significance testing and uncertainty quantification across seeds.

Rules applied here, so that they cannot be quietly broken later:

* A test is only run over **repeated seeds of the same protocol**; there is no
  test that can be computed from a single accuracy number, and asking for one
  raises.
* Paired tests are used because every model sees the identical split for a given
  seed.  Normality is checked (Shapiro-Wilk) and the Wilcoxon signed-rank test
  is reported alongside the paired t-test whenever n is small, which it always
  is at five seeds.
* Multiple comparisons across the baseline set are corrected (Holm-Bonferroni).
* Effect size is Cohen's d_z for the paired case, with a bootstrap CI on the
  mean difference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare two models over matched seeds.

    ``a`` and ``b`` must be aligned: element *i* of each is the metric obtained
    under the same seed and split.
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired inputs: {a.shape} vs {b.shape}")
    if len(a) < 2:
        raise ValueError(
            "A significance test needs repeated seeds. Received "
            f"{len(a)} observation(s) -- run the multi-seed protocol first."
        )
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b

    out: Dict[str, object] = {
        "model_a": name_a,
        "model_b": name_b,
        "n_pairs": int(len(diff)),
        "mean_a": float(a.mean()),
        "std_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mean_b": float(b.mean()),
        "std_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "mean_difference": float(diff.mean()),
        "cohens_dz": cohens_dz(diff),
    }
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    out["difference_ci95"] = [lo, hi]
    out["ci_excludes_zero"] = bool(lo > 0 or hi < 0)

    if np.allclose(diff, 0):
        out.update({"paired_t_p": 1.0, "wilcoxon_p": 1.0,
                    "shapiro_p": None, "note": "identical scores"})
        return out

    try:
        out["shapiro_p"] = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else None
    except Exception:
        out["shapiro_p"] = None

    t_stat, t_p = stats.ttest_rel(a, b)
    out["paired_t_stat"] = float(t_stat)
    out["paired_t_p"] = float(t_p)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
        out["wilcoxon_stat"] = float(w_stat)
        out["wilcoxon_p"] = float(w_p)
    except ValueError as exc:
        out["wilcoxon_stat"] = None
        out["wilcoxon_p"] = None
        out["wilcoxon_note"] = str(exc)

    # With five seeds the Wilcoxon test cannot reach p < 0.05 for some
    # configurations; say so rather than letting a reader over-read the number.
    if len(diff) < 6:
        out["small_sample_warning"] = (
            f"n={len(diff)} paired observations: the exact Wilcoxon test has a "
            f"minimum attainable two-sided p of "
            f"{2 / (2 ** len(diff)):.4f}; interpret alongside the bootstrap CI "
            f"and effect size."
        )
    return out


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, object]:
    """Step-down correction over a family of comparisons."""
    items = [(k, v) for k, v in pvalues.items() if v is not None and not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, object] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        adjusted = min(1.0, max(prev, p * (m - i)))
        prev = adjusted
        out[k] = {
            "p_raw": float(p),
            "p_holm": float(adjusted),
            "threshold": float(thresh),
            "significant_at_alpha": bool(adjusted < alpha),
        }
    for k, v in pvalues.items():
        if k not in out:
            out[k] = {"p_raw": v, "p_holm": None, "significant_at_alpha": None}
    out["_family_size"] = m
    out["_alpha"] = alpha
    return out


def aggregate_over_seeds(
    df,
    group_cols: Sequence[str] = ("model", "representation", "task", "protocol"),
    metrics: Sequence[str] = (
        "accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc", "pr_auc",
        "fpr", "fnr", "balanced_accuracy", "minority_class_recall",
        "precision_macro", "recall_macro",
    ),
    n_boot: int = 10_000,
    seed: int = 42,
):
    """mean / std / 95% bootstrap CI per model-configuration."""
    import pandas as pd

    rows: List[Dict[str, object]] = []
    group_cols = [c for c in group_cols if c in df.columns]
    for key, sub in df.groupby(list(group_cols), dropna=False):
        row: Dict[str, object] = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = int(len(sub))
        row["seeds"] = sorted(sub["seed"].dropna().astype(int).tolist()) \
            if "seed" in sub.columns else []
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = pd.to_numeric(sub[m], errors="coerce").to_numpy()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def compare_against_reference(
    df,
    reference_model: str = "E-GAT-IDS",
    metric: str = "f1_macro",
    group_cols: Sequence[str] = ("representation", "task", "protocol"),
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Every baseline vs the proposed model, paired by seed, Holm-corrected."""
    import pandas as pd

    results: List[Dict[str, object]] = []
    group_cols = [c for c in group_cols if c in df.columns]
    for key, sub in df.groupby(list(group_cols), dropna=False):
        ctx = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        ref = sub[sub["model"] == reference_model]
        if ref.empty:
            continue
        ref_by_seed = ref.set_index("seed")[metric]
        family: Dict[str, float] = {}
        rows: List[Dict[str, object]] = []
        for model, msub in sub.groupby("model"):
            if model == reference_model:
                continue
            other = msub.set_index("seed")[metric]
            common = sorted(set(ref_by_seed.index) & set(other.index))
            if len(common) < 2:
                rows.append({**ctx, "metric": metric, "model_a": reference_model,
                             "model_b": model, "n_pairs": len(common),
                             "note": "insufficient paired seeds for a test"})
                continue
            cmp = paired_comparison(
                ref_by_seed.loc[common].to_numpy(),
                other.loc[common].to_numpy(),
                name_a=reference_model, name_b=model, seed=seed,
            )
            cmp.update(ctx)
            cmp["metric"] = metric
            rows.append(cmp)
            if cmp.get("wilcoxon_p") is not None:
                family[model] = cmp["wilcoxon_p"]
        corrected = holm_bonferroni(family) if family else {}
        for r in rows:
            c = corrected.get(r.get("model_b"))
            if isinstance(c, dict):
                r["wilcoxon_p_holm"] = c.get("p_holm")
                r["significant_holm"] = c.get("significant_at_alpha")
        results.extend(rows)
    return results
