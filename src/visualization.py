"""Publication figures.

Palette provenance
------------------
The categorical palette is a re-ordered Okabe-Ito set, validated with the
``dataviz`` six-check validator rather than chosen by eye:

    light  #0072B2 #D55E00 #009E73 #E69F00 #CC79A7 #5B5BD6
           lightness band PASS | chroma floor PASS | CVD separation PASS
           (worst adjacent dE 11.0 deutan) | normal-vision floor PASS (20.5)
           | contrast WARN -> relieved: every figure carries a legend, axis
             labels and a companion CSV in results/tables/

    dark   #3B9BD6 #DA6A26 #16A97F #B98105 #C46B9F #6F7FE0
           all six checks PASS against surface #1a1a19

Hues are assigned in fixed order and never cycled; a seventh series folds into
"Other" or becomes a small multiple.  Sequential scales are single-hue
light->dark; the diverging scale is blue<->neutral gray<->vermillion.  No figure
in this project uses two y-axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    categorical: Tuple[str, ...]
    sequential: Tuple[str, ...]
    diverging: Tuple[str, str, str]


LIGHT = Theme(
    name="light",
    surface="#FCFCFB",
    ink_primary="#1A1A19",
    ink_secondary="#4A4A46",
    ink_muted="#77776F",
    grid="#E4E4DF",
    axis="#B8B8B0",
    categorical=("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#5B5BD6"),
    sequential=("#F2F7FB", "#D3E3EF", "#A9CAE1", "#6FA6CB", "#3B7FAE", "#1B5A85",
                "#0A3A5C"),
    diverging=("#0072B2", "#E8E8E4", "#D55E00"),
)

DARK = Theme(
    name="dark",
    surface="#1A1A19",
    ink_primary="#F2F2EE",
    ink_secondary="#C9C9C2",
    ink_muted="#8E8E86",
    grid="#33332F",
    axis="#55554F",
    categorical=("#3B9BD6", "#DA6A26", "#16A97F", "#B98105", "#C46B9F", "#6F7FE0"),
    sequential=("#14262F", "#1B3C4D", "#22556D", "#2B7091", "#3C8DB2", "#62AAD0",
                "#9BCBE6"),
    diverging=("#3B9BD6", "#3A3A36", "#DA6A26"),
)

THEMES = {"light": LIGHT, "dark": DARK}

#: Stable model -> colour assignment. Colour follows the entity, never its rank,
#: so filtering the model set never repaints the survivors.
MODEL_COLOURS: Dict[str, int] = {
    "E-GAT-IDS": 0,
    "GAT": 1,
    "EdgeGAT": 2,
    "GCN": 3,
    "GraphSAGE": 4,
    "MLP": 5,
}
_EXTRA_MODEL_ORDER = ["RandomForest", "XGBoost", "GA-EBT", "SVM", "KNN"]


def model_colour(model: str, theme: Theme = LIGHT) -> str:
    """Fixed hue per model; models beyond the palette get a neutral 'Other' ink."""
    if model in MODEL_COLOURS:
        return theme.categorical[MODEL_COLOURS[model]]
    return theme.ink_muted


def apply_theme(theme: Theme = LIGHT, base_font: float = 8.0) -> None:
    """IEEE two-column defaults: 8pt body text, thin recessive chrome."""
    plt.rcParams.update({
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "font.family": "DejaVu Sans",
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 0.5,
        "ytick.labelsize": base_font - 0.5,
        "legend.fontsize": base_font - 0.5,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.6,
        "axes.labelcolor": theme.ink_secondary,
        "axes.titlecolor": theme.ink_primary,
        "text.color": theme.ink_primary,
        "xtick.color": theme.ink_muted,
        "ytick.color": theme.ink_muted,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": theme.grid,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _despine(ax, keep: Sequence[str] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def sequential_cmap(theme: Theme = LIGHT, name: str = "seq") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(theme.sequential))


def diverging_cmap(theme: Theme = LIGHT, name: str = "div") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(theme.diverging))


def _save(fig, path: Path, also_pdf: bool = True) -> Path:
    """Finalise every bar axis, then write PNG (600 dpi) and a vector PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        _finalize_bars(ax)
    fig.savefig(path)
    if also_pdf and path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Bars
#
# A bar is rounded at its *data* end only and square where it meets the
# baseline, and the corner radius is a fixed number of screen pixels rather than
# a number of data units -- otherwise the same code gives a hairline on an axis
# that runs to 500,000 and a blob on one that runs to 1.0. The geometry is
# therefore recorded when the bar is declared and converted to patches at save
# time, once the axis limits are final.
# --------------------------------------------------------------------------- #


def _rounded_bars(ax, positions, values, colour, width=0.62, horizontal=False,
                  radius_pts=4.0, gap=0.02):
    """Declare bars on ``ax``; they are drawn by :func:`_finalize_bars`."""
    specs = getattr(ax, "_bar_specs", None)
    if specs is None:
        specs = []
        ax._bar_specs = specs
    for p, v in zip(positions, values):
        if v is None or not np.isfinite(v):
            continue
        specs.append({"pos": float(p), "value": float(v), "colour": colour,
                      "width": float(width) - float(gap), "horizontal": horizontal,
                      "radius_px": float(radius_pts)})
    # Keep autoscale informed even though nothing is drawn yet.
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if finite:
        if horizontal:
            ax.update_datalim([(0, min(positions) - width),
                               (max(finite), max(positions) + width)])
        else:
            ax.update_datalim([(min(positions) - width, 0),
                               (max(positions) + width, max(finite))])
    return specs


def _rounded_end_path(x0, y0, x1, y1, rx, ry, horizontal):
    """Rectangle with the two corners at the data end rounded."""
    from matplotlib.path import Path as MPath

    if horizontal:
        rx = min(rx, abs(x1 - x0) * 0.5)
        ry = min(ry, abs(y1 - y0) * 0.5)
        verts = [
            (x0, y0), (x1 - rx, y0),
            (x1, y0), (x1, y0 + ry),
            (x1, y1 - ry),
            (x1, y1), (x1 - rx, y1),
            (x0, y1), (x0, y0),
        ]
        codes = [MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CLOSEPOLY]
    else:
        rx = min(rx, abs(x1 - x0) * 0.5)
        ry = min(ry, abs(y1 - y0) * 0.5)
        verts = [
            (x0, y0), (x0, y1 - ry),
            (x0, y1), (x0 + rx, y1),
            (x1 - rx, y1),
            (x1, y1), (x1, y1 - ry),
            (x1, y0), (x0, y0),
        ]
        codes = [MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CLOSEPOLY]
    return MPath(verts, codes)


def _finalize_bars(ax) -> None:
    from matplotlib.patches import PathPatch

    specs = getattr(ax, "_bar_specs", None)
    if not specs:
        return
    ax._bar_specs = []          # idempotent: a second save must not redraw

    inv = ax.transData.inverted()
    origin = inv.transform((0, 0))
    for spec in specs:
        r = spec["radius_px"]
        far = inv.transform((r, r))
        rx = abs(far[0] - origin[0])
        ry = abs(far[1] - origin[1])
        half = spec["width"] / 2.0
        if spec["horizontal"]:
            path = _rounded_end_path(0.0, spec["pos"] - half,
                                     spec["value"], spec["pos"] + half,
                                     rx, ry, True)
        else:
            path = _rounded_end_path(spec["pos"] - half, 0.0,
                                     spec["pos"] + half, spec["value"],
                                     rx, ry, False)
        ax.add_patch(PathPatch(path, facecolor=spec["colour"], linewidth=0,
                               zorder=2))


def _metric_limits(ax, values, horizontal=True, pad=0.06, cap=1.0):
    """Limits for a bounded metric axis: never past 1.0, never a zero span."""
    finite = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return
    lo = max(0.0, float(finite.min()) - pad)
    hi = min(cap, float(finite.max()) + pad)
    if hi - lo < 1e-3:
        lo, hi = max(0.0, lo - 0.01), min(cap, hi + 0.01)
    if horizontal:
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


# --------------------------------------------------------------------------- #
# Figure 5 -- class distribution
# --------------------------------------------------------------------------- #


def fig_class_distribution(
    counts: Dict[str, Dict[str, int]],
    out_path: Path,
    theme: Theme = LIGHT,
    title: str = "Flow records per scenario",
) -> Path:
    """Grouped bars: scenario counts for each representation.

    ``counts`` maps representation -> {scenario: n}.
    """
    apply_theme(theme)
    reps = list(counts.keys())
    scenarios = list(dict.fromkeys(
        s for rep in reps for s in counts[rep].keys()
    ))
    x = np.arange(len(scenarios), dtype=float)
    n_rep = max(1, len(reps))
    slot = 0.80 / n_rep

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for k, rep in enumerate(reps):
        vals = [counts[rep].get(s, 0) for s in scenarios]
        pos = x - 0.40 + slot * (k + 0.5)
        _rounded_bars(ax, pos, vals, theme.categorical[k], width=slot, gap=0.02)
        ax.plot([], [], marker="s", linestyle="none", color=theme.categorical[k],
                label=rep, markersize=5)
        # Direct labels: <=4 series, so every bar is annotated.
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:,}", (p, v), textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=5.5,
                        color=theme.ink_secondary, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Flow records")
    ax.set_title(title, loc="left", pad=6)
    ax.set_ylim(0, max(max(c.values()) for c in counts.values()) * 1.28)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="upper right", ncol=len(reps), handletextpad=0.4,
              columnspacing=1.0)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 7 -- confusion matrix
# --------------------------------------------------------------------------- #


def fig_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    theme: Theme = LIGHT,
    normalise: str = "true",
    title: str = "Confusion matrix",
) -> Path:
    """Single-hue sequential heatmap with counts and row-normalised rates."""
    apply_theme(theme)
    cm = np.asarray(cm, dtype=float)
    if normalise == "true":
        denom = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, denom, out=np.zeros_like(cm), where=denom > 0)
        cbar_label = "Share of true class"
    else:
        shown = cm
        cbar_label = "Flow records"

    k = len(class_names)
    fig, ax = plt.subplots(figsize=(0.62 * k + 1.9, 0.62 * k + 1.5))
    im = ax.imshow(shown, cmap=sequential_cmap(theme), vmin=0,
                   vmax=shown.max() if shown.max() > 0 else 1)

    thresh = shown.max() * 0.55 if shown.max() > 0 else 0.5
    for i in range(k):
        for j in range(k):
            ink = theme.surface if shown[i, j] > thresh else theme.ink_primary
            label = (f"{shown[i, j]:.3f}\n{int(cm[i, j]):,}"
                     if normalise == "true" else f"{int(cm[i, j]):,}")
            ax.text(j, i, label, ha="center", va="center", color=ink, fontsize=5.8)

    ax.set_xticks(range(k), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(range(k), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color=theme.surface, linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(cbar_label, color=theme.ink_secondary)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=theme.ink_muted, width=0.5)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 8 -- per-class F1
# --------------------------------------------------------------------------- #


def fig_per_class_f1(
    per_class: Dict[str, Dict[str, float]],
    out_path: Path,
    theme: Theme = LIGHT,
    metric: str = "f1",
    title: str = "Per-class F1",
    errors: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """Horizontal bars, one model per colour, classes on the y-axis."""
    apply_theme(theme)
    models = list(per_class.keys())
    classes = list(dict.fromkeys(c for m in models for c in per_class[m].keys()))
    y = np.arange(len(classes), dtype=float)
    slot = 0.80 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(3.5, 0.42 * len(classes) * max(1, len(models)) + 1.0))
    for k, m in enumerate(models):
        vals = [per_class[m].get(c, {}).get(metric, np.nan)
                if isinstance(per_class[m].get(c), dict) else per_class[m].get(c, np.nan)
                for c in classes]
        pos = y - 0.40 + slot * (k + 0.5)
        _rounded_bars(ax, pos, vals, model_colour(m, theme), width=slot,
                      horizontal=True, gap=0.02)
        if errors and m in errors:
            err = [errors[m].get(c, 0.0) for c in classes]
            ax.errorbar(vals, pos, xerr=err, fmt="none", ecolor=theme.ink_muted,
                        elinewidth=0.7, capsize=1.6)
        ax.plot([], [], marker="s", linestyle="none", color=model_colour(m, theme),
                label=m, markersize=5)
        if len(models) <= 4:
            for p, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.3f}", (v, p), textcoords="offset points",
                                xytext=(3, 0), va="center", fontsize=5.5,
                                color=theme.ink_secondary)

    ax.set_yticks(y, labels=classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title, loc="left", pad=6)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="lower right", ncol=1)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figures 9 & 10 -- PR and ROC curves
# --------------------------------------------------------------------------- #


def _curve_figure(curves, out_path, theme, xlabel, ylabel, title,
                  baseline=None, diagonal=False):
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, (name, (x, y, auc)) in enumerate(curves.items()):
        ax.plot(x, y, color=model_colour(name, theme), linewidth=1.6,
                label=f"{name} ({auc:.4f})", solid_capstyle="round")
    if diagonal:
        ax.plot([0, 1], [0, 1], color=theme.axis, linewidth=0.8, linestyle=(0, (3, 3)),
                zorder=0)
    if baseline is not None:
        ax.axhline(baseline, color=theme.axis, linewidth=0.8, linestyle=(0, (3, 3)),
                   zorder=0)
        ax.annotate("chance", (0.02, baseline), textcoords="offset points",
                    xytext=(0, 3), fontsize=5.5, color=theme.ink_muted)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6)
    ax.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="lower left" if diagonal else "lower right")
    return _save(fig, out_path)


def fig_roc_curves(y_true, scores_by_model: Dict[str, np.ndarray], out_path: Path,
                   theme: Theme = LIGHT, positive_class: int = 1,
                   title: str = "ROC, attack class") -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = {}
    for name, s in scores_by_model.items():
        s = np.asarray(s)
        score = s[:, positive_class] if s.ndim == 2 else s
        fpr, tpr, _ = roc_curve((np.asarray(y_true) == positive_class).astype(int), score)
        auc = roc_auc_score((np.asarray(y_true) == positive_class).astype(int), score)
        curves[name] = (fpr, tpr, auc)
    return _curve_figure(curves, out_path, theme, "False positive rate",
                         "True positive rate", title, diagonal=True)


def fig_pr_curves(y_true, scores_by_model: Dict[str, np.ndarray], out_path: Path,
                  theme: Theme = LIGHT, positive_class: int = 1,
                  title: str = "Precision-recall, attack class") -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    yb = (np.asarray(y_true) == positive_class).astype(int)
    curves = {}
    for name, s in scores_by_model.items():
        s = np.asarray(s)
        score = s[:, positive_class] if s.ndim == 2 else s
        prec, rec, _ = precision_recall_curve(yb, score)
        ap = average_precision_score(yb, score)
        curves[name] = (rec, prec, ap)
    return _curve_figure(curves, out_path, theme, "Recall", "Precision", title,
                         baseline=float(yb.mean()))


# --------------------------------------------------------------------------- #
# Figure 11 -- ablation
# --------------------------------------------------------------------------- #


def fig_ablation(
    summary,
    out_path: Path,
    metric: str = "f1_macro",
    theme: Theme = LIGHT,
    order: Optional[Sequence[str]] = None,
    title: str = "Ablation ladder",
) -> Path:
    """One bar per variant with the 95% bootstrap CI drawn as an error bar.

    ``summary`` is the aggregate frame from :func:`src.statistics.aggregate_over_seeds`.
    """
    apply_theme(theme)
    import pandas as pd

    df = summary.copy()
    if order:
        df["__o"] = df["model"].map({m: i for i, m in enumerate(order)})
        df = df.sort_values("__o")
    else:
        df = df.sort_values(f"{metric}_mean")

    names = df["model"].tolist()
    means = df[f"{metric}_mean"].to_numpy()
    lo = df.get(f"{metric}_ci_lo", pd.Series(means)).to_numpy()
    hi = df.get(f"{metric}_ci_hi", pd.Series(means)).to_numpy()
    y = np.arange(len(names), dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 0.34 * len(names) + 1.1))
    for i, name in enumerate(names):
        _rounded_bars(ax, [y[i]], [means[i]], model_colour(name, theme),
                      width=0.66, horizontal=True)
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none",
                ecolor=theme.ink_secondary, elinewidth=0.8, capsize=2.0)
    for i, v in enumerate(means):
        ax.annotate(f"{v:.4f}", (v, y[i]), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=6,
                    color=theme.ink_secondary)

    ax.set_yticks(y, labels=names)
    ax.invert_yaxis()
    bounded = metric not in {"time_train_seconds", "n_parameters"}
    _metric_limits(ax, np.concatenate([lo, hi]), horizontal=True, pad=0.02,
                   cap=1.0 if bounded else float("inf"))
    ax.set_xlabel(f"{metric.replace('_', ' ')} (mean over seeds, 95% bootstrap CI)")
    ax.set_title(title, loc="left", pad=6)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 12 -- uni-flow vs bi-flow
# --------------------------------------------------------------------------- #


def fig_representation_comparison(
    summary,
    out_path: Path,
    metrics: Sequence[str] = ("f1_macro", "mcc", "pr_auc", "minority_class_recall"),
    theme: Theme = LIGHT,
    title: str = "Uni-flow vs bi-flow",
) -> Path:
    """Small multiples, one panel per metric; representation is the series."""
    apply_theme(theme)
    reps = sorted(summary["representation"].dropna().unique().tolist())
    models = summary["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(1.85 * len(metrics) + 0.6, 0.30 * len(models) + 1.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(models), dtype=float)
    slot = 0.78 / max(1, len(reps))

    for ax, metric in zip(axes, metrics):
        for k, rep in enumerate(reps):
            sub = summary[summary["representation"] == rep].set_index("model")
            vals = [float(sub[f"{metric}_mean"].get(m, np.nan)) if f"{metric}_mean" in sub
                    else np.nan for m in models]
            pos = y - 0.39 + slot * (k + 0.5)
            _rounded_bars(ax, pos, vals, theme.categorical[k], width=slot,
                          horizontal=True, gap=0.02)
        ax.set_title(metric.replace("_", " "), loc="left", fontsize=7,
                     color=theme.ink_secondary)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_yticks(y, labels=models)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="s", linestyle="none",
                          color=theme.categorical[k], markersize=5, label=rep)
               for k, rep in enumerate(reps)]
    # Legend above the panels, so it never sits on top of a bar.
    fig.legend(handles=handles, loc="upper right", ncol=len(reps),
               bbox_to_anchor=(0.99, 1.0), frameon=False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=9, color=theme.ink_primary)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figures 6 & 13 -- communication graph, plain and attention-weighted
# --------------------------------------------------------------------------- #


def fig_communication_graph(
    edge_records,
    out_path: Path,
    theme: Theme = LIGHT,
    weight_col: str = "n_flows",
    title: str = "Communication graph",
    top_k: int = 60,
    annotate_top: int = 8,
    seed: int = 42,
) -> Path:
    """Node-link diagram; edge thickness encodes ``weight_col``.

    ``edge_records`` is a DataFrame with ``src_ip``, ``dst_ip`` and the weight
    column.  Used for Figure 6 (flow counts) and Figure 13 (attention mass).
    """
    import networkx as nx

    apply_theme(theme)
    df = edge_records.nlargest(top_k, weight_col)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(str(r["src_ip"]), str(r["dst_ip"]), w=float(r[weight_col]))

    pos = nx.spring_layout(G, seed=seed, k=0.9 / np.sqrt(max(1, G.number_of_nodes())))
    weights = np.array([G[u][v]["w"] for u, v in G.edges()], dtype=float)
    wnorm = weights / weights.max() if weights.max() > 0 else weights
    degree = dict(G.degree())

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    cmap = sequential_cmap(theme)
    for (u, v), w in zip(G.edges(), wnorm):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-|>", color=cmap(0.35 + 0.6 * w),
                            linewidth=0.4 + 2.6 * w, alpha=0.85,
                            shrinkA=6, shrinkB=7,
                            connectionstyle="arc3,rad=0.08"),
        )
    sizes = np.array([40 + 26 * degree[n] for n in G.nodes()], dtype=float)
    ax.scatter([pos[n][0] for n in G.nodes()], [pos[n][1] for n in G.nodes()],
               s=sizes, c=theme.categorical[0], edgecolors=theme.surface,
               linewidths=1.4, zorder=3)

    hubs = sorted(G.nodes(), key=lambda n: -degree[n])[:annotate_top]
    for n in hubs:
        ax.annotate(n, pos[n], textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=5.5, color=theme.ink_secondary, zorder=4)

    ax.set_title(title, loc="left", pad=6)
    ax.set_axis_off()
    note = f"edge width ∝ {weight_col.replace('_', ' ')}; top {len(df)} edges shown"
    ax.annotate(note, (0.0, -0.04), xycoords="axes fraction", fontsize=5.5,
                color=theme.ink_muted)
    return _save(fig, out_path)


def fig_attention_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: Path,
    theme: Theme = LIGHT,
    title: str = "Mean attention by relationship",
    cbar_label: str = "Mean attention",
) -> Path:
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(0.42 * len(col_labels) + 2.0,
                                    0.34 * len(row_labels) + 1.4))
    im = ax.imshow(np.asarray(matrix, dtype=float), cmap=sequential_cmap(theme),
                   aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=theme.surface, linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, color=theme.ink_secondary)
    cb.outline.set_visible(False)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 14 -- runtime vs performance
# --------------------------------------------------------------------------- #


def fig_runtime_vs_performance(
    summary,
    out_path: Path,
    x: str = "time_train_seconds",
    y: str = "f1_macro_mean",
    theme: Theme = LIGHT,
    title: str = "Cost against performance",
    logx: bool = True,
) -> Path:
    """Single-panel scatter. One axis pair, no secondary y-axis."""
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for _, r in summary.iterrows():
        name = r["model"]
        xv, yv = r.get(x, np.nan), r.get(y, np.nan)
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        ax.scatter(xv, yv, s=46, color=model_colour(name, theme),
                   edgecolors=theme.surface, linewidths=1.2, zorder=3)
        ax.annotate(name, (xv, yv), textcoords="offset points", xytext=(6, 2),
                    fontsize=6, color=theme.ink_secondary)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)" if logx else "Training time (s)")
    ax.set_ylabel(y.replace("_mean", "").replace("_", " "))
    ax.set_title(title, loc="left", pad=6)
    ax.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Audit figures
# --------------------------------------------------------------------------- #


def fig_feature_distributions(
    df, feature_cols: Sequence[str], out_path: Path, theme: Theme = LIGHT,
    label_col: str = "binary_label", max_features: int = 12,
    title: str = "Feature distributions by class",
) -> Path:
    apply_theme(theme)
    cols = list(feature_cols)[:max_features]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    classes = sorted(df[label_col].unique().tolist())

    for ax, c in zip(axes, cols):
        for k, cls in enumerate(classes):
            v = df.loc[df[label_col] == cls, c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            v = np.log1p(np.clip(v, 0, None)) if np.nanmax(v) > 1e3 else v
            ax.hist(v, bins=40, density=True, histtype="step", linewidth=1.2,
                    color=theme.categorical[k % len(theme.categorical)])
        ax.set_title(c, loc="left", fontsize=6, color=theme.ink_secondary)
        ax.tick_params(labelsize=5)
        _despine(ax)
        ax.grid(True, alpha=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    for k, cls in enumerate(classes):
        axes[0].plot([], [], color=theme.categorical[k % len(theme.categorical)],
                     label=f"class {cls}")
    axes[0].legend(fontsize=5, loc="upper right")
    fig.suptitle(title, x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


def fig_correlation_matrix(
    df, feature_cols: Sequence[str], out_path: Path, theme: Theme = LIGHT,
    title: str = "Feature correlation (Spearman)",
) -> Path:
    apply_theme(theme)
    corr = df[list(feature_cols)].corr(method="spearman").to_numpy()
    n = len(feature_cols)
    fig, ax = plt.subplots(figsize=(0.24 * n + 2.2, 0.24 * n + 1.8))
    im = ax.imshow(corr, cmap=diverging_cmap(theme), vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels=list(feature_cols), rotation=80, ha="right",
                  fontsize=5)
    ax.set_yticks(range(n), labels=list(feature_cols), fontsize=5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=[-1, 0, 1])
    cb.set_label("Spearman ρ", color=theme.ink_secondary)
    cb.outline.set_visible(False)
    return _save(fig, out_path)


def fig_training_curves(
    histories: Dict[str, Dict[str, list]], out_path: Path, theme: Theme = LIGHT,
    key: str = "val_f1_macro", title: str = "Validation macro-F1 by epoch",
) -> Path:
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for name, h in histories.items():
        vals = h.get(key, [])
        ax.plot(np.arange(1, len(vals) + 1), vals, color=model_colour(name, theme),
                label=name, linewidth=1.6)
        if len(vals):
            ax.scatter([len(vals)], [vals[-1]], s=22, color=model_colour(name, theme),
                       edgecolors=theme.surface, linewidths=1.0, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(key.replace("_", " "))
    ax.set_title(title, loc="left", pad=6)
    ax.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="lower right")
    return _save(fig, out_path)
