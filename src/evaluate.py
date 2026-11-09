"""Metric computation.

Accuracy is reported but is deliberately *not* the headline: on this dataset it
saturates for almost any model, which is exactly why the base paper's 99.99%
figure cannot by itself discriminate between methods.  The metrics that carry
the argument are macro-F1, MCC, PR-AUC, the false-positive rate and
minority-class recall (section 23).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.int8)
    out[np.arange(len(y)), y] = 1
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    restrict_to_present: bool = False,
) -> Dict[str, object]:
    """Full metric battery for one (split, model, seed) evaluation.

    ``y_score`` is the ``[n, n_classes]`` probability matrix; if omitted the
    threshold-free metrics are returned as ``None`` rather than approximated.

    ``restrict_to_present`` averages the macro metrics over the classes that
    actually occur in ``y_true``.  An endpoint-grouped split can legitimately
    have no test instances of a class whose victims all sit in the training
    split, and averaging a zero over that absent class would report a penalty
    the model never earned.  The classes averaged over are recorded in
    ``macro_averaged_over`` so the restriction is always visible.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        confusion_matrix, f1_score, matthews_corrcoef, precision_score,
        recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    labels = list(range(n_classes))
    if class_names is None:
        class_names = [str(i) for i in labels]

    present = sorted(np.unique(y_true).tolist())
    macro_labels = present if restrict_to_present else labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # One-vs-rest FP/FN rates from the confusion matrix.
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr_pc = np.where((fp + tn) > 0, fp / (fp + tn), np.nan)
        fnr_pc = np.where((fn + tp) > 0, fn / (fn + tp), np.nan)

    out: Dict[str, object] = {
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=macro_labels,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=macro_labels,
                                           average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=macro_labels,
                                   average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels,
                                                    average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels,
                                              average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.nanmean(fpr_pc)),
        "fnr_macro": float(np.nanmean(fnr_pc)),
        "confusion_matrix": cm.tolist(),
        "classes_present_in_test": present,
        "macro_averaged_over": list(macro_labels),
        "macro_restricted_to_present": bool(restrict_to_present),
    }

    # Binary-specific rates, using class 1 = attack.
    if n_classes == 2:
        tn_b, fp_b, fn_b, tp_b = cm.ravel()
        denom_fpr = fp_b + tn_b
        denom_fnr = fn_b + tp_b
        out["fpr"] = float(fp_b / denom_fpr) if denom_fpr else float("nan")
        out["fnr"] = float(fn_b / denom_fnr) if denom_fnr else float("nan")
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        out["fpr"] = out["fpr_macro"]
        out["fnr"] = out["fnr_macro"]

    # Minority-class recall: the rarest class actually present in y_true.
    present_arr, counts = np.unique(y_true, return_counts=True)
    minority = int(present_arr[np.argmin(counts)])
    out["minority_class"] = minority
    out["minority_class_support"] = int(counts.min())
    out["minority_class_recall"] = float(
        tp[minority] / (tp[minority] + fn[minority])
        if (tp[minority] + fn[minority]) > 0 else np.nan
    )

    # Threshold-free metrics.
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=np.float64)
        Y = _one_hot(y_true, n_classes)
        try:
            if n_classes == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_true, y_score[:, 1]))
            elif len(present) < 2:
                out["roc_auc"] = None
                out["pr_auc"] = None
                out["auc_note"] = ("only one class present in the test split; "
                                   "AUC is undefined")
            else:
                cols = list(present) if restrict_to_present else labels
                sub = y_score[:, cols]
                sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
                out["roc_auc"] = float(roc_auc_score(
                    y_true, sub if len(cols) > 2 else sub[:, 1],
                    multi_class="ovr", average="macro", labels=cols)
                    if len(cols) > 2 else
                    roc_auc_score((y_true == cols[1]).astype(int), sub[:, 1]))
                out["pr_auc"] = float(average_precision_score(
                    Y[:, cols], y_score[:, cols], average="macro"))
        except ValueError as exc:
            # e.g. a class absent from y_true in a source-grouped split.
            out["roc_auc"] = None
            out["pr_auc"] = None
            out["auc_note"] = str(exc)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Optional[Sequence[str]] = None,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, f1_score, precision_score,
        recall_score,
    )

    labels = list(range(n_classes))
    names = list(class_names) if class_names else [str(i) for i in labels]
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + (cm.sum(axis=1) - tp))

    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        entry = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "false_positives": int(fp[i]),
            "fpr": float(fp[i] / (fp[i] + tn[i])) if (fp[i] + tn[i]) > 0 else float("nan"),
        }
        if y_score is not None and support[i] > 0:
            try:
                entry["average_precision"] = float(
                    average_precision_score((np.asarray(y_true) == i).astype(int),
                                            np.asarray(y_score)[:, i])
                )
            except Exception:
                entry["average_precision"] = float("nan")
        out[name] = entry
    return out


def error_analysis_table(
    df,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    max_rows: int = 5000,
):
    """Row-level record of every misclassified test flow (section 35)."""
    import pandas as pd

    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return pd.DataFrame(columns=["note"]).assign(note=["no misclassified flows"])
    sel = wrong if len(wrong) <= max_rows else wrong[:max_rows]
    rows = df.iloc[np.asarray(test_idx)[sel]].copy()
    rows["true_label"] = [class_names[i] for i in y_true[sel]]
    rows["pred_label"] = [class_names[i] for i in y_pred[sel]]
    rows.attrs["n_misclassified_total"] = int(len(wrong))
    rows.attrs["n_rows_written"] = int(len(sel))
    return rows
