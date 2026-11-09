"""Shared experiment driver.

One function, :func:`run_configuration`, executes a single
(representation, task, protocol, seed) cell for a chosen set of models and
writes one :class:`~src.utils.RunRecord` per model.  Every runner in this
directory is a thin loop over that function, so the preprocessing, the split,
the graph and the leakage guards are provably identical across models: the test
split is constructed once and handed to everything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    SCENARIO_LABELS, SCENARIO_PRETTY, feature_columns, file_hashes,
    load_representation,
)
from src.evaluate import compute_metrics, error_analysis_table, per_class_report  # noqa: E402
from src.graph_builder import build_flow_graph  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.models import GRAPH_MODELS  # noqa: E402
from src.preprocessing import (  # noqa: E402
    FlowPreprocessor, assert_no_duplicate_rows_across_splits,
)
from src.split import make_split, verify_split  # noqa: E402
from src.train import (  # noqa: E402
    GraphTensors, TrainConfig, measure_inference, predict_graph_model,
    predict_mlp, train_graph_model, train_mlp,
)
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_CKPT, RESULTS_LOGS, RESULTS_RAW, RunRecord,
    count_parameters, ensure_dirs, get_logger, peak_memory_mb, save_json, set_seed,
)

log = get_logger("experiments")

CLASSICAL = ("EndpointLookup", "RandomForest", "XGBoost", "SVM", "KNN",
             "GA-EBT")
NEURAL_TABULAR = ("MLP",)
GRAPH = tuple(GRAPH_MODELS.keys())
ALL_MODELS = CLASSICAL + NEURAL_TABULAR + GRAPH

BINARY_CLASS_NAMES = ["Normal", "Attack"]
MULTICLASS_NAMES = [SCENARIO_PRETTY[k] for k, _ in
                    sorted(SCENARIO_LABELS.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


class PreparedData:
    """Everything downstream of the split, built exactly once per cell."""

    def __init__(self, representation: str, task: str, protocol: str, seed: int,
                 raw_dir: Optional[Path] = None, scaler: str = "quantile",
                 node_feature_source: str = "train",
                 nrows: Optional[int] = None,
                 split_kwargs: Optional[Dict] = None) -> None:
        self.representation = representation
        self.task = task
        self.protocol = protocol
        #: Label written into the run record and the result filename. Protocol L
        #: puts the fold index here so each fold gets its own file.
        self.seed = seed
        #: Seed that actually drives model initialisation, shuffling and any
        #: estimator's random_state. Normally identical to ``seed``; protocol L
        #: pins it so that variation across folds cannot be initialisation
        #: noise. Recorded separately in every run record.
        self.model_seed = seed
        self.raw_dir = Path(raw_dir) if raw_dir else None

        self.df = load_representation(representation, raw_dir) if raw_dir \
            else load_representation(representation)
        if nrows:
            self.df = self.df.sample(n=min(nrows, len(self.df)),
                                     random_state=seed).reset_index(drop=True)
            self.df.attrs["representation"] = representation
        self.feature_cols = feature_columns(self.df)
        self.y = self.df["binary_label"].to_numpy() if task == "binary" \
            else self.df["multiclass_label"].to_numpy()
        self.class_names = BINARY_CLASS_NAMES if task == "binary" else MULTICLASS_NAMES
        self.n_classes = len(np.unique(self.df["binary_label"])) if task == "binary" \
            else len(MULTICLASS_NAMES)

        split_result = make_split(protocol, self.df, self.y, self.feature_cols,
                                  seed=seed, **(split_kwargs or {}))
        self.splits = split_result.splits
        self.split_description = split_result.description
        self.applicable = bool(self.splits)
        if not self.applicable:
            log.warning("protocol %s not applicable for %s/%s: %s", protocol,
                        representation, task,
                        self.split_description.get("reason"))
            return
        verify_split(self.splits, len(self.df))

        # Fit preprocessing on the training split only.
        self.pre = FlowPreprocessor(scaler=scaler, representation=representation,
                                    random_state=seed)
        self.pre.fit(self.df.iloc[self.splits["train"]], self.feature_cols)
        self.Z = self.pre.transform(self.df)
        self.edge_feature_names = self.pre.feature_names

        self.leakage_report = assert_no_duplicate_rows_across_splits(
            self.df, self.splits, self.feature_cols)
        self.leakage_report["preprocessing_fitted_on"] = "train split only"
        self.leakage_report["node_features_from"] = node_feature_source
        self.leakage_report["message_passing_edges"] = "train split only"

        self.graph, self.node_scaler = build_flow_graph(
            self.df, self.Z, self.edge_feature_names,
            train_idx=self.splits["train"],
            node_feature_source=node_feature_source,
        )
        self.graph_statistics = self.graph.statistics()

    # ------------------------------------------------------------------ #

    def raw_matrix(self) -> np.ndarray:
        """Numeric-coerced, unscaled feature matrix (for GA-EBT's own scaling)."""
        sub = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan)
        med = sub.iloc[self.splits["train"]].median()
        return sub.fillna(med).fillna(0.0).to_numpy(dtype=np.float64)

    def context(self) -> Dict[str, object]:
        ctx = {
            "representation": self.representation,
            "task": self.task,
            "protocol": self.protocol,
            "seed": self.seed,
            "model_seed": self.model_seed,
            "seed_is_fold_index": self.protocol.upper() == "L",
            "n_rows": int(len(self.df)),
            "n_features": len(self.edge_feature_names),
            "feature_names": list(self.edge_feature_names),
            "class_names": list(self.class_names),
            "split": self.split_description,
            "leakage_checks": self.leakage_report,
            "preprocessing_report": self.pre.report.__dict__,
            "graph_statistics": self.graph_statistics,
        }
        return ctx


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #


def _finalise(
    data: PreparedData, model_name: str, y_prob: np.ndarray,
    timings: Dict[str, float], experiment: str,
    n_parameters: Optional[int] = None, extra_config: Optional[Dict] = None,
    extra_notes: str = "", write_errors: bool = True,
    model_label: Optional[str] = None,
) -> RunRecord:
    test_idx = data.splits["test"]
    y_true = data.y[test_idx]
    y_pred = y_prob.argmax(axis=1)

    # An endpoint-grouped split can legitimately lack a class in test; macro
    # averaging is then restricted to the classes present, and the run record
    # says so explicitly.
    restrict = bool(data.split_description.get(
        "evaluate_on_classes_present_in_test_only", False))
    metrics = compute_metrics(y_true, y_pred, y_prob, n_classes=data.n_classes,
                              class_names=data.class_names,
                              restrict_to_present=restrict)
    cm = metrics.pop("confusion_matrix")
    pc = per_class_report(y_true, y_pred, data.n_classes, data.class_names, y_prob)

    cfg = dict(data.context())
    if extra_config:
        cfg.update(extra_config)

    rec = RunRecord(
        experiment=experiment,
        model=model_label or model_name,
        representation=data.representation,
        task=data.task,
        protocol=data.protocol,
        seed=data.seed,
        config=cfg,
        metrics=metrics,
        per_class=pc,
        confusion_matrix=cm,
        timings=timings,
        n_parameters=n_parameters,
        peak_memory_mb=peak_memory_mb(),
        data_hashes=(file_hashes(data.raw_dir) if data.raw_dir
                     else file_hashes()),
        notes=extra_notes,
    )
    path = rec.save()

    if write_errors:
        errors = error_analysis_table(data.df, test_idx, y_true, y_pred,
                                      data.class_names)
        out = RESULTS_RAW / "errors" / f"{rec.run_id()}_errors.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(out, index=False)

    log.info("%-12s %-8s %-10s protocol %s seed %-5d | macroF1 %.4f MCC %.4f "
             "FPR %.5f acc %.5f -> %s",
             (model_label or model_name), data.representation, data.task, data.protocol, data.seed,
             metrics["f1_macro"], metrics["mcc"], metrics["fpr"],
             metrics["accuracy"], path.name)
    return rec


# --------------------------------------------------------------------------- #
# Model runners
# --------------------------------------------------------------------------- #


def run_classical(data: PreparedData, model_name: str, experiment: str,
                  max_train_samples: Optional[int] = None,
                  ga_cache_dir: Optional[Path] = None,
                  ga_budget: Optional[Dict] = None) -> RunRecord:
    from baselines import (
        endpoint_lookup, ga_ebt, knn, random_forest, svm, xgboost_model,
    )

    set_seed(data.model_seed)
    tr, te = data.splits["train"], data.splits["test"]
    timings: Dict[str, float] = {}
    extra: Dict[str, object] = {}
    notes = ""

    if model_name == "EndpointLookup":
        # Deliberately feature-free: it sees only the endpoint pair. Kept as a
        # baseline because a graph model that merely matches it has learned
        # endpoint identity rather than traffic behaviour.
        model = endpoint_lookup.build(seed=data.model_seed,
                                      n_classes=data.n_classes)
        t0 = time.perf_counter()
        model.fit(data.df.iloc[tr], data.y[tr])
        timings["train_seconds"] = time.perf_counter() - t0
        Xte = data.df.iloc[te]
        extra["inputs"] = "ordered (ip_src, ip_dst) pair only; no traffic features"
        notes = ("Feature-free endpoint-identity baseline; see "
                 "baselines/endpoint_lookup.py for why it is reported.")
    elif model_name == "GA-EBT":
        Xraw = data.raw_matrix()
        p = ga_ebt.minmax_fit(Xraw[tr])
        Xs = ga_ebt.minmax_apply(Xraw, p)
        cfg = ga_ebt.GAConfig(**(ga_budget or {}))
        cache = None
        if ga_cache_dir is not None:
            cache = (Path(ga_cache_dir) /
                     f"ga_search_{data.representation}_{data.task}_"
                     f"protocol{data.protocol}.json")
        t0 = time.perf_counter()
        report = ga_ebt.search(Xs[tr], data.y[tr], data.feature_cols, cfg=cfg,
                               seed=42, cache_path=cache)
        timings["ga_search_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model, cols = ga_ebt.fit_final(Xs[tr], data.y[tr], report,
                                       seed=data.model_seed)
        timings["train_seconds"] = time.perf_counter() - t0
        Xte = Xs[te][:, cols]
        extra["ga_ebt_search"] = {k: v for k, v in report.items() if k != "history"}
        extra["ga_ebt_fidelity"] = ga_ebt.FIDELITY
        notes = ("GA-EBT is an approximate reproduction of the base paper's "
                 "method; see config['ga_ebt_fidelity'].")
    else:
        Xtr, ytr = data.Z[tr], data.y[tr]
        if model_name == "XGBoost":
            model = xgboost_model.build(seed=data.model_seed,
                                        n_classes=data.n_classes)
            sw = xgboost_model.balanced_sample_weight(ytr)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr, sample_weight=sw)
            timings["train_seconds"] = time.perf_counter() - t0
        else:
            if model_name == "RandomForest":
                model = random_forest.build(seed=data.model_seed)
            elif model_name == "SVM":
                model = svm.build(seed=data.model_seed)
                Xtr, ytr, info = svm.subsample(
                    Xtr, ytr, max_train_samples or 100_000, seed=data.model_seed)
                extra["svm_subsample"] = info
                extra["svm_variant"] = "rbf_nys"
                notes = ("Kernel SVM does not scale to the full training split; "
                         "see config['svm_subsample'].")
            elif model_name == "KNN":
                model = knn.build(seed=data.model_seed)
                from baselines.svm import subsample as _sub
                Xtr, ytr, info = _sub(Xtr, ytr, max_train_samples or 100_000,
                                      seed=data.model_seed)
                extra["knn_subsample"] = info
                if info["subsampled"]:
                    notes = ("k-NN prediction cost is linear in the training set; "
                             "see config['knn_subsample'].")
            else:
                raise ValueError(f"unknown classical model {model_name!r}")
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            timings["train_seconds"] = time.perf_counter() - t0
        Xte = data.Z[te]

    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        y_prob = np.asarray(model.predict_proba(Xte), dtype=float)
        if y_prob.shape[1] != data.n_classes:
            full = np.zeros((len(Xte), data.n_classes))
            for j, c in enumerate(model.classes_):
                full[:, int(c)] = y_prob[:, j]
            y_prob = full
    else:
        pred = model.predict(Xte)
        y_prob = np.eye(data.n_classes)[np.asarray(pred, dtype=int)]
    timings["inference_seconds"] = time.perf_counter() - t0
    timings.update(measure_inference(lambda: model.predict(Xte[:min(len(Xte), 50_000)]),
                                     n_samples=min(len(Xte), 50_000)))

    n_params = None
    if hasattr(model, "estimators_"):
        try:
            n_params = int(sum(e.tree_.node_count for e in model.estimators_))
            extra["parameter_proxy"] = "total decision-tree nodes in the ensemble"
        except Exception:
            n_params = None

    return _finalise(data, model_name, y_prob, timings, experiment,
                     n_parameters=n_params, extra_config=extra, extra_notes=notes)


def run_neural(data: PreparedData, model_name: str, experiment: str,
               train_cfg: Optional[TrainConfig] = None,
               model_kwargs: Optional[Dict] = None,
               loss: str = "ce", weighting: str = "balanced",
               focal_gamma: float = 2.0,
               save_checkpoint: bool = True,
               collect_attention: bool = False,
               model_label: Optional[str] = None,
               extra_notes: str = "") -> RunRecord:
    import torch

    from src.models import TABULAR_NEURAL_MODELS

    set_seed(data.model_seed)
    cfg = train_cfg or TrainConfig()
    # Model selection must use the same macro-averaging convention as the
    # final evaluation, or early stopping optimises a different quantity.
    restrict = bool(data.split_description.get(
        "evaluate_on_classes_present_in_test_only", False))
    if restrict != cfg.restrict_macro_to_present:
        cfg = TrainConfig(**{**cfg.to_dict(),
                             "restrict_macro_to_present": restrict})
    kw = dict(model_kwargs or {})
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    loss_fn = build_loss(loss, data.y[tr], data.n_classes, weighting=weighting,
                         gamma=focal_gamma)

    extra: Dict[str, object] = {
        "train_config": cfg.to_dict(),
        "model_kwargs": kw,
        "loss": loss,
        "class_weighting": weighting,
    }

    if model_name in TABULAR_NEURAL_MODELS:
        model = TABULAR_NEURAL_MODELS[model_name](
            edge_dim=data.Z.shape[1], n_classes=data.n_classes, **kw)
        model, history = train_mlp(model, data.Z[tr], data.y[tr],
                                   data.Z[va], data.y[va], loss_fn, cfg,
                                   seed=data.model_seed)
        t0 = time.perf_counter()
        y_prob = predict_mlp(model, data.Z[te], cfg.batch_size, cfg.device)
        infer = time.perf_counter() - t0
        timings = {"train_seconds": history["train_seconds"],
                   "inference_seconds": infer}
        timings.update(measure_inference(
            lambda: predict_mlp(model, data.Z[te], cfg.batch_size, cfg.device),
            n_samples=len(te)))
    elif model_name in GRAPH_MODELS:
        model = GRAPH_MODELS[model_name](
            node_dim=data.graph.node_features.shape[1],
            edge_dim=data.graph.edge_attr.shape[1],
            n_classes=data.n_classes, **kw)
        extra["model_config"] = model.cfg.to_dict()
        model, history = train_graph_model(model, data.graph, data.splits,
                                           data.task, loss_fn, cfg,
                                           seed=data.model_seed,
                                           class_names=data.class_names)
        gt = GraphTensors.from_graph(data.graph, data.task, device=cfg.device)
        t0 = time.perf_counter()
        y_prob = predict_graph_model(model, gt, data.graph, te, tr, cfg,
                                     seed=data.model_seed)
        infer = time.perf_counter() - t0
        timings = {"train_seconds": history["train_seconds"],
                   "inference_seconds": infer}
        timings.update(measure_inference(
            lambda: predict_graph_model(model, gt, data.graph, te, tr, cfg,
                                        seed=data.model_seed),
            n_samples=len(te)))
    else:
        raise ValueError(f"unknown neural model {model_name!r}")

    extra["history"] = {k: v for k, v in history.items() if k != "lr"}
    rec = _finalise(data, model_name, y_prob, timings, experiment,
                    n_parameters=count_parameters(model), extra_config=extra,
                    model_label=model_label, extra_notes=extra_notes)

    if save_checkpoint:
        ck = RESULTS_CKPT / f"{rec.run_id()}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_kwargs": kw,
                    "run_id": rec.run_id()}, ck)
    return rec


# --------------------------------------------------------------------------- #
# Cell driver
# --------------------------------------------------------------------------- #


def run_configuration(
    representation: str,
    task: str,
    protocol: str,
    seed: int,
    models: Sequence[str] = ALL_MODELS,
    experiment: str = "main",
    train_cfg: Optional[TrainConfig] = None,
    model_kwargs: Optional[Dict[str, Dict]] = None,
    nrows: Optional[int] = None,
    ga_budget: Optional[Dict] = None,
    node_feature_source: str = "train",
    raw_dir: Optional[Path] = None,
    split_kwargs: Optional[Dict] = None,
) -> List[RunRecord]:
    ensure_dirs()
    set_seed(seed)
    data = PreparedData(representation, task, protocol, seed, nrows=nrows,
                        node_feature_source=node_feature_source,
                        raw_dir=raw_dir, split_kwargs=split_kwargs)
    if not data.applicable:
        save_json(data.split_description,
                  RESULTS_RAW / f"SKIPPED__{experiment}__{representation}__{task}"
                                f"__protocol{protocol}__seed{seed}.json")
        return []

    save_json(data.context(),
              DATA_METADATA / f"context__{experiment}__{representation}__{task}"
                              f"__protocol{protocol}__seed{seed}.json")

    records: List[RunRecord] = []
    mk = model_kwargs or {}
    for name in models:
        try:
            if name in CLASSICAL:
                rec = run_classical(data, name, experiment,
                                    ga_cache_dir=DATA_METADATA / "ga_search",
                                    ga_budget=ga_budget)
            else:
                rec = run_neural(data, name, experiment, train_cfg=train_cfg,
                                 model_kwargs=mk.get(name))
            records.append(rec)
        except Exception as exc:  # a failed model must not erase the others
            log.exception("model %s failed on %s/%s/protocol%s/seed%d: %s",
                          name, representation, task, protocol, seed, exc)
            save_json(
                {"model": name, "error": repr(exc), "representation": representation,
                 "task": task, "protocol": protocol, "seed": seed},
                RESULTS_LOGS / f"FAILED__{experiment}__{name}__{representation}"
                               f"__{task}__protocol{protocol}__seed{seed}.json")
    return records
