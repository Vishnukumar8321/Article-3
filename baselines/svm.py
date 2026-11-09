"""SVM baseline.

A kernel SVM is O(n^2)-O(n^3) in the number of training flows, which is not
tractable at ~350k training rows.  Two honest options are provided and the one
used is recorded with the run:

``linear``  -- LinearSVC (liblinear), full training set.
``rbf_nys`` -- Nystroem kernel approximation + linear classifier, which keeps
               the RBF geometry at linear cost.
``rbf_exact`` -- true SVC on a stratified subsample capped at
               ``max_train_samples``; the cap is reported in the results file
               and must be stated in the manuscript.

Never report this baseline without also reporting which variant produced it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def build(seed: int = 42, variant: str = "rbf_nys", n_components: int = 512,
          C: float = 1.0, gamma: str = "scale", max_iter: int = 5000, **kw):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC, LinearSVC

    if variant == "linear":
        base = LinearSVC(C=C, random_state=seed, max_iter=max_iter,
                         class_weight="balanced", dual="auto")
        return Pipeline([("clf", CalibratedClassifierCV(base, cv=3))])
    if variant == "rbf_nys":
        return Pipeline([
            ("nys", Nystroem(kernel="rbf", n_components=n_components,
                             random_state=seed)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=C, random_state=seed, max_iter=max_iter,
                          class_weight="balanced", dual="auto"), cv=3)),
        ])
    if variant == "rbf_exact":
        return Pipeline([("clf", SVC(C=C, gamma=gamma, probability=True,
                                     class_weight="balanced", random_state=seed))])
    raise ValueError(f"unknown SVM variant {variant!r}")


def subsample(X: np.ndarray, y: np.ndarray, max_train_samples: int,
              seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Stratified cap on the training set, reported with the run."""
    if len(y) <= max_train_samples:
        return X, y, {"subsampled": False, "n_train": int(len(y))}
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(X, y, train_size=max_train_samples,
                                    stratify=y, random_state=seed)
    return Xs, ys, {"subsampled": True, "n_train": int(len(ys)),
                    "n_train_available": int(len(y)),
                    "reason": "kernel SVM does not scale to the full split"}


def search_space() -> Dict[str, list]:
    return {"C": [0.1, 1.0, 10.0], "n_components": [256, 512, 1024]}
