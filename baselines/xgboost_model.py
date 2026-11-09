"""XGBoost baseline -- the strong tabular reference for RQ1."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


DEFAULT = dict(n_estimators=600, max_depth=8, learning_rate=0.1,
               subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
               tree_method="hist", n_jobs=-1)


def build(seed: int = 42, n_classes: int = 2, sample_weight_balanced: bool = True,
          **kw):
    from xgboost import XGBClassifier
    params = {**DEFAULT, **kw}
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    extra = {} if n_classes == 2 else {"num_class": n_classes}
    return XGBClassifier(random_state=seed, objective=objective,
                         eval_metric="mlogloss" if n_classes > 2 else "logloss",
                         **extra, **params)


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Per-sample weights: XGBoost has no class_weight argument."""
    counts = np.bincount(np.asarray(y))
    counts[counts == 0] = 1
    w = len(y) / (len(counts) * counts)
    return w[np.asarray(y)]


def search_space() -> Dict[str, list]:
    return {
        "n_estimators": [300, 600, 1000],
        "max_depth": [4, 8, 12],
        "learning_rate": [0.03, 0.1, 0.3],
        "subsample": [0.7, 0.9, 1.0],
    }
