"""Random Forest baseline."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


DEFAULT = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
               max_features="sqrt", n_jobs=-1, class_weight="balanced_subsample")


def build(seed: int = 42, **kw):
    from sklearn.ensemble import RandomForestClassifier
    params = {**DEFAULT, **kw}
    return RandomForestClassifier(random_state=seed, **params)


def search_space() -> Dict[str, list]:
    return {
        "n_estimators": [100, 300, 600],
        "max_depth": [None, 16, 32],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5],
    }
