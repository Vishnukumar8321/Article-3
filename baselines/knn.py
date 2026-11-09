"""k-NN baseline.

Prediction cost is linear in the training-set size per query, so the training
set is capped by the runner exactly as for the SVM, and the cap is recorded.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

DEFAULT = dict(n_neighbors=5, weights="distance", algorithm="auto", n_jobs=-1)


def build(seed: int = 42, **kw):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**{**DEFAULT, **kw})


def search_space() -> Dict[str, list]:
    return {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]}
