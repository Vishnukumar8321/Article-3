"""Baseline models.

The classical-ML baselines live here as self-contained modules.  The neural and
graph baselines are **not** re-implemented: they are the same code paths as the
proposed model, exposed from :mod:`src.models` with different flags, so that the
ablation compares mechanisms rather than codebases.  ``baselines/gcn.py`` and
friends are therefore thin re-exports, kept so the directory matches the project
specification and so a reader looking for "the GCN baseline" finds a file.
"""

from __future__ import annotations

from typing import Dict

from . import endpoint_lookup, ga_ebt, knn, random_forest, svm, xgboost_model

CLASSICAL_BASELINES: Dict[str, object] = {
    "EndpointLookup": endpoint_lookup,
    "RandomForest": random_forest,
    "XGBoost": xgboost_model,
    "SVM": svm,
    "KNN": knn,
    "GA-EBT": ga_ebt,
}

__all__ = ["CLASSICAL_BASELINES", "endpoint_lookup", "ga_ebt", "knn",
           "random_forest", "svm", "xgboost_model"]
