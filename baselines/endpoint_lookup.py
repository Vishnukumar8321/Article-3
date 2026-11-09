"""Endpoint-lookup baseline: the label from the communication pair, nothing else.

This baseline exists because it is the first thing a reviewer should ask for.
It uses **no traffic features at all** -- only a lookup table from the ordered
(source IP, destination IP) pair to the majority training label, falling back to
the majority class for a pair never seen in training.

On MQTT-IoT-IDS2020 it reaches 0.956 accuracy and 0.856 macro-F1 on a random
split, because 86 of the 106 ordered pairs carry exactly one class and those
pairs cover 85% of the flows.  Any graph model must be compared against this
number: a GNN that beats the tabular baselines but only matches this one has
learned endpoint identity, not traffic behaviour.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class EndpointLookup:
    """Majority-class lookup on the ordered endpoint pair.

    Implements the scikit-learn ``fit``/``predict``/``predict_proba`` surface so
    the experiment runner treats it like any other baseline.
    """

    def __init__(self, n_classes: int = 2, key: str = "pair") -> None:
        if key not in {"pair", "src", "dst"}:
            raise ValueError("key must be 'pair', 'src' or 'dst'")
        self.key = key
        self.n_classes = n_classes
        self.table_: Optional[Dict[str, int]] = None
        self.distribution_: Optional[Dict[str, np.ndarray]] = None
        self.fallback_: int = 0
        self.classes_ = np.arange(n_classes)

    # -- key construction ---------------------------------------------- #

    def _keys(self, df: pd.DataFrame) -> np.ndarray:
        src = df["ip_src"].astype(str)
        dst = df["ip_dst"].astype(str)
        if self.key == "pair":
            return (src + "->" + dst).to_numpy()
        return (src if self.key == "src" else dst).to_numpy()

    # -- sklearn-ish surface ------------------------------------------- #

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "EndpointLookup":
        keys = self._keys(df)
        y = np.asarray(y)
        frame = pd.DataFrame({"k": keys, "y": y})
        counts = frame.groupby(["k", "y"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=range(self.n_classes), fill_value=0)
        self.table_ = counts.values.argmax(axis=1)
        self.index_ = {k: i for i, k in enumerate(counts.index)}
        rows = counts.values.astype(np.float64)
        self.distribution_ = rows / np.clip(rows.sum(axis=1, keepdims=True), 1, None)
        self.fallback_ = int(np.bincount(y, minlength=self.n_classes).argmax())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("EndpointLookup used before fit")
        idx = pd.Series(self._keys(df)).map(self.index_)
        known = idx.notna().to_numpy()
        out = np.full(len(df), self.fallback_, dtype=np.int64)
        out[known] = self.table_[idx[known].astype(int).to_numpy()]
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.distribution_ is None:
            raise RuntimeError("EndpointLookup used before fit")
        idx = pd.Series(self._keys(df)).map(self.index_)
        known = idx.notna().to_numpy()
        out = np.zeros((len(df), self.n_classes), dtype=np.float64)
        out[:, self.fallback_] = 1.0
        out[known] = self.distribution_[idx[known].astype(int).to_numpy()]
        return out


def build(seed: int = 42, n_classes: int = 2, key: str = "pair", **_):
    return EndpointLookup(n_classes=n_classes, key=key)
