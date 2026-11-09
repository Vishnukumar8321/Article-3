"""Preprocessing with hard leakage guards.

Every transformation that has *parameters* (scaling, encoding, imputation) is
fitted on the training split alone and then applied to validation/test.  The
:class:`FlowPreprocessor` refuses to transform before it has been fitted and
records every decision it makes into a report that is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import NON_PREDICTIVE_COLUMNS
from .utils import get_logger

log = get_logger(__name__)


# Ports and protocol numbers are categorical identifiers whose *magnitude* is
# meaningless. They are still informative (port 22 vs 1883), so we keep them but
# treat them as categorical-ish: see `PORT_COLUMNS` handling below.
PORT_COLUMNS = ("prt_src", "prt_dst")
PROTO_COLUMNS = ("proto",)

# Well-known ports that carry semantic meaning for this dataset's scenarios.
_KNOWN_PORTS = (22, 80, 443, 1883, 8883, 53, 123, 21, 23, 25, 3306, 5683)


@dataclass
class PreprocessReport:
    representation: str
    n_rows_in: int = 0
    n_cols_in: int = 0
    dropped_constant: List[str] = field(default_factory=list)
    dropped_near_zero_variance: List[str] = field(default_factory=list)
    dropped_all_nan: List[str] = field(default_factory=list)
    non_numeric_coerced: Dict[str, int] = field(default_factory=dict)
    infinities_replaced: Dict[str, int] = field(default_factory=dict)
    nan_imputed: Dict[str, int] = field(default_factory=dict)
    engineered_columns: List[str] = field(default_factory=list)
    final_feature_columns: List[str] = field(default_factory=list)
    scaler: str = ""
    fitted_on_n_rows: int = 0
    notes: List[str] = field(default_factory=list)


class FlowPreprocessor:
    """Fit-on-train-only preprocessing for flow feature tables.

    Parameters
    ----------
    scaler:
        ``"quantile"`` (default) maps each feature to an approximately normal
        distribution and is robust to the extremely heavy-tailed inter-arrival
        and byte-count features in this dataset.  ``"standard"`` and
        ``"robust"`` are provided for the feature-subset ablation.
    drop_near_zero_variance:
        Whether to drop features whose training-set variance is below
        ``nzv_threshold``.  Dropping is decided on the *training* split only.
    engineer_port_features:
        Replace raw port magnitudes with (a) a log-scaled value, (b) a
        well-known-port indicator, (c) a registered/ephemeral bucket.
    """

    def __init__(
        self,
        scaler: str = "quantile",
        drop_constant: bool = True,
        drop_near_zero_variance: bool = True,
        nzv_threshold: float = 1e-8,
        engineer_port_features: bool = True,
        representation: str = "uniflow",
        random_state: int = 42,
    ) -> None:
        self.scaler_name = scaler
        self.drop_constant = drop_constant
        self.drop_near_zero_variance = drop_near_zero_variance
        self.nzv_threshold = nzv_threshold
        self.engineer_port_features = engineer_port_features
        self.random_state = random_state
        self.report = PreprocessReport(representation=representation)

        self._fitted = False
        self._feature_cols: List[str] = []
        self._scaler = None
        self._fill_values: Dict[str, float] = {}
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    # Feature engineering (deterministic, row-wise, no fitted parameters)
    # ------------------------------------------------------------------ #

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise derived features. Deterministic => safe before splitting."""
        out = df.copy()
        new_cols: List[str] = []

        if self.engineer_port_features:
            for col in PORT_COLUMNS:
                if col not in out.columns:
                    continue
                port = pd.to_numeric(out[col], errors="coerce").fillna(-1)
                out[f"{col}_log"] = np.log1p(port.clip(lower=0))
                out[f"{col}_is_wellknown"] = port.isin(_KNOWN_PORTS).astype(np.int8)
                out[f"{col}_is_system"] = ((port >= 0) & (port < 1024)).astype(np.int8)
                out[f"{col}_is_ephemeral"] = (port >= 49152).astype(np.int8)
                new_cols += [f"{col}_log", f"{col}_is_wellknown",
                             f"{col}_is_system", f"{col}_is_ephemeral"]

        # Ratio features that are well defined for both representations.
        if {"num_bytes", "num_pkts"} <= set(out.columns):
            pkts = pd.to_numeric(out["num_pkts"], errors="coerce")
            byts = pd.to_numeric(out["num_bytes"], errors="coerce")
            out["bytes_per_pkt"] = byts / pkts.replace(0, np.nan)
            new_cols.append("bytes_per_pkt")

        if {"max_iat", "mean_iat"} <= set(out.columns):
            mean_iat = pd.to_numeric(out["mean_iat"], errors="coerce")
            max_iat = pd.to_numeric(out["max_iat"], errors="coerce")
            out["iat_burstiness"] = max_iat / mean_iat.replace(0, np.nan)
            new_cols.append("iat_burstiness")

        if {"max_pkt_len", "min_pkt_len"} <= set(out.columns):
            out["pkt_len_range"] = (
                pd.to_numeric(out["max_pkt_len"], errors="coerce")
                - pd.to_numeric(out["min_pkt_len"], errors="coerce")
            )
            new_cols.append("pkt_len_range")

        self.report.engineered_columns = new_cols
        return out

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, feature_cols: Sequence[str]) -> "FlowPreprocessor":
        from sklearn.preprocessing import (
            QuantileTransformer, RobustScaler, StandardScaler,
        )

        leaked = [c for c in feature_cols if c.lower() in NON_PREDICTIVE_COLUMNS]
        if leaked:
            raise AssertionError(
                f"Refusing to fit on non-predictive columns: {leaked}. "
                "ip_src/ip_dst/is_attack must never enter the feature matrix."
            )

        self.report.n_rows_in = len(train_df)
        self.report.n_cols_in = len(feature_cols)

        eng = self._engineer(train_df[list(feature_cols)])
        cols = [c for c in eng.columns]

        # Coerce to numeric, recording how much had to be coerced.
        num = pd.DataFrame(index=eng.index)
        for c in cols:
            s = eng[c]
            if not pd.api.types.is_numeric_dtype(s):
                coerced = pd.to_numeric(s, errors="coerce")
                self.report.non_numeric_coerced[c] = int(
                    coerced.isna().sum() - s.isna().sum()
                )
                s = coerced
            num[c] = s.astype(np.float64)

        # Infinities -> NaN (recorded), then median imputation fitted on train.
        for c in cols:
            n_inf = int(np.isinf(num[c]).sum())
            if n_inf:
                self.report.infinities_replaced[c] = n_inf
                num[c] = num[c].replace([np.inf, -np.inf], np.nan)

        all_nan = [c for c in cols if num[c].isna().all()]
        if all_nan:
            self.report.dropped_all_nan = all_nan
            cols = [c for c in cols if c not in all_nan]
            num = num[cols]

        for c in cols:
            n_nan = int(num[c].isna().sum())
            if n_nan:
                self.report.nan_imputed[c] = n_nan
            self._fill_values[c] = float(np.nanmedian(num[c].values))
            num[c] = num[c].fillna(self._fill_values[c])

        # Winsorisation bounds from the *training* distribution only. This tames
        # the heavy tails without letting test values define the clip range.
        for c in cols:
            lo, hi = np.percentile(num[c].values, [0.05, 99.95])
            if hi <= lo:
                lo, hi = float(num[c].min()), float(num[c].max())
            self._clip_bounds[c] = (float(lo), float(hi))
            num[c] = num[c].clip(lo, hi)

        if self.drop_constant:
            const = [c for c in cols if float(num[c].std(ddof=0)) == 0.0]
            if const:
                self.report.dropped_constant = const
                cols = [c for c in cols if c not in const]
                num = num[cols]

        if self.drop_near_zero_variance:
            nzv = [c for c in cols if float(num[c].var(ddof=0)) < self.nzv_threshold]
            if nzv:
                self.report.dropped_near_zero_variance = nzv
                cols = [c for c in cols if c not in nzv]
                num = num[cols]

        if self.scaler_name == "quantile":
            self._scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, max(10, len(num))),
                subsample=200_000,
                random_state=self.random_state,
            )
        elif self.scaler_name == "standard":
            self._scaler = StandardScaler()
        elif self.scaler_name == "robust":
            self._scaler = RobustScaler()
        elif self.scaler_name == "none":
            self._scaler = None
        else:
            raise ValueError(f"unknown scaler {self.scaler_name!r}")

        if self._scaler is not None:
            self._scaler.fit(num.values)

        self._feature_cols = cols
        self.report.final_feature_columns = list(cols)
        self.report.scaler = self.scaler_name
        self.report.fitted_on_n_rows = len(train_df)
        self._fitted = True
        log.info(
            "preprocessor fitted on %d train rows -> %d features (scaler=%s)",
            len(train_df), len(cols), self.scaler_name,
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "FlowPreprocessor.transform called before fit. Fit on the "
                "training split only."
            )
        eng = self._engineer(df)
        num = pd.DataFrame(index=eng.index)
        for c in self._feature_cols:
            if c not in eng.columns:
                raise KeyError(f"column {c!r} missing at transform time")
            s = eng[c]
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors="coerce")
            s = s.astype(np.float64).replace([np.inf, -np.inf], np.nan)
            s = s.fillna(self._fill_values[c])
            lo, hi = self._clip_bounds[c]
            num[c] = s.clip(lo, hi)
        X = num.values
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, train_df: pd.DataFrame,
                      feature_cols: Sequence[str]) -> np.ndarray:
        return self.fit(train_df, feature_cols).transform(train_df)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_cols)


# --------------------------------------------------------------------------- #
# Leakage assertions used by the experiment runners
# --------------------------------------------------------------------------- #


def assert_no_index_overlap(*index_arrays: np.ndarray) -> None:
    seen: set = set()
    for arr in index_arrays:
        s = set(np.asarray(arr).tolist())
        overlap = seen & s
        if overlap:
            raise AssertionError(
                f"{len(overlap)} rows appear in more than one split "
                f"(e.g. {sorted(list(overlap))[:5]})"
            )
        seen |= s


def assert_no_duplicate_rows_across_splits(
    df: pd.DataFrame, splits: Dict[str, np.ndarray], feature_cols: Sequence[str],
) -> Dict[str, int]:
    """Report exact-duplicate feature vectors shared between train and test.

    Duplicate flow records genuinely occur in scan traffic, so this is reported
    rather than enforced -- but the number must appear in the manuscript's
    threats-to-validity section.
    """
    sub = df[list(feature_cols)]
    keys = pd.util.hash_pandas_object(sub, index=False).values
    out: Dict[str, int] = {}
    train_keys = set(keys[splits["train"]].tolist())
    for name in ("val", "test"):
        if name not in splits:
            continue
        other = set(keys[splits[name]].tolist())
        out[f"train_{name}_shared_feature_vectors"] = len(train_keys & other)
    return out
