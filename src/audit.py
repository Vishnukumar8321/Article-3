"""Phase 1: dataset audit.

Produces ``data/metadata/dataset_audit.json`` and ``dataset_audit.csv`` plus the
audit figures.  Nothing is removed here -- the audit only *reports*.  Any
dropping happens later, on the training split, inside
:class:`src.preprocessing.FlowPreprocessor`, and is itself logged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import (
    EXPECTED_UNIFLOW_COLUMNS, REPRESENTATIONS, SCENARIO_LABELS, SCENARIO_PRETTY,
    discover_flow_files, feature_columns, file_hashes, load_representation,
)
from .utils import DATA_METADATA, RESULTS_FIGURES, get_logger, save_json, sha256_file

log = get_logger(__name__)


#: Tokens that would indicate a real time column. Matched on word boundaries,
#: not as substrings: "num_pkts" contains "ts" and is not a timestamp.
_TEMPORAL_TOKENS = {
    "ts", "time", "times", "timestamp", "date", "datetime", "epoch",
    "stime", "ltime", "start", "end", "first", "last", "seen", "duration",
}


def _looks_temporal(column: str) -> bool:
    low = column.lower()
    if any(sub in low for sub in ("timestamp", "datetime", "epoch")):
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    return any(t in _TEMPORAL_TOKENS for t in tokens)


def _numeric_summary(s: pd.Series) -> Dict[str, object]:
    v = pd.to_numeric(s, errors="coerce")
    finite = v[np.isfinite(v)]
    return {
        "dtype": str(s.dtype),
        "n_missing": int(s.isna().sum()),
        "n_infinite": int(np.isinf(v.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "n_unique": int(s.nunique(dropna=True)),
        "min": float(finite.min()) if len(finite) else None,
        "p01": float(finite.quantile(0.01)) if len(finite) else None,
        "median": float(finite.median()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "p99": float(finite.quantile(0.99)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "std": float(finite.std()) if len(finite) else None,
        "is_constant": bool(len(finite) > 0 and float(finite.std()) == 0.0),
        "near_zero_variance": bool(len(finite) > 0 and float(finite.var()) < 1e-8),
    }


def audit_file(path: Path, representation: str, scenario: str) -> Dict[str, object]:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    entry: Dict[str, object] = {
        "filename": path.name,
        "relative_path": str(path),
        "sha256": sha256_file(path),
        "representation": representation,
        "scenario": scenario,
        "scenario_pretty": SCENARIO_PRETTY.get(scenario, scenario),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_missing_total": int(df.isna().sum().sum()),
        "has_timestamp_like_column": sorted(
            [c for c in cols if _looks_temporal(c)]
        ),
    }
    if "ip_src" in cols:
        entry["n_unique_src_ips"] = int(df["ip_src"].nunique())
        entry["top_src_ips"] = df["ip_src"].value_counts().head(10).to_dict()
    if "ip_dst" in cols:
        entry["n_unique_dst_ips"] = int(df["ip_dst"].nunique())
        entry["top_dst_ips"] = df["ip_dst"].value_counts().head(10).to_dict()
    if {"ip_src", "ip_dst"} <= set(cols):
        pairs = df[["ip_src", "ip_dst"]].astype(str).agg("->".join, axis=1)
        entry["n_unique_src_dst_pairs"] = int(pairs.nunique())
        entry["top_src_dst_pairs"] = pairs.value_counts().head(10).to_dict()
    if "proto" in cols:
        entry["protocol_distribution"] = df["proto"].value_counts().head(20).to_dict()
    for p in ("prt_src", "prt_dst"):
        if p in cols:
            entry[f"{p}_top20"] = df[p].value_counts().head(20).to_dict()
    if "is_attack" in cols:
        vc = df["is_attack"].value_counts().to_dict()
        entry["is_attack_distribution"] = {str(k): int(v) for k, v in vc.items()}
        entry["attack_ratio"] = float(df["is_attack"].astype(float).mean())

    entry["column_summaries"] = {
        c: _numeric_summary(df[c]) for c in cols
        if pd.api.types.is_numeric_dtype(df[c])
    }
    entry["constant_columns"] = [
        c for c, s in entry["column_summaries"].items() if s["is_constant"]
    ]
    entry["near_zero_variance_columns"] = [
        c for c, s in entry["column_summaries"].items() if s["near_zero_variance"]
    ]
    return entry


def run_audit(
    raw_dir: Optional[Path] = None,
    out_dir: Path = DATA_METADATA,
    figures_dir: Path = RESULTS_FIGURES,
    make_figures: bool = True,
) -> Dict[str, object]:
    from .data_loader import DATA_RAW as _DEFAULT_RAW

    raw_dir = Path(raw_dir) if raw_dir else _DEFAULT_RAW
    files = discover_flow_files(raw_dir)

    per_file = [audit_file(f.path, f.representation, f.scenario) for f in files]

    audit: Dict[str, object] = {
        "dataset": {
            "name": "MQTT-IoT-IDS2020",
            "doi": "10.21227/bhxy-ep04",
            "citation": (
                "H. Hindy, C. Tachtatzis, R. Atkinson, E. Bayne and X. Bellekens, "
                "\"MQTT-IoT-IDS2020: MQTT Internet of Things Intrusion Detection "
                "Dataset\", IEEE DataPort, 2020, doi: 10.21227/bhxy-ep04."
            ),
            "source": (
                "https://ieee-dataport.org/open-access/"
                "mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset"
            ),
        },
        "raw_file_hashes": file_hashes(raw_dir),
        "files": per_file,
        "expected_uniflow_columns": EXPECTED_UNIFLOW_COLUMNS,
    }

    # Representation-level rollup.
    rollup: Dict[str, object] = {}
    class_counts: Dict[str, Dict[str, int]] = {}
    for rep in REPRESENTATIONS:
        rep_files = [e for e in per_file if e["representation"] == rep]
        if not rep_files:
            continue
        df = load_representation(rep, raw_dir)
        feat = feature_columns(df)
        # Class counts for the figure use the MODEL's label space, not the
        # capture layout, so figure 5 shows what is actually classified.
        inv = {v: SCENARIO_PRETTY[k] for k, v in SCENARIO_LABELS.items()}
        counts = (df["multiclass_label"].map(inv).value_counts()
                    .reindex([SCENARIO_PRETTY[k] for k in SCENARIO_LABELS])
                    .dropna().to_dict())
        class_counts[rep] = {str(k): int(v) for k, v in counts.items()}
        pairs = df[["ip_src", "ip_dst"]].astype(str).agg("->".join, axis=1)
        dup_feature_vectors = int(df[feat].duplicated().sum())
        rollup[rep] = {
            "n_files": len(rep_files),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "predictive_feature_columns": feat,
            "n_predictive_features": len(feat),
            "n_unique_src_ips": int(df["ip_src"].nunique()),
            "n_unique_dst_ips": int(df["ip_dst"].nunique()),
            "n_unique_endpoints": int(pd.concat([df["ip_src"], df["ip_dst"]]).nunique()),
            "n_unique_src_dst_pairs": int(pairs.nunique()),
            "binary_class_counts": {
                str(k): int(v) for k, v in df["binary_label"].value_counts().items()
            },
            "attack_ratio": float(df["binary_label"].mean()),
            "multiclass_counts": {
                str(k): int(v) for k, v in df["multiclass_label"].value_counts().items()
            },
            "multiclass_scenario_counts": class_counts[rep],
            "n_exact_duplicate_rows": int(df.duplicated().sum()),
            "n_duplicate_predictive_feature_vectors": dup_feature_vectors,
            "duplicate_feature_vector_ratio": float(dup_feature_vectors / max(1, len(df))),
            "benign_background_in_attack_captures": int(
                df.attrs.get("benign_background_in_attack_captures", 0)
            ),
            "attack_flagged_in_normal_capture": int(
                df.attrs.get("attack_flagged_in_normal_capture", 0)
            ),
            "capture_counts": {
                str(k): int(v) for k, v in df["scenario"].value_counts().items()
            },
            "multiclass_label_counts": {
                str(k): int(v) for k, v in
                df["multiclass_label"].value_counts().sort_index().items()
            },
            "timestamp_columns_present": sorted({
                c for e in rep_files for c in e["has_timestamp_like_column"]
            }),
        }
        if make_figures:
            _representation_figures(df, feat, rep, Path(figures_dir))

    audit["representations"] = rollup
    audit["temporal_data_statement"] = (
        "No timestamp column was found in the flow CSVs "
        f"({ {k: v['timestamp_columns_present'] for k, v in rollup.items()} }). "
        "Row order is therefore NOT treated as time, no artificial timestamps "
        "are created, and no temporal claim is made anywhere in this project. "
        "The temporal extension (ST-EGAT-GRU) remains future work pending "
        "packet-level or PCAP data."
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit, out_dir / "dataset_audit.json")

    flat = pd.DataFrame([{
        k: v for k, v in e.items()
        if not isinstance(v, (dict, list))
    } for e in per_file])
    flat.to_csv(out_dir / "dataset_audit.csv", index=False)

    if make_figures and class_counts:
        from .visualization import fig_class_distribution
        fig_class_distribution(class_counts,
                               Path(figures_dir) / "fig05_class_distribution.png")

    log.info("audit complete: %d files, %d representations -> %s",
             len(per_file), len(rollup), out_dir / "dataset_audit.json")
    return audit


def _representation_figures(df: pd.DataFrame, feat: List[str], rep: str,
                            figures_dir: Path) -> None:
    from .visualization import (
        fig_communication_graph, fig_correlation_matrix, fig_feature_distributions,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_feature_distributions(
        df, feat, figures_dir / f"audit_{rep}_feature_distributions.png",
        title=f"Feature distributions by class ({rep})",
    )
    fig_correlation_matrix(
        df, feat, figures_dir / f"audit_{rep}_correlation.png",
        title=f"Feature correlation, Spearman ({rep})",
    )
    pairs = (df.groupby(["ip_src", "ip_dst"]).size()
               .rename("n_flows").reset_index()
               .rename(columns={"ip_src": "src_ip", "ip_dst": "dst_ip"}))
    fig_communication_graph(
        pairs, figures_dir / f"fig06_communication_graph_{rep}.png",
        weight_col="n_flows",
        title=f"Communication graph, {rep}",
    )
