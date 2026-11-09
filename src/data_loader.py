"""Discovery and loading of the MQTT-IoT-IDS2020 flow CSVs.

Design rules enforced here
--------------------------
1.  Nothing about the on-disk layout is assumed.  Files are *discovered*, their
    schema is *reported*, and a mismatch raises rather than being silently
    patched.
2.  The multi-class label uses the source capture to say **which** attack a flow
    belongs to and ``is_attack`` to say **whether** it is an attack at all.
    Neither source is used alone: every attack capture also contains the
    testbed's benign background traffic (41% of uni-flow rows), so labelling by
    capture alone would relabel 204,350 benign flows as attacks, while
    ``is_attack`` alone cannot name an attack class.  See
    :func:`load_representation` for the full rationale.
3.  ``is_attack`` and the IP columns never enter the predictive feature matrix.
    They are carried as metadata columns and stripped downstream.

Dataset
-------
H. Hindy, C. Tachtatzis, R. Atkinson, E. Bayne and X. Bellekens,
"MQTT-IoT-IDS2020: MQTT Internet of Things Intrusion Detection Dataset",
IEEE DataPort, 2020.  DOI: 10.21227/bhxy-ep04
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import DATA_RAW, get_logger, sha256_file

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Scenario vocabulary
# --------------------------------------------------------------------------- #

# Canonical multi-class label space (section 9 of the project specification).
SCENARIO_LABELS: Dict[str, int] = {
    "normal": 0,
    "mqtt_bruteforce": 1,
    "scan_A": 2,       # aggressive scan
    "scan_sU": 3,      # UDP scan
    "sparta": 4,       # Sparta SSH brute-force
}

SCENARIO_PRETTY: Dict[str, str] = {
    "normal": "Normal",
    "mqtt_bruteforce": "MQTT Brute-force",
    "scan_A": "Aggressive Scan",
    "scan_sU": "UDP Scan",
    "sparta": "Sparta SSH Brute-force",
}

# Filename fragments -> canonical scenario key.  Ordered: the first match wins,
# so the more specific patterns must come first.
_SCENARIO_PATTERNS: List[Tuple[str, str]] = [
    (r"mqtt[_\-]?brute", "mqtt_bruteforce"),
    (r"sparta", "sparta"),
    (r"scan[_\-]?su\b|scan[_\-]?su[_\.]|su[_\-]?scan|udp[_\-]?scan", "scan_sU"),
    (r"scan[_\-]?a\b|scan[_\-]?a[_\.]|aggress", "scan_A"),
    (r"normal|benign", "normal"),
]

REPRESENTATIONS = ("uniflow", "biflow")

# Columns that must never be used as predictive features.
NON_PREDICTIVE_COLUMNS = {"ip_src", "ip_dst", "is_attack", "label", "class"}

# Canonical uni-flow schema as documented by the dataset authors.  Used only for
# reporting agreement/disagreement, never to coerce the data.
EXPECTED_UNIFLOW_COLUMNS = [
    "ip_src", "ip_dst", "prt_src", "prt_dst", "proto", "num_pkts",
    "mean_iat", "std_iat", "min_iat", "max_iat", "mean_pkt_len", "num_bytes",
    "num_psh_flags", "num_rst_flags", "num_urg_flags",
    "std_pkt_len", "min_pkt_len", "max_pkt_len", "is_attack",
]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class FlowFile:
    path: Path
    representation: str
    scenario: str
    scenario_label: int

    @property
    def pretty(self) -> str:
        return SCENARIO_PRETTY[self.scenario]


def infer_scenario(name: str) -> Optional[str]:
    """Map a file name to a canonical scenario key, or None if undecidable."""
    low = name.lower()
    for pattern, key in _SCENARIO_PATTERNS:
        if re.search(pattern, low):
            return key
    return None


def infer_representation(path: Path) -> Optional[str]:
    """Decide uniflow vs biflow from the path, preferring directory evidence."""
    joined = "/".join(part.lower() for part in path.parts)
    if "biflow" in joined or "bidirectional" in joined:
        return "biflow"
    if "uniflow" in joined or "unidirectional" in joined:
        return "uniflow"
    return None


def extract_archives(raw_dir: Path = DATA_RAW) -> List[Path]:
    """Unpack any *.zip found under data/raw into a sibling directory.

    The extracted tree is left in place so the SHA256 of both the archive and
    each CSV can be recorded for the reproducibility section.
    """
    raw_dir = Path(raw_dir)
    extracted: List[Path] = []
    for zp in sorted(raw_dir.rglob("*.zip")):
        target = zp.with_suffix("")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(target, members=members)
        log.info("extracted %s -> %s (%d members, sha256=%s)",
                 zp.name, target.name, len(members), sha256_file(zp)[:16])
        extracted.append(target)
    return extracted


def discover_flow_files(raw_dir: Path = DATA_RAW,
                        representation: Optional[str] = None) -> List[FlowFile]:
    """Find every scenario CSV under data/raw and label it.

    Raises if a CSV cannot be assigned to a scenario or representation, because
    silently dropping a scenario would corrupt the multi-class label space.
    """
    raw_dir = Path(raw_dir)
    found: List[FlowFile] = []
    unassigned: List[str] = []

    # Unpack the released archives on first use, so the documented workflow
    # ("drop the two zips in data/raw") works without a manual step.
    if not any(raw_dir.rglob("*.csv")) and any(raw_dir.rglob("*.zip")):
        extract_archives(raw_dir)

    for csv_path in sorted(raw_dir.rglob("*.csv")):
        rep = infer_representation(csv_path)
        scen = infer_scenario(csv_path.name) or infer_scenario(csv_path.parent.name)
        if rep is None or scen is None:
            unassigned.append(str(csv_path.relative_to(raw_dir)))
            continue
        if representation is not None and rep != representation:
            continue
        found.append(
            FlowFile(path=csv_path, representation=rep, scenario=scen,
                     scenario_label=SCENARIO_LABELS[scen])
        )

    if unassigned:
        raise ValueError(
            "Could not assign a representation and/or scenario to the following "
            "files. Rename them or extend _SCENARIO_PATTERNS rather than letting "
            "them be dropped:\n  " + "\n  ".join(unassigned)
        )
    if not found:
        raise FileNotFoundError(
            f"No scenario CSVs found under {raw_dir}. Place uniflow_features.zip "
            f"and biflow_features.zip there (or their extracted contents) first."
        )
    return found


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_representation(representation: str,
                        raw_dir: Path = DATA_RAW,
                        nrows: Optional[int] = None) -> pd.DataFrame:
    """Load and concatenate every scenario file for one representation.

    Adds five provenance/label columns and nothing else:

    ``source_file``             -- file the row came from
    ``scenario``                -- canonical capture key (from the file name)
    ``capture_scenario_label``  -- 0..4 for the capture, kept for provenance
    ``binary_label``            -- the dataset's own ``is_attack`` column
    ``multiclass_label``        -- 0 for any benign flow, else the capture's
                                   attack class (see the note below)
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    files = discover_flow_files(raw_dir, representation=representation)
    frames = []
    for ff in files:
        df = pd.read_csv(ff.path, nrows=nrows, low_memory=False)
        df["source_file"] = ff.path.name
        df["scenario"] = ff.scenario
        df["capture_scenario_label"] = ff.scenario_label
        frames.append(df)
        log.info("loaded %-28s rows=%-8d cols=%-3d scenario=%s",
                 ff.path.name, len(df), df.shape[1] - 3, ff.scenario)

    # Column union check: a schema mismatch between scenario files is a real
    # problem, not something to paper over with an outer join.
    col_sets = [frozenset(f.columns) for f in frames]
    if len(set(col_sets)) != 1:
        ref = set(col_sets[0])
        diffs = {
            files[i].path.name: {
                "missing": sorted(ref - set(cs)),
                "extra": sorted(set(cs) - ref),
            }
            for i, cs in enumerate(col_sets)
            if set(cs) != ref
        }
        raise ValueError(f"Scenario files disagree on columns: {diffs}")

    data = pd.concat(frames, ignore_index=True)

    if "is_attack" not in data.columns:
        raise ValueError("Expected an 'is_attack' column; none found.")
    data["binary_label"] = data["is_attack"].astype(int)

    # --------------------------------------------------------------------- #
    # Multi-class label.
    #
    # Each attack capture also contains the testbed's ordinary background
    # traffic: uniflow_sparta.csv, for example, holds 154,175 flows with
    # is_attack = 0 alongside 28,232 attack flows. Labelling every row of that
    # file "Sparta SSH Brute-force" would therefore relabel 154,175 benign
    # flows as an attack and make the five-class task meaningless.
    #
    # The label is consequently built from BOTH sources, each doing the job it
    # is fit for:
    #   * ``is_attack`` decides whether a flow is an attack at all;
    #   * the source capture names WHICH attack it is.
    # The attack class is never inferred from ``is_attack`` alone, and the
    # capture identity is never used to override a benign flag.
    # ``capture_scenario_label`` is retained so the provenance of every row
    # stays inspectable.
    # --------------------------------------------------------------------- #
    normal = SCENARIO_LABELS["normal"]
    data["multiclass_label"] = np.where(
        data["binary_label"] == 1, data["capture_scenario_label"], normal
    ).astype(int)

    background = int(((data["capture_scenario_label"] != normal)
                      & (data["binary_label"] == 0)).sum())
    if background:
        log.info(
            "%d/%d rows (%.2f%%) are benign background traffic captured during "
            "an attack scenario; they are labelled Normal for the multi-class "
            "task, with the capture retained in capture_scenario_label.",
            background, len(data), 100 * background / len(data),
        )
    mislabelled_normal = int(((data["capture_scenario_label"] == normal)
                              & (data["binary_label"] == 1)).sum())
    if mislabelled_normal:
        log.warning(
            "%d rows in the normal capture are flagged is_attack=1; reported in "
            "the audit, not modified.", mislabelled_normal,
        )

    data.attrs["benign_background_in_attack_captures"] = background
    data.attrs["attack_flagged_in_normal_capture"] = mislabelled_normal
    data.attrs["representation"] = representation
    return data


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Predictive columns only: everything except labels, IPs and provenance."""
    banned = NON_PREDICTIVE_COLUMNS | {
        "source_file", "scenario", "multiclass_label", "binary_label",
        "capture_scenario_label",
    }
    cols = [c for c in df.columns if c not in banned]
    leaked = [c for c in cols if c.lower() in NON_PREDICTIVE_COLUMNS]
    if leaked:  # pragma: no cover - defensive
        raise AssertionError(f"Leakage guard tripped: {leaked}")
    return cols


def file_hashes(raw_dir: Path = DATA_RAW) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in {".csv", ".zip"}:
            out[str(p.relative_to(raw_dir))] = sha256_file(p)
    return out
