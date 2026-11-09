"""Shared utilities: determinism, logging, timing, hashing, artefact persistence.

Every experiment in this project writes its artefacts through the helpers in this
module so that *no* number that appears in the manuscript can exist without a
corresponding file on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:  # torch is optional for the classical-ML-only code paths
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Absent classes in an endpoint-grouped test split raise UndefinedMetricWarning
# by design; the condition is recorded in each run record instead.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
try:
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"
RESULTS = PROJECT_ROOT / "results"
RESULTS_RAW = RESULTS / "raw"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_LOGS = RESULTS / "logs"
RESULTS_CKPT = RESULTS / "checkpoints"


def ensure_dirs() -> None:
    for p in (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_METADATA,
        RESULTS_RAW,
        RESULTS_TABLES,
        RESULTS_FIGURES,
        RESULTS_LOGS,
        RESULTS_CKPT,
    ):
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

SEEDS = [42, 123, 2024, 3407, 2026]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG this project touches.

    Note: full bitwise determinism on CPU is achievable for the operations used
    here; we still record the seed with every result so any run is reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_CONFIGURED = False


def get_logger(name: str, logfile: Optional[Path] = None) -> logging.Logger:
    global _LOG_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOG_CONFIGURED:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logging.getLogger().addHandler(sh)
        logging.getLogger().setLevel(logging.INFO)
        _LOG_CONFIGURED = True
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# Timing / resources
# --------------------------------------------------------------------------- #


@contextmanager
def timer(store: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0


def peak_memory_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 ** 2)
    except Exception:  # pragma: no cover
        return float("nan")


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------------------------------------------------------- #
# Hashing / provenance
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir(directory: Path, pattern: str = "*.csv") -> Dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(Path(directory).rglob(pattern))}


def environment_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    if torch is not None:
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------------- #
# Result persistence
# --------------------------------------------------------------------------- #


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=NumpyJSONEncoder)
    return path


def load_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


@dataclass
class RunRecord:
    """Everything needed to trace a single number in the manuscript back to a run."""

    experiment: str
    model: str
    representation: str          # uniflow | biflow
    task: str                    # binary | multiclass
    protocol: str                # A | B | C | D
    seed: int
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_class: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix: Optional[list] = None
    timings: Dict[str, float] = field(default_factory=dict)
    n_parameters: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    data_hashes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=environment_fingerprint)
    notes: str = ""

    def run_id(self) -> str:
        return (
            f"{self.experiment}__{self.model}__{self.representation}"
            f"__{self.task}__protocol{self.protocol}__seed{self.seed}"
        )

    def save(self, root: Path = RESULTS_RAW) -> Path:
        path = Path(root) / f"{self.run_id()}.json"
        return save_json(asdict(self), path)


def collect_runs(root: Path = RESULTS_RAW) -> "Any":
    """Load every run record into a tidy DataFrame (used to build all tables)."""
    import pandas as pd

    rows = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            rec = load_json(p)
        except Exception:
            continue
        if "metrics" not in rec:
            continue
        row = {
            "run_id": p.stem,
            "experiment": rec.get("experiment"),
            "model": rec.get("model"),
            "representation": rec.get("representation"),
            "task": rec.get("task"),
            "protocol": rec.get("protocol"),
            "seed": rec.get("seed"),
            "n_parameters": rec.get("n_parameters"),
            "peak_memory_mb": rec.get("peak_memory_mb"),
            "source_file": str(p),
        }
        row.update({f"{k}": v for k, v in (rec.get("metrics") or {}).items()})
        row.update({f"time_{k}": v for k, v in (rec.get("timings") or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.3g}{unit}"
        n /= 1000.0
    return f"{n:.3g}T"
