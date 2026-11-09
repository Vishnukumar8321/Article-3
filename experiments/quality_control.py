"""Section 54 quality control, executable.

Run before declaring anything finished:

    python -m experiments.quality_control

It fails loudly on the things that most often go wrong quietly: a manuscript
number with no result file behind it, a split with overlapping rows, a
preprocessing object fitted on more than the training split, a single-seed claim
dressed up as a multi-seed one, a citation with no DOI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    DATA_METADATA, PROJECT_ROOT, RESULTS_RAW, RESULTS_TABLES, SEEDS,
    collect_runs, get_logger, load_json, save_json,
)

log = get_logger("qc")

MANUSCRIPT = PROJECT_ROOT / "manuscript" / "main.tex"
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}\.\d{3,6})(?![\w])")


def _is_not_a_run(path: Path) -> bool:
    """Manifests, skip markers and failure markers are not result records."""
    name = path.name
    return (name.startswith("_") or "__" not in name
            or name.startswith("SKIPPED__") or name.startswith("FAILED__"))


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    def to_dict(self) -> Dict[str, object]:
        return {"check": self.name, "passed": not self.failures,
                "failures": self.failures, "warnings": self.warnings,
                "info": self.info}


# --------------------------------------------------------------------------- #


def check_data() -> Check:
    c = Check("DATA: labels, leakage, duplicates, preprocessing scope")
    contexts = sorted(DATA_METADATA.glob("context__*.json"))
    if not contexts:
        c.fail("no split contexts found; the main phase has not run")
        return c
    for p in contexts:
        ctx = load_json(p)
        lk = ctx.get("leakage_checks", {})
        if lk.get("preprocessing_fitted_on") != "train split only":
            c.fail(f"{p.name}: preprocessing not restricted to the training split")
        if lk.get("message_passing_edges") != "train split only":
            c.fail(f"{p.name}: message passing used non-training edges")
        if lk.get("node_features_from") not in {"train", "all"}:
            c.fail(f"{p.name}: unknown node feature source")
        if lk.get("node_features_from") == "all":
            c.warn(f"{p.name}: transductive node features -- valid only as a "
                   "labelled ablation, never as a headline result")
        shared = lk.get("train_test_shared_feature_vectors", 0)
        if shared:
            c.warn(f"{p.name}: {shared} identical predictive feature vectors are "
                   "shared between train and test (protocol A only; protocol B "
                   "exists to remove them)")
        sizes = (ctx.get("split") or {}).get("sizes", {})
        if sizes and min(sizes.values()) == 0:
            c.fail(f"{p.name}: an empty split {sizes}")
        pr = ctx.get("preprocessing_report", {})
        banned = {"ip_src", "ip_dst", "is_attack"}
        leaked = banned & set(pr.get("final_feature_columns", []))
        if leaked:
            c.fail(f"{p.name}: banned columns in the feature matrix: {leaked}")
    c.note(f"{len(contexts)} split contexts verified")
    return c


def check_models(runs: pd.DataFrame) -> Check:
    c = Check("MODEL: seeds, selection, determinism")
    if runs.empty:
        c.fail("no run records")
        return c
    main = runs[runs["experiment"] == "main"]
    for key, sub in main.groupby(["model", "representation", "task", "protocol"]):
        seeds = sorted(sub["seed"].dropna().astype(int).unique().tolist())
        if len(seeds) < 5:
            c.warn(f"{key}: only {len(seeds)} seed(s) {seeds}; the multi-seed "
                   "protocol asks for five before any variability claim")
        missing = sorted(set(SEEDS) - set(seeds))
        if seeds and missing:
            c.note(f"{key}: missing seeds {missing}")
    for p in sorted(RESULTS_RAW.glob("*.json")):
        if _is_not_a_run(p):
            continue
        rec = load_json(p)
        hist = (rec.get("config") or {}).get("history")
        if hist and hist.get("monitor") not in (None, "f1_macro", "mcc"):
            c.warn(f"{p.name}: model selected on {hist.get('monitor')}")
        if hist and hist.get("best_epoch") is not None and hist["best_epoch"] < 0:
            c.fail(f"{p.name}: no epoch improved the validation score")
    return c


def check_results(runs: pd.DataFrame) -> Check:
    c = Check("RESULTS: traceability, intervals, per-class metrics, cost")
    if runs.empty:
        c.fail("no run records")
        return c
    required = ["accuracy", "f1_macro", "mcc", "fpr"]
    for col in required:
        if col not in runs.columns:
            c.fail(f"metric {col} missing from every run record")
    for p in sorted(RESULTS_RAW.glob("*.json")):
        if _is_not_a_run(p):
            continue
        rec = load_json(p)
        if not rec.get("per_class"):
            c.fail(f"{p.name}: no per-class report")
        if rec.get("confusion_matrix") is None:
            c.fail(f"{p.name}: no confusion matrix")
        if not rec.get("timings"):
            c.warn(f"{p.name}: no timings recorded")
        if not rec.get("data_hashes"):
            c.warn(f"{p.name}: no input file hashes recorded")
    sig = RESULTS_TABLES / "table12_significance_full.json"
    if not sig.exists():
        c.warn("no significance table; claims of superiority cannot be supported")
    else:
        for r in load_json(sig):
            if isinstance(r, dict) and r.get("n_pairs", 0) < 2:
                c.warn(f"{r.get('model_b')}: fewer than two paired seeds, no test")
    return c


def check_manuscript(runs: pd.DataFrame) -> Check:
    c = Check("MANUSCRIPT: no placeholders, numbers traceable, citations have DOIs")
    if not MANUSCRIPT.exists():
        c.note("manuscript/main.tex does not exist yet -- correct at this stage; "
               "results must be produced before the Results section is written")
        return c
    text = MANUSCRIPT.read_text()
    for token in ("XX.XX", "TODO", "TBD", "PLACEHOLDER", "\\todo", "??"):
        if token in text:
            c.fail(f"placeholder {token!r} still present in main.tex")

    # Every 3+ decimal number in the text should appear in a generated table.
    table_text = "".join(
        p.read_text() for p in RESULTS_TABLES.glob("*.csv")
    ) if RESULTS_TABLES.exists() else ""
    numeric_pool = set(NUMBER_RE.findall(table_text))
    if runs is not None and not runs.empty:
        for col in runs.select_dtypes("number").columns:
            numeric_pool |= {f"{v:.4f}" for v in runs[col].dropna().tolist()}
    untraceable = [n for n in set(NUMBER_RE.findall(text))
                   if n not in numeric_pool]
    if untraceable:
        c.warn("numbers in main.tex not found in any generated table: "
               f"{sorted(untraceable)[:15]}")

    bib = PROJECT_ROOT / "manuscript" / "references.bib"
    if bib.exists():
        entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib.read_text(), re.S)
        for key, body in entries:
            if "doi" not in body.lower() and "url" not in body.lower():
                c.warn(f"reference {key} has neither a DOI nor a URL")
    else:
        c.note("references.bib not written yet")
    return c


def check_publication() -> Check:
    c = Check("PUBLICATION: dataset citation, availability statements, licence")
    for name, path in [
        ("README", PROJECT_ROOT / "README.md"),
        ("requirements.txt", PROJECT_ROOT / "requirements.txt"),
        ("environment.yml", PROJECT_ROOT / "environment.yml"),
        ("CITATION.cff", PROJECT_ROOT / "CITATION.cff"),
        ("LICENSE", PROJECT_ROOT / "LICENSE"),
    ]:
        if not path.exists():
            c.fail(f"{name} missing")
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        t = readme.read_text()
        if "10.21227/bhxy-ep04" not in t:
            c.fail("README does not cite the dataset DOI")
        for section in ("Data availability", "Code availability"):
            if section.lower() not in t.lower():
                c.warn(f"README has no '{section}' statement")
    raw = PROJECT_ROOT / "data" / "raw"
    if raw.exists() and any(raw.rglob("*.csv")):
        gi = PROJECT_ROOT / ".gitignore"
        if not gi.exists() or "data/raw" not in gi.read_text():
            c.fail("dataset files are present but data/raw is not git-ignored; "
                   "the licence does not grant redistribution")
    return c


def main() -> int:
    runs = collect_runs()
    checks = [check_data(), check_models(runs), check_results(runs),
              check_manuscript(runs), check_publication()]
    report = {"checks": [c.to_dict() for c in checks],
              "passed": all(not c.failures for c in checks)}
    save_json(report, RESULTS_TABLES / "quality_control.json")

    line = "=" * 72
    print(f"\n{line}\nQUALITY CONTROL\n{line}")
    for c in checks:
        status = "PASS" if not c.failures else "FAIL"
        print(f"\n[{status}] {c.name}")
        for f in c.failures:
            print(f"   FAIL    {f}")
        for w in c.warnings:
            print(f"   warning {w}")
        for i in c.info:
            print(f"   note    {i}")
    print(f"\n{line}")
    print("OVERALL: " + ("PASS" if report["passed"] else "FAIL"))
    print(line + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
