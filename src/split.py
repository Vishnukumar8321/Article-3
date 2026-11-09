"""Evaluation protocols A-D.

Four protocols, because a single random split on this dataset flatters every
model.  Each returns ``{"train": idx, "val": idx, "test": idx}`` of positional
row indices and a JSON-serialisable description that is stored with the run.

A -- Random stratified 70/15/15.  Comparable with the bulk of the literature,
     and the protocol under which the base paper's near-perfect numbers were
     obtained.  Its weakness is that identical or near-identical flow records
     (very common in scan traffic) straddle the split.

B -- Duplicate-aware grouped split.  Rows whose *exact* predictive feature
     vector coincides are treated as one group and assigned to a single split,
     which removes the trivial-memorisation route that inflates Protocol A.

C -- Endpoint-grouped generalisation.  Whole endpoints are held out, so the
     test set contains devices the model has never seen.  The default holds out
     *victims* (``ip_dst``), which is the well-posed direction here; ``CS``
     holds out *sources* and is reported for completeness, since every attack
     class on this testbed originates from one or two fixed addresses and a
     source holdout therefore cannot put unseen sources and attack traffic in
     the same test set.  A draw that leaves a class in test but not in train, or
     that puts most of the data in test, is rejected; if no draw qualifies the
     protocol reports itself not applicable rather than degrading silently.

D -- Cross-representation: handled by the runners, not here, since uni-flow and
     bi-flow are different feature spaces and are never merged.

L -- Leave-one-endpoint-out. Protocol C draws a random set of endpoints to
     withhold, and on a 24-endpoint testbed the composition of that draw
     dominates the result: the same model can score 0.48 or 0.99 depending on
     which victims land in the test split. Averaging five random draws does not
     fix this, it only reports a noisy mean. Protocol L instead enumerates the
     folds -- each candidate endpoint is withheld exactly once, with the model
     seed held fixed -- so that variation across folds is attributable to
     holdout composition alone, and so that a paired test over models has as
     many matched observations as there are endpoints rather than as many as
     there are seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger(__name__)

SplitDict = Dict[str, np.ndarray]


@dataclass
class SplitResult:
    splits: SplitDict
    description: Dict[str, object]

    def sizes(self) -> Dict[str, int]:
        return {k: int(len(v)) for k, v in self.splits.items()}


def _class_balance(y: np.ndarray, idx: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(y[idx], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def _report(splits: SplitDict, y: np.ndarray, extra: Dict[str, object]) -> Dict[str, object]:
    desc = dict(extra)
    desc["sizes"] = {k: int(len(v)) for k, v in splits.items()}
    desc["class_balance"] = {k: _class_balance(y, v) for k, v in splits.items()}
    return desc


# --------------------------------------------------------------------------- #
# Protocol A
# --------------------------------------------------------------------------- #


def protocol_a_random_stratified(
    y: np.ndarray,
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitResult:
    from sklearn.model_selection import train_test_split

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")

    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(
        idx, train_size=train_size, stratify=y, random_state=seed, shuffle=True,
    )
    rel_val = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=rel_val, stratify=y[rest_idx],
        random_state=seed, shuffle=True,
    )
    splits = {"train": np.sort(train_idx), "val": np.sort(val_idx),
              "test": np.sort(test_idx)}
    return SplitResult(splits, _report(splits, y, {
        "protocol": "A",
        "name": "random stratified",
        "fractions": [train_size, val_size, test_size],
        "seed": seed,
        "stratified_on": "task label",
    }))


# --------------------------------------------------------------------------- #
# Protocol B
# --------------------------------------------------------------------------- #


def protocol_b_duplicate_aware(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitResult:
    """Group exact-duplicate feature vectors, then split groups stratified.

    Groups are assigned by the group's majority label so stratification remains
    meaningful; a group is never divided between splits.
    """
    from sklearn.model_selection import train_test_split

    keys = pd.util.hash_pandas_object(df[list(feature_cols)], index=False).to_numpy()
    uniq_keys, group_of_row = np.unique(keys, return_inverse=True)
    n_groups = len(uniq_keys)

    # Majority label per group, for stratification.
    order = np.argsort(group_of_row, kind="stable")
    grp_sorted = group_of_row[order]
    y_sorted = y[order]
    boundaries = np.flatnonzero(np.diff(grp_sorted)) + 1
    group_label = np.empty(n_groups, dtype=np.int64)
    for g, chunk in zip(np.unique(grp_sorted), np.split(y_sorted, boundaries)):
        vals, counts = np.unique(chunk, return_counts=True)
        group_label[g] = int(vals[np.argmax(counts)])

    gidx = np.arange(n_groups)
    # Stratification needs >=2 members per class among groups.
    strat = group_label if np.min(np.bincount(group_label)) >= 2 else None
    g_train, g_rest = train_test_split(
        gidx, train_size=train_size, stratify=strat, random_state=seed,
    )
    strat_rest = group_label[g_rest] if strat is not None else None
    if strat_rest is not None and np.min(np.bincount(strat_rest,
                                                     minlength=group_label.max() + 1)[
            np.unique(strat_rest)]) < 2:
        strat_rest = None
    g_val, g_test = train_test_split(
        g_rest, train_size=val_size / (val_size + test_size),
        stratify=strat_rest, random_state=seed,
    )

    assign = np.empty(n_groups, dtype=np.int8)
    assign[g_train], assign[g_val], assign[g_test] = 0, 1, 2
    row_assign = assign[group_of_row]
    splits = {
        "train": np.flatnonzero(row_assign == 0),
        "val": np.flatnonzero(row_assign == 1),
        "test": np.flatnonzero(row_assign == 2),
    }
    n_dup_rows = int(len(df) - n_groups)
    log.info("protocol B: %d rows -> %d duplicate-groups (%d redundant rows)",
             len(df), n_groups, n_dup_rows)
    return SplitResult(splits, _report(splits, y, {
        "protocol": "B",
        "name": "duplicate-aware grouped",
        "n_groups": int(n_groups),
        "n_redundant_rows": n_dup_rows,
        "grouping_key": "exact predictive feature vector",
        "seed": seed,
    }))


# --------------------------------------------------------------------------- #
# Protocol C
# --------------------------------------------------------------------------- #


def protocol_c_endpoint_grouped(
    df: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    group_col: str = "ip_dst",
    test_fraction: float = 0.25,
    val_fraction: float = 0.15,
    min_classes_in_test: int = 3,
    max_test_row_fraction: float = 0.35,
    exclude_groups: Optional[Sequence[str]] = None,
    max_draws: int = 500,
) -> SplitResult:
    """Hold out whole endpoints, so the test set contains devices never seen.

    ``group_col="ip_dst"`` holds out **victim** endpoints; ``"ip_src"`` holds out
    **source** endpoints.

    On MQTT-IoT-IDS2020 these two are not equally well posed, and the asymmetry
    is a property of the testbed rather than of the protocol:

    * every attack class originates from one or two fixed addresses
      (192.168.2.5 and 10.0.0.5), so a source-grouped holdout puts either all of
      an attack class in train or all of it in test -- it cannot produce a test
      set containing unseen sources *and* attack traffic;
    * victims are better spread -- Aggressive Scan reaches 18 destinations and
      UDP Scan 18 -- so a destination-grouped holdout is well posed for those,
      while MQTT Brute-force (4 victims) and Sparta SSH (2 victims) cannot be
      held out this way and remain train-only.

    A draw is accepted only when the test split has at least
    ``min_classes_in_test`` classes, holds no more than
    ``max_test_row_fraction`` of the rows, and contains no class absent from
    train.  When no draw qualifies the protocol reports itself **not
    applicable** with the reason, rather than returning a degraded split.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)

    # Endpoints that only ever carry one class make poor holdouts: removing them
    # teaches the model nothing and can empty a class from train.
    per_group_classes = {
        g: set(np.unique(y[groups == g]).tolist()) for g in uniq
    }
    candidates = np.array([g for g in uniq
                           if len(per_group_classes[g]) >= 2
                           and (exclude_groups is None or g not in exclude_groups)])
    if len(candidates) < 4:
        return SplitResult({}, {
            "protocol": "C", "group_col": group_col, "applicable": False,
            "reason": (f"only {len(candidates)} {group_col} values carry more than "
                       "one class; an endpoint holdout is not well posed"),
            "n_groups_total": int(len(uniq)), "seed": seed,
        })

    n_test_groups = max(1, int(round(test_fraction * len(candidates))))
    n_val_groups = max(1, int(round(val_fraction * len(candidates))))
    all_classes = set(np.unique(y).tolist())
    rejections: Dict[str, int] = {}

    for attempt in range(max_draws):
        perm = rng.permutation(candidates)
        test_g = set(perm[:n_test_groups].tolist())
        val_g = set(perm[n_test_groups:n_test_groups + n_val_groups].tolist())
        test_idx = np.flatnonzero(np.isin(groups, list(test_g)))
        val_idx = np.flatnonzero(np.isin(groups, list(val_g)))
        train_idx = np.flatnonzero(~np.isin(groups, list(test_g | val_g)))
        if not (len(test_idx) and len(val_idx) and len(train_idx)):
            rejections["empty split"] = rejections.get("empty split", 0) + 1
            continue
        if len(test_idx) / len(y) > max_test_row_fraction:
            rejections["test too large"] = rejections.get("test too large", 0) + 1
            continue
        test_classes = set(np.unique(y[test_idx]).tolist())
        train_classes = set(np.unique(y[train_idx]).tolist())
        if len(test_classes) < min_classes_in_test:
            rejections["too few test classes"] = \
                rejections.get("too few test classes", 0) + 1
            continue
        if not test_classes <= train_classes:
            rejections["test class absent from train"] = \
                rejections.get("test class absent from train", 0) + 1
            continue

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        return SplitResult(splits, _report(splits, y, {
            "protocol": "C",
            "name": f"endpoint-grouped generalisation ({group_col})",
            "applicable": True,
            "group_col": group_col,
            "n_groups_total": int(len(uniq)),
            "n_candidate_groups": int(len(candidates)),
            "test_groups": sorted(test_g),
            "val_groups": sorted(val_g),
            "classes_in_test": sorted(test_classes),
            "classes_train_only": sorted(all_classes - test_classes),
            "evaluate_on_classes_present_in_test_only": True,
            "test_row_fraction": float(len(test_idx) / len(y)),
            "draw_attempts": attempt + 1,
            "seed": seed,
        }))

    return SplitResult({}, {
        "protocol": "C",
        "name": f"endpoint-grouped generalisation ({group_col})",
        "applicable": False,
        "reason": (f"no endpoint holdout satisfied the validity constraints in "
                   f"{max_draws} draws"),
        "rejections": rejections,
        "n_groups_total": int(len(uniq)),
        "n_candidate_groups": int(len(candidates)),
        "seed": seed,
    })


#: Backwards-compatible alias.
protocol_c_source_grouped = protocol_c_endpoint_grouped


# --------------------------------------------------------------------------- #
# Protocol L
# --------------------------------------------------------------------------- #


def candidate_holdout_endpoints(df: pd.DataFrame, y: np.ndarray,
                                group_col: str = "ip_dst",
                                min_classes: int = 2) -> List[str]:
    """Endpoints eligible to be withheld: those carrying more than one class.

    A single-class endpoint makes a degenerate test split and teaches nothing
    by its removal, so it stays in training for every fold.
    """
    groups = df[group_col].astype(str).to_numpy()
    out = []
    for g in np.unique(groups):
        if len(np.unique(y[groups == g])) >= min_classes:
            out.append(str(g))
    return sorted(out)


def protocol_l_leave_one_endpoint_out(
    df: pd.DataFrame,
    y: np.ndarray,
    fold: int,
    group_col: str = "ip_dst",
    val_fraction: float = 0.15,
    seed: int = 42,
    min_classes_in_test: int = 2,
) -> SplitResult:
    """Withhold endpoint ``fold`` of the candidate list; validate on another.

    The validation split is drawn from the *remaining* endpoints, also at
    endpoint granularity, so that model selection never sees the held-out
    device either.
    """
    candidates = candidate_holdout_endpoints(df, y, group_col)
    if not candidates:
        return SplitResult({}, {"protocol": "L", "applicable": False,
                                "reason": f"no {group_col} carries >1 class"})
    if fold >= len(candidates):
        return SplitResult({}, {
            "protocol": "L", "applicable": False,
            "reason": f"fold {fold} out of range; {len(candidates)} folds exist",
            "n_folds": len(candidates)})

    groups = df[group_col].astype(str).to_numpy()
    test_group = candidates[fold]
    rest = [g for g in candidates if g != test_group]
    rng = np.random.default_rng(seed + 1000 * fold)
    n_val = max(1, int(round(val_fraction * len(rest))))
    val_groups = set(rng.choice(rest, size=min(n_val, len(rest)),
                                replace=False).tolist())

    test_idx = np.flatnonzero(groups == test_group)
    val_idx = np.flatnonzero(np.isin(groups, list(val_groups)))
    train_idx = np.flatnonzero((groups != test_group)
                               & ~np.isin(groups, list(val_groups)))

    test_classes = set(np.unique(y[test_idx]).tolist()) if len(test_idx) else set()
    train_classes = set(np.unique(y[train_idx]).tolist()) if len(train_idx) else set()
    if (not len(test_idx) or not len(val_idx) or not len(train_idx)
            or len(test_classes) < min_classes_in_test
            or not test_classes <= train_classes):
        return SplitResult({}, {
            "protocol": "L", "applicable": False,
            "fold": fold, "held_out_endpoint": test_group,
            "reason": ("fold rejected: test classes "
                       f"{sorted(test_classes)} vs train {sorted(train_classes)}, "
                       f"sizes {len(train_idx)}/{len(val_idx)}/{len(test_idx)}"),
            "n_folds": len(candidates)})

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return SplitResult(splits, _report(splits, y, {
        "protocol": "L",
        "name": f"leave-one-endpoint-out ({group_col})",
        "applicable": True,
        "group_col": group_col,
        "fold": fold,
        "n_folds": len(candidates),
        "held_out_endpoint": test_group,
        "val_endpoints": sorted(val_groups),
        "candidate_endpoints": candidates,
        "classes_in_test": sorted(test_classes),
        "classes_train_only": sorted(set(np.unique(y).tolist()) - test_classes),
        "evaluate_on_classes_present_in_test_only": True,
        "test_row_fraction": float(len(test_idx) / len(y)),
        "seed": seed,
    }))


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def make_split(
    protocol: str,
    df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    seed: int = 42,
    **kwargs,
) -> SplitResult:
    protocol = protocol.upper()
    if protocol == "A":
        return protocol_a_random_stratified(y, seed=seed, **kwargs)
    if protocol == "B":
        return protocol_b_duplicate_aware(df, feature_cols, y, seed=seed, **kwargs)
    if protocol == "C":
        # Victim-endpoint holdout: the well-posed direction on this testbed.
        kwargs.setdefault("group_col", "ip_dst")
        return protocol_c_endpoint_grouped(df, y, seed=seed, **kwargs)
    if protocol == "L":
        fold = kwargs.pop("fold", None)
        if fold is None:
            raise ValueError("protocol L requires a fold index")
        kwargs.setdefault("group_col", "ip_dst")
        return protocol_l_leave_one_endpoint_out(df, y, fold=fold, seed=seed,
                                                 **kwargs)
    if protocol == "CS":
        # Source-endpoint holdout, reported for completeness. Expected to be
        # ill-posed for the multi-class task; the returned description says why.
        kwargs["group_col"] = "ip_src"
        return protocol_c_endpoint_grouped(df, y, seed=seed, **kwargs)
    raise ValueError(
        f"protocol {protocol!r} is not produced here "
        "(D is a runner-level cross-representation study)"
    )


def verify_split(splits: SplitDict, n_rows: int) -> None:
    """Fail loudly on any overlap or coverage error."""
    if not splits:
        return
    seen = np.zeros(n_rows, dtype=np.int8)
    for name, idx in splits.items():
        idx = np.asarray(idx)
        if idx.size and (idx.min() < 0 or idx.max() >= n_rows):
            raise AssertionError(f"split {name} has out-of-range indices")
        seen[idx] += 1
    if (seen > 1).any():
        raise AssertionError(
            f"{int((seen > 1).sum())} rows appear in more than one split"
        )
    log.info("split verified: %s, %d/%d rows used",
             {k: int(len(v)) for k, v in splits.items()}, int((seen == 1).sum()), n_rows)
