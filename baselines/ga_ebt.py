"""GA-EBT: reproduction of the base paper's Genetic-Algorithm + Ensemble
Bagging Trees detector.

Source
------
M. Norouzi and Z. Gurkas-Aydin, "A reliable cyber-attack detection architecture
for cyber-physical systems in SDN-enabled internet of vehicles", *Cluster
Computing*, vol. 29, no. 110, 2026, doi: 10.1007/s10586-025-05929-2.
Algorithm 1 of that paper specifies the GA; Section 3.5 and Algorithm 2 specify
the EBT.

Fidelity statement -- this is an **approximate reproduction**, not an exact
replication.  ``FIDELITY`` below enumerates, item by item, what is reproduced
exactly, what is modified and what could not be reproduced.  The manuscript must
carry this statement verbatim and must never present these numbers as the
original authors' results.

The chromosome, the operators, the bounds and the stopping rule follow the
published pseudocode literally:

    chromosome = ( feature bit-mask F[1..d] , H = {n_estimators, max_depth,
                   min_samples_split} )
    P = 40, G_max = 100, p_c = 0.80, p_m = 0.05, tournament = 3,
    no_improve_patience = 10, elitism keep = 2, min selected features = 10
    n_estimators in [50, 300], max_depth in [3, 20], min_samples_split in [2, 10]
    mutation jitter: n_estimators +-20, max_depth +-2, min_samples_split +-1
    fitness = mean k-fold CV F1 on the training split
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Published hyperparameters (Algorithm 1)
# --------------------------------------------------------------------------- #

PAPER_GA = dict(
    population_size=40,
    max_generations=100,
    crossover_rate=0.80,
    mutation_rate=0.05,
    tournament_size=3,
    no_improve_patience=10,
    elitism_keep=2,
    min_selected_features=10,
    n_estimators_bounds=(50, 300),
    max_depth_bounds=(3, 20),
    min_samples_split_bounds=(2, 10),
    n_estimators_jitter=20,
    max_depth_jitter=2,
    min_samples_split_jitter=1,
    fitness_cv_folds=10,
    fitness_metric="f1",
)

FIDELITY: Dict[str, object] = {
    "classification": "approximate reproduction",
    "reproduced_exactly": [
        "chromosome encoding (feature bit-mask + EBT hyperparameters)",
        "population size 40, max generations 100, crossover 0.80, mutation 0.05",
        "tournament selection with tournament size 3",
        "uniform crossover on the feature mask, parameter-wise crossover on H",
        "bit-flip mutation on F and bounded jitter mutation on H",
        "elitism keeping the best 2 individuals",
        "minimum of 10 selected features enforced after mutation",
        "hyperparameter bounds n_estimators [50,300], max_depth [3,20], "
        "min_samples_split [2,10]",
        "stopping on max generations or 10 generations without fitness gain",
        "bagging ensemble of decision trees with majority-vote aggregation",
    ],
    "modified": [
        "scikit-learn BaggingClassifier(DecisionTreeClassifier) is used; the "
        "original was run in WEKA, whose default tree induction (REPTree/J48) "
        "differs in pruning and split criteria",
        "GA fitness is evaluated on a stratified subsample of the training "
        "split and with a reduced number of CV folds, because a 40x100x10 "
        "grid of bagged-ensemble fits over the full ~350k-row training split "
        "is computationally infeasible here; the exact budget used is recorded "
        "in the run's result file",
        "the GA search is executed once per (representation, task) using the "
        "search seed, and the resulting feature subset and hyperparameters are "
        "then refitted per evaluation seed; the original paper does not state "
        "a multi-seed protocol",
        "min-max scaling to [0,1] as described in the original is applied "
        "inside this baseline rather than the project's default quantile "
        "scaler, so that the baseline matches its own published preprocessing",
    ],
    "not_reproducible": [
        "the exact training/test partition, which the original does not "
        "publish",
        "the exact 33,079-instance subset of MQTT-IoT-IDS2020 used in the "
        "original (the released uni-flow feature set contains substantially "
        "more records); the sampling procedure is not described",
        "the WEKA random seed and internal implementation details",
        "the original SMOTE configuration beyond 'applied within each "
        "training fold'",
    ],
    "consequence": (
        "Numbers produced by this module are a good-faith reimplementation "
        "under this project's protocol. They are reported as 'GA-EBT "
        "(reproduced)' and are never substituted for, or presented as, the "
        "99.997% figure reported in the original paper. The original figure is "
        "quoted only as a literature value with its own protocol attached."
    ),
}


# --------------------------------------------------------------------------- #
# Chromosome
# --------------------------------------------------------------------------- #


@dataclass
class Chromosome:
    mask: np.ndarray                 # bool [d]
    n_estimators: int
    max_depth: int
    min_samples_split: int
    fitness: float = -np.inf

    def key(self) -> str:
        return (f"{''.join('1' if b else '0' for b in self.mask)}"
                f"|{self.n_estimators}|{self.max_depth}|{self.min_samples_split}")

    def selected(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    def to_dict(self, feature_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
        out = {
            "n_selected_features": int(self.mask.sum()),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "min_samples_split": int(self.min_samples_split),
            "fitness": float(self.fitness),
        }
        if feature_names is not None:
            out["selected_features"] = [feature_names[i] for i in self.selected()]
        return out


@dataclass
class GAConfig:
    population_size: int = PAPER_GA["population_size"]
    max_generations: int = PAPER_GA["max_generations"]
    crossover_rate: float = PAPER_GA["crossover_rate"]
    mutation_rate: float = PAPER_GA["mutation_rate"]
    tournament_size: int = PAPER_GA["tournament_size"]
    no_improve_patience: int = PAPER_GA["no_improve_patience"]
    elitism_keep: int = PAPER_GA["elitism_keep"]
    min_selected_features: int = PAPER_GA["min_selected_features"]
    fitness_cv_folds: int = 3
    fitness_subsample: int = 20_000
    fitness_metric: str = "f1_macro"
    use_smote: bool = True
    max_fitness_evaluations: Optional[int] = None
    budget_note: str = (
        "Reduced from the published 10-fold CV over the full training split; "
        "see FIDELITY['modified']."
    )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# EBT
# --------------------------------------------------------------------------- #


def build_ebt(n_estimators: int, max_depth: int, min_samples_split: int,
              seed: int = 42, n_jobs: int = -1):
    """Ensemble Bagging Trees: bagged decision trees, majority vote."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    base = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=seed,
    )
    return BaggingClassifier(
        estimator=base,
        n_estimators=int(n_estimators),
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    from sklearn.metrics import f1_score

    if metric == "f1":
        return float(f1_score(y_true, y_pred,
                              average="binary" if len(np.unique(y_true)) == 2 else "macro",
                              zero_division=0))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"unknown fitness metric {metric!r}")


# --------------------------------------------------------------------------- #
# Genetic algorithm
# --------------------------------------------------------------------------- #


class GAEBTSearch:
    """Algorithm 1 of the base paper, run on the **training split only**."""

    def __init__(self, cfg: Optional[GAConfig] = None, seed: int = 42,
                 verbose: bool = True) -> None:
        self.cfg = cfg or GAConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.history: List[Dict[str, object]] = []
        self._cache: Dict[str, float] = {}
        self.n_evaluations = 0
        self.best: Optional[Chromosome] = None

    # -- population ---------------------------------------------------- #

    def _random_chromosome(self, d: int) -> Chromosome:
        p = self.rng.uniform(0.3, 0.9)
        mask = self.rng.random(d) < p
        mask = self._repair(mask, d)
        return Chromosome(
            mask=mask,
            n_estimators=int(self.rng.integers(*PAPER_GA["n_estimators_bounds"],
                                               endpoint=True)),
            max_depth=int(self.rng.integers(*PAPER_GA["max_depth_bounds"],
                                            endpoint=True)),
            min_samples_split=int(self.rng.integers(
                *PAPER_GA["min_samples_split_bounds"], endpoint=True)),
        )

    def _repair(self, mask: np.ndarray, d: int) -> np.ndarray:
        """Enforce the published minimum-feature constraint."""
        need = min(self.cfg.min_selected_features, d)
        if mask.sum() >= need:
            return mask
        off = np.flatnonzero(~mask)
        add = self.rng.choice(off, size=int(need - mask.sum()), replace=False)
        mask = mask.copy()
        mask[add] = True
        return mask

    # -- fitness ------------------------------------------------------- #

    def _fitness(self, ch: Chromosome, X: np.ndarray, y: np.ndarray) -> float:
        key = ch.key()
        if key in self._cache:
            return self._cache[key]
        if (self.cfg.max_fitness_evaluations is not None
                and self.n_evaluations >= self.cfg.max_fitness_evaluations):
            return -np.inf

        from sklearn.model_selection import StratifiedKFold

        cols = ch.selected()
        Xs, ys = X[:, cols], y
        if self.cfg.fitness_subsample and len(ys) > self.cfg.fitness_subsample:
            from sklearn.model_selection import train_test_split
            Xs, _, ys, _ = train_test_split(
                Xs, ys, train_size=self.cfg.fitness_subsample, stratify=ys,
                random_state=self.seed)

        folds = min(self.cfg.fitness_cv_folds,
                    int(np.min(np.bincount(ys)[np.bincount(ys) > 0])))
        folds = max(2, folds)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        scores: List[float] = []
        for tr, va in skf.split(Xs, ys):
            Xtr, ytr = Xs[tr], ys[tr]
            if self.cfg.use_smote:
                Xtr, ytr = _maybe_smote(Xtr, ytr, self.seed)
            model = build_ebt(ch.n_estimators, ch.max_depth, ch.min_samples_split,
                              seed=self.seed)
            model.fit(Xtr, ytr)
            scores.append(_score(ys[va], model.predict(Xs[va]),
                                 self.cfg.fitness_metric))
        val = float(np.mean(scores))
        self._cache[key] = val
        self.n_evaluations += 1
        return val

    # -- operators ----------------------------------------------------- #

    def _tournament(self, pop: List[Chromosome]) -> Chromosome:
        picks = self.rng.choice(len(pop), size=min(self.cfg.tournament_size, len(pop)),
                                replace=False)
        return max((pop[i] for i in picks), key=lambda c: c.fitness)

    def _crossover(self, a: Chromosome, b: Chromosome, d: int) -> Tuple[Chromosome, Chromosome]:
        c1, c2 = Chromosome(a.mask.copy(), a.n_estimators, a.max_depth,
                            a.min_samples_split), \
                 Chromosome(b.mask.copy(), b.n_estimators, b.max_depth,
                            b.min_samples_split)
        if self.rng.random() >= self.cfg.crossover_rate:
            return c1, c2
        swap = self.rng.random(d) < 0.5          # uniform crossover on F
        c1.mask[swap], c2.mask[swap] = b.mask[swap], a.mask[swap]
        for attr in ("n_estimators", "max_depth", "min_samples_split"):
            if self.rng.random() < 0.5:          # parameter-wise crossover on H
                va, vb = getattr(a, attr), getattr(b, attr)
                setattr(c1, attr, vb)
                setattr(c2, attr, va)
        c1.mask = self._repair(c1.mask, d)
        c2.mask = self._repair(c2.mask, d)
        return c1, c2

    def _mutate(self, ch: Chromosome, d: int) -> Chromosome:
        flip = self.rng.random(d) < self.cfg.mutation_rate
        ch.mask = self._repair(np.where(flip, ~ch.mask, ch.mask), d)
        if self.rng.random() < self.cfg.mutation_rate:
            ch.n_estimators = int(np.clip(
                ch.n_estimators + self.rng.integers(-PAPER_GA["n_estimators_jitter"],
                                                    PAPER_GA["n_estimators_jitter"] + 1),
                *PAPER_GA["n_estimators_bounds"]))
        if self.rng.random() < self.cfg.mutation_rate:
            ch.max_depth = int(np.clip(
                ch.max_depth + self.rng.integers(-PAPER_GA["max_depth_jitter"],
                                                 PAPER_GA["max_depth_jitter"] + 1),
                *PAPER_GA["max_depth_bounds"]))
        if self.rng.random() < self.cfg.mutation_rate:
            ch.min_samples_split = int(np.clip(
                ch.min_samples_split + self.rng.integers(
                    -PAPER_GA["min_samples_split_jitter"],
                    PAPER_GA["min_samples_split_jitter"] + 1),
                *PAPER_GA["min_samples_split_bounds"]))
        ch.fitness = -np.inf
        return ch

    # -- search -------------------------------------------------------- #

    def run(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[Sequence[str]] = None) -> Chromosome:
        d = X.shape[1]
        t0 = time.perf_counter()
        pop = [self._random_chromosome(d) for _ in range(self.cfg.population_size)]
        for ch in pop:
            ch.fitness = self._fitness(ch, X, y)
        best = max(pop, key=lambda c: c.fitness)
        stale = 0

        for gen in range(self.cfg.max_generations):
            if stale >= self.cfg.no_improve_patience:
                break
            if (self.cfg.max_fitness_evaluations is not None
                    and self.n_evaluations >= self.cfg.max_fitness_evaluations):
                break

            mating = [self._tournament(pop) for _ in range(self.cfg.population_size)]
            offspring: List[Chromosome] = []
            for i in range(0, len(mating) - 1, 2):
                c1, c2 = self._crossover(mating[i], mating[i + 1], d)
                offspring += [self._mutate(c1, d), self._mutate(c2, d)]
            for ch in offspring:
                ch.fitness = self._fitness(ch, X, y)

            elite = sorted(pop, key=lambda c: c.fitness, reverse=True)[:self.cfg.elitism_keep]
            pool = sorted(elite + offspring, key=lambda c: c.fitness, reverse=True)
            pop = pool[:self.cfg.population_size]

            curr = pop[0]
            if curr.fitness > best.fitness + 1e-9:
                best, stale = curr, 0
            else:
                stale += 1

            self.history.append({
                "generation": gen,
                "best_fitness": float(best.fitness),
                "generation_best": float(curr.fitness),
                "mean_fitness": float(np.mean([c.fitness for c in pop])),
                "n_selected_features": int(best.mask.sum()),
                "stale": stale,
                "evaluations": self.n_evaluations,
            })
            if self.verbose:
                print(f"[GA-EBT] gen {gen:3d} best {best.fitness:.5f} "
                      f"(d={int(best.mask.sum())}/{d}, evals={self.n_evaluations})",
                      flush=True)

        self.best = best
        self.search_seconds = time.perf_counter() - t0
        return best

    def report(self, feature_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
        return {
            "algorithm": "GA-EBT (reproduced)",
            "fidelity": FIDELITY,
            "published_hyperparameters": PAPER_GA,
            "search_config": self.cfg.to_dict(),
            "search_seed": self.seed,
            "n_fitness_evaluations": self.n_evaluations,
            "search_seconds": getattr(self, "search_seconds", None),
            "generations_run": len(self.history),
            "history": self.history,
            "best": self.best.to_dict(feature_names) if self.best else None,
        }


def _maybe_smote(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE inside the training fold, as the original describes.

    Falls back to returning the fold unchanged (and says so) when imbalanced-learn
    is unavailable or a class is too small to interpolate.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except Exception:
        return X, y
    counts = np.bincount(y)
    smallest = counts[counts > 0].min()
    if smallest < 6 or len(counts[counts > 0]) < 2:
        return X, y
    try:
        sm = SMOTE(random_state=seed, k_neighbors=min(5, smallest - 1))
        return sm.fit_resample(X, y)
    except Exception:
        return X, y


# --------------------------------------------------------------------------- #
# Public API used by the runners
# --------------------------------------------------------------------------- #


def minmax_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    """The original's [0,1] min-max scaling, fitted on the training split."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    return {"lo": lo, "span": span}


def minmax_apply(X: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    return np.clip((X - p["lo"]) / p["span"], 0.0, 1.0)


def search(X_train: np.ndarray, y_train: np.ndarray,
           feature_names: Sequence[str], cfg: Optional[GAConfig] = None,
           seed: int = 42, cache_path: Optional[Path] = None,
           verbose: bool = True) -> Dict[str, object]:
    """Run (or load) the GA feature/hyperparameter search."""
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as fh:
            return json.load(fh)
    ga = GAEBTSearch(cfg=cfg, seed=seed, verbose=verbose)
    ga.run(X_train, y_train, feature_names)
    rep = ga.report(feature_names)
    rep["selected_indices"] = ga.best.selected().tolist()
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(rep, fh, indent=2)
    return rep


def fit_final(X_train: np.ndarray, y_train: np.ndarray,
              search_report: Dict[str, object], seed: int = 42,
              use_smote: bool = True):
    """Train the final EBT on the full training split with the chosen genome."""
    cols = np.asarray(search_report["selected_indices"], dtype=int)
    best = search_report["best"]
    Xtr, ytr = X_train[:, cols], y_train
    if use_smote:
        Xtr, ytr = _maybe_smote(Xtr, ytr, seed)
    model = build_ebt(best["n_estimators"], best["max_depth"],
                      best["min_samples_split"], seed=seed)
    model.fit(Xtr, ytr)
    return model, cols
