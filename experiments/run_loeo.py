"""Leave-one-endpoint-out evaluation -- the decisive model comparison.

Why this exists. Protocol C withholds a random set of endpoints, and on a
24-endpoint testbed the composition of that set dominates the outcome: across
three random draws the same GCN scored 0.4756, 0.5440 and 0.9927 macro-F1.
A mean over five draws is not a ranking, it is a noisy point estimate with a
standard deviation of the same order as the differences between models.

Protocol L enumerates instead of sampling. Each candidate endpoint is withheld
exactly once, giving 20 folds on the uni-flow release, and the model seed is
held fixed at 42 across folds so that variation between folds is attributable
to holdout composition alone. Every model sees exactly the same 20 splits,
which turns the model comparison into a paired test over 20 matched
observations rather than 5 -- enough for the Wilcoxon signed-rank test to
resolve a difference, which at n=5 it cannot.

Fold-level results are retained, not just their mean, because the spread across
folds is itself a finding: it says how much an endpoint-grouped result depends
on which device happens to be held out, and therefore how much weight a
single-split result in the literature can bear.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    CLASSICAL, PreparedData, run_classical, run_neural,
)
from src.split import candidate_holdout_endpoints  # noqa: E402
from src.train import TrainConfig  # noqa: E402
from src.utils import (  # noqa: E402
    DATA_METADATA, RESULTS_RAW, get_logger, save_json,
)

log = get_logger("loeo")

MODELS = ["EndpointLookup", "MLP", "GCN", "GraphSAGE", "E-GraphSAGE", "GAT",
          "EdgeGAT", "E-GAT-IDS", "XGBoost", "RandomForest", "SVM", "KNN",
          "GA-EBT"]

#: Fixed across folds on purpose: the point is to isolate holdout composition.
MODEL_SEED = 42

GA_BUDGET = {"population_size": 16, "max_generations": 8,
             "fitness_subsample": 15_000, "fitness_cv_folds": 3,
             "no_improve_patience": 3}


def prepare_fold(representation: str, task: str, fold: int,
                 seed: int = MODEL_SEED) -> PreparedData:
    """Build a protocol-L cell for one fold.

    The fold index reaches the splitter through ``split_kwargs`` and is echoed
    into the stored split description, so nothing about which endpoint was
    withheld is implicit in the saved result.
    """
    return PreparedData(representation, task, "L", seed,
                        split_kwargs={"fold": fold})


def run(representation: str = "uniflow", task: str = "multiclass",
        models: Sequence[str] = MODELS,
        folds: Optional[Sequence[int]] = None) -> None:
    from src.data_loader import load_representation

    df = load_representation(representation)
    y = (df["binary_label"] if task == "binary"
         else df["multiclass_label"]).to_numpy()
    candidates = candidate_holdout_endpoints(df, y)
    fold_list = list(folds) if folds is not None else list(range(len(candidates)))
    del df, y

    cfg = TrainConfig(epochs=60, patience=12, batch_size=8192, log_every=30)
    manifest = {
        "representation": representation,
        "task": task,
        "protocol": "L",
        "model_seed_fixed_at": MODEL_SEED,
        "candidate_endpoints": candidates,
        "n_folds_attempted": len(fold_list),
        "models": list(models),
        "rationale": (
            "Each candidate endpoint is withheld exactly once with the model "
            "seed fixed, so variation across folds reflects holdout "
            "composition rather than initialisation. Every model sees the "
            "identical fold set, giving a paired comparison over as many "
            "observations as there are endpoints."
        ),
        "folds": [],
    }
    t_all = time.perf_counter()

    for fold in fold_list:
        stem = f"__{representation}__{task}__protocolL__seed{fold}.json"
        if all((RESULTS_RAW / f"loeo__{m}{stem}").exists() for m in models):
            log.info("fold %d already complete, skipping", fold)
            continue
        t0 = time.perf_counter()
        data = prepare_fold(representation, task, fold)
        if not data.applicable:
            log.warning("fold %d not applicable: %s", fold,
                        data.split_description.get("reason"))
            save_json(data.split_description,
                      RESULTS_RAW / f"SKIPPED__loeo{stem}")
            manifest["folds"].append({"fold": fold, "applicable": False,
                                      "reason": data.split_description.get("reason")})
            continue

        # The fold index occupies the record's seed field so that each fold
        # gets its own file, while model_seed stays pinned at MODEL_SEED so
        # that fold-to-fold variation cannot be initialisation noise. Both are
        # written into every run record.
        data.seed = fold
        data.model_seed = MODEL_SEED
        held = data.split_description.get("held_out_endpoint")
        log.info("fold %d/%d: holding out %s, test=%d rows, classes=%s",
                 fold, len(fold_list), held, len(data.splits["test"]),
                 data.split_description.get("classes_in_test"))

        n_ok = 0
        for name in models:
            try:
                if name in CLASSICAL:
                    run_classical(data, name, "loeo",
                                  ga_cache_dir=DATA_METADATA / "ga_search_loeo",
                                  ga_budget=GA_BUDGET)
                else:
                    run_neural(data, name, "loeo", train_cfg=cfg,
                               save_checkpoint=False)
                n_ok += 1
            except Exception as exc:
                log.exception("fold %d model %s failed: %s", fold, name, exc)
        dt = time.perf_counter() - t0
        manifest["folds"].append({
            "fold": fold, "applicable": True, "held_out_endpoint": held,
            "test_rows": int(len(data.splits["test"])),
            "classes_in_test": data.split_description.get("classes_in_test"),
            "models_completed": n_ok, "seconds": dt,
        })
        save_json(manifest, DATA_METADATA / f"loeo_manifest_{representation}_{task}.json")
        log.info("fold %d done: %d/%d models in %.0fs (elapsed %.2f h)",
                 fold, n_ok, len(models), dt, (time.perf_counter() - t_all) / 3600)

    log.info("leave-one-endpoint-out complete in %.2f h",
             (time.perf_counter() - t_all) / 3600)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", default="uniflow")
    p.add_argument("--task", default="multiclass")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--folds", default=None,
                   help="comma-separated fold indices; default is all")
    a = p.parse_args()
    run(a.representation, a.task, [m for m in a.models.split(",") if m],
        [int(x) for x in a.folds.split(",")] if a.folds else None)
