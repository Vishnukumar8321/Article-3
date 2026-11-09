"""Component ablation of E-GAT-IDS (section 25).

Two ladders are run.

**Architecture ladder** (A-F) is executed by the main phase, since those are the
same six models as the headline comparison: MLP, GCN, GraphSAGE, GAT, EdgeGAT,
E-GAT-IDS.

**Component ablation** is what this module adds: starting from E-GAT-IDS, one
component is removed or swapped at a time so that any change in macro-F1 is
attributable to that component and nothing else.  Every variant uses the same
split, the same preprocessing, the same graph and the same training budget.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import PreparedData, run_neural  # noqa: E402
from src.train import TrainConfig  # noqa: E402
from src.utils import RESULTS_RAW, SEEDS, ensure_dirs, get_logger, save_json  # noqa: E402

log = get_logger("ablation")


#: variant name -> (model key, model kwargs override, loss settings, what it tests)
VARIANTS: Dict[str, Dict[str, object]] = {
    "E-GAT-IDS (full)": {
        "model": "E-GAT-IDS", "kwargs": {}, "loss": "ce",
        "tests": "reference configuration",
    },
    "E-GAT w/o edge-aware attention": {
        "model": "E-GAT-IDS", "kwargs": {"edge_in_attention": False}, "loss": "ce",
        "tests": "the single methodological claim: z_ij inside the attention logit",
    },
    "E-GAT w/o edge features in message": {
        "model": "E-GAT-IDS", "kwargs": {"edge_in_message": False}, "loss": "ce",
        "tests": "whether flow attributes need to reach the aggregated message",
    },
    "E-GAT w/o edge features in classifier": {
        "model": "E-GAT-IDS", "kwargs": {"edge_in_classifier": False}, "loss": "ce",
        "tests": "how much of the decision rests on the raw flow vector",
    },
    "E-GAT w/o node behavioural features": {
        "model": "E-GAT-IDS", "kwargs": {"use_node_features": False}, "loss": "ce",
        "tests": "whether engineered endpoint behaviour matters or topology suffices",
    },
    "E-GAT w/o residual connections": {
        "model": "E-GAT-IDS", "kwargs": {"residual": False}, "loss": "ce",
        "tests": "optimisation aid vs representational contribution",
    },
    "E-GAT single layer": {
        "model": "E-GAT-IDS", "kwargs": {"n_layers": 1}, "loss": "ce",
        "tests": "depth of the neighbourhood the model actually needs",
    },
    "E-GAT single head": {
        "model": "E-GAT-IDS", "kwargs": {"heads": 1}, "loss": "ce",
        "tests": "contribution of multi-head attention",
    },
    "E-GAT + focal loss": {
        "model": "E-GAT-IDS", "kwargs": {}, "loss": "focal",
        "tests": "loss choice under class imbalance",
    },
    "E-GAT + unweighted CE": {
        "model": "E-GAT-IDS", "kwargs": {}, "loss": "unweighted_ce",
        "tests": "whether class weighting is doing the minority-class work",
    },
    "E-GAT + balanced edge sampling": {
        "model": "E-GAT-IDS", "kwargs": {}, "loss": "ce", "balanced_sampling": True,
        "tests": "graph-aware resampling of real edges instead of synthesis",
    },
}


def run(representation: str, task: str, protocol: str, seeds: Sequence[int],
        train_cfg: Optional[TrainConfig] = None, nrows: Optional[int] = None,
        variants: Optional[Sequence[str]] = None,
        raw_dir: Optional[Path] = None, experiment: str = "ablation") -> None:
    ensure_dirs()
    names = list(variants) if variants else list(VARIANTS)
    save_json({k: {"model": v["model"], "kwargs": v["kwargs"],
                   "loss": v.get("loss"), "tests": v["tests"]}
               for k, v in VARIANTS.items()},
              RESULTS_RAW / "_ablation_variants.json")

    for seed in seeds:
        data = PreparedData(representation, task, protocol, seed, nrows=nrows,
                            raw_dir=raw_dir)
        if not data.applicable:
            log.warning("skipping ablation for protocol %s seed %d", protocol, seed)
            continue
        for name in names:
            spec = VARIANTS[name]
            cfg = train_cfg or TrainConfig()
            if spec.get("balanced_sampling"):
                cfg = TrainConfig(**{**cfg.to_dict(), "balanced_sampling": True})
            try:
                run_neural(
                    data, spec["model"], experiment=experiment,
                    train_cfg=cfg, model_kwargs=dict(spec["kwargs"]),
                    loss=str(spec.get("loss", "ce")),
                    weighting="none" if spec.get("loss") == "unweighted_ce"
                    else "balanced",
                    save_checkpoint=False,
                    # The record is written once, under the variant name, so the
                    # ablation table reads by variant rather than by model.
                    model_label=name,
                    extra_notes=f"ablation variant; tests: {spec['tests']}",
                )
            except Exception as exc:
                log.exception("ablation variant %r failed: %s", name, exc)


def main(args) -> None:
    cfg = TrainConfig(epochs=getattr(args, "epochs", 100),
                      patience=getattr(args, "patience", 15),
                      batch_size=getattr(args, "batch_size", 8192),
                      device=getattr(args, "device", "cpu"))
    reps = getattr(args, "representations", ["uniflow"])
    tasks = getattr(args, "tasks", ["multiclass"])
    seeds = getattr(args, "seeds", SEEDS)
    protocols = getattr(args, "protocols", ["A"])
    for rep in reps:
        for task in tasks:
            run(rep, task, protocols[0], seeds, train_cfg=cfg,
                nrows=getattr(args, "nrows", None))


if __name__ == "__main__":
    from experiments.run_all import build_parser, normalise

    main(normalise(build_parser().parse_args()))
