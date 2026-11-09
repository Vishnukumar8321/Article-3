"""Training loops for the graph models and the neural tabular baseline.

Two things in here matter for the scientific claim and are worth stating
plainly in the manuscript:

**Neighbour sampling.**  The communication graph has a few dozen endpoints and
hundreds of thousands of parallel flow-edges, so a full-adjacency forward pass
is dominated by a handful of hub nodes.  Message passing therefore uses a
per-destination sample of at most ``max_msg_edges_per_node`` incoming training
edges, redrawn each step during training and fixed (seeded) at evaluation.
This is neighbour sampling in the GraphSAGE sense, not a change to the model.

**Model selection.**  Early stopping and every hyperparameter decision use the
*validation* macro-F1.  The test split is touched exactly once, by
:func:`evaluate_graph_model`, after training has finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .evaluate import compute_metrics
from .graph_builder import FlowGraph
from .utils import count_parameters, get_logger, peak_memory_mb

log = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    max_msg_edges_per_node: int = 2000
    eval_msg_edges_per_node: int = 4000
    grad_clip: float = 5.0
    scheduler: str = "cosine"          # cosine | plateau | none
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    balanced_sampling: bool = False
    device: str = "cpu"
    log_every: int = 10
    monitor: str = "f1_macro"
    restrict_macro_to_present: bool = False
    """Average the validation macro metrics over the classes present in the
    validation split.

    An endpoint-grouped split can leave a class entirely inside the training
    split.  Averaging a structural zero over that class caps validation
    macro-F1 at (k_present / k) regardless of how well the model does, which
    makes early stopping insensitive exactly where the protocol is hardest.
    The flag is set from the split description, never by hand."""

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Neighbour sampling
# --------------------------------------------------------------------------- #


def sample_message_edges(
    edge_index: np.ndarray,
    allowed: np.ndarray,
    n_nodes: int,
    max_per_node: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick at most ``max_per_node`` incoming edges per destination node.

    ``allowed`` is the pool of edges that may be used for message passing --
    always the training edges.  Returns positional indices into ``edge_index``.
    """
    allowed = np.asarray(allowed)
    if max_per_node <= 0 or len(allowed) == 0:
        return allowed
    dst = edge_index[1][allowed]
    order = np.argsort(dst, kind="stable")
    dst_sorted = dst[order]
    allowed_sorted = allowed[order]
    bounds = np.searchsorted(dst_sorted, np.arange(n_nodes + 1))
    picked: List[np.ndarray] = []
    for node in range(n_nodes):
        lo, hi = bounds[node], bounds[node + 1]
        if hi <= lo:
            continue
        block = allowed_sorted[lo:hi]
        if len(block) > max_per_node:
            block = rng.choice(block, size=max_per_node, replace=False)
        picked.append(block)
    if not picked:
        return allowed
    return np.concatenate(picked)


# --------------------------------------------------------------------------- #
# Tensor packaging
# --------------------------------------------------------------------------- #


@dataclass
class GraphTensors:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor

    @classmethod
    def from_graph(cls, graph: FlowGraph, task: str, device: str = "cpu") -> "GraphTensors":
        return cls(
            x=torch.tensor(graph.node_features, dtype=torch.float32, device=device),
            edge_index=torch.tensor(graph.edge_index, dtype=torch.long, device=device),
            edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float32, device=device),
            y=torch.tensor(graph.labels(task), dtype=torch.long, device=device),
        )


# --------------------------------------------------------------------------- #
# Graph model training
# --------------------------------------------------------------------------- #


def _forward_batch(
    model: nn.Module,
    gt: GraphTensors,
    msg_idx: torch.Tensor,
    target_idx: torch.Tensor,
    collect_attention: bool = False,
) -> torch.Tensor:
    msg_edge_index = gt.edge_index[:, msg_idx]
    msg_edge_attr = gt.edge_attr[msg_idx]
    h, _ = model.encode(gt.x, msg_edge_index, msg_edge_attr,
                        collect_attention=collect_attention)
    return model.classify_edges(h, gt.edge_index[:, target_idx],
                                gt.edge_attr[target_idx])


@torch.no_grad()
def predict_graph_model(
    model: nn.Module,
    gt: GraphTensors,
    graph: FlowGraph,
    target_idx: np.ndarray,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    seed: int = 0,
) -> np.ndarray:
    """Probabilities for ``target_idx``, with a fixed seeded message graph."""
    model.eval()
    rng = np.random.default_rng(seed)
    msg = sample_message_edges(graph.edge_index, train_idx, graph.n_nodes,
                               cfg.eval_msg_edges_per_node, rng)
    msg_t = torch.tensor(msg, dtype=torch.long, device=gt.x.device)
    msg_edge_index = gt.edge_index[:, msg_t]
    msg_edge_attr = gt.edge_attr[msg_t]
    h, _ = model.encode(gt.x, msg_edge_index, msg_edge_attr)

    probs: List[np.ndarray] = []
    for start in range(0, len(target_idx), cfg.batch_size):
        chunk = torch.tensor(target_idx[start:start + cfg.batch_size],
                             dtype=torch.long, device=gt.x.device)
        logits = model.classify_edges(h, gt.edge_index[:, chunk], gt.edge_attr[chunk])
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_graph_model(
    model: nn.Module,
    graph: FlowGraph,
    splits: Dict[str, np.ndarray],
    task: str,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train with early stopping on validation macro-F1.

    Returns the model restored to its best validation checkpoint plus a history
    dict containing per-epoch losses, validation metrics and timings.
    """
    import time

    device = torch.device(cfg.device)
    model = model.to(device)
    gt = GraphTensors.from_graph(graph, task, device=str(device))

    train_idx = np.asarray(splits["train"])
    val_idx = np.asarray(splits["val"])
    n_classes = int(gt.y.max().item()) + 1

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs), eta_min=cfg.min_lr)
    elif cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(2, cfg.patience // 3))
    else:
        sched = None

    rng = np.random.default_rng(seed)
    sampler_p = None
    if cfg.balanced_sampling:
        from .losses import balanced_edge_sampler_weights
        sampler_p = balanced_edge_sampler_weights(
            gt.y[torch.tensor(train_idx)].cpu().numpy(), n_classes)

    history: Dict[str, object] = {
        "train_loss": [], "val_f1_macro": [], "val_mcc": [],
        "val_accuracy": [], "lr": [], "epoch_seconds": [],
    }
    best_score = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    bad_epochs = 0
    t_train0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        if sampler_p is None:
            order = rng.permutation(len(train_idx))
            epoch_targets = train_idx[order]
        else:
            epoch_targets = rng.choice(train_idx, size=len(train_idx),
                                       replace=True, p=sampler_p)

        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(epoch_targets), cfg.batch_size):
            batch = epoch_targets[start:start + cfg.batch_size]
            if len(batch) < 2:
                continue
            msg = sample_message_edges(graph.edge_index, train_idx, graph.n_nodes,
                                       cfg.max_msg_edges_per_node, rng)
            msg_t = torch.tensor(msg, dtype=torch.long, device=device)
            tgt_t = torch.tensor(batch, dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            logits = _forward_batch(model, gt, msg_t, tgt_t)
            loss = loss_fn(logits, gt.y[tgt_t])
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(1, n_batches)

        val_prob = predict_graph_model(model, gt, graph, val_idx, train_idx,
                                       cfg, seed=seed)
        val_pred = val_prob.argmax(axis=1)
        val_true = gt.y[torch.tensor(val_idx)].cpu().numpy()
        vm = compute_metrics(val_true, val_pred, val_prob, n_classes=n_classes,
                             restrict_to_present=cfg.restrict_macro_to_present)

        history["train_loss"].append(mean_loss)
        history["val_f1_macro"].append(vm["f1_macro"])
        history["val_mcc"].append(vm["mcc"])
        history["val_accuracy"].append(vm["accuracy"])
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        history["epoch_seconds"].append(time.perf_counter() - t0)

        score = float(vm[cfg.monitor])
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if sched is not None:
            sched.step(score) if cfg.scheduler == "plateau" else sched.step()

        if epoch % cfg.log_every == 0 or bad_epochs >= cfg.patience:
            log.info("epoch %3d | loss %.4f | val macroF1 %.4f | val MCC %.4f | %s",
                     epoch, mean_loss, vm["f1_macro"], vm["mcc"],
                     "best" if best_epoch == epoch else f"patience {bad_epochs}")
        if bad_epochs >= cfg.patience:
            log.info("early stop at epoch %d (best epoch %d, val %s=%.4f)",
                     epoch, best_epoch, cfg.monitor, best_score)
            break

    train_seconds = time.perf_counter() - t_train0
    model.load_state_dict(best_state)
    history.update({
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "monitor": cfg.monitor,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": train_seconds,
        "n_parameters": count_parameters(model),
        "peak_memory_mb": peak_memory_mb(),
    })
    return model, history


# --------------------------------------------------------------------------- #
# Tabular neural baseline
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_mlp(model: nn.Module, Z: np.ndarray, batch_size: int = 8192,
                device: str = "cpu") -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(Z), batch_size):
        chunk = torch.tensor(Z[start:start + batch_size], dtype=torch.float32,
                             device=device)
        out.append(torch.softmax(model(chunk), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_mlp(
    model: nn.Module,
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_val: np.ndarray, y_val: np.ndarray,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, object]]:
    import time

    device = torch.device(cfg.device)
    model = model.to(device)
    n_classes = int(max(y_train.max(), y_val.max())) + 1
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.epochs), eta_min=cfg.min_lr)
    rng = np.random.default_rng(seed)

    Zt = torch.tensor(Z_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    history: Dict[str, object] = {"train_loss": [], "val_f1_macro": [],
                                  "epoch_seconds": []}
    best_score, best_epoch, bad = -np.inf, -1, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0_all = time.perf_counter()

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(Z_train))
        total, nb = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = torch.tensor(order[start:start + cfg.batch_size],
                               dtype=torch.long, device=device)
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Zt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()

        prob = predict_mlp(model, Z_val, cfg.batch_size, str(device))
        vm = compute_metrics(y_val, prob.argmax(axis=1), prob, n_classes=n_classes,
                             restrict_to_present=cfg.restrict_macro_to_present)
        history["train_loss"].append(total / max(1, nb))
        history["val_f1_macro"].append(vm["f1_macro"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

        if vm["f1_macro"] > best_score + 1e-6:
            best_score, best_epoch, bad = vm["f1_macro"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    history.update({
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": time.perf_counter() - t0_all,
        "n_parameters": count_parameters(model),
        "peak_memory_mb": peak_memory_mb(),
    })
    return model, history


def measure_inference(
    predict_callable, n_samples: int, repeats: int = 3,
) -> Dict[str, float]:
    """Wall-clock inference latency and throughput (section 34)."""
    import time

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_callable()
        times.append(time.perf_counter() - t0)
    best = float(np.min(times))
    return {
        "inference_seconds_mean": float(np.mean(times)),
        "inference_seconds_min": best,
        "inference_seconds_std": float(np.std(times)),
        "flows_per_second": float(n_samples / best) if best > 0 else float("nan"),
        "latency_ms_per_flow": float(1000.0 * best / n_samples) if n_samples else float("nan"),
        "repeats": repeats,
    }
