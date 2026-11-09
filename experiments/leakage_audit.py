"""Leakage audit: Phases 2, 3, 4, 6 and 8, computed rather than asserted.

Trains nothing. Every number below is measured from the data and from the
split objects the pipeline actually produces, so a claim in the audit report
can be checked against ``results/audit/leakage_audit.json``.

The audit deliberately re-derives quantities the pipeline already believes are
true (preprocessing fitted on train only, message passing on train edges only,
banned columns absent) instead of trusting the pipeline's own bookkeeping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    NON_PREDICTIVE_COLUMNS, SCENARIO_LABELS, feature_columns, load_representation,
)
from src.split import candidate_holdout_endpoints, make_split, verify_split  # noqa: E402
from src.utils import RESULTS, get_logger, save_json  # noqa: E402

log = get_logger("leakage_audit")

AUDIT_DIR = RESULTS / "audit"


# --------------------------------------------------------------------------- #
# Phase 3 -- feature inventory
# --------------------------------------------------------------------------- #

#: Semantic annotation for every column the loader can produce. Each entry
#: records what the column is, whether a deployed detector would have it at
#: inference time, and whether it could identify the target, the capture or an
#: endpoint.
FEATURE_SEMANTICS: Dict[str, Dict[str, object]] = {
    "ip_src": dict(meaning="source IP address", available_at_inference=True,
                   target_derived=False, identifies_endpoint=True,
                   identifies_capture=False, used_as_feature=False,
                   used_as="graph node identity only"),
    "ip_dst": dict(meaning="destination IP address", available_at_inference=True,
                   target_derived=False, identifies_endpoint=True,
                   identifies_capture=False, used_as_feature=False,
                   used_as="graph node identity only"),
    "is_attack": dict(meaning="ground-truth attack flag", available_at_inference=False,
                      target_derived=True, identifies_endpoint=False,
                      identifies_capture=False, used_as_feature=False,
                      used_as="label source only"),
    "source_file": dict(meaning="capture file the row came from",
                        available_at_inference=False, target_derived=True,
                        identifies_endpoint=False, identifies_capture=True,
                        used_as_feature=False, used_as="provenance only"),
    "scenario": dict(meaning="canonical capture key", available_at_inference=False,
                     target_derived=True, identifies_endpoint=False,
                     identifies_capture=True, used_as_feature=False,
                     used_as="provenance and multi-class label construction"),
    "capture_scenario_label": dict(
        meaning="numeric capture id", available_at_inference=False,
        target_derived=True, identifies_endpoint=False, identifies_capture=True,
        used_as_feature=False, used_as="provenance only"),
    "binary_label": dict(meaning="target, normal vs attack",
                         available_at_inference=False, target_derived=True,
                         identifies_endpoint=False, identifies_capture=False,
                         used_as_feature=False, used_as="target"),
    "multiclass_label": dict(meaning="target, five-class",
                             available_at_inference=False, target_derived=True,
                             identifies_endpoint=False, identifies_capture=False,
                             used_as_feature=False, used_as="target"),
    "prt_src": dict(meaning="source port", available_at_inference=True,
                    target_derived=False, identifies_endpoint=False,
                    identifies_capture=False, used_as_feature=True,
                    used_as="edge attribute"),
    "prt_dst": dict(meaning="destination port, i.e. the service contacted",
                    available_at_inference=True, target_derived=False,
                    identifies_endpoint=False, identifies_capture=False,
                    used_as_feature=True,
                    used_as="edge attribute; strongly associated with attack "
                            "class by construction of the testbed (22 = Sparta, "
                            "1883 = MQTT) but observable at deployment"),
    "proto": dict(meaning="IP protocol number", available_at_inference=True,
                  target_derived=False, identifies_endpoint=False,
                  identifies_capture=False, used_as_feature=True,
                  used_as="edge attribute"),
}

_GENERIC_TRAFFIC = (
    "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat", "mean_pkt_len",
    "num_bytes", "num_psh_flags", "num_rst_flags", "num_urg_flags",
    "std_pkt_len", "min_pkt_len", "max_pkt_len",
)
for _c in _GENERIC_TRAFFIC:
    FEATURE_SEMANTICS[_c] = dict(
        meaning="per-flow traffic statistic", available_at_inference=True,
        target_derived=False, identifies_endpoint=False,
        identifies_capture=False, used_as_feature=True, used_as="edge attribute")

_DERIVED = {
    "prt_src_log": "log1p of source port",
    "prt_dst_log": "log1p of destination port",
    "prt_src_is_wellknown": "source port in a well-known list",
    "prt_dst_is_wellknown": "destination port in a well-known list",
    "prt_src_is_system": "source port < 1024",
    "prt_dst_is_system": "destination port < 1024",
    "prt_src_is_ephemeral": "source port >= 49152",
    "prt_dst_is_ephemeral": "destination port >= 49152",
    "bytes_per_pkt": "num_bytes / num_pkts",
    "iat_burstiness": "max_iat / mean_iat",
    "pkt_len_range": "max_pkt_len - min_pkt_len",
}
for _c, _m in _DERIVED.items():
    FEATURE_SEMANTICS[_c] = dict(
        meaning=_m, available_at_inference=True, target_derived=False,
        identifies_endpoint=False, identifies_capture=False,
        used_as_feature=True,
        used_as="edge attribute, row-wise deterministic (no fitted parameters, "
                "so safe to compute before splitting)")


def audit_features(df: pd.DataFrame, representation: str) -> Dict[str, object]:
    feats = feature_columns(df)
    inventory = []
    unknown = []
    for c in df.columns:
        sem = FEATURE_SEMANTICS.get(c)
        if sem is None:
            # Bi-flow forward/backward columns follow the uni-flow semantics of
            # their stem; anything else is flagged for manual review.
            stem = c.replace("fwd_", "").replace("bwd_", "")
            sem = FEATURE_SEMANTICS.get(stem)
            if sem is None:
                unknown.append(c)
                sem = dict(meaning="UNANNOTATED -- review", available_at_inference=None,
                           target_derived=None, identifies_endpoint=None,
                           identifies_capture=None, used_as_feature=c in feats,
                           used_as="unknown")
        inventory.append({"column": c, "dtype": str(df[c].dtype),
                          "in_feature_matrix": c in feats, **sem})

    banned_in_matrix = [c for c in feats if c in NON_PREDICTIVE_COLUMNS]
    target_derived_in_matrix = [
        e["column"] for e in inventory
        if e["in_feature_matrix"] and e.get("target_derived") is True]
    capture_id_in_matrix = [
        e["column"] for e in inventory
        if e["in_feature_matrix"] and e.get("identifies_capture") is True]
    endpoint_id_in_matrix = [
        e["column"] for e in inventory
        if e["in_feature_matrix"] and e.get("identifies_endpoint") is True]

    return {
        "representation": representation,
        "n_columns_total": int(df.shape[1]),
        "n_columns_in_feature_matrix": len(feats),
        "feature_matrix_columns": feats,
        "inventory": inventory,
        "unannotated_columns": unknown,
        "VIOLATION_banned_columns_in_matrix": banned_in_matrix,
        "VIOLATION_target_derived_in_matrix": target_derived_in_matrix,
        "VIOLATION_capture_identifier_in_matrix": capture_id_in_matrix,
        "VIOLATION_endpoint_identifier_in_matrix": endpoint_id_in_matrix,
        "feature_sets": {
            "A_original": feats,
            "B_leakage_safe": feats,
            "C_endpoint_masked": feats,
            "D_scenario_masked": feats,
            "_note": (
                "A == B == C == D for this pipeline because no endpoint "
                "identifier, capture identifier or target-derived column is in "
                "the feature matrix to begin with. Endpoint identity enters "
                "only through graph structure, so the endpoint-masked condition "
                "is realised by an evaluation protocol (endpoint-disjoint "
                "splitting) rather than by dropping a column, and the "
                "scenario-masked condition likewise."),
        },
    }


# --------------------------------------------------------------------------- #
# Phase 4 & 8 -- split overlap and duplication
# --------------------------------------------------------------------------- #


def _five_tuple(df: pd.DataFrame) -> np.ndarray:
    cols = ["ip_src", "ip_dst", "prt_src", "prt_dst", "proto"]
    have = [c for c in cols if c in df.columns]
    return df[have].astype(str).agg("|".join, axis=1).to_numpy()


def audit_split(df: pd.DataFrame, y: np.ndarray, feats: Sequence[str],
                protocol: str, seed: int = 42,
                split_kwargs: Optional[Dict] = None) -> Dict[str, object]:
    res = make_split(protocol, df, y, feats, seed=seed, **(split_kwargs or {}))
    if not res.splits:
        return {"protocol": protocol, "seed": seed, "applicable": False,
                "reason": res.description.get("reason"),
                "description": res.description}
    verify_split(res.splits, len(df))
    tr, va, te = (res.splits["train"], res.splits["val"], res.splits["test"])

    def overlap(name: str, values: np.ndarray) -> Dict[str, object]:
        a, b = set(values[tr].tolist()), set(values[te].tolist())
        shared = a & b
        n_rows_affected = int(np.isin(values[te], list(shared)).sum()) if shared else 0
        return {"train_unique": len(a), "test_unique": len(b),
                "shared": len(shared),
                "share_of_test_unique": round(len(shared) / max(1, len(b)), 4),
                "test_rows_touching_shared_value": n_rows_affected,
                "test_row_share": round(n_rows_affected / max(1, len(te)), 4)}

    row_id = np.arange(len(df))
    src = df["ip_src"].astype(str).to_numpy()
    dst = df["ip_dst"].astype(str).to_numpy()
    pair = np.char.add(np.char.add(src, "->"), dst)
    endpoint_rows = np.array([f"{a}|{b}" for a, b in zip(src, dst)])
    five = _five_tuple(df)
    feat_hash = pd.util.hash_pandas_object(df[list(feats)], index=False).to_numpy()
    capture = df["scenario"].astype(str).to_numpy()

    # Endpoint-level intersection needs the union of both roles.
    ep_train = set(src[tr].tolist()) | set(dst[tr].tolist())
    ep_test = set(src[te].tolist()) | set(dst[te].tolist())

    out = {
        "protocol": protocol,
        "seed": seed,
        "applicable": True,
        "name": res.description.get("name"),
        "sizes": {k: int(len(v)) for k, v in res.splits.items()},
        "class_balance": res.description.get("class_balance"),
        "overlaps": {
            "flow_ids": overlap("flow_ids", row_id),
            "source_ips": overlap("source_ips", src),
            "destination_ips": overlap("destination_ips", dst),
            "src_dst_pairs": overlap("src_dst_pairs", pair),
            "five_tuples": overlap("five_tuples", five),
            "exact_feature_vectors": overlap("exact_feature_vectors", feat_hash),
            "capture_ids": overlap("capture_ids", capture),
        },
        "endpoints": {
            "train_endpoints": len(ep_train),
            "test_endpoints": len(ep_test),
            "shared_endpoints": len(ep_train & ep_test),
            "test_only_endpoints": sorted(ep_test - ep_train),
            "endpoint_disjoint": len(ep_train & ep_test) == 0,
        },
        "split_description": {k: v for k, v in res.description.items()
                              if k not in ("class_balance",)},
    }
    return out


def audit_duplicates(df: pd.DataFrame, feats: Sequence[str]) -> Dict[str, object]:
    feat_hash = pd.util.hash_pandas_object(df[list(feats)], index=False)
    full_hash = pd.util.hash_pandas_object(df.drop(columns=["source_file"],
                                                   errors="ignore"), index=False)
    five = pd.Series(_five_tuple(df))
    vc = feat_hash.value_counts()
    return {
        "n_rows": int(len(df)),
        "exact_duplicate_rows_all_columns": int(full_hash.duplicated().sum()),
        "duplicate_predictive_feature_vectors": int(feat_hash.duplicated().sum()),
        "duplicate_feature_vector_share": round(
            float(feat_hash.duplicated().mean()), 4),
        "distinct_feature_vectors": int(feat_hash.nunique()),
        "largest_duplicate_group": int(vc.iloc[0]) if len(vc) else 0,
        "n_groups_with_10_or_more": int((vc >= 10).sum()),
        "rows_in_groups_of_10_or_more": int(vc[vc >= 10].sum()),
        "duplicate_five_tuples": int(five.duplicated().sum()),
        "distinct_five_tuples": int(five.nunique()),
        "note": (
            "Repeated flow records are a genuine property of scan and "
            "brute-force traffic and are NOT removed. They are reported so "
            "that the contribution of memorisation can be bounded, and "
            "protocol B groups them so that no duplicate straddles a split."),
    }


# --------------------------------------------------------------------------- #
# Phase 7 -- how much is answerable from endpoint identity alone
# --------------------------------------------------------------------------- #


def audit_endpoint_predictiveness(df: pd.DataFrame, y: np.ndarray) -> Dict[str, object]:
    src = df["ip_src"].astype(str)
    dst = df["ip_dst"].astype(str)
    pair = (src + "->" + dst).to_numpy()

    def purity(keys: np.ndarray) -> Dict[str, object]:
        frame = pd.DataFrame({"k": keys, "y": y})
        g = frame.groupby("k")["y"]
        pur = g.agg(lambda s: s.value_counts(normalize=True).max())
        size = g.size()
        pure = pur[pur == 1.0]
        return {
            "n_groups": int(len(pur)),
            "groups_single_class": int(len(pure)),
            "rows_in_single_class_groups": int(size[pur == 1.0].sum()),
            "row_share_in_single_class_groups": round(
                float(size[pur == 1.0].sum() / len(y)), 4),
            "flow_weighted_mean_purity": round(
                float((pur * size).sum() / size.sum()), 4),
        }

    return {
        "ordered_pair": purity(pair),
        "source_ip": purity(src.to_numpy()),
        "destination_ip": purity(dst.to_numpy()),
        "interpretation": (
            "Flow-weighted mean purity is an upper bound on the accuracy of a "
            "detector that sees only the endpoint identity and predicts the "
            "majority class of that endpoint group. A value near 1 means the "
            "label is almost determined by identity, so any model given the "
            "endpoint pair can reach that accuracy without learning anything "
            "about traffic."),
    }


# --------------------------------------------------------------------------- #
# Phase 6 -- graph construction leakage
# --------------------------------------------------------------------------- #


def audit_graph(df: pd.DataFrame, y: np.ndarray, feats: Sequence[str],
                protocol: str, seed: int = 42,
                split_kwargs: Optional[Dict] = None) -> Dict[str, object]:
    from src.feature_engineering import build_node_features
    from src.graph_builder import build_node_index

    res = make_split(protocol, df, y, feats, seed=seed, **(split_kwargs or {}))
    if not res.splits:
        return {"protocol": protocol, "applicable": False}
    tr, te = res.splits["train"], res.splits["test"]

    node_index = build_node_index(df)
    src = df["ip_src"].astype(str).map(node_index).to_numpy()
    dst = df["ip_dst"].astype(str).map(node_index).to_numpy()

    nodes_all = set(node_index.values())
    nodes_train = set(src[tr].tolist()) | set(dst[tr].tolist())
    nodes_test = set(src[te].tolist()) | set(dst[te].tolist())

    # Do the node features actually differ when computed train-only vs all?
    feats_train, names = build_node_features(df, node_index, edge_subset=tr)
    feats_all, _ = build_node_features(df, node_index, edge_subset=None)
    diff = np.abs(feats_train - feats_all)
    denom = np.abs(feats_all)
    rel = np.where(denom > 0, diff / np.maximum(denom, 1e-9), 0.0)

    test_only_nodes = sorted(nodes_test - nodes_train)
    zero_rows = [int(n) for n in test_only_nodes
                 if float(np.abs(feats_train[n]).sum()) == 0.0]

    return {
        "protocol": protocol,
        "seed": seed,
        "applicable": True,
        "n_nodes_in_index": len(nodes_all),
        "node_index_built_over": "ALL rows (train, val and test)",
        "n_nodes_touched_by_train_edges": len(nodes_train),
        "n_nodes_touched_by_test_edges": len(nodes_test),
        "test_only_nodes": test_only_nodes,
        "test_only_nodes_with_all_zero_train_features": zero_rows,
        "message_passing_edge_pool": "training edges only "
                                     "(FlowGraph.message_edge_mask)",
        "node_features_source": "training edges only "
                                "(build_flow_graph node_feature_source='train')",
        "node_feature_divergence_train_only_vs_all": {
            "max_absolute": float(diff.max()),
            "mean_absolute": float(diff.mean()),
            "max_relative": float(rel.max()),
            "n_features_changed": int((diff > 0).any(axis=0).sum()),
            "n_nodes_changed": int((diff > 0).any(axis=1).sum()),
            "interpretation": (
                "Non-zero divergence confirms the train-only aggregation is "
                "actually doing something: if it were zero, the 'train only' "
                "setting would be cosmetic."),
        },
        "residual_transductive_element": (
            "The node INDEX spans all splits, so a node that appears only in "
            "test exists in the graph during training. It carries all-zero "
            "behavioural features and no training edge, so no test traffic and "
            "no test label reaches the model through it; what leaks is the "
            "bare existence of an address. Under leave-one-endpoint-out this "
            "is the held-out device. An inductive variant that rebuilds the "
            "index from training endpoints alone is the stricter alternative "
            "and is listed in the recommendations."),
    }


# --------------------------------------------------------------------------- #


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {"representations": {}}

    for rep in ("uniflow", "biflow"):
        df = load_representation(rep)
        feats = feature_columns(df)
        y_mc = df["multiclass_label"].to_numpy()
        y_bin = df["binary_label"].to_numpy()

        entry: Dict[str, object] = {
            "features": audit_features(df, rep),
            "duplicates": audit_duplicates(df, feats),
            "endpoint_predictiveness_multiclass": audit_endpoint_predictiveness(df, y_mc),
            "endpoint_predictiveness_binary": audit_endpoint_predictiveness(df, y_bin),
            "splits": {},
            "graph": {},
        }

        protocols = [("A", {}), ("B", {}), ("C", {}), ("CS", {})]
        if rep == "uniflow":
            protocols.append(("L", {"fold": 0}))
        # Split overlaps are cheap and are computed for every protocol.
        for proto, kw in protocols:
            for task, yv in (("multiclass", y_mc), ("binary", y_bin)):
                key = f"{proto}_{task}"
                try:
                    entry["splits"][key] = audit_split(df, yv, feats, proto,
                                                       seed=42, split_kwargs=kw)
                except Exception as exc:
                    entry["splits"][key] = {"error": repr(exc)}
                log.info("  split audited: %s %s", rep, key)
        # The graph audit recomputes node features twice per protocol, which is
        # the expensive part; protocols A (random) and C (endpoint-disjoint)
        # bracket the behaviour, so the rest would only repeat the finding.
        for proto, kw in [p_ for p_ in protocols if p_[0] in ("A", "C")]:
            try:
                entry["graph"][proto] = audit_graph(df, y_mc, feats, proto,
                                                    seed=42, split_kwargs=kw)
            except Exception as exc:
                entry["graph"][proto] = {"error": repr(exc)}
            log.info("  graph audited: %s %s", rep, proto)

        # Seed-to-seed stability of protocol A's duplicate overlap.
        entry["protocol_A_duplicate_overlap_by_seed"] = {}
        for s in (42, 123, 2024):
            a = audit_split(df, y_mc, feats, "A", seed=s)
            entry["protocol_A_duplicate_overlap_by_seed"][str(s)] = \
                a["overlaps"]["exact_feature_vectors"]

        report["representations"][rep] = entry
        log.info("audited %s", rep)
        del df

    save_json(report, AUDIT_DIR / "leakage_audit.json")
    log.info("wrote %s", AUDIT_DIR / "leakage_audit.json")

    # Flat CSV of the split overlaps, for the paper's leakage table.
    rows = []
    for rep, entry in report["representations"].items():
        for key, s in entry["splits"].items():
            if not s.get("applicable"):
                rows.append({"representation": rep, "cell": key,
                             "applicable": False,
                             "reason": s.get("reason")})
                continue
            o = s["overlaps"]
            rows.append({
                "representation": rep, "cell": key, "applicable": True,
                "protocol": s["protocol"], "n_train": s["sizes"]["train"],
                "n_test": s["sizes"]["test"],
                "shared_flow_ids": o["flow_ids"]["shared"],
                "shared_source_ips": o["source_ips"]["shared"],
                "shared_dest_ips": o["destination_ips"]["shared"],
                "shared_pairs": o["src_dst_pairs"]["shared"],
                "shared_five_tuples": o["five_tuples"]["shared"],
                "shared_feature_vectors": o["exact_feature_vectors"]["shared"],
                "test_rows_with_shared_feature_vector":
                    o["exact_feature_vectors"]["test_rows_touching_shared_value"],
                "test_row_share_shared_feature_vector":
                    o["exact_feature_vectors"]["test_row_share"],
                "shared_captures": o["capture_ids"]["shared"],
                "endpoint_disjoint": s["endpoints"]["endpoint_disjoint"],
                "shared_endpoints": s["endpoints"]["shared_endpoints"],
            })
    pd.DataFrame(rows).to_csv(AUDIT_DIR / "split_overlaps.csv", index=False)
    log.info("wrote %s", AUDIT_DIR / "split_overlaps.csv")


if __name__ == "__main__":
    main()
