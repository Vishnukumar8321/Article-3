"""E-GAT-IDS: edge-aware graph attention for MQTT-IoT intrusion detection.

Modules
-------
``data_loader``        discovery, scenario labelling, schema checks
``audit``              phase 1 dataset audit (reports, never modifies)
``preprocessing``      fit-on-train-only cleaning, scaling, leakage guards
``feature_engineering``behaviour-aware node features
``graph_builder``      flow table -> directed communication multigraph
``split``              evaluation protocols A-C (D is runner-level)
``models``             MLP, GCN, GraphSAGE, GAT, EdgeGAT, E-GAT-IDS
``losses``             weighted cross entropy, focal loss, graph-aware sampling
``train``              training loops, neighbour sampling, early stopping
``evaluate``           metric battery and error analysis
``statistics``         multi-seed aggregation and paired significance tests
``explainability``     attention-based attribution, permutation importance
``visualization``      publication figures on a validated palette
``utils``              determinism, provenance, result persistence
"""

__version__ = "0.1.0"
__all__ = [
    "audit", "data_loader", "evaluate", "explainability", "feature_engineering",
    "graph_builder", "losses", "models", "preprocessing", "split", "statistics",
    "train", "utils", "visualization",
]
