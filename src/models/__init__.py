"""Model registry.

The keys here are the exact strings that appear in result filenames and in the
manuscript's tables, so a reader can go from a table row to a constructor.
"""

from __future__ import annotations

from typing import Callable, Dict

from . import egat, gat, gcn, graphsage, mlp
from .base import EdgeGNN, EdgeGNNConfig, FlowMLP
from .layers import GATLayer, GCNLayer, SAGELayer

#: name -> builder.  Graph builders take (node_dim, edge_dim, n_classes, **kw);
#: the MLP builder takes (edge_dim, n_classes, **kw).
GRAPH_MODELS: Dict[str, Callable] = {
    "GCN": gcn.build,
    "GraphSAGE": graphsage.build,
    "E-GraphSAGE": graphsage.build_e_graphsage,
    "GAT": gat.build,
    "EdgeGAT": gat.build_edge_gat,
    "E-GAT-IDS": egat.build,
}

TABULAR_NEURAL_MODELS: Dict[str, Callable] = {
    "MLP": mlp.build,
}

#: The ablation ladder of section 25, in order. Read as two parallel rungs:
#: topology-only aggregation (GCN, GraphSAGE, GAT) and the same architectures
#: given the flow attributes (E-GraphSAGE, EdgeGAT, E-GAT-IDS), with the
#: graph-free MLP as the control.
ABLATION_LADDER = ["MLP", "GCN", "GraphSAGE", "GAT",
                   "E-GraphSAGE", "EdgeGAT", "E-GAT-IDS"]

#: Which rung each model sits on, used to facet the ladder figure rather than
#: cycle a seventh hue.
LADDER_GROUP = {
    "MLP": "no graph",
    "GCN": "topology only", "GraphSAGE": "topology only", "GAT": "topology only",
    "E-GraphSAGE": "with edge features", "EdgeGAT": "with edge features",
    "E-GAT-IDS": "with edge features",
}

__all__ = [
    "GRAPH_MODELS",
    "TABULAR_NEURAL_MODELS",
    "ABLATION_LADDER",
    "LADDER_GROUP",
    "EdgeGNN",
    "EdgeGNNConfig",
    "FlowMLP",
    "GATLayer",
    "GCNLayer",
    "SAGELayer",
]
