"""GAT baselines -- re-export of ``src.models.gat``.

``build`` is the topology-only attention baseline (model D);
``build_edge_gat`` is the edge-in-message variant (model E).
"""
from src.models.gat import build, build_edge_gat  # noqa: F401
