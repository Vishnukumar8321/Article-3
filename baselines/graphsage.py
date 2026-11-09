"""GraphSAGE baselines -- re-export of ``src.models.graphsage``.

``build`` is topology-only; ``build_e_graphsage`` is the published E-GraphSAGE
configuration (Lo et al., NOMS 2022).
"""
from src.models.graphsage import build, build_e_graphsage  # noqa: F401
