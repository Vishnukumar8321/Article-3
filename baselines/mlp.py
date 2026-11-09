"""MLP baseline -- re-export of the shared implementation.

Kept as a file so the directory matches the project specification. The model
itself lives in ``src.models.mlp`` so that the tabular DL baseline and the
graph models share one training loop, one metric path and one result writer.
"""
from src.models.mlp import build  # noqa: F401
