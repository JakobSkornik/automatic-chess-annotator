"""Guid-style weighted positional feature vector (the Knowledge Module).

Pure python-chess; computed for every ply. Split into: weights (config + model),
geometry (board helpers), vector (the feature vector), squares (grounded lookups).
"""

from __future__ import annotations

from .squares import (
    bad_bishop_squares,
    doubled_pawn_files,
    outpost_squares,
    passed_pawn_squares,
    rooks_on_seventh_squares,
)
from .vector import compute_feature_vector, compute_feature_vector_fen, vector_to_plain
from .weights import CHART_FEATURES, FeatureValue, FeatureVector

__all__ = [
    "CHART_FEATURES",
    "FeatureValue",
    "FeatureVector",
    "bad_bishop_squares",
    "compute_feature_vector",
    "compute_feature_vector_fen",
    "doubled_pawn_files",
    "outpost_squares",
    "passed_pawn_squares",
    "rooks_on_seventh_squares",
    "vector_to_plain",
]
