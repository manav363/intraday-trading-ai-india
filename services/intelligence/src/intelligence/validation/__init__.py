"""Validation: purged CV, CPCV, and the statistics that make a number defensible."""

from .cpcv import CombinatorialPurgedCV, CPCVSplit, summarise_paths
from .purged import (
    PurgedSplit,
    PurgedTimeSeriesSplit,
    purged_oof_predictions,
)
from .statistics import (
    PermutationResult,
    brier_score,
    deflated_sharpe_ratio,
    permutation_test,
    probabilistic_sharpe_ratio,
    reliability_curve,
    sharpe_ratio,
    shuffle_labels_within_folds,
)

__all__ = [
    "CPCVSplit",
    "CombinatorialPurgedCV",
    "PermutationResult",
    "PurgedSplit",
    "PurgedTimeSeriesSplit",
    "brier_score",
    "deflated_sharpe_ratio",
    "permutation_test",
    "probabilistic_sharpe_ratio",
    "purged_oof_predictions",
    "reliability_curve",
    "sharpe_ratio",
    "shuffle_labels_within_folds",
    "summarise_paths",
]
