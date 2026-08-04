"""The quality gate: rules, the gate that applies them, and per-symbol scoring."""

from .gate import GateOutcome, run_gate
from .rules import (
    CIRCUIT_LIMIT,
    MIN_SESSION_COVERAGE,
    ROW_RULES,
    SIGMA_THRESHOLD,
    STALE_BAR_THRESHOLD,
    VOLUME_SPIKE_MULTIPLE,
)
from .scoring import MIN_SCORE, eligible_universe, score_symbol

__all__ = [
    "CIRCUIT_LIMIT",
    "MIN_SCORE",
    "MIN_SESSION_COVERAGE",
    "ROW_RULES",
    "SIGMA_THRESHOLD",
    "STALE_BAR_THRESHOLD",
    "VOLUME_SPIKE_MULTIPLE",
    "GateOutcome",
    "eligible_universe",
    "run_gate",
    "score_symbol",
]
