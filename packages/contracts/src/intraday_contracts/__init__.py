"""Shared contracts for the intraday trading AI services.

This package is the *only* thing the services import from one another. Everything
else stays behind a service boundary.
"""

from .errors import ContractViolation, ErrorCode, ErrorEnvelope
from .intelligence import (
    BarrierOutcome,
    CPCVResult,
    Direction,
    ExplanationFact,
    ModelCard,
    Prediction,
    Side,
)
from .lake import (
    BAR_COLUMNS,
    CORP_ACTION_COLUMNS,
    EOD_COLUMNS,
    QUALITY_SCORE_COLUMNS,
    SCHEMA_VERSION,
    UNIVERSE_COLUMNS,
    LakePaths,
)
from .market_data import (
    IST,
    NSE_SESSION_CLOSE,
    NSE_SESSION_MINUTES,
    NSE_SESSION_OPEN,
    BarSource,
    Interval,
    OHLCVBar,
    SymbolMeta,
    UniverseSnapshot,
)
from .quality import QualityRun, QualityScore, RuleResult, Severity, Tier, Verdict

__version__ = "2.0.0"

__all__ = [
    "BAR_COLUMNS",
    "CORP_ACTION_COLUMNS",
    "EOD_COLUMNS",
    "IST",
    "NSE_SESSION_CLOSE",
    "NSE_SESSION_MINUTES",
    "NSE_SESSION_OPEN",
    "QUALITY_SCORE_COLUMNS",
    "SCHEMA_VERSION",
    "UNIVERSE_COLUMNS",
    "BarSource",
    "BarrierOutcome",
    "CPCVResult",
    "ContractViolation",
    "Direction",
    "ErrorCode",
    "ErrorEnvelope",
    "ExplanationFact",
    "Interval",
    "LakePaths",
    "ModelCard",
    "OHLCVBar",
    "Prediction",
    "QualityRun",
    "QualityScore",
    "RuleResult",
    "Severity",
    "Side",
    "SymbolMeta",
    "Tier",
    "UniverseSnapshot",
    "Verdict",
    "__version__",
]
