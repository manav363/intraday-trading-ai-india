"""market-data service.

Owns ingestion, the quality gate, corporate-action adjustment, and the parquet
lake. It is the **single writer** to the lake; every other service reads it.
"""

from .calendar import CalendarUnavailable, NSECalendar
from .providers import (
    KNOWN_PROVIDERS,
    HistoryWindowExceeded,
    MarketDataProvider,
    ProviderNotImplemented,
    ProviderUnavailable,
    build_provider,
)

__version__ = "2.0.0"

__all__ = [
    "KNOWN_PROVIDERS",
    "CalendarUnavailable",
    "HistoryWindowExceeded",
    "MarketDataProvider",
    "NSECalendar",
    "ProviderNotImplemented",
    "ProviderUnavailable",
    "__version__",
    "build_provider",
]
