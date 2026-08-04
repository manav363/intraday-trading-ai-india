"""The parquet lake layout, published as a contract.

Bulk data does not cross the network. `market-data` is the single writer;
`intelligence` reads these paths directly. Moving a million rows of OHLCV as
JSON to train a model is minutes of serialisation for no benefit — in
production ML systems the feature store *is* the interface.

Because the layout is the interface, it lives here with a schema-freeze test.
A column rename or a partition change then breaks both services at build time
rather than at integration time.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final

from .market_data import Interval

# Canonical column sets. The freeze test pins these; changing one is a
# deliberate, visible act.

BAR_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)

EOD_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "symbol",
    "series",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover",
    "trades",
    "delivery_qty",
    "delivery_pct",
    "isin",
    "source",
)
"""`delivery_pct` is a genuine institutional-participation signal and it comes
free with the delivery bhavcopy — v1 never had it."""

CORP_ACTION_COLUMNS: Final[tuple[str, ...]] = (
    "ex_date",
    "symbol",
    "action_type",
    "ratio_from",
    "ratio_to",
    "adj_factor",
)

QUALITY_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "as_of",
    "symbol",
    "score",
    "history_completeness",
    "validity_clean_rate",
    "continuity_clean_rate",
    "liquidity_adequacy",
    "recency",
    "eligible",
)

UNIVERSE_COLUMNS: Final[tuple[str, ...]] = ("as_of", "index", "symbol")


class LakePaths:
    """Resolves lake partition paths. Layout is part of the contract.

    Intraday bars partition by ``interval / symbol / year=YYYY / month=MM``.
    Symbol comes before the date partitions because every read is
    symbol-scoped or universe-scoped, and putting it first lets a single-symbol
    read touch one directory instead of scanning every month.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- intraday bars -------------------------------------------------

    def bars_dir(self, interval: Interval) -> Path:
        return self.root / "bars" / interval.value

    def bars_partition(self, interval: Interval, symbol: str, day: dt.date) -> Path:
        return (
            self.bars_dir(interval)
            / f"symbol={symbol}"
            / f"year={day.year:04d}"
            / f"month={day.month:02d}"
        )

    def bars_file(self, interval: Interval, symbol: str, day: dt.date) -> Path:
        return self.bars_partition(interval, symbol, day) / f"{day.isoformat()}.parquet"

    # -- end of day ----------------------------------------------------

    def eod_partition(self, day: dt.date) -> Path:
        return self.root / "eod" / f"year={day.year:04d}"

    def eod_file(self, day: dt.date) -> Path:
        return self.eod_partition(day) / f"{day.isoformat()}.parquet"

    # -- reference -----------------------------------------------------

    @property
    def corp_actions(self) -> Path:
        return self.root / "corp_actions"

    @property
    def symbols_meta(self) -> Path:
        return self.root / "reference" / "symbols.parquet"

    def universe_pit(self, day: dt.date) -> Path:
        return self.root / "universe_pit" / f"as_of={day.isoformat()}" / "members.parquet"

    # -- quality -------------------------------------------------------

    @property
    def quality_runs(self) -> Path:
        return self.root / "quality" / "runs"

    @property
    def quality_scores(self) -> Path:
        return self.root / "quality" / "scores"

    def quarantine(self, day: dt.date) -> Path:
        """Quarantined rows keep their failing rule attached.

        Nothing is deleted — fix an adjustment bug and the quarantine can be
        replayed.
        """
        return self.root / "quality" / "quarantine" / f"date={day.isoformat()}"

    # -- models --------------------------------------------------------

    @property
    def models(self) -> Path:
        return self.root / "models"

    def model_dir(self, model_id: str) -> Path:
        return self.models / model_id


SCHEMA_VERSION: Final[str] = "2.0.0"
"""Bump on any change to the column tuples or the partition layout above."""
