"""Market data contracts.

Design notes worth keeping:

* **Prices are `float`, not `Decimal`.** This is a research system — no money
  moves, no ledger balances. Parquet and every numerical library work in
  float64, and forcing Decimal through pandas costs a great deal for a
  precision guarantee nothing here needs. (The ai_trade rule that money is
  Decimal applies to a ledger; this is not one.)

* **Provenance is required, never defaulted.** `source` has no default. A bar
  whose origin is unknown cannot be constructed, so "unknown provenance is
  treated as synthetic" is enforced by the type rather than by a convention
  somebody remembers.

* **Validity is enforced here, at the boundary.** high >= low and friends are
  invariants of a bar, not opinions a downstream quality rule holds. Tier-2
  checks in the quality gate exist for data arriving as raw rows; anything that
  becomes an `OHLCVBar` has already passed them.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Interval(StrEnum):
    """Bar intervals. Values match what the NSE charting endpoint accepts."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        """Bar duration in minutes. `D1` is one NSE equity session."""
        return {
            Interval.M1: 1,
            Interval.M5: 5,
            Interval.M15: 15,
            Interval.M30: 30,
            Interval.H1: 60,
            Interval.D1: NSE_SESSION_MINUTES,
        }[self]

    @property
    def is_intraday(self) -> bool:
        return self is not Interval.D1

    @property
    def bars_per_session(self) -> int:
        """How many bars one full NSE equity session yields at this interval.

        Used to size the panel and to sanity-check ingested day counts.
        """
        return max(1, NSE_SESSION_MINUTES // self.minutes)


# NSE equity session: 09:15–15:30 IST inclusive of the opening bar = 375 minutes.
NSE_SESSION_MINUTES = 375
NSE_SESSION_OPEN = dt.time(9, 15)
NSE_SESSION_CLOSE = dt.time(15, 30)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30), name="IST")


class BarSource(StrEnum):
    """Where a bar came from. Travels on the row, per-bar.

    Provenance you have to look up separately is provenance that will not be
    looked up.
    """

    NSE_CHARTING = "nse_charting"
    """NSE's own public charting endpoint, via the OpenChart client."""

    BHAVCOPY = "bhavcopy"
    """NSE's official published end-of-day file."""

    REPLAY = "replay"
    """A committed fixture slice. Real data, frozen, for offline tests."""

    SYNTHETIC = "synthetic"
    """Generated. Never permitted to reach a trained production model."""

    @property
    def is_synthetic(self) -> bool:
        return self is BarSource.SYNTHETIC

    @property
    def is_real_market_data(self) -> bool:
        """True for sources that came from the exchange at some point.

        `REPLAY` counts — it is real data, merely frozen.
        """
        return self is not BarSource.SYNTHETIC


class OHLCVBar(BaseModel):
    """One bar. Validity invariants are enforced on construction."""

    model_config = {"frozen": True}

    symbol: str = Field(min_length=1)
    timestamp: dt.datetime = Field(description="Bar OPEN time, timezone-aware, IST.")
    interval: Interval

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    source: BarSource
    """Required. There is deliberately no default — see module docstring."""

    @model_validator(mode="after")
    def _check_bar_geometry(self) -> OHLCVBar:
        if self.high < self.low:
            raise ValueError(
                f"high {self.high} < low {self.low} for {self.symbol} @ {self.timestamp}"
            )
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high {self.high} below body max {max(self.open, self.close)} "
                f"for {self.symbol} @ {self.timestamp}"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low {self.low} above body min {min(self.open, self.close)} "
                f"for {self.symbol} @ {self.timestamp}"
            )
        return self

    @model_validator(mode="after")
    def _check_timezone_aware(self) -> OHLCVBar:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"naive timestamp for {self.symbol}; bars must carry IST offset "
                "(a naive intraday timestamp silently shifts the session)"
            )
        return self

    @property
    def is_synthetic(self) -> bool:
        return self.source.is_synthetic

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def turnover(self) -> float:
        """Approximate traded value. Used for dollar bars and liquidity screens."""
        return self.typical_price * self.volume


class SymbolMeta(BaseModel):
    """Reference data for one symbol."""

    model_config = {"frozen": True}

    symbol: str = Field(min_length=1)
    name: str
    isin: str | None = None
    sector: str | None = Field(
        default=None,
        description="Used for sector-neutral z-scores. None means the symbol is "
        "excluded from sector neutralisation rather than pooled into a bogus bucket.",
    )
    listing_date: dt.date | None = None
    delisting_date: dt.date | None = Field(
        default=None,
        description="Set means the symbol stopped trading. Its presence in "
        "history is correct and must not be treated as missing data.",
    )

    def was_listed_on(self, day: dt.date) -> bool:
        before_listing = self.listing_date is not None and day < self.listing_date
        after_delisting = self.delisting_date is not None and day >= self.delisting_date
        return not (before_listing or after_delisting)


class UniverseSnapshot(BaseModel):
    """Point-in-time universe membership.

    Point-in-time is the whole reason this type exists. Building a universe from
    today's index membership makes every delisted company invisible and inflates
    every backtest by an unknown amount.
    """

    model_config = {"frozen": True}

    as_of: dt.date
    index: str = Field(description="e.g. 'NIFTY50'. Free-form; not validated against a registry.")
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)
