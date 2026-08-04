"""Deterministic synthetic bars.

This is what the test suite runs on, so it has to be realistic in the ways the
pipeline actually depends on:

* bars exist only inside the NSE session, on trading days
* intraday volatility and volume are **U-shaped** — high at the open and the
  close, quiet around midday. A generator with flat intraday volatility makes
  every time-of-day feature look useless and every session-aware bug invisible.
* OHLC geometry is correct by construction
* **regimes vary, including falling ones.** A sibling project registered a
  regime classifier trained on synthetic GBM that had no bear regime in it; the
  model mapped two of three states to UPTREND and none to DOWNTREND, and so was
  structurally unable to see a falling market. A generator that only goes up
  produces models that only know up.

Determinism is per `(symbol, day)` rather than per stream, so fetching a date
range in two calls returns identical bars for the overlapping days.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from typing import ClassVar

import numpy as np
from intraday_contracts import IST, BarSource, Interval, OHLCVBar

from ..calendar import NSECalendar
from .base import MarketDataProvider

# Annualised vol targets, converted to per-bar below. Chosen to bracket what
# NSE large-caps actually do (roughly 18%-45% annualised).
_ANNUAL_VOL_LOW = 0.18
_ANNUAL_VOL_HIGH = 0.45
_TRADING_DAYS = 252

# Regime drift, expressed as annualised return. The bear case is deliberately
# as strong as the bull case.
_REGIME_DRIFT = {
    "bull": 0.35,
    "ranging": 0.0,
    "bear": -0.35,
}
_REGIME_LENGTH_DAYS = 21
"""Regimes persist about a month, so a 60-day fetch spans two or three."""


def _stable_seed(*parts: object) -> int:
    """A stable 64-bit seed from arbitrary parts.

    `hash()` is salted per process, so it cannot be used for reproducibility.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _intraday_shape(fraction_through_session: float) -> float:
    """U-shaped multiplier for volatility and volume.

    Returns ~2.0 at the open, ~0.6 at midday, ~1.6 into the close — the shape
    every intraday equity market shows.
    """
    x = fraction_through_session
    return 0.6 + 1.4 * math.exp(-8.0 * x) + 1.0 * math.exp(-8.0 * (1.0 - x))


class SyntheticProvider(MarketDataProvider):
    """Generates bars. Never permitted to train a served model."""

    name: ClassVar[str] = "synthetic"
    source: ClassVar[BarSource] = BarSource.SYNTHETIC

    def __init__(self, seed: int = 0, calendar: NSECalendar | None = None) -> None:
        self.seed = seed
        self.calendar = calendar or NSECalendar()

    def max_history_days(self, interval: Interval) -> int:
        """Unbounded — it is generated. Kept large rather than infinite so that
        window-limit handling is still exercised in tests."""
        return 36_500

    # -- price process --------------------------------------------------

    def _base_price(self, symbol: str) -> float:
        rng = np.random.default_rng(_stable_seed("base", symbol, self.seed))
        return float(rng.uniform(150.0, 3500.0))

    def _annual_vol(self, symbol: str) -> float:
        rng = np.random.default_rng(_stable_seed("vol", symbol, self.seed))
        return float(rng.uniform(_ANNUAL_VOL_LOW, _ANNUAL_VOL_HIGH))

    def _regime_for(self, symbol: str, day: dt.date) -> str:
        """Which regime `day` falls in. Blocks of ~21 trading days."""
        block = day.toordinal() // _REGIME_LENGTH_DAYS
        rng = np.random.default_rng(_stable_seed("regime", symbol, block, self.seed))
        return str(rng.choice(list(_REGIME_DRIFT)))

    def _day_open(self, symbol: str, day: dt.date) -> float:
        """Opening price for a day, built by compounding regime drift from a
        fixed epoch so that consecutive days join up continuously."""
        base = self._base_price(symbol)
        annual_vol = self._annual_vol(symbol)
        epoch = dt.date(2024, 1, 1)

        days_elapsed = max(0, (day - epoch).days)
        rng = np.random.default_rng(_stable_seed("path", symbol, day, self.seed))

        drift = _REGIME_DRIFT[self._regime_for(symbol, day)] / _TRADING_DAYS
        daily_vol = annual_vol / math.sqrt(_TRADING_DAYS)

        # Deterministic slow component + a day-specific shock.
        trend = drift * (days_elapsed % (_REGIME_LENGTH_DAYS * 6))
        shock = float(rng.normal(0.0, daily_vol)) * math.sqrt(days_elapsed % 30 + 1)
        return float(base * math.exp(trend + shock))

    # -- fetch ----------------------------------------------------------

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[OHLCVBar]:
        bars: list[OHLCVBar] = []
        for day in self.calendar.trading_days(start.date(), end.date()):
            bars.extend(self._session_bars(symbol, interval, day))
        return [b for b in bars if start <= b.timestamp <= end]

    def _session_bars(self, symbol: str, interval: Interval, day: dt.date) -> list[OHLCVBar]:
        n_bars = interval.bars_per_session
        if not interval.is_intraday:
            n_bars = 1

        rng = np.random.default_rng(_stable_seed("session", symbol, day, interval.value, self.seed))
        annual_vol = self._annual_vol(symbol)
        per_bar_vol = annual_vol / math.sqrt(_TRADING_DAYS * max(n_bars, 1))
        drift_per_bar = _REGIME_DRIFT[self._regime_for(symbol, day)] / (_TRADING_DAYS * n_bars)

        base_volume = float(rng.uniform(20_000, 400_000)) / max(n_bars, 1)
        price = self._day_open(symbol, day)
        out: list[OHLCVBar] = []

        for i in range(n_bars):
            shape = _intraday_shape(i / n_bars if n_bars > 1 else 0.5)
            bar_vol = per_bar_vol * shape

            open_ = price
            close = open_ * math.exp(drift_per_bar + float(rng.normal(0.0, bar_vol)))

            # Wicks extend beyond the body by a positive amount, so geometry
            # holds by construction rather than by a later repair step.
            body_hi, body_lo = max(open_, close), min(open_, close)
            high = body_hi * (1.0 + abs(float(rng.normal(0.0, bar_vol))) * 0.6)
            low = body_lo * (1.0 - abs(float(rng.normal(0.0, bar_vol))) * 0.6)

            volume = base_volume * shape * float(rng.uniform(0.6, 1.5))

            out.append(
                OHLCVBar(
                    symbol=symbol,
                    timestamp=self._bar_time(day, interval, i),
                    interval=interval,
                    open=round(open_, 2),
                    high=round(max(high, body_hi), 2),
                    low=round(min(low, body_lo), 2),
                    close=round(close, 2),
                    volume=float(round(volume)),
                    source=BarSource.SYNTHETIC,
                )
            )
            price = close

        return out

    @staticmethod
    def _bar_time(day: dt.date, interval: Interval, index: int) -> dt.datetime:
        """Bar OPEN time. Intraday bars start at 09:15; a daily bar is stamped
        at the open too, so daily and intraday share one convention."""
        open_dt = dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST)
        if not interval.is_intraday:
            return open_dt
        return open_dt + dt.timedelta(minutes=interval.minutes * index)
