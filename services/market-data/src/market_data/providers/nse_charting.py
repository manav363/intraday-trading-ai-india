"""NSE public charting endpoint, via the OpenChart client (MIT, no auth).

See `docs/adr/0002-data-sources.md` for why this source and not yfinance.

Two things this adapter is careful about:

* **The rolling window is real.** Minute bars are served for roughly the last
  60 days, 5-minute for ~90. Asking for more does not error upstream — it
  quietly returns less, which would make the lake's start date a lie. So the
  window is checked here and exceeding it raises.

* **Column names are normalised defensively.** The upstream response shape is
  not a published contract and has changed across releases. Rather than index
  fixed column names and fail with a KeyError months later, the frame is
  normalised case-insensitively and a missing required column raises with the
  columns it actually saw.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar

from intraday_contracts import IST, BarSource, Interval, OHLCVBar

from .base import (
    MarketDataProvider,
    ProviderHealth,
    ProviderNotImplemented,
    ProviderUnavailable,
)

if TYPE_CHECKING:
    import pandas as pd

# Measured against the public endpoint. Conservative — under-claiming costs a
# little history, over-claiming silently truncates.
_MAX_HISTORY_DAYS: dict[Interval, int] = {
    Interval.M1: 55,
    Interval.M5: 85,
    Interval.M15: 85,
    Interval.M30: 85,
    Interval.H1: 85,
    Interval.D1: 3650,
}

_REQUIRED = ("open", "high", "low", "close", "volume")
_TIMESTAMP_ALIASES = ("timestamp", "date", "datetime", "time", "index")


class NSEChartingProvider(MarketDataProvider):
    """Intraday and daily bars from NSE's own public charting API."""

    name: ClassVar[str] = "nse_charting"
    source: ClassVar[BarSource] = BarSource.NSE_CHARTING

    def __init__(self, segment: str = "EQ") -> None:
        try:
            from openchart import NSEData
        except ImportError as exc:  # pragma: no cover - depends on install state
            # Refuse at construction. Never degrade to synthetic: that is how
            # generated bars reach a real screen with every log line healthy.
            raise ProviderNotImplemented(
                "provider 'nse_charting' selected but the 'openchart' package is "
                "not installed. Install it or select a different provider — this "
                "will not fall back to synthetic data."
            ) from exc

        self._client = NSEData()
        self.segment = segment

    def max_history_days(self, interval: Interval) -> int:
        return _MAX_HISTORY_DAYS[interval]

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name,
            available=self._client is not None,
            detail=f"segment={self.segment}",
        )

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[OHLCVBar]:
        self.check_window(interval, start, dt.datetime.now(tz=IST))

        try:
            frame = self._client.historical(
                symbol=symbol,
                segment=self.segment,
                start=start.replace(tzinfo=None),
                end=end.replace(tzinfo=None),
                interval=interval.value,
            )
        except Exception as exc:
            raise ProviderUnavailable(f"NSE charting fetch failed for {symbol}: {exc}") from exc

        if frame is None or len(frame) == 0:
            # An empty range is a legitimate answer (holiday, suspension). It is
            # NOT the same as a failure, and the two must not collapse.
            return []

        return self._to_bars(frame, symbol, interval)

    # -- normalisation ---------------------------------------------------

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reset_index()
        out.columns = [str(c).strip().lower() for c in out.columns]
        return out

    def _to_bars(self, frame: pd.DataFrame, symbol: str, interval: Interval) -> list[OHLCVBar]:
        import pandas as pd

        df = self._normalise(frame)

        ts_col = next((c for c in _TIMESTAMP_ALIASES if c in df.columns), None)
        if ts_col is None:
            raise ProviderUnavailable(
                f"no timestamp column for {symbol}; saw columns {list(df.columns)}"
            )

        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ProviderUnavailable(
                f"missing columns {missing} for {symbol}; saw {list(df.columns)}"
            )

        bars: list[OHLCVBar] = []
        for row in df.itertuples(index=False):
            record: dict[str, Any] = dict(zip(df.columns, row, strict=True))
            timestamp = pd.Timestamp(record[ts_col])
            timestamp = (
                timestamp.tz_localize(IST)
                if timestamp.tzinfo is None
                else timestamp.tz_convert(IST)
            )

            try:
                bars.append(
                    OHLCVBar(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        interval=interval,
                        open=float(record["open"]),
                        high=float(record["high"]),
                        low=float(record["low"]),
                        close=float(record["close"]),
                        volume=float(record["volume"]),
                        source=BarSource.NSE_CHARTING,
                    )
                )
            except ValueError:
                # A row that violates bar geometry is bad upstream data, not a
                # crash. It is dropped here and the quality gate counts the gap;
                # repairing it silently would hide a real feed problem.
                continue

        bars.sort(key=lambda b: b.timestamp)
        return bars
