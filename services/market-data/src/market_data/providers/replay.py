"""Replays bars already ingested into the local lake.

This is the offline development path: ingest once from the live endpoint, then
work against the frozen result without touching the network again.

The data it reads is **never committed** — it is whatever the operator has in
their own lake. Tests do not use this provider; they use `synthetic`, so that no
NSE data has to enter the repository at all.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar

from intraday_contracts import IST, BarSource, Interval, LakePaths, OHLCVBar

from ..calendar import NSECalendar
from .base import MarketDataProvider, ProviderHealth, ProviderUnavailable


class ReplayProvider(MarketDataProvider):
    """Serves bars from a local lake directory."""

    name: ClassVar[str] = "replay"
    source: ClassVar[BarSource] = BarSource.REPLAY

    def __init__(self, lake_root: Path | str, calendar: NSECalendar | None = None) -> None:
        self.paths = LakePaths(lake_root)
        self.calendar = calendar or NSECalendar()
        if not self.paths.root.exists():
            raise ProviderUnavailable(
                f"replay provider points at {self.paths.root}, which does not exist. "
                "Ingest once with a live provider first."
            )

    def max_history_days(self, interval: Interval) -> int:
        """Whatever has accumulated. The lake is the deep-history answer — it
        outgrows the live endpoint's rolling window precisely because it keeps
        what the endpoint later drops."""
        return 36_500

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name,
            available=self.paths.root.exists(),
            detail=f"lake={self.paths.root}",
        )

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[OHLCVBar]:
        import pandas as pd

        frames = []
        for day in self.calendar.trading_days(start.date(), end.date()):
            path = self.paths.bars_file(interval, symbol, day)
            if path.exists():
                frames.append(pd.read_parquet(path))

        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        bars = [
            OHLCVBar(
                symbol=symbol,
                timestamp=pd.Timestamp(r.timestamp).tz_convert(IST).to_pydatetime(),
                interval=interval,
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=float(r.volume),
                source=BarSource(r.source),
            )
            for r in df.itertuples(index=False)
        ]
        bars = [b for b in bars if start <= b.timestamp <= end]
        bars.sort(key=lambda b: b.timestamp)
        return bars
