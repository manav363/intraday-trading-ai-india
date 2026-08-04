"""The parquet lake writer.

`market-data` is the **single writer**. `intelligence` mounts the same directory
read-only. Two writers under one path is how you get two differently-shaped
tables with the same name, where the loser becomes a silent no-op.

Writes are **idempotent per (interval, symbol, day)**: re-ingesting a day
replaces that day's file rather than appending duplicates. This matters because
the ingest job runs daily against a rolling window that overlaps what is already
stored — without idempotence every run would double the overlap.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from intraday_contracts import BAR_COLUMNS, Interval, LakePaths, OHLCVBar, QualityScore


class LakeWriteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WriteReport:
    days_written: int
    rows_written: int
    symbols: tuple[str, ...]


def bars_to_frame(bars: list[OHLCVBar]) -> pd.DataFrame:
    """Canonical column order, enforced. `interval` is a partition key carried
    by the path, not a column repeated a million times."""
    if not bars:
        return pd.DataFrame(columns=list(BAR_COLUMNS))

    frame = pd.DataFrame(
        {
            "timestamp": [b.timestamp for b in bars],
            "symbol": [b.symbol for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "source": [b.source.value for b in bars],
        }
    )
    return frame[list(BAR_COLUMNS)]


class LakeWriter:
    """Writes bars, quality runs, and scores into the lake."""

    def __init__(self, root: Path | str) -> None:
        self.paths = LakePaths(root)

    # -- bars ------------------------------------------------------------

    def write_bars(self, frame: pd.DataFrame, interval: Interval) -> WriteReport:
        if frame.empty:
            return WriteReport(days_written=0, rows_written=0, symbols=())

        missing = [c for c in BAR_COLUMNS if c not in frame.columns]
        if missing:
            raise LakeWriteError(f"frame missing canonical columns {missing}")

        frame = frame[list(BAR_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

        days = 0
        rows = 0
        for (symbol, day), chunk in frame.groupby(
            [frame["symbol"], frame["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date]
        ):
            path = self.paths.bars_file(interval, str(symbol), day)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Replace, never append: the daily job re-reads an overlapping
            # window, and appending would duplicate every overlapping bar.
            chunk.sort_values("timestamp").to_parquet(path, index=False, compression="snappy")
            days += 1
            rows += len(chunk)

        return WriteReport(
            days_written=days,
            rows_written=rows,
            symbols=tuple(sorted(frame["symbol"].unique())),
        )

    def read_bars(
        self,
        interval: Interval,
        symbol: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        """Read back a symbol's bars over a date range."""
        base = self.paths.bars_dir(interval) / f"symbol={symbol}"
        if not base.exists():
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        frames = [
            pd.read_parquet(p)
            for p in sorted(base.rglob("*.parquet"))
            if start.isoformat() <= p.stem <= end.isoformat()
        ]
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def stored_days(self, interval: Interval, symbol: str) -> list[dt.date]:
        base = self.paths.bars_dir(interval) / f"symbol={symbol}"
        if not base.exists():
            return []
        return sorted(dt.date.fromisoformat(p.stem) for p in base.rglob("*.parquet"))

    def stored_symbols(self, interval: Interval) -> list[str]:
        base = self.paths.bars_dir(interval)
        if not base.exists():
            return []
        return sorted(p.name.removeprefix("symbol=") for p in base.iterdir() if p.is_dir())

    # -- quality ---------------------------------------------------------

    def write_quarantine(self, frame: pd.DataFrame, day: dt.date) -> int:
        """Quarantined rows keep the rule that condemned them. Nothing is
        deleted — fix the bug and replay."""
        if frame.empty:
            return 0
        path = self.paths.quarantine(day)
        path.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path / "rows.parquet", index=False)
        return len(frame)

    def write_scores(self, scores: list[QualityScore]) -> int:
        if not scores:
            return 0
        self.paths.quality_scores.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([s.model_dump() for s in scores])
        as_of = scores[0].as_of
        frame.to_parquet(self.paths.quality_scores / f"{as_of.isoformat()}.parquet", index=False)
        return len(scores)

    def read_scores(self, as_of: dt.date) -> pd.DataFrame:
        path = self.paths.quality_scores / f"{as_of.isoformat()}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
