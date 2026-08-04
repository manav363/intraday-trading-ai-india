"""Ingest orchestration: provider -> quality gate -> lake.

This is the append-only accumulation loop that makes the project's history
outgrow the source's rolling window. The public endpoint serves ~60-90 days; run
this daily for a year and the lake holds a year.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from intraday_contracts import IST, Interval, QualityScore, Verdict

from ..calendar import NSECalendar
from ..providers import MarketDataProvider, ProviderUnavailable
from ..quality import run_gate, score_symbol
from ..store import LakeWriter, bars_to_frame

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestReport:
    interval: Interval
    requested: tuple[str, ...]
    ingested: dict[str, int] = field(default_factory=dict)
    quarantined: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    scores: list[QualityScore] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.ingested.values())

    @property
    def eligible(self) -> tuple[str, ...]:
        return tuple(sorted(s.symbol for s in self.scores if s.eligible))

    def summary(self) -> str:
        return (
            f"{self.interval.value}: {len(self.ingested)}/{len(self.requested)} symbols, "
            f"{self.total_rows} rows, {sum(self.quarantined.values())} quarantined, "
            f"{len(self.eligible)} eligible, {len(self.failed)} failed"
        )


def ingest_symbols(
    provider: MarketDataProvider,
    symbols: list[str],
    interval: Interval,
    *,
    lake_root: Path | str,
    days: int = 30,
    as_of: dt.date | None = None,
    calendar: NSECalendar | None = None,
) -> IngestReport:
    """Fetch, gate, and store bars for each symbol.

    One symbol failing does not abort the batch — but it is recorded in
    `failed`, not silently skipped. A run that reports 40/50 symbols with the
    ten names listed is actionable; one that quietly reports 40 is not.
    """
    calendar = calendar or NSECalendar()
    writer = LakeWriter(lake_root)
    as_of = as_of or dt.datetime.now(tz=IST).date()

    start = dt.datetime.combine(as_of - dt.timedelta(days=days), dt.time(0, 0), tzinfo=IST)
    end = dt.datetime.combine(as_of, dt.time(23, 59), tzinfo=IST)
    expected_sessions = calendar.count_trading_days(start.date(), as_of)

    report = IngestReport(interval=interval, requested=tuple(symbols))

    for symbol in symbols:
        try:
            bars = provider.fetch_bars(symbol, interval, start, end)
        except ProviderUnavailable as exc:
            report.failed[symbol] = str(exc)
            logger.warning("provider unavailable for %s: %s", symbol, exc)
            continue

        if not bars:
            report.failed[symbol] = "no bars returned for the requested range"
            continue

        frame = bars_to_frame(bars)
        outcome = run_gate(frame, interval, source_ref=f"{provider.name}:{symbol}")

        if outcome.verdict is Verdict.REJECTED:
            report.failed[symbol] = "quality gate rejected the batch"
            continue

        if not outcome.quarantined.empty:
            written = writer.write_quarantine(outcome.quarantined, as_of)
            report.quarantined[symbol] = written

        written = writer.write_bars(outcome.clean, interval)
        report.ingested[symbol] = written.rows_written

        report.scores.append(
            score_symbol(
                outcome.clean,
                symbol,
                interval,
                as_of=as_of,
                expected_sessions=expected_sessions,
                quarantined_rows=report.quarantined.get(symbol, 0),
            )
        )

    writer.write_scores(report.scores)
    logger.info(report.summary())
    return report
