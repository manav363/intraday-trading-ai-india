"""Scheduled jobs.

The important one is `append_session`. The public NSE endpoint serves a rolling
60–90 day window, so the only way this project ever holds more history than that
is by appending every session and keeping what the source later drops. The lake
outgrows its own source, and that is by design rather than by luck.

Jobs report what they did and what they failed to do. A job that silently
skipped ten symbols is indistinguishable from one that had ten fewer symbols to
process.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from intraday_contracts import IST, Interval
from market_data.calendar import CalendarUnavailable, NSECalendar
from market_data.ingest import ingest_symbols
from market_data.providers import MarketDataProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JobReport:
    job: str
    started_at: dt.datetime
    ok: bool
    detail: str
    metrics: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "ok" if self.ok else "FAILED"
        return f"[{status}] {self.job}: {self.detail}"


def append_session(
    provider: MarketDataProvider,
    symbols: list[str],
    *,
    lake_root: Path | str,
    interval: Interval = Interval.M15,
    as_of: dt.date | None = None,
    lookback_days: int = 5,
    calendar: NSECalendar | None = None,
) -> JobReport:
    """Append the most recent sessions to the lake.

    A short lookback rather than one day: the endpoint occasionally backfills a
    session late, and lake writes are idempotent per (interval, symbol, day), so
    re-reading a few days repairs gaps without duplicating anything.
    """
    started = dt.datetime.now(tz=IST)
    calendar = calendar or NSECalendar()
    as_of = as_of or started.date()

    try:
        if not calendar.is_trading_day(as_of):
            return JobReport(
                job="append_session",
                started_at=started,
                ok=True,
                detail=f"{as_of} was not a trading day; nothing to append",
            )
    except CalendarUnavailable as exc:
        # Refuse rather than guess. Assuming the market was open would
        # manufacture a missing-data finding for every symbol.
        return JobReport(job="append_session", started_at=started, ok=False, detail=str(exc))

    report = ingest_symbols(
        provider,
        symbols,
        interval,
        lake_root=lake_root,
        days=lookback_days,
        as_of=as_of,
        calendar=calendar,
    )

    return JobReport(
        job="append_session",
        started_at=started,
        ok=not report.failed,
        detail=report.summary(),
        metrics={
            "rows": report.total_rows,
            "ingested": len(report.ingested),
            "failed": list(report.failed),
            "eligible": list(report.eligible),
        },
    )


def check_drift(
    lake_root: Path | str,
    *,
    interval: Interval = Interval.M15,
    as_of: dt.date | None = None,
    max_stale_days: int = 3,
) -> JobReport:
    """Alert when the lake stops growing.

    A stalled ingest is invisible from the outside: the API keeps serving the
    last model against increasingly old bars, and every log line reads healthy.
    """
    from market_data.store import LakeWriter

    started = dt.datetime.now(tz=IST)
    as_of = as_of or started.date()
    writer = LakeWriter(lake_root)

    symbols = writer.stored_symbols(interval)
    if not symbols:
        return JobReport(job="check_drift", started_at=started, ok=False, detail="lake is empty")

    stale: dict[str, int] = {}
    for symbol in symbols:
        days = writer.stored_days(interval, symbol)
        if not days:
            stale[symbol] = 9999
            continue
        age = (as_of - days[-1]).days
        if age > max_stale_days:
            stale[symbol] = age

    ok = not stale
    detail = (
        f"{len(symbols)} symbols current"
        if ok
        else f"{len(stale)}/{len(symbols)} symbols stale by more than {max_stale_days}d"
    )
    return JobReport(
        job="check_drift",
        started_at=started,
        ok=ok,
        detail=detail,
        metrics={"stale": stale, "symbols": len(symbols)},
    )
