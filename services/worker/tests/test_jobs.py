from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from intraday_contracts import Interval
from market_data.calendar import NSECalendar
from market_data.providers import SyntheticProvider
from worker import append_session, check_drift

SYMBOLS = ["AAA", "BBB"]


@pytest.fixture
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=21)


def test_append_writes_sessions(tmp_path: Path, provider: SyntheticProvider) -> None:
    report = append_session(
        provider, SYMBOLS, lake_root=tmp_path / "lake", as_of=dt.date(2025, 3, 7)
    )
    assert report.ok
    assert report.metrics["rows"] > 0


def test_append_is_idempotent(tmp_path: Path, provider: SyntheticProvider) -> None:
    """The job re-reads an overlapping window every run; appending rather than
    replacing would double every overlapping bar daily."""
    from market_data.store import LakeWriter

    for _ in range(3):
        append_session(provider, ["AAA"], lake_root=tmp_path / "lake", as_of=dt.date(2025, 3, 7))

    writer = LakeWriter(tmp_path / "lake")
    back = writer.read_bars(Interval.M15, "AAA", dt.date(2025, 3, 3), dt.date(2025, 3, 7))
    assert len(back) == len(back.drop_duplicates(subset=["timestamp"]))


def test_non_trading_day_is_a_success_not_a_failure(
    tmp_path: Path, provider: SyntheticProvider
) -> None:
    report = append_session(
        provider, SYMBOLS, lake_root=tmp_path / "lake", as_of=dt.date(2025, 12, 25)
    )
    assert report.ok
    assert "not a trading day" in report.detail


def test_unknown_calendar_year_refuses_rather_than_guessing(
    tmp_path: Path, provider: SyntheticProvider
) -> None:
    report = append_session(
        provider,
        SYMBOLS,
        lake_root=tmp_path / "lake",
        as_of=dt.date(2031, 6, 10),
        calendar=NSECalendar(),
    )
    assert not report.ok
    assert "holiday list" in report.detail


def test_drift_flags_an_empty_lake(tmp_path: Path) -> None:
    (tmp_path / "lake").mkdir(parents=True)
    report = check_drift(tmp_path / "lake")
    assert not report.ok
    assert "empty" in report.detail


def test_drift_flags_a_stalled_ingest(tmp_path: Path, provider: SyntheticProvider) -> None:
    """A stalled ingest is invisible from outside: the API keeps serving the
    last model against increasingly old bars."""
    append_session(provider, ["AAA"], lake_root=tmp_path / "lake", as_of=dt.date(2025, 3, 7))

    fresh = check_drift(tmp_path / "lake", as_of=dt.date(2025, 3, 8))
    stale = check_drift(tmp_path / "lake", as_of=dt.date(2025, 4, 30))

    assert fresh.ok
    assert not stale.ok
    assert "stale" in stale.detail
