from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from intraday_contracts import BAR_COLUMNS, IST, Interval
from market_data.calendar import NSECalendar
from market_data.ingest.pipeline import ingest_symbols
from market_data.providers import SyntheticProvider
from market_data.store import LakeWriter, bars_to_frame

START = dt.datetime(2025, 3, 3, 0, 0, tzinfo=IST)
END = dt.datetime(2025, 3, 7, 23, 59, tzinfo=IST)


@pytest.fixture
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=11)


@pytest.fixture
def writer(tmp_path: Path) -> LakeWriter:
    return LakeWriter(tmp_path / "lake")


def test_bars_to_frame_uses_canonical_column_order(provider: SyntheticProvider) -> None:
    bars = provider.fetch_bars("RELIANCE", Interval.M15, START, END)
    frame = bars_to_frame(bars)
    assert list(frame.columns) == list(BAR_COLUMNS)


def test_empty_bars_still_yield_canonical_columns() -> None:
    assert list(bars_to_frame([]).columns) == list(BAR_COLUMNS)


def test_round_trip_preserves_rows(provider: SyntheticProvider, writer: LakeWriter) -> None:
    bars = provider.fetch_bars("RELIANCE", Interval.M15, START, END)
    report = writer.write_bars(bars_to_frame(bars), Interval.M15)

    assert report.days_written == 5
    assert report.rows_written == len(bars)

    back = writer.read_bars(Interval.M15, "RELIANCE", START.date(), END.date())
    assert len(back) == len(bars)
    assert back["close"].tolist() == pytest.approx([b.close for b in bars])


def test_writes_are_idempotent_not_appending(
    provider: SyntheticProvider, writer: LakeWriter
) -> None:
    """The daily job re-reads an overlapping window. Appending would double
    every overlapping bar on every run."""
    frame = bars_to_frame(provider.fetch_bars("TCS", Interval.M15, START, END))

    writer.write_bars(frame, Interval.M15)
    writer.write_bars(frame, Interval.M15)
    writer.write_bars(frame, Interval.M15)

    back = writer.read_bars(Interval.M15, "TCS", START.date(), END.date())
    assert len(back) == len(frame)


def test_partition_layout_matches_the_contract(
    provider: SyntheticProvider, writer: LakeWriter
) -> None:
    frame = bars_to_frame(provider.fetch_bars("INFY", Interval.M5, START, END))
    writer.write_bars(frame, Interval.M5)

    expected = writer.paths.root / "bars/5m/symbol=INFY/year=2025/month=03/2025-03-03.parquet"
    assert expected.exists()


def test_stored_days_and_symbols(provider: SyntheticProvider, writer: LakeWriter) -> None:
    for symbol in ("RELIANCE", "TCS"):
        writer.write_bars(
            bars_to_frame(provider.fetch_bars(symbol, Interval.M15, START, END)), Interval.M15
        )

    assert writer.stored_symbols(Interval.M15) == ["RELIANCE", "TCS"]
    assert writer.stored_days(Interval.M15, "TCS") == [
        dt.date(2025, 3, 3),
        dt.date(2025, 3, 4),
        dt.date(2025, 3, 5),
        dt.date(2025, 3, 6),
        dt.date(2025, 3, 7),
    ]


def test_reading_an_absent_symbol_returns_empty_not_error(writer: LakeWriter) -> None:
    out = writer.read_bars(Interval.M15, "NOPE", START.date(), END.date())
    assert out.empty
    assert list(out.columns) == list(BAR_COLUMNS)


def test_missing_canonical_column_raises(writer: LakeWriter, provider: SyntheticProvider) -> None:
    frame = bars_to_frame(provider.fetch_bars("ITC", Interval.M15, START, END))
    with pytest.raises(Exception, match="missing canonical columns"):
        writer.write_bars(frame.drop(columns=["source"]), Interval.M15)


# ------------------------------------------------------------- pipeline


def test_ingest_pipeline_end_to_end(tmp_path: Path, provider: SyntheticProvider) -> None:
    report = ingest_symbols(
        provider,
        ["RELIANCE", "TCS", "INFY"],
        Interval.M15,
        lake_root=tmp_path / "lake",
        days=20,
        as_of=dt.date(2025, 3, 7),
        calendar=NSECalendar(),
    )

    assert set(report.ingested) == {"RELIANCE", "TCS", "INFY"}
    assert report.total_rows > 300
    assert not report.failed
    assert len(report.scores) == 3

    writer = LakeWriter(tmp_path / "lake")
    assert writer.stored_symbols(Interval.M15) == ["INFY", "RELIANCE", "TCS"]


def test_pipeline_records_failures_rather_than_skipping_silently(tmp_path: Path) -> None:
    """A run reporting 2/3 with the failing name listed is actionable; one
    quietly reporting 2 is not."""

    class OneBadSymbol(SyntheticProvider):
        def fetch_bars(self, symbol, interval, start, end):  # type: ignore[no-untyped-def]
            if symbol == "BROKEN":
                return []
            return super().fetch_bars(symbol, interval, start, end)

    report = ingest_symbols(
        OneBadSymbol(seed=5),
        ["RELIANCE", "BROKEN"],
        Interval.M15,
        lake_root=tmp_path / "lake",
        days=10,
        as_of=dt.date(2025, 3, 7),
    )

    assert "RELIANCE" in report.ingested
    assert "BROKEN" in report.failed
    assert "no bars" in report.failed["BROKEN"]


def test_scores_are_persisted_and_readable(tmp_path: Path, provider: SyntheticProvider) -> None:
    as_of = dt.date(2025, 3, 7)
    ingest_symbols(
        provider,
        ["RELIANCE"],
        Interval.M15,
        lake_root=tmp_path / "lake",
        days=20,
        as_of=as_of,
    )
    scores = LakeWriter(tmp_path / "lake").read_scores(as_of)
    assert len(scores) == 1
    assert scores.iloc[0]["symbol"] == "RELIANCE"
    assert "eligible" in scores.columns
