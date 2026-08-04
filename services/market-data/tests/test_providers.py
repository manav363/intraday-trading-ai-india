from __future__ import annotations

import datetime as dt

import pytest
from intraday_contracts import IST, BarSource, Interval
from market_data.providers import (
    HistoryWindowExceeded,
    ProviderNotImplemented,
    SyntheticProvider,
    build_provider,
)

START = dt.datetime(2025, 3, 3, 0, 0, tzinfo=IST)
END = dt.datetime(2025, 3, 7, 23, 59, tzinfo=IST)


@pytest.fixture
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=42)


# ------------------------------------------------------- registry rules


def test_unknown_provider_raises_and_does_not_default() -> None:
    with pytest.raises(ProviderNotImplemented, match="no fallback by design"):
        build_provider("definitely_not_a_provider")


def test_replay_without_lake_root_raises() -> None:
    with pytest.raises(ProviderNotImplemented, match="requires lake_root"):
        build_provider("replay")


def test_synthetic_must_be_named_explicitly() -> None:
    p = build_provider("synthetic")
    assert p.source is BarSource.SYNTHETIC


def test_nse_charting_without_openchart_would_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-fallback rule, at the point it matters.

    A missing client must stop the process, not silently serve generated bars.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openchart":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderNotImplemented, match="will not fall back to synthetic"):
        build_provider("nse_charting")


# ------------------------------------------------------------ synthetic


def test_bars_are_generated_only_on_trading_days(provider: SyntheticProvider) -> None:
    bars = provider.fetch_bars("RELIANCE", Interval.M15, START, END)
    days = {b.timestamp.date() for b in bars}
    assert dt.date(2025, 3, 8) not in days  # Saturday
    assert days == {
        dt.date(2025, 3, 3),
        dt.date(2025, 3, 4),
        dt.date(2025, 3, 5),
        dt.date(2025, 3, 6),
        dt.date(2025, 3, 7),
    }


def test_full_session_bar_count(provider: SyntheticProvider) -> None:
    one_day = provider.fetch_bars(
        "TCS",
        Interval.M15,
        dt.datetime(2025, 3, 3, 0, 0, tzinfo=IST),
        dt.datetime(2025, 3, 3, 23, 59, tzinfo=IST),
    )
    assert len(one_day) == 25  # 375 session minutes / 15


def test_session_boundaries(provider: SyntheticProvider) -> None:
    bars = provider.fetch_bars(
        "TCS",
        Interval.M15,
        dt.datetime(2025, 3, 3, 0, 0, tzinfo=IST),
        dt.datetime(2025, 3, 3, 23, 59, tzinfo=IST),
    )
    assert bars[0].timestamp.time() == dt.time(9, 15)
    assert bars[-1].timestamp.time() == dt.time(15, 15)  # last bar opens 15:15, closes 15:30


def test_bars_are_ordered_and_timezone_aware(provider: SyntheticProvider) -> None:
    bars = provider.fetch_bars("INFY", Interval.M15, START, END)
    assert bars == sorted(bars, key=lambda b: b.timestamp)
    assert all(b.timestamp.tzinfo is not None for b in bars)


def test_geometry_holds_for_every_generated_bar(provider: SyntheticProvider) -> None:
    """OHLCVBar validates on construction, so generating a range at all is the
    assertion. Made explicit because a generator that emitted invalid geometry
    would fail far away from here."""
    bars = provider.fetch_bars("HDFCBANK", Interval.M5, START, END)
    assert len(bars) > 300
    for b in bars:
        assert b.high >= max(b.open, b.close)
        assert b.low <= min(b.open, b.close)
        assert b.volume >= 0


def test_determinism_same_seed_same_bars() -> None:
    a = SyntheticProvider(seed=7).fetch_bars("RELIANCE", Interval.M15, START, END)
    b = SyntheticProvider(seed=7).fetch_bars("RELIANCE", Interval.M15, START, END)
    assert [x.close for x in a] == [x.close for x in b]


def test_different_seeds_diverge() -> None:
    a = SyntheticProvider(seed=1).fetch_bars("RELIANCE", Interval.M15, START, END)
    b = SyntheticProvider(seed=2).fetch_bars("RELIANCE", Interval.M15, START, END)
    assert [x.close for x in a] != [x.close for x in b]


def test_range_subsetting_is_consistent(provider: SyntheticProvider) -> None:
    """Fetching a wide range and a narrow one must agree on the overlap."""
    wide = provider.fetch_bars("ITC", Interval.M15, START, END)
    narrow = provider.fetch_bars(
        "ITC",
        Interval.M15,
        dt.datetime(2025, 3, 5, 0, 0, tzinfo=IST),
        dt.datetime(2025, 3, 5, 23, 59, tzinfo=IST),
    )
    overlap = [b for b in wide if b.timestamp.date() == dt.date(2025, 3, 5)]
    assert [b.close for b in overlap] == [b.close for b in narrow]


def test_generator_produces_falling_markets_too() -> None:
    """A generator that only trends up produces models that only know 'up'.

    A sibling project registered a regime classifier trained on synthetic GBM
    with no bear regime; it mapped 2/3 states to UPTREND and none to DOWNTREND,
    and so could not see a falling market at all.
    """
    provider = SyntheticProvider(seed=3)
    regimes = {
        provider._regime_for("RELIANCE", dt.date(2025, 1, 1) + dt.timedelta(days=21 * k))
        for k in range(40)
    }
    assert "bear" in regimes, "generator never produces a falling regime"
    assert "bull" in regimes
    assert "ranging" in regimes


def test_intraday_volume_is_u_shaped(provider: SyntheticProvider) -> None:
    """Open and close are busier than midday — every intraday equity market
    shows this, and a flat generator hides every time-of-day bug."""
    bars = provider.fetch_bars(
        "SBIN",
        Interval.M15,
        dt.datetime(2025, 3, 3, 0, 0, tzinfo=IST),
        dt.datetime(2025, 3, 3, 23, 59, tzinfo=IST),
    )
    volumes = [b.volume for b in bars]
    opening = sum(volumes[:3]) / 3
    midday = sum(volumes[10:15]) / 5
    closing = sum(volumes[-3:]) / 3

    assert opening > midday * 1.5
    assert closing > midday


def test_history_window_is_enforced() -> None:
    """Silently truncating to the available window makes the lake's start date
    a lie, so exceeding it raises."""
    provider = SyntheticProvider()
    now = dt.datetime(2025, 3, 10, tzinfo=IST)

    class Narrow(SyntheticProvider):
        def max_history_days(self, interval: Interval) -> int:
            return 55

    with pytest.raises(HistoryWindowExceeded, match="serves at most 55d"):
        Narrow().check_window(Interval.M1, now - dt.timedelta(days=90), now)

    provider.check_window(Interval.M1, now - dt.timedelta(days=90), now)  # unbounded: fine


def test_empty_range_returns_empty_not_error(provider: SyntheticProvider) -> None:
    """A holiday is a legitimate empty answer, distinct from a failure."""
    bars = provider.fetch_bars(
        "RELIANCE",
        Interval.M15,
        dt.datetime(2025, 12, 25, 0, 0, tzinfo=IST),
        dt.datetime(2025, 12, 25, 23, 59, tzinfo=IST),
    )
    assert bars == []


def test_daily_interval_yields_one_bar_per_session(provider: SyntheticProvider) -> None:
    bars = provider.fetch_bars("LT", Interval.D1, START, END)
    assert len(bars) == 5
    assert all(b.timestamp.time() == dt.time(9, 15) for b in bars)
