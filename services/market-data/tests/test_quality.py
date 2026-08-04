"""Adversarial fixtures for the quality gate.

Every rule gets a hand-built frame that it must catch. A quality gate without
failing fixtures is decoration — you assert it catches a `high < low` row by
handing it one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from intraday_contracts import IST, Interval, Verdict
from market_data.quality import eligible_universe, run_gate, score_symbol
from market_data.quality.rules import (
    check_circuit_limit,
    check_duplicate_timestamps,
    check_geometry,
    check_sigma_outliers,
    check_staleness,
    check_volume_spike,
    check_zero_volume_with_price_move,
)

DAY = dt.date(2025, 3, 3)


def _frame(rows: list[dict[str, object]], symbol: str = "TESTCO") -> pd.DataFrame:
    """Build a bar frame; missing fields default to a calm, valid bar."""
    records = []
    for i, row in enumerate(rows):
        base = {
            "symbol": symbol,
            "timestamp": pd.Timestamp(
                dt.datetime.combine(DAY, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=15 * i)
            ),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10_000.0,
            "source": "synthetic",
        }
        records.append(base | row)
    return pd.DataFrame(records)


def _calm(n: int, symbol: str = "TESTCO") -> pd.DataFrame:
    """n valid bars with a gently varying close, so trailing stats exist."""
    return _frame(
        [
            {
                "close": 100.0 + (i % 7) * 0.1,
                "high": 101.5 + (i % 7) * 0.1,
                "low": 98.5 + (i % 7) * 0.1,
                "volume": 10_000.0 + (i % 5) * 100,
            }
            for i in range(n)
        ],
        symbol=symbol,
    )


def _spike(frame: pd.DataFrame, index: int, close: float) -> None:
    """Move one bar's close a long way while keeping the bar geometrically valid.

    Setting only `close` would push it outside high/low, and the geometry rule
    (fatal) would condemn the row before the plausibility rule under test ever
    saw it.
    """
    frame.loc[index, "close"] = close
    frame.loc[index, "high"] = max(close, float(frame.loc[index, "open"])) + 0.5
    frame.loc[index, "low"] = min(close, float(frame.loc[index, "open"])) - 0.5


# ------------------------------------------------------ Tier 2 fixtures


def test_catches_high_below_low() -> None:
    frame = _frame([{"high": 98.0, "low": 99.0}])
    result, mask = check_geometry(frame)
    assert not result.passed
    assert mask.sum() == 1
    assert result.evidence  # never empty on failure


def test_catches_high_below_body() -> None:
    frame = _frame([{"open": 100.0, "close": 105.0, "high": 102.0, "low": 99.0}])
    result, mask = check_geometry(frame)
    assert not result.passed
    assert mask.sum() == 1


def test_catches_low_above_body() -> None:
    frame = _frame([{"open": 100.0, "close": 95.0, "high": 101.0, "low": 97.0}])
    result, _ = check_geometry(frame)
    assert not result.passed


def test_valid_geometry_passes() -> None:
    result, mask = check_geometry(_calm(5))
    assert result.passed
    assert mask.sum() == 0
    assert result.evidence == {}  # no evidence needed when passing


# ------------------------------------------------------ Tier 3 fixtures


def test_catches_duplicate_bars() -> None:
    frame = _calm(3)
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    result, mask = check_duplicate_timestamps(frame)
    assert not result.passed
    assert mask.sum() == 1


def test_short_session_is_flagged() -> None:
    outcome = run_gate(_calm(5), Interval.M15)  # 5 bars, expected 25
    coverage = next(
        r for r in outcome.run.results if r.rule_id == "T3.completeness.session_coverage"
    )
    assert not coverage.passed
    assert coverage.evidence["expected_bars_per_session"] == 25


def test_full_session_passes_coverage() -> None:
    outcome = run_gate(_calm(25), Interval.M15)
    coverage = next(
        r for r in outcome.run.results if r.rule_id == "T3.completeness.session_coverage"
    )
    assert coverage.passed


# ------------------------------------------------------ Tier 5 fixtures


def test_catches_circuit_limit_breach() -> None:
    frame = _calm(30)
    _spike(frame, 20, 200.0)  # +100% in one bar
    result, mask = check_circuit_limit(frame)
    assert not result.passed
    assert mask.sum() >= 1


def test_catches_sigma_outlier() -> None:
    frame = _calm(120)
    _spike(frame, 110, 118.0)  # far outside trailing vol
    result, mask = check_sigma_outliers(frame)
    assert not result.passed
    assert mask.sum() >= 1


def test_sigma_rule_uses_only_past_data() -> None:
    """A full-sample sigma would leak the future into a quality decision, and
    that decision determines which rows the model trains on."""
    frame = _calm(120)
    _spike(frame, 110, 118.0)

    truncated = frame.iloc[:100].copy()
    _, mask_full = check_sigma_outliers(frame)
    _, mask_trunc = check_sigma_outliers(truncated)

    # Verdicts on the first 100 rows must not change because of later rows.
    assert list(mask_full.iloc[:100]) == list(mask_trunc)


def test_catches_stale_run() -> None:
    frame = _frame([{"close": 100.0} for _ in range(40)])
    result, mask = check_staleness(frame)
    assert not result.passed
    assert mask.sum() >= 20


def test_short_flat_run_is_not_stale() -> None:
    frame = _frame([{"close": 100.0} for _ in range(5)])
    result, _ = check_staleness(frame)
    assert result.passed


def test_catches_price_move_on_zero_volume() -> None:
    frame = _calm(10)
    frame.loc[5, "volume"] = 0.0
    frame.loc[5, "close"] = 103.0
    result, mask = check_zero_volume_with_price_move(frame)
    assert not result.passed
    assert mask.sum() == 1


def test_zero_volume_without_price_move_is_fine() -> None:
    frame = _frame([{"close": 100.0, "volume": 0.0} for _ in range(4)])
    result, _ = check_zero_volume_with_price_move(frame)
    assert result.passed


def test_catches_volume_unit_error() -> None:
    frame = _calm(120)
    frame.loc[110, "volume"] = 10_000_000.0
    result, _ = check_volume_spike(frame)
    assert not result.passed


# ---------------------------------------------------------------- gate


def test_empty_batch_is_rejected_not_passed() -> None:
    """An empty batch reporting clean is indistinguishable from a healthy quiet
    market, which is how a dead ingestion path stays invisible."""
    outcome = run_gate(pd.DataFrame(), Interval.M15)
    assert outcome.verdict is Verdict.REJECTED


def test_clean_batch_passes() -> None:
    outcome = run_gate(_calm(25), Interval.M15)
    assert outcome.verdict is Verdict.PASS
    assert len(outcome.clean) == 25
    assert outcome.quarantined.empty


def test_fatal_rows_are_quarantined_with_their_rule() -> None:
    frame = _calm(25)
    frame.loc[10, "high"] = 1.0  # geometry violation

    outcome = run_gate(frame, Interval.M15)

    assert outcome.verdict is Verdict.QUARANTINED
    assert len(outcome.quarantined) == 1
    assert len(outcome.clean) == 24
    assert outcome.quarantined.iloc[0]["failed_rule"] == "T2.validity.geometry"


def test_nothing_is_deleted_clean_plus_quarantined_is_the_input() -> None:
    frame = _calm(25)
    frame.loc[10, "high"] = 1.0
    frame.loc[15, "close"] = -5.0

    outcome = run_gate(frame, Interval.M15)
    assert len(outcome.clean) + len(outcome.quarantined) == len(frame)


def test_non_fatal_flags_do_not_quarantine() -> None:
    """A 6-sigma move is often a real event. Dropping it would silently remove
    exactly the observations a volatility model most needs."""
    frame = _calm(120)
    _spike(frame, 110, 118.0)

    outcome = run_gate(frame, Interval.M15)
    assert len(outcome.clean) == 120
    assert outcome.verdict is not Verdict.QUARANTINED


# ------------------------------------------------------------- scoring


def test_thin_symbol_is_ineligible_and_explains_why() -> None:
    score = score_symbol(_calm(25), "TESTCO", Interval.M15, as_of=DAY, expected_sessions=20)
    assert not score.eligible
    assert any("sessions of history" in r for r in score.ineligible_reasons)


def test_liquid_complete_symbol_is_eligible() -> None:
    frames = []
    for d in range(20):
        f = _calm(25)
        f["timestamp"] = f["timestamp"] + pd.Timedelta(days=d)
        f["volume"] = 5_000_000.0  # ~₹50 crore/day at ~100/share
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)

    as_of = frame["timestamp"].dt.date.max()
    score = score_symbol(frame, "TESTCO", Interval.M15, as_of=as_of, expected_sessions=20)

    assert score.eligible, score.ineligible_reasons
    assert eligible_universe([score]) == ("TESTCO",)


def test_absent_symbol_scores_zero_with_a_reason() -> None:
    score = score_symbol(_calm(25), "MISSING", Interval.M15, as_of=DAY, expected_sessions=20)
    assert score.score == 0.0
    assert score.ineligible_reasons == ("no data",)


def test_illiquid_symbol_is_ineligible() -> None:
    frames = []
    for d in range(20):
        f = _calm(25)
        f["timestamp"] = f["timestamp"] + pd.Timedelta(days=d)
        f["volume"] = 10.0  # nearly nothing trades
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)

    as_of = frame["timestamp"].dt.date.max()
    score = score_symbol(frame, "TESTCO", Interval.M15, as_of=as_of, expected_sessions=20)

    assert not score.eligible
    assert any("turnover" in r for r in score.ineligible_reasons)


@pytest.mark.parametrize("stale_days", [0, 3, 10])
def test_recency_decays_with_staleness(stale_days: int) -> None:
    frame = _calm(25)
    as_of = frame["timestamp"].dt.date.max() + dt.timedelta(days=stale_days)
    score = score_symbol(frame, "TESTCO", Interval.M15, as_of=as_of, expected_sessions=1)
    if stale_days == 0:
        assert score.recency == 1.0
    elif stale_days == 10:
        assert score.recency == 0.0
