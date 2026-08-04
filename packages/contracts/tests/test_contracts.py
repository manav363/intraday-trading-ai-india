"""Contract invariant tests.

Several of these exist specifically to make a v1 defect unrepresentable. Those
are marked with the bug they close.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from intraday_contracts import (
    IST,
    BarrierOutcome,
    BarSource,
    ExplanationFact,
    Interval,
    LakePaths,
    ModelCard,
    OHLCVBar,
    Prediction,
    QualityScore,
    RuleResult,
    Severity,
    Side,
    SymbolMeta,
    Tier,
)
from intraday_contracts.intelligence import Direction

TS = dt.datetime(2026, 8, 3, 10, 15, tzinfo=IST)


def _bar(**overrides: object) -> OHLCVBar:
    base: dict[str, object] = {
        "symbol": "RELIANCE",
        "timestamp": TS,
        "interval": Interval.M15,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 5000.0,
        "source": BarSource.REPLAY,
    }
    return OHLCVBar(**(base | overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------- bars


def test_valid_bar_constructs() -> None:
    bar = _bar()
    assert bar.symbol == "RELIANCE"
    assert not bar.is_synthetic


def test_bar_rejects_high_below_low() -> None:
    with pytest.raises(ValidationError, match="high"):
        _bar(high=98.0, low=99.0)


def test_bar_rejects_high_below_body() -> None:
    with pytest.raises(ValidationError, match="below body max"):
        _bar(high=100.5, open=100.0, close=101.0)


def test_bar_rejects_low_above_body() -> None:
    with pytest.raises(ValidationError, match="above body min"):
        _bar(low=100.5, open=100.0, close=101.0)


def test_bar_rejects_non_positive_close() -> None:
    with pytest.raises(ValidationError):
        _bar(close=0.0)


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        _bar(volume=-1.0)


def test_bar_rejects_naive_timestamp() -> None:
    """A naive intraday timestamp silently shifts the whole NSE session."""
    with pytest.raises(ValidationError, match="naive timestamp"):
        _bar(timestamp=dt.datetime(2026, 8, 3, 10, 15))  # noqa: DTZ001 — naive is the point


def test_provenance_has_no_default() -> None:
    """Unknown provenance must be unconstructable, not silently defaulted."""
    with pytest.raises(ValidationError):
        OHLCVBar(
            symbol="RELIANCE",
            timestamp=TS,
            interval=Interval.M15,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1.0,
        )  # type: ignore[call-arg]


def test_synthetic_source_is_flagged() -> None:
    assert _bar(source=BarSource.SYNTHETIC).is_synthetic
    assert not BarSource.SYNTHETIC.is_real_market_data
    assert BarSource.REPLAY.is_real_market_data


def test_interval_bars_per_session() -> None:
    assert Interval.M1.bars_per_session == 375
    assert Interval.M15.bars_per_session == 25
    assert Interval.D1.bars_per_session == 1
    assert Interval.M15.is_intraday
    assert not Interval.D1.is_intraday


# ------------------------------------------------------------ symbols


def test_delisted_symbol_is_listed_before_delisting() -> None:
    meta = SymbolMeta(
        symbol="DEAD",
        name="Delisted Co",
        listing_date=dt.date(2010, 1, 1),
        delisting_date=dt.date(2020, 6, 1),
    )
    assert meta.was_listed_on(dt.date(2015, 5, 5))
    assert not meta.was_listed_on(dt.date(2021, 1, 1))
    assert not meta.was_listed_on(dt.date(2009, 1, 1))


# --------------------------------------------------- prediction (v1 bugs)


def _prediction(**overrides: object) -> Prediction:
    p_up = float(overrides.pop("p_up", 0.72))  # type: ignore[arg-type]
    base: dict[str, object] = {
        "symbol": "RELIANCE",
        "as_of": TS,
        "horizon_bars": 4,
        "p_up": p_up,
        "certainty": max(p_up, 1.0 - p_up),
        "side": Side.LONG if p_up > 0.5 else Side.SHORT,
        "size_fraction": 0.02,
        "is_calibrated": True,
    }
    return Prediction(**(base | overrides))  # type: ignore[arg-type]


def test_short_call_is_representable() -> None:
    """v1 bug: `Confidence` held P(up) and the gate was `< 0.55`, so a short
    call could never clear it. The system was long-only by accident."""
    short = _prediction(p_up=0.19)
    assert short.side is Side.SHORT
    assert short.certainty == pytest.approx(0.81)


def test_certainty_below_half_is_rejected_by_the_field_bound() -> None:
    """v1 bug, first guard: storing P(up) in the certainty field.

    Certainty is `max(p, 1-p)` and so is >= 0.5 by construction. Putting a raw
    P(up) of 0.19 there — exactly what v1 did — cannot even satisfy the bound.
    """
    with pytest.raises(ValidationError, match="greater than or equal to 0.5"):
        Prediction(
            symbol="RELIANCE",
            as_of=TS,
            horizon_bars=4,
            p_up=0.19,
            certainty=0.19,
            side=Side.SHORT,
            size_fraction=0.01,
            is_calibrated=True,
        )


def test_certainty_cannot_drift_from_probability() -> None:
    """v1 bug, second guard: a certainty that is plausible but not derived.

    0.70 clears the >= 0.5 bound, so only the cross-field validator catches that
    it disagrees with p_up=0.19 (whose certainty is 0.81).
    """
    with pytest.raises(ValidationError, match="v1 confidence bug"):
        Prediction(
            symbol="RELIANCE",
            as_of=TS,
            horizon_bars=4,
            p_up=0.19,
            certainty=0.70,
            side=Side.SHORT,
            size_fraction=0.01,
            is_calibrated=True,
        )


def test_side_cannot_contradict_probability() -> None:
    with pytest.raises(ValidationError, match="contradicts p_up"):
        Prediction(
            symbol="RELIANCE",
            as_of=TS,
            horizon_bars=4,
            p_up=0.19,
            certainty=0.81,
            side=Side.LONG,
            size_fraction=0.01,
            is_calibrated=True,
        )


def test_flat_call_must_have_zero_size() -> None:
    with pytest.raises(ValidationError, match="size_fraction"):
        Prediction(
            symbol="RELIANCE",
            as_of=TS,
            horizon_bars=4,
            p_up=0.51,
            certainty=0.51,
            side=Side.FLAT,
            size_fraction=0.05,
            is_calibrated=True,
        )


def test_flat_call_with_zero_size_is_valid() -> None:
    flat = _prediction(p_up=0.51, side=Side.FLAT, size_fraction=0.0)
    assert flat.side is Side.FLAT


def test_probability_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        _prediction(p_up=1.4)


def test_explanation_fact_is_numeric() -> None:
    fact = ExplanationFact(
        feature="ofi_z",
        display_name="Order-flow imbalance",
        value=1.83,
        display_value="+1.8σ",
        shap=0.041,
        direction=Direction.SUPPORTS_UP,
        rank=1,
    )
    assert fact.rank == 1


# ------------------------------------------------------------- labels


def test_unresolved_barrier_is_not_trainable() -> None:
    """v1 bug: `(NaN > 0).astype(int)` is 0, so unresolved rows silently became
    'down' labels and trained the model."""
    assert not BarrierOutcome.UNRESOLVED.is_trainable
    assert BarrierOutcome.PROFIT_TAKE.is_trainable
    assert BarrierOutcome.STOP_LOSS.is_trainable
    assert BarrierOutcome.TIME.is_trainable


# ------------------------------------------------------------ quality


def test_failing_rule_must_carry_evidence() -> None:
    with pytest.raises(ValidationError, match="without evidence"):
        RuleResult(
            rule_id="T4.continuity.prev_close",
            tier=Tier.CONTINUITY,
            severity=Severity.FATAL,
            passed=False,
        )


def test_passing_rule_needs_no_evidence() -> None:
    result = RuleResult(
        rule_id="T2.validity.high_low",
        tier=Tier.VALIDITY,
        severity=Severity.FATAL,
        passed=True,
    )
    assert result.passed


def test_rule_id_tier_prefix_must_match_tier_field() -> None:
    with pytest.raises(ValidationError, match="declares tier"):
        RuleResult(
            rule_id="T2.validity.high_low",
            tier=Tier.CONTINUITY,
            severity=Severity.FATAL,
            passed=True,
        )


def test_rule_id_format_is_enforced() -> None:
    with pytest.raises(ValidationError):
        RuleResult(
            rule_id="whatever",
            tier=Tier.VALIDITY,
            severity=Severity.WARNING,
            passed=True,
        )


def test_ineligible_symbol_must_explain_itself() -> None:
    with pytest.raises(ValidationError, match="no reason"):
        QualityScore(
            symbol="THIN",
            as_of=dt.date(2026, 8, 3),
            score=0.4,
            history_completeness=0.4,
            validity_clean_rate=1.0,
            continuity_clean_rate=1.0,
            liquidity_adequacy=0.2,
            recency=1.0,
            eligible=False,
        )


# ------------------------------------------------------------- model card


def _card(**overrides: object) -> ModelCard:
    base: dict[str, object] = {
        "model_id": "m1",
        "trained_at": TS,
        "n_samples": 1_100_000,
        "n_symbols": 50,
        "n_features": 60,
        "train_start": TS - dt.timedelta(days=60),
        "train_end": TS,
        "oos_accuracy": 0.53,
        "oos_auc": 0.55,
        "brier_score": 0.248,
        "permutation_p_value": 0.0597,
        "permutation_n": 200,
        "trained_on_synthetic": False,
    }
    return ModelCard(**(base | overrides))  # type: ignore[arg-type]


def test_p_value_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        _card(permutation_p_value=0.0)


def test_p_value_respects_permutation_floor() -> None:
    """With N permutations the smallest achievable p is 1/(N+1). Reporting
    anything below that claims significance the test cannot deliver."""
    with pytest.raises(ValidationError, match="below the floor"):
        _card(permutation_p_value=0.001, permutation_n=200)


def test_p_value_at_the_floor_is_allowed() -> None:
    card = _card(permutation_p_value=1.0 / 201, permutation_n=200)
    assert card.is_significant


def test_significance_verdict() -> None:
    assert not _card(permutation_p_value=0.0597).is_significant
    assert _card(permutation_p_value=0.01).is_significant


def test_baseline_comparison_is_tri_state() -> None:
    assert _card().beats_baseline is None
    assert _card(baseline_oos_auc=0.50).beats_baseline is True
    assert _card(baseline_oos_auc=0.60).beats_baseline is False


# --------------------------------------------------------------- lake


def test_lake_paths_layout() -> None:
    paths = LakePaths("/lake")
    day = dt.date(2026, 8, 3)

    assert paths.bars_file(Interval.M15, "RELIANCE", day) == (
        __import__("pathlib").Path(
            "/lake/bars/15m/symbol=RELIANCE/year=2026/month=08/2026-08-03.parquet"
        )
    )
    assert paths.eod_file(day).name == "2026-08-03.parquet"
    assert "as_of=2026-08-03" in str(paths.universe_pit(day))
    assert "date=2026-08-03" in str(paths.quarantine(day))
