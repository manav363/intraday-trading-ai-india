"""Validation tests.

The purge/embargo tests are the ones that matter: an unpurged split does not
fail, it just makes everything look better.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from intelligence.backtest import (
    CostConfig,
    breakeven_move,
    cost_in_return_terms,
    round_trip_cost,
)
from intelligence.validation import (
    CombinatorialPurgedCV,
    PurgedTimeSeriesSplit,
    brier_score,
    deflated_sharpe_ratio,
    permutation_test,
    reliability_curve,
    sharpe_ratio,
    shuffle_labels_within_folds,
    summarise_paths,
)
from intraday_contracts import IST


def _panel(n_times: int = 200, n_symbols: int = 4, horizon: int = 5) -> pd.DataFrame:
    start = dt.datetime(2025, 3, 3, 9, 15, tzinfo=IST)
    times = [pd.Timestamp(start + dt.timedelta(minutes=15 * i)) for i in range(n_times)]

    rows = []
    for t_i, t in enumerate(times):
        for s in range(n_symbols):
            resolve = times[min(t_i + horizon, n_times - 1)]
            rows.append({"timestamp": t, "symbol": f"S{s}", "t1": resolve, "x": float(t_i + s)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- splitting


def test_split_is_by_timestamp_never_by_row() -> None:
    """In a pooled panel, RELIANCE at 10:15 and TCS at 10:15 share the
    market-wide move. Splitting rows at random puts one in train and the other
    in test, which leaks across the cross-section."""
    panel = _panel()
    splitter = PurgedTimeSeriesSplit(n_splits=4)

    for split in splitter.split(panel["timestamp"], panel["t1"]):
        train_times = set(panel.iloc[split.train]["timestamp"])
        test_times = set(panel.iloc[split.test]["timestamp"])
        assert train_times.isdisjoint(test_times), "a timestamp appeared in both train and test"


def test_every_symbol_on_a_timestamp_lands_in_the_same_fold() -> None:
    panel = _panel()
    splitter = PurgedTimeSeriesSplit(n_splits=4)

    for split in splitter.split(panel["timestamp"], panel["t1"]):
        test_rows = panel.iloc[split.test]
        counts = test_rows.groupby("timestamp", observed=True)["symbol"].nunique()
        assert counts.eq(counts.iloc[0]).all()


def test_purge_removes_labels_that_resolve_into_the_test_window() -> None:
    """A training row observed before the test window but resolving inside it
    had its outcome partly determined by test-period prices."""
    panel = _panel(horizon=10)
    splitter = PurgedTimeSeriesSplit(n_splits=4)

    for split in splitter.split(panel["timestamp"], panel["t1"]):
        if split.n_test == 0 or split.n_train == 0:
            continue
        test_start = panel.iloc[split.test]["timestamp"].min()
        train_rows = panel.iloc[split.train]
        assert (train_rows["t1"] < test_start).all(), (
            "a training label resolves inside the test set"
        )


def test_without_t1_no_purge_is_silently_applied() -> None:
    """Callers must not get an unpurged split that looks purged."""
    panel = _panel(horizon=10)
    splitter = PurgedTimeSeriesSplit(n_splits=4)

    purged = list(splitter.split(panel["timestamp"], panel["t1"]))
    unpurged = list(splitter.split(panel["timestamp"], None))

    assert sum(s.n_train for s in purged) < sum(s.n_train for s in unpurged)


def test_embargo_removes_rows_after_the_test_window() -> None:
    panel = _panel()
    rolling = PurgedTimeSeriesSplit(n_splits=4, embargo_pct=0.05, expanding=False)

    for split in rolling.split(panel["timestamp"], panel["t1"]):
        if split.n_test == 0:
            continue
        test_end = panel.iloc[split.test]["timestamp"].max()
        train_after = panel.iloc[split.train]
        just_after = train_after[
            (train_after["timestamp"] > test_end)
            & (train_after["timestamp"] <= test_end + dt.timedelta(minutes=15 * 5))
        ]
        assert just_after.empty


def test_expanding_window_never_trains_on_the_future() -> None:
    panel = _panel()
    splitter = PurgedTimeSeriesSplit(n_splits=5, expanding=True)

    for split in splitter.split(panel["timestamp"], panel["t1"]):
        if split.n_train == 0 or split.n_test == 0:
            continue
        assert (
            panel.iloc[split.train]["timestamp"].max() < panel.iloc[split.test]["timestamp"].min()
        )


def test_folds_cover_the_timeline() -> None:
    panel = _panel()
    splitter = PurgedTimeSeriesSplit(n_splits=5)
    covered = set()
    for split in splitter.split(panel["timestamp"], panel["t1"]):
        covered |= set(panel.iloc[split.test]["timestamp"])
    assert covered == set(panel["timestamp"])


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedTimeSeriesSplit(n_splits=1)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedTimeSeriesSplit(embargo_pct=0.9)


def test_too_few_timestamps_raises_rather_than_producing_degenerate_folds() -> None:
    tiny = _panel(n_times=6)
    with pytest.raises(ValueError, match="too few"):
        list(PurgedTimeSeriesSplit(n_splits=5).split(tiny["timestamp"], tiny["t1"]))


# ------------------------------------------------------------------ CPCV


def test_cpcv_yields_many_configurations_not_one_path() -> None:
    """One Sharpe cannot distinguish mean 0.4 sd 0.1 from mean 0.4 sd 0.8."""
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    assert cv.n_combinations == 15
    assert cv.n_paths == 5

    panel = _panel(n_times=240)
    splits = list(cv.split(panel["timestamp"], panel["t1"]))
    assert len(splits) == 15
    assert all(s.n_test > 0 for s in splits)


def test_cpcv_train_and_test_never_share_a_timestamp() -> None:
    panel = _panel(n_times=240)
    for split in CombinatorialPurgedCV(n_groups=6, n_test_groups=2).split(
        panel["timestamp"], panel["t1"]
    ):
        train_times = set(panel.iloc[split.train]["timestamp"])
        test_times = set(panel.iloc[split.test]["timestamp"])
        assert train_times.isdisjoint(test_times)


def test_cpcv_summary_reports_a_distribution() -> None:
    result = summarise_paths([0.4, 0.1, 0.9, -0.2, 0.5])
    assert result.n_paths == 5
    assert result.sharpe_std > 0
    assert result.sharpe_p05 < result.sharpe_p50 < result.sharpe_p95


def test_cpcv_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="n_test_groups"):
        CombinatorialPurgedCV(n_groups=4, n_test_groups=4)


# ------------------------------------------------------------ statistics


def test_sharpe_uses_periodic_returns_not_per_trade_roi() -> None:
    """v1 annualised per-trade ROI by sqrt(252), which scales a strategy taking
    5 trades a day and one taking 5 a year identically."""
    daily = pd.Series(np.full(252, 0.001))
    assert sharpe_ratio(daily) > 10  # zero variance -> huge; sanity only

    noisy = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252))
    value = sharpe_ratio(noisy)
    assert -5 < value < 5


def test_sharpe_of_constant_series_is_zero_not_infinite() -> None:
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_permutation_p_value_never_zero() -> None:
    """With N permutations the smallest achievable p is 1/(N+1). Reporting 0
    claims infinite significance from a finite experiment."""
    result = permutation_test(10.0, [0.0] * 200)
    assert result.p_value == pytest.approx(1 / 201)
    assert result.p_value > 0


def test_permutation_p_value_when_null_dominates() -> None:
    result = permutation_test(0.0, [1.0] * 99)
    assert result.p_value == pytest.approx(1.0)
    assert not result.is_significant
    assert "NOT significant" in result.verdict()


def test_permutation_shuffles_within_folds_not_globally() -> None:
    """Global shuffling can leave train and test sharing shuffled-label
    structure, which drags the null toward the observed value."""
    labels = pd.Series([0, 1] * 50)
    folds = pd.Series([0] * 50 + [1] * 50)
    rng = np.random.default_rng(0)

    shuffled = shuffle_labels_within_folds(labels, folds, rng)

    for fold in (0, 1):
        mask = folds == fold
        assert shuffled[mask].sum() == labels[mask].sum()


def test_deflated_sharpe_penalises_many_trials() -> None:
    """The direct answer to 'did you pick the best of 200 runs?'

    Uses a marginal per-period Sharpe. A strong one (say 1.5 over 500
    observations) is overwhelmingly significant either way, so both figures
    saturate at 1.0 and the test would assert nothing.
    """
    single = deflated_sharpe_ratio(0.1, 250, n_trials=1)
    many = deflated_sharpe_ratio(0.1, 250, n_trials=200)

    assert single > 0.9, "a marginal edge should look good when tried once"
    assert many < 0.5, "the same edge should not survive 200 attempts"
    assert many < single


def test_deflated_sharpe_still_credits_a_genuinely_strong_result() -> None:
    """Deflation is a discount, not a veto."""
    assert deflated_sharpe_ratio(0.5, 1000, n_trials=200) > 0.95


def test_deflated_sharpe_rejects_zero_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(1.0, 100, n_trials=0)


def test_brier_score_of_a_coin_flip() -> None:
    p = pd.Series([0.5] * 100)
    y = pd.Series([0, 1] * 50)
    assert brier_score(p, y) == pytest.approx(0.25)


def test_brier_score_of_a_perfect_forecast() -> None:
    y = pd.Series([0, 1, 1, 0])
    assert brier_score(y.astype(float), y) == pytest.approx(0.0)


def test_reliability_curve_exposes_overconfidence() -> None:
    """A model whose 0.7 bucket is right 55% of the time is overconfident, and
    Kelly sizing on that probability allocates far too much."""
    rng = np.random.default_rng(3)
    p = pd.Series(np.full(400, 0.7))
    y = pd.Series((rng.random(400) < 0.55).astype(int))

    curve = reliability_curve(p, y, n_bins=10)
    row = curve[curve["count"] > 0].iloc[0]
    assert row["predicted"] == pytest.approx(0.7)
    assert row["observed"] < 0.65


# ----------------------------------------------------------------- costs


def test_stt_is_charged_on_the_sell_side_only() -> None:
    """Charging it both sides roughly doubles that component."""
    cost = round_trip_cost(entry_price=1000.0, exit_price=1000.0, quantity=100)
    assert cost.stt == pytest.approx(1000.0 * 100 * 0.00025)


def test_stamp_duty_is_charged_on_the_buy_side_only() -> None:
    cost = round_trip_cost(entry_price=1000.0, exit_price=1000.0, quantity=100)
    assert cost.stamp_duty == pytest.approx(1000.0 * 100 * 0.00003)


def test_gst_excludes_stt_and_stamp_duty() -> None:
    """They are taxes, and taxes are not themselves taxable."""
    cost = round_trip_cost(entry_price=1000.0, exit_price=1000.0, quantity=100)
    taxable = cost.brokerage + cost.exchange + cost.sebi
    assert cost.gst == pytest.approx(taxable * 0.18)


def test_brokerage_is_capped_per_order() -> None:
    big = round_trip_cost(entry_price=10_000.0, exit_price=10_000.0, quantity=1000)
    assert big.brokerage == pytest.approx(40.0)  # ₹20 cap x 2 legs


def test_round_trip_cost_is_material_against_an_intraday_target() -> None:
    """If the profit target is 0.3% and the round trip costs 0.1%, a third of
    the edge is gone before the model is right about anything."""
    cost_fraction = cost_in_return_terms(1000.0, 100)
    assert 0.0005 < cost_fraction < 0.005


def test_breakeven_move_is_reported() -> None:
    breakeven = breakeven_move()
    assert breakeven > 0
    # A profit target below breakeven is not a strategy.
    assert breakeven < 0.01


def test_slippage_scales_with_the_configured_bps() -> None:
    cheap = round_trip_cost(1000.0, 1000.0, 100, config=CostConfig(slippage_bps=1.0))
    dear = round_trip_cost(1000.0, 1000.0, 100, config=CostConfig(slippage_bps=10.0))
    assert dear.slippage == pytest.approx(cheap.slippage * 10)


def test_statutory_charges_can_be_isolated_for_comparison() -> None:
    gross = round_trip_cost(
        1000.0, 1000.0, 100, config=CostConfig(include_statutory=False, slippage_bps=0.0)
    )
    assert gross.stt == 0.0
    assert gross.total == pytest.approx(gross.brokerage)
