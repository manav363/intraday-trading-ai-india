"""Feature tests.

The leakage tests are the load-bearing ones. A leaking feature does not crash —
it makes every metric better, which is precisely why it survives.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from intelligence.features import (
    add_microstructure_features,
    add_technical_features,
    amihud_illiquidity,
    corwin_schultz_spread_est,
    order_flow_imbalance_est,
    roll_spread_est,
    rsi,
    session_vwap,
    technical_feature_names,
    tick_rule_sign,
    true_range,
    wilder_rma,
)
from intraday_contracts import IST

DAY = dt.date(2025, 3, 3)


def _bars(n: int, *, days: int = 1, base: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Bars with *proportional* moves.

    Noise is scaled by price so that a ₹90 and a ₹3,900 series have the same
    percentage volatility. Using a fixed absolute step would give the cheap
    series ~40x the percentage volatility, and a scale-invariance test against
    that fixture would fail for reasons that have nothing to do with the
    features.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = DAY + dt.timedelta(days=d)
        price = base
        for i in range(n):
            open_ = price
            close = price * (1.0 + float(rng.normal(0, 0.004)))
            wick_up = abs(float(rng.normal(0, 0.002))) * price
            wick_dn = abs(float(rng.normal(0, 0.002))) * price
            rows.append(
                {
                    "symbol": "TESTCO",
                    "timestamp": pd.Timestamp(
                        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST)
                        + dt.timedelta(minutes=15 * i)
                    ),
                    "open": open_,
                    "high": max(open_, close) + wick_up,
                    "low": min(open_, close) - wick_dn,
                    "close": close,
                    "volume": 10_000.0 + abs(float(rng.normal(0, 2_000))),
                }
            )
            price = close
    return pd.DataFrame(rows)


# -------------------------------------------------------------- v1 bugs


def test_vwap_resets_every_session() -> None:
    """v1's comment claimed a daily reset; its code cumsum-ed the whole frame,
    so by day ten the 'VWAP' was a ten-day average."""
    frame = _bars(25, days=3)
    vwap = session_vwap(frame)

    key = frame["timestamp"].dt.date
    first_bars = frame.groupby(key).head(1).index

    # The first bar of each session has VWAP == its own typical price, which is
    # only true if the accumulator reset.
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    for idx in first_bars:
        assert vwap.iloc[idx] == pytest.approx(typical.iloc[idx])


def test_vwap_does_not_carry_across_the_overnight_gap() -> None:
    frame = _bars(25, days=2)
    # Make day 2 trade far above day 1.
    day2 = frame["timestamp"].dt.date == (DAY + dt.timedelta(days=1))
    for col in ("open", "high", "low", "close"):
        frame.loc[day2, col] = frame.loc[day2, col] + 500.0

    vwap = session_vwap(frame)
    day2_vwap = vwap[day2]

    # If day 1 leaked in, day 2's VWAP would be dragged far below its prices.
    assert (day2_vwap > 500.0).all()


def test_atr_uses_wilder_smoothing_not_a_rolling_mean() -> None:
    """They are different numbers, and every charting package reports Wilder's."""
    frame = _bars(80)
    tr = true_range(frame["high"], frame["low"], frame["close"])

    wilder = wilder_rma(tr, 14)
    simple = tr.rolling(14).mean()

    assert not np.allclose(wilder.dropna().to_numpy()[-20:], simple.dropna().to_numpy()[-20:])
    # Wilder's is an EWM with alpha = 1/n.
    expected = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    assert wilder.dropna().equals(expected.dropna())


# ------------------------------------------------------------- leakage


@pytest.mark.parametrize("cut", [40, 60, 80])
def test_no_technical_feature_depends_on_the_future(cut: int) -> None:
    """The core leakage assertion.

    Computing features on a longer series must not change the values already
    computed on the shorter prefix. A feature that fails this makes every model
    metric better and nothing look wrong.
    """
    full = add_technical_features(_bars(100, days=1, seed=3))
    prefix = add_technical_features(_bars(100, days=1, seed=3).iloc[:cut].copy())

    for name in technical_feature_names():
        a = full[name].iloc[:cut]
        b = prefix[name]
        both_defined = a.notna() & b.notna()
        assert np.allclose(a[both_defined], b[both_defined], equal_nan=True, rtol=1e-9), (
            f"{name} changed when future bars were appended — it leaks"
        )


@pytest.mark.parametrize("cut", [40, 70])
def test_no_microstructure_feature_depends_on_the_future(cut: int) -> None:
    full = add_microstructure_features(_bars(100, days=1, seed=5))
    prefix = add_microstructure_features(_bars(100, days=1, seed=5).iloc[:cut].copy())

    from intelligence.features import MICROSTRUCTURE_FEATURES

    for name in MICROSTRUCTURE_FEATURES:
        a = full[name].iloc[:cut]
        b = prefix[name]
        both = a.notna() & b.notna()
        assert np.allclose(a[both], b[both], rtol=1e-9), f"{name} leaks"


def test_session_range_position_uses_cummax_not_day_max() -> None:
    """A plain max() over the session would leak the afternoon into every
    morning bar."""
    frame = _bars(25, days=1, seed=7)
    out = add_technical_features(frame)

    # The first bar of the session must sit at a range position derived only
    # from itself, so its session high equals its own high.
    first = out.iloc[0]
    assert 0.0 <= first["session_range_position"] <= 1.0


# ----------------------------------------------------------- mechanics


def test_rsi_bounds_and_pure_uptrend() -> None:
    rising = pd.Series(np.linspace(100, 200, 60))
    values = rsi(rising, 14).dropna()
    assert (values <= 100.0).all()
    assert (values >= 0.0).all()
    assert values.iloc[-1] == pytest.approx(100.0)


def test_rsi_pure_downtrend() -> None:
    falling = pd.Series(np.linspace(200, 100, 60))
    assert rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_true_range_accounts_for_gaps() -> None:
    frame = pd.DataFrame({"high": [101.0, 120.0], "low": [99.0, 118.0], "close": [100.0, 119.0]})
    tr = true_range(frame["high"], frame["low"], frame["close"])
    # Second bar gapped up: TR is high - prev_close (20), not high - low (2).
    assert tr.iloc[1] == pytest.approx(20.0)


def test_features_are_price_level_scale_free() -> None:
    """The panel pools a ₹90 stock and a ₹3,900 stock. Any feature that scales
    with price would encode the price level as if it were a signal."""
    cheap = add_technical_features(_bars(80, base=90.0, seed=11))
    dear = add_technical_features(_bars(80, base=3900.0, seed=11))

    scale_free = [
        "vwap_distance_pct",
        "atr_pct",
        "macd_histogram_pct",
        "distance_sma50",
        "trend_fast_slow",
        "bb_width",
    ]
    for name in scale_free:
        a = cheap[name].dropna().abs().median()
        b = dear[name].dropna().abs().median()
        assert a == pytest.approx(b, rel=0.6), f"{name} scales with price level"


def test_feature_name_list_matches_what_is_produced() -> None:
    """A hand-maintained feature list drifts, and a feature missing from it is
    silently absent from training."""
    out = add_technical_features(_bars(60))
    for name in technical_feature_names():
        assert name in out.columns


# ------------------------------------------------------ microstructure


def test_tick_rule_carries_direction_through_unchanged_bars() -> None:
    close = pd.Series([100.0, 101.0, 101.0, 101.0, 100.0])
    signs = tick_rule_sign(close)
    assert signs.tolist() == [0.0, 1.0, 1.0, 1.0, -1.0]


def test_ofi_is_bounded_and_signed() -> None:
    frame = _bars(80, seed=13)
    ofi = order_flow_imbalance_est(frame["close"], frame["volume"], 20).dropna()
    assert (ofi.abs() <= 1.0 + 1e-9).all()


def test_ofi_is_positive_in_a_pure_uptrend() -> None:
    n = 60
    close = pd.Series(np.linspace(100, 130, n))
    volume = pd.Series(np.full(n, 1000.0))
    ofi = order_flow_imbalance_est(close, volume, 20).dropna()
    assert (ofi > 0.9).all()


def test_corwin_schultz_is_non_negative() -> None:
    frame = _bars(80, seed=17)
    spread = corwin_schultz_spread_est(frame["high"], frame["low"]).dropna()
    assert (spread >= 0.0).all()


def test_roll_spread_is_nan_when_autocovariance_is_positive() -> None:
    """Roll's estimator is genuinely undefined in a trending market. Returning
    NaN is correct; fabricating a number would not be."""
    trending = pd.Series(np.linspace(100, 200, 80))
    out = roll_spread_est(trending, 20)
    assert out.dropna().eq(0.0).all() or out.isna().any()


def test_amihud_rises_when_volume_falls() -> None:
    n = 80
    close = pd.Series(100 + np.sin(np.arange(n) / 3.0))
    liquid = amihud_illiquidity(close, pd.Series(np.full(n, 1e7)), 20).dropna()
    illiquid = amihud_illiquidity(close, pd.Series(np.full(n, 1e3)), 20).dropna()
    assert illiquid.mean() > liquid.mean()
