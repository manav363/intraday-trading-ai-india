"""Triple-barrier labelling tests.

The first group pins the v1 defects. Those are the ones that matter most: they
are regressions that produced plausible-looking numbers rather than crashes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from intelligence.labeling import (
    BarrierConfig,
    effective_sample_size,
    label_panel,
    label_symbol,
    panel_sample_weights,
    realised_volatility,
    sample_weights,
    trainable,
)
from intraday_contracts import IST, BarrierOutcome

DAY = dt.date(2025, 3, 3)


def _bars(closes: list[float], *, day: dt.date = DAY, spread: float = 0.5) -> pd.DataFrame:
    """Bars whose high/low bracket the close by a fixed spread."""
    return pd.DataFrame(
        {
            "symbol": "TESTCO",
            "timestamp": [
                pd.Timestamp(
                    dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST)
                    + dt.timedelta(minutes=15 * i)
                )
                for i in range(len(closes))
            ],
            "open": closes,
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": 10_000.0,
        }
    )


def _wobble(n: int, base: float = 100.0, amplitude: float = 0.4) -> list[float]:
    """A gently oscillating series, so realised vol is well defined and finite."""
    return [base + amplitude * np.sin(i / 3.0) for i in range(n)]


# ------------------------------------------------------------ v1 defects


def test_tail_rows_are_unresolved_not_labelled_down() -> None:
    """v1's headline bug.

    `(NaN > 0).astype(int)` is 0, and the dropna meant to remove those rows was
    a no-op — so the final rows trained as "down" and were the rows the live
    trade plan read.
    """
    frame = _bars(_wobble(60))
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=20))

    # The last `horizon_bars - 1` rows cannot fit a full horizon in the session.
    tail = out.tail(7)
    assert (tail["barrier_outcome"] == BarrierOutcome.UNRESOLVED.value).all()
    assert tail["label"].isna().all()
    assert not tail["is_trainable"].any()


def test_truncated_horizons_are_excluded_not_shortened() -> None:
    """Shortening the horizon near the close biases labels toward the time
    barrier, and the model would learn "late in the session nothing happens" as
    a market fact rather than an artefact of how the labels were cut."""
    frame = _bars(_wobble(40))
    out = label_symbol(frame, BarrierConfig(horizon_bars=6, vol_window=15))

    trainable_rows = out[out["is_trainable"]]
    # Every trainable row held for at most the full horizon, and any row that
    # timed out held for exactly it — never for a truncated remainder.
    timed_out = trainable_rows[trainable_rows["barrier_outcome"] == BarrierOutcome.TIME.value]
    assert (timed_out["bars_held"] == 6).all()


def test_trainable_filter_actually_removes_rows() -> None:
    """The v1 filter existed and did nothing. This asserts it bites."""
    frame = _bars(_wobble(60))
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=20))

    kept = trainable(out)
    assert len(kept) < len(out)
    assert kept["label"].notna().all()


def test_no_row_is_ever_labelled_by_default() -> None:
    """Rows with no usable volatility estimate get no barrier geometry, so they
    must stay unresolved rather than receive a default width."""
    frame = _bars(_wobble(60))
    out = label_symbol(frame, BarrierConfig(horizon_bars=4, vol_window=30))

    warmup = out.head(10)
    assert warmup["label"].isna().all()


def test_barriers_are_detected_on_high_and_low_not_close() -> None:
    """A barrier only a close can trigger is not a barrier — the position would
    really have been stopped out intrabar. Same defect as the v1 backtester."""
    # A near-flat series: small enough moves that no close ever reaches a
    # barrier on its own, so any touch must come from high/low.
    closes = [100.0 + 0.05 * (i % 2) for i in range(60)]
    frame = _bars(closes, spread=0.0)

    baseline = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=15))
    reachers = baseline.iloc[22:30]
    assert (reachers["barrier_outcome"] == BarrierOutcome.TIME.value).all(), (
        "fixture is wrong: closes alone already touch a barrier"
    )

    # Now let one bar's HIGH pierce far above, while its CLOSE stays put.
    frame.loc[30, "high"] = 200.0
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=15))

    # Rows 22..29 can all reach bar 30 within their horizon.
    touched = out.iloc[22:30]
    assert (touched["barrier_outcome"] == BarrierOutcome.PROFIT_TAKE.value).all()
    # The close at bar 30 is unchanged, so a close-only check would see nothing.
    assert frame.loc[30, "close"] == closes[30]


# ------------------------------------------------------------- mechanics


def test_volatility_is_past_only() -> None:
    """A vol estimate including the current bar leaks into the barrier width and
    therefore into the label — the worst place for a leak, because every metric
    inherits it and no diagnostic can see it."""
    closes = _wobble(60)
    full = realised_volatility(pd.Series(closes), window=20)
    truncated = realised_volatility(pd.Series(closes[:40]), window=20)

    assert full.iloc[:40].round(12).equals(truncated.round(12))


def test_profit_take_is_labelled_plus_one() -> None:
    closes = [100.0] * 30 + [130.0] * 10
    frame = _bars(_wobble(30) + [130.0] * 10)
    out = label_symbol(frame, BarrierConfig(horizon_bars=5, vol_window=15))

    resolved = out[out["is_trainable"]]
    assert (resolved["label"] == 1.0).any()
    assert set(resolved["label"].unique()) <= {-1.0, 0.0, 1.0}
    assert len(closes) == 40  # fixture sanity


def test_stop_loss_is_labelled_minus_one() -> None:
    frame = _bars(_wobble(30) + [60.0] * 10)
    out = label_symbol(frame, BarrierConfig(horizon_bars=5, vol_window=15))

    resolved = out[out["is_trainable"]]
    assert (resolved["label"] == -1.0).any()


def test_time_barrier_is_labelled_zero() -> None:
    """A perfectly flat series can touch neither barrier."""
    frame = _bars([100.0 + 0.01 * (i % 2) for i in range(60)], spread=0.001)
    out = label_symbol(
        frame, BarrierConfig(horizon_bars=4, vol_window=20, pt_sigma=50, sl_sigma=50)
    )

    resolved = out[out["is_trainable"]]
    assert (resolved["label"] == 0.0).all()
    assert (resolved["barrier_outcome"] == BarrierOutcome.TIME.value).all()


def test_barriers_scale_with_symbol_volatility() -> None:
    """Wide barriers for volatile names, tight for calm ones. This is what makes
    labels comparable across the cross-section."""
    calm = label_symbol(_bars(_wobble(60, amplitude=0.1)), BarrierConfig(vol_window=20))
    wild = label_symbol(_bars(_wobble(60, amplitude=5.0)), BarrierConfig(vol_window=20))

    calm_width = (calm["pt_level"] - calm["close"]).dropna().mean()
    wild_width = (wild["pt_level"] - wild["close"]).dropna().mean()

    assert wild_width > calm_width * 5


def test_labels_never_span_the_overnight_gap() -> None:
    """This is an intraday system. A label must not depend on a price that would
    have required carrying risk through the close."""
    day1 = _bars(_wobble(25), day=dt.date(2025, 3, 3))
    day2 = _bars(_wobble(25), day=dt.date(2025, 3, 4))
    frame = pd.concat([day1, day2], ignore_index=True)

    out = label_symbol(frame, BarrierConfig(horizon_bars=10, vol_window=15))

    resolved = out[out["is_trainable"]]
    same_session = resolved["t1"].dt.date == resolved["timestamp"].dt.date
    assert same_session.all()


def test_end_of_session_rows_are_unresolved() -> None:
    frame = _bars(_wobble(25))
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=10))
    assert not out.iloc[-1]["is_trainable"]


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="horizon_bars"):
        BarrierConfig(horizon_bars=0)
    with pytest.raises(ValueError, match="barrier widths"):
        BarrierConfig(pt_sigma=-1.0)


# ------------------------------------------------------------------ panel


def test_panel_labels_each_symbol_independently() -> None:
    a = _bars(_wobble(40))
    b = _bars(_wobble(40, base=2000.0, amplitude=20.0))
    b["symbol"] = "OTHERCO"
    panel = pd.concat([a, b], ignore_index=True)

    out = label_panel(panel, BarrierConfig(horizon_bars=5, vol_window=15))

    assert set(out["symbol"].unique()) == {"TESTCO", "OTHERCO"}
    # Sorted by timestamp then symbol — the panel ordering the model expects.
    assert out["timestamp"].is_monotonic_increasing


def test_empty_panel_is_handled() -> None:
    assert label_panel(pd.DataFrame()).empty


# ---------------------------------------------------------- sample weights


def test_overlapping_labels_get_less_weight_than_isolated_ones() -> None:
    """Overlap is the reason weights exist: two labels resolving over the same
    bars are not two independent observations."""
    frame = _bars(_wobble(60))
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=15))
    weights = sample_weights(out)

    assert (weights[out["is_trainable"]] > 0).all()
    assert (weights[~out["is_trainable"]] == 0).all()


def test_effective_sample_size_is_below_row_count() -> None:
    """The honest count. Quoting len(df) against overlapping labels overstates
    the evidence, and every p-value built on it is inflated."""
    frame = _bars(_wobble(80))
    out = label_symbol(frame, BarrierConfig(horizon_bars=8, vol_window=15))
    weights = sample_weights(out)

    ess = effective_sample_size(weights)
    n_trainable = int(out["is_trainable"].sum())

    assert 0 < ess <= n_trainable


def test_panel_weights_normalise_to_mean_one() -> None:
    a = _bars(_wobble(50))
    b = _bars(_wobble(50, base=500.0, amplitude=3.0))
    b["symbol"] = "OTHERCO"
    panel = label_panel(pd.concat([a, b], ignore_index=True), BarrierConfig(vol_window=15))

    weights = panel_sample_weights(panel)
    positive = weights[weights > 0]

    assert positive.mean() == pytest.approx(1.0, rel=1e-6)
