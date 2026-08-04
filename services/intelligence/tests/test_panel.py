"""Cross-sectional panel tests.

`test_ranks_are_computed_per_timestamp_not_globally` is the one that matters.
Ranking the whole panel at once leaks the future distribution into every row,
improves every metric, and looks completely normal.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from intelligence.panel import (
    PanelConfig,
    base_feature_names,
    build_panel,
    cross_sectional_rank,
    model_feature_names,
    panel_summary,
    sector_neutral_z,
)
from intraday_contracts import IST

DAY = dt.date(2025, 3, 3)
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def _multi_symbol_bars(
    symbols: list[str] = SYMBOLS, n: int = 80, days: int = 2, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, symbol in enumerate(symbols):
        base = 100.0 * (s_i + 1)
        for d in range(days):
            day = DAY + dt.timedelta(days=d)
            price = base
            for i in range(n):
                open_ = price
                close = price * (1.0 + float(rng.normal(0, 0.004)))
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(
                            dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST)
                            + dt.timedelta(minutes=15 * i)
                        ),
                        "open": open_,
                        "high": max(open_, close) * 1.002,
                        "low": min(open_, close) * 0.998,
                        "close": close,
                        "volume": 10_000.0 + abs(float(rng.normal(0, 2_000))),
                    }
                )
                price = close
    return pd.DataFrame(rows)


# ------------------------------------------------------------- leakage


def test_ranks_are_computed_per_timestamp_not_globally() -> None:
    """The panel's sharpest leak.

    A rank computed across the whole panel encodes where a value sits relative
    to values that had not happened yet. It raises every metric and nothing
    about it looks wrong.
    """
    bars = _multi_symbol_bars()
    panel = build_panel(bars, PanelConfig(sector_neutral=False))

    ranked_col = "rsi_14_pct"
    per_timestamp = panel.groupby("timestamp", observed=True)[ranked_col]

    # Within any timestamp that has the full cross-section, ranks must span the
    # unit interval and average to the midpoint — the signature of a rank taken
    # among peers rather than against the whole history.
    full = panel.groupby("timestamp", observed=True)[ranked_col].transform("count") == len(SYMBOLS)
    sample = panel[full]

    assert not sample.empty
    means = sample.groupby("timestamp", observed=True)[ranked_col].mean()
    assert np.allclose(means.to_numpy(), 0.5, atol=1e-9)
    # Mid-rank keeps every value strictly inside (0, 1) and symmetric.
    assert per_timestamp.max().max() < 1.0
    assert per_timestamp.min().min() > 0.0


def test_appending_future_bars_does_not_change_past_ranks() -> None:
    bars = _multi_symbol_bars(n=80, days=2, seed=4)
    cutoff = bars["timestamp"].sort_values().unique()[100]

    full = build_panel(bars, PanelConfig(sector_neutral=False))
    prefix = build_panel(
        bars[bars["timestamp"] <= cutoff].copy(), PanelConfig(sector_neutral=False)
    )

    a = full[full["timestamp"] <= cutoff].set_index(["timestamp", "symbol"])["rsi_14_pct"]
    b = prefix.set_index(["timestamp", "symbol"])["rsi_14_pct"]
    common = a.index.intersection(b.index)

    assert len(common) > 50
    assert np.allclose(a.loc[common], b.loc[common], equal_nan=True)


def test_thin_cross_sections_yield_nan_not_a_degenerate_rank() -> None:
    """With three symbols a percentile can only be 0, 0.5 or 1.0 — that says
    more about the sample size than about the market."""
    bars = _multi_symbol_bars(["AAA", "BBB", "CCC"], n=60, days=1)
    panel = build_panel(bars, PanelConfig(sector_neutral=False, drop_incomplete_rows=False))
    assert panel["rsi_14_pct"].isna().all()


# ------------------------------------------------------------ mechanics


def test_panel_is_pooled_across_symbols() -> None:
    bars = _multi_symbol_bars()
    panel = build_panel(bars, PanelConfig(sector_neutral=False))

    assert panel["symbol"].nunique() == len(SYMBOLS)
    assert panel["timestamp"].is_monotonic_increasing
    summary = panel_summary(panel)
    assert summary["symbols"] == len(SYMBOLS)
    assert summary["median_symbols_per_timestamp"] == len(SYMBOLS)


def test_rank_is_scale_free_across_price_levels() -> None:
    """Symbols priced ₹100 and ₹800 must be comparable after ranking — that is
    the entire purpose of the transformation."""
    bars = _multi_symbol_bars()
    panel = build_panel(bars, PanelConfig(sector_neutral=False))

    per_symbol_mean = panel.groupby("symbol", observed=True)["rsi_14_pct"].mean()
    # No symbol should sit permanently at the top or bottom purely from price.
    assert per_symbol_mean.min() > 0.2
    assert per_symbol_mean.max() < 0.8


def test_model_features_exclude_raw_levels_when_ranking() -> None:
    """A pooled model handed both rsi_14 and rsi_14_pct will key off the raw
    level, re-introducing the per-symbol thinking the panel removes."""
    bars = _multi_symbol_bars()
    panel = build_panel(bars, PanelConfig(sector_neutral=False))
    features = model_feature_names(panel, PanelConfig(sector_neutral=False))

    assert "rsi_14" not in features
    assert "rsi_14_pct" in features
    assert not any(f in features for f in ("vwap", "atr_14", "close", "open"))


def test_session_position_is_not_ranked() -> None:
    """ "Ten minutes after the open" means the same thing for every symbol.
    Ranking it across the cross-section destroys its only information."""
    bars = _multi_symbol_bars()
    panel = build_panel(bars, PanelConfig(sector_neutral=False))
    features = model_feature_names(panel, PanelConfig(sector_neutral=False))

    assert "session_sin" in features
    assert "session_cos" in features


def test_sector_neutral_z_excludes_symbols_without_a_sector() -> None:
    """Pooling sector-less symbols into a catch-all bucket would invent a
    'sector' whose members share nothing."""
    bars = _multi_symbol_bars()
    meta = pd.DataFrame(
        {
            "symbol": SYMBOLS,
            "sector": ["IT"] * 5 + [None] * 3,
        }
    )
    panel = build_panel(bars, PanelConfig(sector_neutral=True), symbol_meta=meta)

    sectorless = panel[panel["sector"].isna()]
    assert sectorless["rsi_14_z_sector"].isna().all()


def test_empty_input_returns_empty() -> None:
    assert build_panel(pd.DataFrame()).empty


def test_base_feature_names_are_all_produced() -> None:
    bars = _multi_symbol_bars(n=60, days=1)
    panel = build_panel(bars, PanelConfig(rank_features=False, drop_incomplete_rows=False))
    for name in base_feature_names():
        assert name in panel.columns, f"{name} declared but not produced"


def test_cross_sectional_rank_helper_respects_min_symbols() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-03-03 09:15"] * 3, utc=True),
            "x": [1.0, 2.0, 3.0],
        }
    )
    assert cross_sectional_rank(frame, "x", min_symbols=5).isna().all()
    ranked = cross_sectional_rank(frame, "x", min_symbols=3)
    assert ranked.notna().all()
    # (1-0.5)/3, (2-0.5)/3, (3-0.5)/3 — centred on 0.5 for any n.
    assert ranked.tolist() == pytest.approx([1 / 6, 0.5, 5 / 6])


def test_sector_neutral_helper_without_sector_column_is_nan() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-03-03 09:15"] * 3, utc=True), "x": [1.0, 2.0, 3.0]}
    )
    assert sector_neutral_z(frame, "x", min_symbols=2).isna().all()


@pytest.mark.slow
def test_panel_reaches_useful_scale() -> None:
    """The whole point of pooling: v1 trained on ~1,200 rows per ticker, which
    forecloses a sub-1% effect before any modelling happens."""
    bars = _multi_symbol_bars(SYMBOLS, n=100, days=6)
    panel = build_panel(bars, PanelConfig(sector_neutral=False))
    assert len(panel) > 3_000
