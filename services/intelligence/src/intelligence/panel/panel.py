"""Cross-sectional panel construction (L0).

**The single highest-value change in the rebuild.**

v1 trained one model per ticker on that ticker's own history — roughly 1,200
rows against a sub-1% effect. That design forecloses the result before any
modelling happens: you cannot extract a tiny effect from a thousand noisy
samples, no matter how good the model is. The literature is unambiguous that the
standard is a pooled cross-sectional panel; nobody serious fits one model per
stock.

Pooling changes what the features *mean*, not just how many rows there are:

    raw:             rsi_14 = 68.2
    cross-sectional: rsi_14_pct = 0.83    <- 83rd percentile among all
                                             symbols AT THIS TIMESTAMP
    sector-neutral:  rsi_14_z_sector = 1.2

The percentile is the point. A model trained on ranks learns *relative*
attractiveness, which is the question a screener actually asks — you are
choosing between symbols, not judging one in isolation.

**Where the leak hides.** Cross-sectional ranks must be computed per timestamp,
using only that timestamp's rows. Ranking the whole panel at once leaks the
future distribution into every row, and it leaks silently: the model gets better,
every metric improves, and nothing looks wrong. `test_ranks_are_computed_per_timestamp`
exists solely to pin this.

Scale reached here: 50 symbols x 60 days x 375 one-minute bars is about 1.1M
rows — the ~10^6 order the literature treats as the minimum for this effect
size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.microstructure import MICROSTRUCTURE_FEATURES, add_microstructure_features
from ..features.technical import add_technical_features, technical_feature_names

MIN_SYMBOLS_FOR_RANKING = 5
"""Below this, a cross-sectional percentile is noise dressed as a number.

With three symbols the ranks can only ever be 0, 0.5 and 1.0, which tells the
model nothing about the market and quite a lot about the sample size.
"""


@dataclass(frozen=True, slots=True)
class PanelConfig:
    rank_features: bool = True
    sector_neutral: bool = True
    min_symbols: int = MIN_SYMBOLS_FOR_RANKING
    drop_incomplete_rows: bool = True
    extra_features: tuple[str, ...] = field(default_factory=tuple)


def base_feature_names() -> tuple[str, ...]:
    """Per-symbol features, before any cross-sectional transformation."""
    return tuple(technical_feature_names()) + MICROSTRUCTURE_FEATURES


def build_symbol_features(frame: pd.DataFrame) -> pd.DataFrame:
    """All per-symbol features for one symbol's bars."""
    out = add_technical_features(frame)
    return add_microstructure_features(out)


def cross_sectional_rank(frame: pd.DataFrame, column: str, *, min_symbols: int) -> pd.Series:
    """Percentile rank of `column` within each timestamp.

    `groupby("timestamp")` is what makes this point-in-time. Timestamps with too
    few symbols yield NaN rather than a degenerate rank — an unknown must not
    borrow the weight of a measurement.
    """
    grouped = frame.groupby("timestamp", observed=True)[column]
    counts = grouped.transform("count")

    # Mid-rank, (r - 0.5) / n, rather than pandas' pct=True, which is r / n.
    #
    # pct=True puts the mean rank at (n+1)/2n — 0.5625 for 8 symbols, 0.51 for
    # 50. The cross-section size varies bar to bar (a symbol misses a print, a
    # name is suspended), so with pct=True the feature's centre drifts with how
    # many symbols happened to trade. Mid-rank sits at exactly 0.5 for every n,
    # so the same feature value means the same thing on every bar.
    raw_rank = grouped.rank(method="average")
    ranks = (raw_rank - 0.5) / counts

    return ranks.where(counts >= min_symbols)


def sector_neutral_z(frame: pd.DataFrame, column: str, *, min_symbols: int) -> pd.Series:
    """Z-score within (timestamp, sector).

    Symbols with no sector are excluded rather than pooled into a catch-all
    bucket, which would invent a "sector" whose members share nothing.
    """
    if "sector" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)

    grouped = frame.groupby(["timestamp", "sector"], observed=True)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    count = grouped.transform("count")

    z = (frame[column] - mean) / std.replace(0.0, np.nan)
    return z.where(count >= min_symbols)


def build_panel(
    bars: pd.DataFrame,
    config: PanelConfig | None = None,
    *,
    symbol_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the pooled (timestamp x symbol) panel.

    `bars` must carry `symbol`, `timestamp`, and OHLCV. `symbol_meta` optionally
    supplies `sector` for sector-neutral z-scores.
    """
    config = config or PanelConfig()

    if bars.empty:
        return bars

    per_symbol = [
        build_symbol_features(chunk)
        for _, chunk in bars.groupby("symbol", sort=True, observed=True)
    ]
    panel = pd.concat(per_symbol, ignore_index=True)

    if symbol_meta is not None and "sector" in symbol_meta.columns:
        panel = panel.merge(symbol_meta[["symbol", "sector"]], on="symbol", how="left")

    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if config.rank_features:
        panel = _add_cross_sectional(panel, config)

    if config.drop_incomplete_rows:
        panel = _drop_warmup_rows(panel, config)

    return panel.reset_index(drop=True)


def _add_cross_sectional(panel: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    features = [f for f in base_feature_names() if f in panel.columns]

    ranked = {
        f"{name}_pct": cross_sectional_rank(panel, name, min_symbols=config.min_symbols)
        for name in features
    }
    panel = panel.assign(**ranked)

    if config.sector_neutral and "sector" in panel.columns:
        neutral = {
            f"{name}_z_sector": sector_neutral_z(panel, name, min_symbols=config.min_symbols)
            for name in features
        }
        panel = panel.assign(**neutral)

    return panel


def _drop_warmup_rows(panel: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Drop rows where the model's own feature set is not yet defined.

    Only modelling features are considered. Dropping on *any* NaN would also
    discard rows whose display-only columns are missing, which silently shrinks
    the training set for a cosmetic reason.
    """
    features = model_feature_names(panel, config)
    if not features:
        return panel
    return panel.dropna(subset=features)


def model_feature_names(panel: pd.DataFrame, config: PanelConfig | None = None) -> list[str]:
    """The columns actually fed to a model.

    Raw levels are deliberately excluded when ranking is on. A pooled model
    handed `rsi_14` alongside `rsi_14_pct` will happily key off the raw level,
    which re-introduces exactly the per-symbol thinking the panel exists to
    remove.
    """
    config = config or PanelConfig()

    if not config.rank_features:
        return [f for f in base_feature_names() if f in panel.columns]

    names = [c for c in panel.columns if c.endswith("_pct")]
    if config.sector_neutral:
        names += [c for c in panel.columns if c.endswith("_z_sector")]

    # Session position is a genuine absolute quantity: "10 minutes after the
    # open" means the same thing for every symbol, so ranking it across the
    # cross-section would destroy the only information it carries.
    names += [c for c in ("session_sin", "session_cos", "session_fraction") if c in panel.columns]

    names += [c for c in config.extra_features if c in panel.columns]
    return sorted(set(names))


def panel_summary(panel: pd.DataFrame) -> dict[str, object]:
    """Shape facts worth printing after a build."""
    if panel.empty:
        return {"rows": 0, "symbols": 0, "timestamps": 0}

    return {
        "rows": len(panel),
        "symbols": int(panel["symbol"].nunique()),
        "timestamps": int(panel["timestamp"].nunique()),
        "start": str(panel["timestamp"].min()),
        "end": str(panel["timestamp"].max()),
        "median_symbols_per_timestamp": float(
            panel.groupby("timestamp", observed=True)["symbol"].nunique().median()
        ),
    }
