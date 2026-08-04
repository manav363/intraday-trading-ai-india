"""Triple-barrier labelling (L2).

This replaces v1's labelling entirely. v1 asked:

    y = (close[t + h] / close[t] - 1) > 0

which has three defects, all of which this module fixes:

1. **It ignores the path.** A position that fell 4% and recovered to +0.1% got
   the same label as one that rose smoothly. Nobody holds through the first
   path, so the label described a trade that would never have happened.
2. **It used one horizon for every symbol in every regime.** A 45-minute hold
   on a calm IT name and on a volatile mid-cap are not the same bet.
3. **It silently mislabelled the tail.** `(NaN > 0).astype(int)` is `0`, and
   the `dropna(subset=["Target"])` meant to remove those rows was a verified
   no-op. The last `h` rows trained the model as "down", entered the
   out-of-sample metric, and were the rows the live trade plan read.

Here each observation gets three barriers, and the label is **whichever is
touched first**:

        ┌──────────────── profit-take   close_t · (1 + k₁·σ_t)
        │
   close_t ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤ time / session end
        │
        └──────────────── stop-loss     close_t · (1 - k₂·σ_t)

Two intraday-specific decisions:

* **Touches are detected on High and Low, not Close.** A barrier that only a
  close can trigger is not a barrier — the position would really have been
  stopped out intrabar. This is the same defect the v1 backtester had.
* **The time barrier is capped at the end of the session.** This is an intraday
  system; it does not hold overnight, so a label may never depend on a price
  that would have required carrying risk through the close.

Rows whose barriers do not resolve inside the available data are labelled
`UNRESOLVED` and are excluded from training and evaluation by construction —
never coerced to a direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from intraday_contracts import BarrierOutcome

DEFAULT_HORIZON_BARS = 8
DEFAULT_PT_SIGMA = 2.0
DEFAULT_SL_SIGMA = 2.0
DEFAULT_VOL_WINDOW = 50
MIN_VOL = 1e-5


@dataclass(frozen=True, slots=True)
class BarrierConfig:
    """Barrier geometry.

    `pt_sigma` and `sl_sigma` are hyperparameters and belong **inside** the CV
    loop. Tuning them against the same data used to report performance is a
    textbook overfit. Symmetric is the honest default: asymmetric barriers
    encode a directional view, which is the model's job, not the label's.
    """

    horizon_bars: int = DEFAULT_HORIZON_BARS
    pt_sigma: float = DEFAULT_PT_SIGMA
    sl_sigma: float = DEFAULT_SL_SIGMA
    vol_window: int = DEFAULT_VOL_WINDOW
    confine_to_session: bool = True

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.pt_sigma <= 0 or self.sl_sigma <= 0:
            raise ValueError("barrier widths must be positive")


def realised_volatility(close: pd.Series, window: int = DEFAULT_VOL_WINDOW) -> pd.Series:
    """Past-only rolling volatility of log returns.

    `shift(1)` is the whole point. A volatility estimate that includes the
    current bar leaks into the barrier width, and therefore into the label —
    which is the worst possible place for a leak, because every downstream
    metric inherits it and no model diagnostic can detect it.
    """
    log_returns = np.log(close).diff()
    return log_returns.shift(1).rolling(window, min_periods=max(10, window // 5)).std()


def _first_touch_indices(
    high: np.ndarray,
    low: np.ndarray,
    pt_level: np.ndarray,
    sl_level: np.ndarray,
    max_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bars-ahead at which each barrier is first touched, or -1.

    Walks forward one step at a time rather than materialising an (n x horizon)
    matrix. For 10^6 rows and a horizon of 8 that is 8 vectorised passes over
    length-n arrays instead of a 64MB intermediate.
    """
    n = len(high)
    horizon = int(max_steps.max()) if n else 0

    first_pt = np.full(n, -1, dtype=np.int32)
    first_sl = np.full(n, -1, dtype=np.int32)

    for step in range(1, horizon + 1):
        # Forward-shifted high/low; tail positions have no data and stay NaN.
        fwd_high = np.full(n, np.nan)
        fwd_low = np.full(n, np.nan)
        if step < n:
            fwd_high[: n - step] = high[step:]
            fwd_low[: n - step] = low[step:]

        in_window = step <= max_steps

        hit_pt = (first_pt < 0) & in_window & (fwd_high >= pt_level)
        first_pt[hit_pt] = step

        hit_sl = (first_sl < 0) & in_window & (fwd_low <= sl_level)
        first_sl[hit_sl] = step

    return first_pt, first_sl


def _bars_left_in_session(timestamps: pd.Series) -> np.ndarray:
    """How many bars remain after each bar, within its own session."""
    day = timestamps.dt.date
    position = day.groupby(day).cumcount() if hasattr(day, "groupby") else None
    if position is None:  # pragma: no cover - defensive
        raise TypeError("timestamps must be a datetime Series")
    counts = day.map(day.value_counts())
    return (counts.to_numpy() - position.to_numpy() - 1).astype(np.int32)


def label_symbol(
    frame: pd.DataFrame,
    config: BarrierConfig | None = None,
) -> pd.DataFrame:
    """Label one symbol's bars.

    Expects columns `timestamp`, `high`, `low`, `close`, sorted ascending.
    Returns the frame with labelling columns appended.
    """
    config = config or BarrierConfig()
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    close = frame["close"]
    sigma = realised_volatility(close, config.vol_window)

    pt_level = (close * (1.0 + config.pt_sigma * sigma)).to_numpy()
    sl_level = (close * (1.0 - config.sl_sigma * sigma)).to_numpy()

    max_steps = np.full(len(frame), config.horizon_bars, dtype=np.int32)
    if config.confine_to_session:
        # A row is labellable only if the FULL horizon fits inside its own
        # session — not merely "as many bars as happen to be left".
        #
        # Truncating the horizon near the close looks harmless and is not: a
        # 2-bar horizon almost never reaches a volatility-scaled barrier, so
        # those rows resolve to the time barrier (label 0) far more often than
        # mid-session rows do. That injects a systematic, time-of-day-dependent
        # bias into the label distribution, and the model would happily learn
        # "late in the session, nothing happens" as if it were a market fact
        # rather than an artefact of how we cut the labels.
        #
        # Cost: the last `horizon_bars - 1` bars of each session are not
        # trainable. That is the correct price.
        bars_left = _bars_left_in_session(frame["timestamp"])
        max_steps = np.where(bars_left >= config.horizon_bars, max_steps, 0).astype(np.int32)

    first_pt, first_sl = _first_touch_indices(
        frame["high"].to_numpy(),
        frame["low"].to_numpy(),
        pt_level,
        sl_level,
        max_steps,
    )

    outcome, steps = _resolve(first_pt, first_sl, max_steps, sigma.to_numpy())

    out = frame.copy()
    out["vol"] = sigma
    out["pt_level"] = pt_level
    out["sl_level"] = sl_level
    out["barrier_outcome"] = outcome
    out["bars_held"] = steps

    # Resolution timestamp. Needed for sample weights: two labels that resolve
    # over the same bars are not independent observations.
    #
    # Built through pandas rather than np.where so the column keeps its
    # timezone-aware dtype. Going via numpy would drop the IST offset, and a
    # naive resolution time silently shifts every label's span.
    resolved_index = np.where(steps > 0, np.arange(len(out)) + steps, -1)
    resolved_index = np.clip(resolved_index, -1, len(out) - 1)

    t1 = pd.Series(pd.NaT, index=out.index, dtype=frame["timestamp"].dtype)
    resolved = resolved_index >= 0
    if resolved.any():
        t1.loc[resolved] = frame["timestamp"].to_numpy()[resolved_index[resolved]]
    out["t1"] = t1

    out["label"] = _labels_from_outcome(outcome)
    out["is_trainable"] = out["barrier_outcome"] != BarrierOutcome.UNRESOLVED.value

    exit_price = np.where(
        outcome == BarrierOutcome.PROFIT_TAKE.value,
        pt_level,
        np.where(
            outcome == BarrierOutcome.STOP_LOSS.value,
            sl_level,
            close.to_numpy()[np.maximum(resolved_index, 0)],
        ),
    )
    out["exit_price"] = np.where(out["is_trainable"], exit_price, np.nan)
    out["label_return"] = np.where(out["is_trainable"], out["exit_price"] / close - 1.0, np.nan)

    return out


def _resolve(
    first_pt: np.ndarray,
    first_sl: np.ndarray,
    max_steps: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide which barrier came first."""
    n = len(first_pt)
    outcome = np.full(n, BarrierOutcome.UNRESOLVED.value, dtype=object)
    steps = np.zeros(n, dtype=np.int32)

    # A row with no usable volatility estimate has no barrier geometry at all,
    # so it cannot be labelled. It stays UNRESOLVED rather than being handed a
    # default width.
    has_geometry = np.isfinite(sigma) & (sigma > MIN_VOL) & (max_steps > 0)

    pt_hit = first_pt > 0
    sl_hit = first_sl > 0

    pt_first = has_geometry & pt_hit & (~sl_hit | (first_pt <= first_sl))
    sl_first = has_geometry & sl_hit & (~pt_hit | (first_sl < first_pt))
    timed_out = has_geometry & ~pt_hit & ~sl_hit

    outcome[pt_first] = BarrierOutcome.PROFIT_TAKE.value
    steps[pt_first] = first_pt[pt_first]

    outcome[sl_first] = BarrierOutcome.STOP_LOSS.value
    steps[sl_first] = first_sl[sl_first]

    outcome[timed_out] = BarrierOutcome.TIME.value
    steps[timed_out] = max_steps[timed_out]

    return outcome, steps


def _labels_from_outcome(outcome: np.ndarray) -> np.ndarray:
    """+1 profit-take, -1 stop-loss, 0 time barrier, NaN unresolved."""
    label = np.full(len(outcome), np.nan)
    label[outcome == BarrierOutcome.PROFIT_TAKE.value] = 1.0
    label[outcome == BarrierOutcome.STOP_LOSS.value] = -1.0
    label[outcome == BarrierOutcome.TIME.value] = 0.0
    return label


def label_panel(frame: pd.DataFrame, config: BarrierConfig | None = None) -> pd.DataFrame:
    """Label every symbol in a panel, independently.

    Labelling is per-symbol because the barriers are scaled to each symbol's own
    volatility. That scaling is what makes labels comparable across the
    cross-section, which is exactly what a pooled panel model needs.
    """
    if frame.empty:
        return frame

    labelled = [
        label_symbol(chunk, config)
        for _, chunk in frame.groupby("symbol", sort=True, observed=True)
    ]
    out = pd.concat(labelled, ignore_index=True)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def trainable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for training or evaluation.

    Filtering is done here, once, rather than left to each caller to remember.
    The v1 bug was precisely that this filter existed but did nothing.
    """
    return frame[frame["is_trainable"]].copy()
