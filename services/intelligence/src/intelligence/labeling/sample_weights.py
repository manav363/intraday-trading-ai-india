"""Sample weights by average uniqueness (L3).

Triple-barrier labels **overlap in time**. The observation at bar t and the one
at t+1 can both resolve at t+8, which means they are driven by largely the same
price path. Treating them as independent observations inflates the effective
sample size, and therefore inflates every confidence interval, every p-value,
and every significance claim built on top of them.

This is not a refinement. Without it the statistics are wrong in a direction
that flatters the model, and nothing in a training curve reveals it.

Two quantities:

* **Concurrency** c_t — how many labels are "live" over bar t.
* **Average uniqueness** ū_i — the mean of 1/c_t over label i's own span. A
  label that had the bar to itself scores 1.0; one that shared every bar with
  three others scores ~0.25.

Weights are ū_i normalised to mean 1, so that turning weighting on changes the
relative importance of observations without changing the effective scale of the
loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(index: pd.Series, t1: pd.Series) -> pd.Series:
    """Number of labels spanning each bar.

    Computed with a difference array rather than by counting intervals per bar,
    so it is O(n) instead of O(n · horizon).
    """
    valid = t1.notna()
    starts = index[valid]
    ends = t1[valid]

    timeline = pd.Index(sorted(set(index) | set(ends.dropna())))
    delta = pd.Series(0, index=timeline, dtype=np.int64)

    delta.loc[starts] = delta.loc[starts].add(starts.groupby(starts).size(), fill_value=0)

    # A label spanning [t0, t1] stops contributing *after* t1.
    end_counts = ends.groupby(ends).size()
    for end_ts, count in end_counts.items():
        position = timeline.searchsorted(end_ts, side="right")
        if position < len(timeline):
            delta.iloc[position] -= count

    counts = delta.cumsum()
    return counts.reindex(index).fillna(0).astype(float)


def average_uniqueness(index: pd.Series, t1: pd.Series) -> pd.Series:
    """Mean of 1/concurrency over each label's own span."""
    counts = concurrency(index, t1)
    safe_counts = counts.replace(0, np.nan)
    inverse = (1.0 / safe_counts).fillna(0.0)

    # Cumulative sum lets each label's span be summed in O(1).
    cumulative = inverse.cumsum()
    position = pd.Series(np.arange(len(index)), index=index.to_numpy())

    out = np.zeros(len(index))
    values = cumulative.to_numpy()

    for i, (start, end) in enumerate(zip(index, t1, strict=True)):
        if pd.isna(end):
            continue
        i0 = int(position.loc[start]) if start in position.index else i
        i1 = int(position.loc[end]) if end in position.index else i0
        i1 = max(i1, i0)
        span_sum = values[i1] - (values[i0 - 1] if i0 > 0 else 0.0)
        out[i] = span_sum / (i1 - i0 + 1)

    return pd.Series(out, index=index.to_numpy())


def sample_weights(frame: pd.DataFrame, *, normalise: bool = True) -> pd.Series:
    """Per-row training weights for one symbol.

    Expects `timestamp`, `t1`, and `is_trainable`.
    """
    index = frame["timestamp"]
    t1 = frame["t1"].where(frame["is_trainable"])

    uniqueness = average_uniqueness(index, t1)
    weights = pd.Series(uniqueness.to_numpy(), index=frame.index)
    weights = weights.where(frame["is_trainable"], 0.0)

    if normalise:
        positive = weights[weights > 0]
        if len(positive):
            weights = weights / positive.mean()

    return weights


def panel_sample_weights(frame: pd.DataFrame) -> pd.Series:
    """Weights for a multi-symbol panel.

    Uniqueness is computed **per symbol**, because overlap is a property of one
    symbol's own label spans. Cross-sectional correlation between symbols on the
    same bar is a separate problem, handled by splitting folds on timestamp so
    that correlated rows never straddle a train/test boundary.
    """
    if frame.empty:
        return pd.Series(dtype=float)

    parts = [
        sample_weights(chunk, normalise=False)
        for _, chunk in frame.groupby("symbol", sort=True, observed=True)
    ]
    weights = pd.concat(parts).sort_index()

    positive = weights[weights > 0]
    if len(positive):
        weights = weights / positive.mean()
    return weights


def effective_sample_size(weights: pd.Series) -> float:
    """Kish effective sample size.

    The number worth quoting instead of `len(df)`. With heavily overlapping
    labels the honest count can be a small fraction of the row count, and every
    significance claim should be made against this figure.
    """
    positive = weights[weights > 0].to_numpy()
    if not len(positive):
        return 0.0
    return float(positive.sum() ** 2 / np.square(positive).sum())
