"""Purged cross-validation with embargo, split by timestamp.

Three things must all be true or the whole evaluation is fiction:

1. **Splits are by timestamp, not by row.** In a pooled panel, every symbol on
   a given bar goes to the same fold. Splitting rows at random puts RELIANCE at
   10:15 in train and TCS at 10:15 in test — and those two rows share the
   market-wide move that dominates both. That is a leak across the
   cross-section, and it is invisible in any per-row shuffle check.

2. **Purge.** A triple-barrier label at time t resolves at t1 > t. If t is in
   train and t1 falls inside the test window, that training row's outcome was
   partly determined by test-period prices. Those rows are removed.

3. **Embargo.** Even after purging, training rows *immediately after* the test
   window carry features (rolling means, volatility) computed from bars inside
   it. A short embargo after each test fold removes them.

v1 did none of this. It used a 60/40 chronological split with no purge and no
embargo, which is the floor and not the bar — and its stacking step, had it had
one, would have leaked outright.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_PCT = 0.01


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    """One fold's positional indices into the panel."""

    fold: int
    train: np.ndarray
    test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


class PurgedTimeSeriesSplit:
    """K-fold over time with purging and an embargo.

    Parameters
    ----------
    n_splits
        Number of test folds, taken in chronological blocks.
    embargo_pct
        Fraction of the total timeline embargoed after each test window.
    expanding
        True gives an expanding train window (all data before the fold);
        False gives a rolling window that also uses data after the embargo.
        Expanding is the honest default for a system that would retrain as
        data arrives.
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        *,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
        expanding: bool = True,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def split(
        self,
        timestamps: pd.Series,
        t1: pd.Series | None = None,
    ) -> Iterator[PurgedSplit]:
        """Yield folds.

        `timestamps` is each row's observation time; `t1` is when its label
        resolves. Without `t1` no purge can be done, and the caller is told so
        rather than silently getting an unpurged split.
        """
        timestamps = pd.Series(timestamps).reset_index(drop=True)
        unique_times = np.sort(timestamps.unique())
        n_times = len(unique_times)

        if n_times < self.n_splits * 2:
            raise ValueError(f"{n_times} distinct timestamps is too few for {self.n_splits} folds")

        embargo_steps = int(n_times * self.embargo_pct)
        fold_bounds = np.array_split(np.arange(n_times), self.n_splits)

        for fold, bounds in enumerate(fold_bounds):
            test_start_time = unique_times[bounds[0]]
            test_end_time = unique_times[bounds[-1]]

            test_mask = (timestamps >= test_start_time) & (timestamps <= test_end_time)

            embargo_end_idx = min(bounds[-1] + embargo_steps, n_times - 1)
            embargo_end_time = unique_times[embargo_end_idx]

            if self.expanding:
                train_mask = timestamps < test_start_time
            else:
                train_mask = (timestamps < test_start_time) | (timestamps > embargo_end_time)

            train_mask = train_mask & ~test_mask

            if t1 is not None:
                train_mask &= ~self._overlaps_test(
                    timestamps, pd.Series(t1).reset_index(drop=True), test_start_time, test_end_time
                )

            if not self.expanding:
                # Embargo the window immediately after the test fold: those rows
                # carry rolling features computed from bars inside it.
                in_embargo = (timestamps > test_end_time) & (timestamps <= embargo_end_time)
                train_mask &= ~in_embargo

            yield PurgedSplit(
                fold=fold,
                train=np.flatnonzero(train_mask.to_numpy()),
                test=np.flatnonzero(test_mask.to_numpy()),
            )

    @staticmethod
    def _overlaps_test(
        t0: pd.Series,
        t1: pd.Series,
        test_start: object,
        test_end: object,
    ) -> pd.Series:
        """Training rows whose label span reaches into the test window.

        A row observed before the test window but resolving inside it had its
        outcome partly set by test-period prices.
        """
        resolves_into_test = t1.notna() & (t1 >= test_start) & (t0 < test_start)
        straddles = (t0 <= test_end) & t1.notna() & (t1 >= test_start)
        return resolves_into_test | straddles


def purged_oof_predictions(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    fit_predict,
    *,
    splitter: PurgedTimeSeriesSplit | None = None,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Out-of-fold predictions produced under purge and embargo.

    **This is the #1 leakage site in a stacked ensemble.** If a meta-learner is
    trained on base-learner OOF predictions that were not purged *and*
    embargoed, the entire stack is silently overfit and every downstream number
    — accuracy, AUC, Sharpe, p-value — is fiction. Producing OOF predictions
    only through this function is what keeps that from happening.
    """
    splitter = splitter or PurgedTimeSeriesSplit()

    out = pd.Series(np.nan, index=panel.index, dtype=float)
    X = panel[features].to_numpy()
    y = panel[target].to_numpy()
    w = weights.to_numpy() if weights is not None else np.ones(len(panel))

    t1 = panel["t1"] if "t1" in panel.columns else None

    for split in splitter.split(panel["timestamp"], t1):
        if split.n_train < 50 or split.n_test == 0:
            continue
        proba = fit_predict(X[split.train], y[split.train], w[split.train], X[split.test])
        out.iloc[split.test] = proba

    return out
