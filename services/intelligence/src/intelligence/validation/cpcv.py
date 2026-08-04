"""Combinatorial Purged Cross-Validation.

Walk-forward gives **one** Sharpe per configuration. That is not enough to
survive scrutiny, because a strategy with mean Sharpe 0.4 and standard deviation
0.1 is a completely different object from one with mean 0.4 and standard
deviation 0.8 — and single-path walk-forward cannot tell them apart.

CPCV generates many backtest paths from the same data by holding out different
*combinations* of fold groups, purging and embargoing each. The output is a
distribution.

With N groups held out K at a time you get C(N, K) train/test configurations and
C(N,K)·K/N distinct backtest paths. N=6, K=2 gives 15 configurations and 5
paths, which is a reasonable default: enough spread to be informative without
15 model fits becoming the slowest thing in the suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from intraday_contracts import CPCVResult

DEFAULT_N_GROUPS = 6
DEFAULT_N_TEST_GROUPS = 2


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    combination: int
    test_groups: tuple[int, ...]
    train: np.ndarray
    test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


class CombinatorialPurgedCV:
    """Purged CV over combinations of held-out time groups."""

    def __init__(
        self,
        n_groups: int = DEFAULT_N_GROUPS,
        n_test_groups: int = DEFAULT_N_TEST_GROUPS,
        *,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        if n_groups < 3:
            raise ValueError("n_groups must be >= 3")

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    @property
    def n_combinations(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return self.n_combinations * self.n_test_groups // self.n_groups

    def split(
        self,
        timestamps: pd.Series,
        t1: pd.Series | None = None,
    ) -> Iterator[CPCVSplit]:
        timestamps = pd.Series(timestamps).reset_index(drop=True)
        unique_times = np.sort(timestamps.unique())
        n_times = len(unique_times)

        if n_times < self.n_groups * 2:
            raise ValueError(f"{n_times} timestamps is too few for {self.n_groups} groups")

        group_slices = np.array_split(np.arange(n_times), self.n_groups)
        embargo_steps = int(n_times * self.embargo_pct)

        # Group id per row, so membership is a lookup rather than a scan.
        time_to_group = {}
        for gid, indices in enumerate(group_slices):
            for i in indices:
                time_to_group[unique_times[i]] = gid
        row_group = timestamps.map(time_to_group)

        for combo_id, test_groups in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            test_mask = row_group.isin(test_groups)
            train_mask = ~test_mask

            for gid in test_groups:
                bounds = group_slices[gid]
                start_time = unique_times[bounds[0]]
                end_time = unique_times[bounds[-1]]

                if t1 is not None:
                    overlaps = (
                        pd.Series(t1).reset_index(drop=True).notna()
                        & (pd.Series(t1).reset_index(drop=True) >= start_time)
                        & (timestamps <= end_time)
                    )
                    train_mask &= ~overlaps

                embargo_end_idx = min(bounds[-1] + embargo_steps, n_times - 1)
                embargo_end = unique_times[embargo_end_idx]
                train_mask &= ~((timestamps > end_time) & (timestamps <= embargo_end))

            yield CPCVSplit(
                combination=combo_id,
                test_groups=test_groups,
                train=np.flatnonzero(train_mask.to_numpy()),
                test=np.flatnonzero(test_mask.to_numpy()),
            )


def summarise_paths(path_sharpes: list[float]) -> CPCVResult:
    """Turn per-path Sharpes into the contract the UI publishes."""
    values = np.asarray([s for s in path_sharpes if np.isfinite(s)], dtype=float)
    if len(values) == 0:
        raise ValueError("no finite path Sharpe ratios")

    return CPCVResult(
        n_paths=len(values),
        sharpe_mean=float(values.mean()),
        sharpe_std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        sharpe_p05=float(np.percentile(values, 5)),
        sharpe_p50=float(np.percentile(values, 50)),
        sharpe_p95=float(np.percentile(values, 95)),
    )
