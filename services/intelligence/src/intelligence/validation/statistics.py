"""The statistics that decide whether a number means anything.

| Tool | Question it answers |
|------|--------------------|
| Purged walk-forward | Did I train on the future? |
| CPCV | How stable is this across paths? |
| Deflated Sharpe | How much of this is selection luck? |
| Permutation test | Could this be random? |

v1 reported a single out-of-sample accuracy and an AUC. Neither answers any of
the four, and a lone accuracy near 0.55 on a binary problem is consistent with
both a small real edge and pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe from a **periodic return series**.

    v1 computed `mean(per-trade ROI) / std * sqrt(252)`. That annualisation
    factor is for daily returns; applied to per-trade ROI it produces a number
    with no interpretation — a strategy taking 5 trades a day and one taking 5 a
    year would be scaled identically.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std = clean.std(ddof=1)
    if std == 0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    *,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark), correcting for skew, kurtosis and sample size.

    Returns are not normal — they are skewed and fat-tailed — and a Sharpe
    computed as if they were overstates confidence.
    """
    if n_observations < 2:
        return 0.5

    denominator = np.sqrt(
        1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.5

    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1) / denominator
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    *,
    trial_sharpe_std: float | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe discounted for how many configurations were tried.

    This is the direct answer to "did you just pick the best of 200 runs?" —
    which is the first question anyone serious asks, and the honest answer is
    usually yes. The expected maximum of N draws from a null distribution grows
    with N, so the benchmark the observed Sharpe must clear grows too.

    Returns P(true Sharpe > 0) after that correction. Below ~0.95 the result
    does not survive the multiple-testing it went through.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return probabilistic_sharpe_ratio(
            observed_sharpe, n_observations, skew=skew, kurtosis=kurtosis
        )

    # Expected maximum of n_trials standard normals (Bailey & López de Prado).
    euler_mascheroni = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_z = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2

    spread = trial_sharpe_std if trial_sharpe_std is not None else 1.0 / np.sqrt(n_observations)
    benchmark = spread * expected_max_z

    return probabilistic_sharpe_ratio(
        observed_sharpe,
        n_observations,
        benchmark_sharpe=benchmark,
        skew=skew,
        kurtosis=kurtosis,
    )


@dataclass(frozen=True, slots=True)
class PermutationResult:
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self) -> str:
        if self.is_significant:
            return f"significant (p={self.p_value:.4f})"
        return f"NOT significant (p={self.p_value:.4f}) — not distinguishable from chance"


def permutation_test(
    observed_statistic: float,
    null_statistics: list[float] | np.ndarray,
) -> PermutationResult:
    """One-sided permutation p-value with the finite-sample correction.

    The `+1` in numerator and denominator is not a rounding detail. With N
    permutations the smallest achievable p-value is `1/(N+1)`; computing
    `count/N` can return exactly 0, which claims infinite significance from a
    finite experiment. `ModelCard` rejects a p-value below that floor, so this
    and that validator agree by construction.
    """
    null = np.asarray(null_statistics, dtype=float)
    null = null[np.isfinite(null)]
    n = len(null)

    if n == 0:
        raise ValueError("no valid null statistics")

    n_at_least_as_extreme = int((null >= observed_statistic).sum())
    p_value = (n_at_least_as_extreme + 1) / (n + 1)

    return PermutationResult(
        observed=float(observed_statistic),
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)) if n > 1 else 0.0,
        p_value=float(p_value),
        n_permutations=n,
    )


def shuffle_labels_within_folds(
    labels: pd.Series,
    fold_ids: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle labels **within each fold**, not globally.

    Global shuffling can produce folds where train and test share the same
    shuffled-label structure, which drags the null distribution toward the
    observed statistic and destroys the test's power. Shuffling inside folds
    preserves the fold structure while destroying the feature-label link, which
    is the null being tested.
    """
    out = labels.copy()
    for fold in fold_ids.unique():
        mask = fold_ids == fold
        # .copy() because pandas may hand back a read-only view, and shuffling
        # in place would then raise rather than silently do nothing.
        values = out[mask].to_numpy().copy()
        rng.shuffle(values)
        out[mask] = values
    return out


def brier_score(probabilities: pd.Series, outcomes: pd.Series) -> float:
    """Mean squared error of probabilistic forecasts. 0.25 is a coin flip."""
    p = probabilities.to_numpy(dtype=float)
    y = outcomes.to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if not valid.any():
        return float("nan")
    return float(np.mean((p[valid] - y[valid]) ** 2))


def reliability_curve(
    probabilities: pd.Series,
    outcomes: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs realised frequency, for the calibration plot.

    A model whose 0.70 bucket is right 55% of the time is not 70% confident, it
    is overconfident — and Kelly sizing built on that probability allocates far
    too much.
    """
    frame = pd.DataFrame({"p": probabilities, "y": outcomes}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin_lower", "bin_upper", "predicted", "observed", "count"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["p"], edges, include_lowest=True)

    grouped = frame.groupby("bin", observed=True)
    out = pd.DataFrame(
        {
            "predicted": grouped["p"].mean(),
            "observed": grouped["y"].mean(),
            "count": grouped["y"].size(),
        }
    ).reset_index()

    out["bin_lower"] = out["bin"].apply(lambda b: float(b.left))
    out["bin_upper"] = out["bin"].apply(lambda b: float(b.right))
    return out[["bin_lower", "bin_upper", "predicted", "observed", "count"]]
