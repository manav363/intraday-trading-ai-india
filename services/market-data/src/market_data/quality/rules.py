"""Quality rules for an intraday bar stream.

Each rule takes a frame of bars for one symbol and returns a `RuleResult` plus
the row mask it condemns. Evidence is always populated on failure — a rule that
reports "failed" without saying what it saw cannot be debugged later.

**Which tiers apply here.** The six-tier model is designed for a daily bhavcopy
file. For an intraday bar stream the applicable set is narrower, and saying so
is better than shipping empty rule stubs:

* **Tier 1 (structural)** — handled by `OHLCVBar` at construction. A bar that
  reaches this module already parsed.
* **Tier 2 (validity)** — applies. Raw frames read back from parquet have not
  been through the model, so geometry is re-checked here.
* **Tier 3 (completeness)** — applies, and matters most: a session missing 40%
  of its bars is the common real failure of the public endpoint.
* **Tier 4 (continuity)** — lives in the bhavcopy path, not here. Intraday bars
  have no `prev_close` field to check against.
* **Tier 5 (plausibility)** — applies.
* **Tier 6 (cross-source)** — not implemented. There is no second free
  intraday source to confirm against, and inventing one would be decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from intraday_contracts import Interval, RuleResult, Severity, Tier

# NSE applies circuit limits of 2/5/10/20% depending on the scrip. 20% is the
# widest, so a single-bar move beyond it is either a corporate action or bad data.
CIRCUIT_LIMIT = 0.20
SIGMA_THRESHOLD = 6.0
STALE_BAR_THRESHOLD = 20
VOLUME_SPIKE_MULTIPLE = 50.0
MIN_SESSION_COVERAGE = 0.80


def _result(
    rule_id: str,
    tier: Tier,
    severity: Severity,
    mask: pd.Series,
    frame: pd.DataFrame,
    evidence: dict[str, object],
) -> RuleResult:
    failed = int(mask.sum())
    return RuleResult(
        rule_id=rule_id,
        tier=tier,
        severity=severity,
        passed=failed == 0,
        affected_symbols=tuple(sorted(frame.loc[mask, "symbol"].unique())) if failed else (),
        affected_rows=failed,
        evidence=evidence if failed else {},
    )


# ------------------------------------------------------------ Tier 2


def check_geometry(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """high >= low, high >= body max, low <= body min."""
    body_hi = frame[["open", "close"]].max(axis=1)
    body_lo = frame[["open", "close"]].min(axis=1)
    bad = (frame["high"] < frame["low"]) | (frame["high"] < body_hi) | (frame["low"] > body_lo)

    return (
        _result(
            "T2.validity.geometry",
            Tier.VALIDITY,
            Severity.FATAL,
            bad,
            frame,
            {
                "failing_rows": int(bad.sum()),
                "example": frame.loc[bad, ["symbol", "open", "high", "low", "close"]]
                .head(3)
                .to_dict("records"),
            },
        ),
        bad,
    )


def check_positive_prices(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    price_cols = ["open", "high", "low", "close"]
    bad = (frame[price_cols] <= 0).any(axis=1) | frame[price_cols].isna().any(axis=1)

    return (
        _result(
            "T2.validity.positive_price",
            Tier.VALIDITY,
            Severity.FATAL,
            bad,
            frame,
            {"failing_rows": int(bad.sum())},
        ),
        bad,
    )


def check_volume_non_negative(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    bad = (frame["volume"] < 0) | frame["volume"].isna()
    return (
        _result(
            "T2.validity.volume_sign",
            Tier.VALIDITY,
            Severity.FATAL,
            bad,
            frame,
            {"failing_rows": int(bad.sum())},
        ),
        bad,
    )


# ------------------------------------------------------------ Tier 3


def check_session_coverage(
    frame: pd.DataFrame,
    interval: Interval,
    *,
    minimum: float = MIN_SESSION_COVERAGE,
) -> RuleResult:
    """Each trading day should carry close to a full session of bars.

    The usual real-world failure of the public endpoint is a partial session,
    not an absent one — and a partial session is far more dangerous, because it
    looks like data.
    """
    expected = interval.bars_per_session
    per_day = frame.groupby(frame["timestamp"].dt.date).size()
    short_days = per_day[per_day < expected * minimum]

    return RuleResult(
        rule_id="T3.completeness.session_coverage",
        tier=Tier.COMPLETENESS,
        severity=Severity.WARNING,
        passed=short_days.empty,
        affected_symbols=tuple(sorted(frame["symbol"].unique())) if not short_days.empty else (),
        affected_rows=int(short_days.sum()),
        evidence={
            "expected_bars_per_session": expected,
            "minimum_fraction": minimum,
            "short_days": {str(d): int(n) for d, n in short_days.head(10).items()},
        }
        if not short_days.empty
        else {},
    )


def check_duplicate_timestamps(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    bad = frame.duplicated(subset=["symbol", "timestamp"], keep="first")
    return (
        _result(
            "T3.completeness.duplicate_bars",
            Tier.COMPLETENESS,
            Severity.ERROR,
            bad,
            frame,
            {"duplicate_rows": int(bad.sum())},
        ),
        bad,
    )


# ------------------------------------------------------------ Tier 5


def check_circuit_limit(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """Bar-to-bar moves beyond the widest circuit band are not real prices."""
    returns = frame.groupby("symbol")["close"].pct_change()
    bad = returns.abs() > CIRCUIT_LIMIT
    bad = bad.fillna(value=False)

    return (
        _result(
            "T5.plausibility.circuit_limit",
            Tier.PLAUSIBILITY,
            Severity.ERROR,
            bad,
            frame,
            {
                "limit": CIRCUIT_LIMIT,
                "max_abs_return": float(returns.abs().max()) if len(returns) else 0.0,
            },
        ),
        bad,
    )


def check_sigma_outliers(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """Moves beyond 6 sigma of trailing volatility.

    The trailing window is strictly past-only. A rule that used the full-sample
    standard deviation would be leaking the future into a quality decision,
    which then propagates into which rows the model trains on.
    """
    returns = frame.groupby("symbol")["close"].pct_change()
    trailing_sigma = returns.groupby(frame["symbol"]).transform(
        lambda s: s.shift(1).rolling(100, min_periods=30).std()
    )
    bad = (returns.abs() > SIGMA_THRESHOLD * trailing_sigma).fillna(value=False)

    return (
        _result(
            "T5.plausibility.sigma_outlier",
            Tier.PLAUSIBILITY,
            Severity.WARNING,
            bad,
            frame,
            {"threshold_sigma": SIGMA_THRESHOLD, "flagged": int(bad.sum())},
        ),
        bad,
    )


def check_staleness(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """An identical close for many consecutive bars means halted or illiquid."""
    unchanged = frame.groupby("symbol")["close"].diff() == 0
    run_id = (~unchanged).cumsum()
    run_length = unchanged.groupby([frame["symbol"], run_id]).transform("sum")
    bad = unchanged & (run_length >= STALE_BAR_THRESHOLD)

    return (
        _result(
            "T5.plausibility.staleness",
            Tier.PLAUSIBILITY,
            Severity.WARNING,
            bad,
            frame,
            {
                "threshold_bars": STALE_BAR_THRESHOLD,
                "longest_run": int(run_length.max()) if len(run_length) else 0,
            },
        ),
        bad,
    )


def check_zero_volume_with_price_move(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """A price cannot change without a trade."""
    moved = frame.groupby("symbol")["close"].diff().abs() > 0
    bad = (frame["volume"] == 0) & moved.fillna(value=False)

    return (
        _result(
            "T5.plausibility.zero_volume_move",
            Tier.PLAUSIBILITY,
            Severity.ERROR,
            bad,
            frame,
            {"failing_rows": int(bad.sum())},
        ),
        bad,
    )


def check_volume_spike(frame: pd.DataFrame) -> tuple[RuleResult, pd.Series]:
    """Volume far beyond the trailing median usually means a unit error."""
    median = frame.groupby("symbol")["volume"].transform(
        lambda s: s.shift(1).rolling(100, min_periods=30).median()
    )
    bad = ((frame["volume"] > VOLUME_SPIKE_MULTIPLE * median) & (median > 0)).fillna(value=False)

    return (
        _result(
            "T5.plausibility.volume_spike",
            Tier.PLAUSIBILITY,
            Severity.WARNING,
            bad,
            frame,
            {"multiple": VOLUME_SPIKE_MULTIPLE, "flagged": int(bad.sum())},
        ),
        bad,
    )


ROW_RULES = (
    check_geometry,
    check_positive_prices,
    check_volume_non_negative,
    check_duplicate_timestamps,
    check_circuit_limit,
    check_sigma_outliers,
    check_staleness,
    check_zero_volume_with_price_move,
    check_volume_spike,
)
"""Rules that condemn individual rows. Order is irrelevant — masks are unioned."""


def empty_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
