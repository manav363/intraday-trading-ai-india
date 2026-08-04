"""Position sizing (L9) and the decision gate.

This module is where v1's confidence bug actually caused damage, so the fix is
explicit here as well as in the contract:

* v1 stored P(up) in a field called `Confidence` and gated on `< 0.55`. Since
  `Prediction == 0` implies P(up) < 0.5, a short call could never pass. The
  system was long-only by accident, and the only route to a SELL was bullish
  news lifting a bearish call over the threshold.

Here the three quantities stay separate and each does one job:

* **direction** comes from `p_up` vs 0.5
* **the gate** is `certainty = max(p_up, 1 - p_up)`
* **the size** is driven by `P(correct)` from the meta-labelling model, because
  that is the question Kelly actually needs answered

Sizing is half-Kelly, volatility-scaled, hard-capped. Never full Kelly: it
assumes you know the true win rate and payoff, you do not, and the drawdowns are
brutal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from intraday_contracts import Prediction, Side

KELLY_FRACTION = 0.5
MAX_POSITION_FRACTION = 0.10
MIN_CERTAINTY = 0.55
MIN_META_PROBABILITY = 0.50
TARGET_VOLATILITY = 0.01


@dataclass(frozen=True, slots=True)
class SizingConfig:
    kelly_fraction: float = KELLY_FRACTION
    max_position_fraction: float = MAX_POSITION_FRACTION
    min_certainty: float = MIN_CERTAINTY
    min_meta_probability: float = MIN_META_PROBABILITY
    target_volatility: float = TARGET_VOLATILITY
    payoff_ratio: float = 1.0
    """Reward:risk of the barrier geometry. Symmetric barriers give 1.0."""


def certainty_of(p_up: float) -> float:
    """Conviction, regardless of side.

    The single line v1 was missing.
    """
    return max(p_up, 1.0 - p_up)


def kelly_fraction(win_probability: float, payoff_ratio: float = 1.0) -> float:
    """Full Kelly: f* = p - (1-p)/b. Negative means no edge — do not bet."""
    if payoff_ratio <= 0:
        raise ValueError("payoff_ratio must be positive")
    edge = win_probability - (1.0 - win_probability) / payoff_ratio
    return max(0.0, edge)


def position_fraction(
    p_up: float,
    *,
    meta_probability: float | None = None,
    volatility: float | None = None,
    config: SizingConfig | None = None,
) -> float:
    """Fraction of capital to commit. Zero means decline.

    `meta_probability` is preferred over `p_up` as the Kelly input when it is
    available: "will this specific call be right?" is the question Kelly needs,
    and it is a better-posed question than "which way is the market going?".
    """
    config = config or SizingConfig()

    certainty = certainty_of(p_up)
    if certainty < config.min_certainty:
        return 0.0

    win_probability = meta_probability if meta_probability is not None else certainty
    if meta_probability is not None and meta_probability < config.min_meta_probability:
        return 0.0

    full = kelly_fraction(win_probability, config.payoff_ratio)
    fraction = full * config.kelly_fraction

    # Volatility scaling: the same conviction on a name moving twice as much
    # warrants half the exposure.
    if volatility is not None and volatility > 0:
        fraction *= min(1.0, config.target_volatility / volatility)

    return float(np.clip(fraction, 0.0, config.max_position_fraction))


def decide(
    symbol: str,
    as_of,
    p_up: float,
    *,
    horizon_bars: int,
    meta_probability: float | None = None,
    volatility: float | None = None,
    is_calibrated: bool = True,
    facts: tuple = (),
    config: SizingConfig | None = None,
) -> Prediction:
    """Turn a probability into a sized, contract-valid decision.

    Constructing a `Prediction` is itself the assertion: the contract rejects a
    certainty that is not `max(p, 1-p)`, a side that contradicts its
    probability, and a FLAT call with a non-zero size.
    """
    p_up = float(np.clip(p_up, 0.0, 1.0))
    certainty = certainty_of(p_up)

    fraction = position_fraction(
        p_up,
        meta_probability=meta_probability,
        volatility=volatility,
        config=config,
    )

    if fraction <= 0.0:
        side = Side.FLAT
        fraction = 0.0
    else:
        side = Side.LONG if p_up > 0.5 else Side.SHORT

    return Prediction(
        symbol=symbol,
        as_of=as_of,
        horizon_bars=horizon_bars,
        p_up=p_up,
        certainty=certainty,
        meta_probability=meta_probability,
        side=side,
        size_fraction=fraction,
        is_calibrated=is_calibrated,
        facts=facts,
    )


def quantity_for(
    capital: float,
    price: float,
    fraction: float,
    *,
    lot_size: int = 1,
) -> int:
    """Whole shares for a capital fraction. Rounds **down** — survival first."""
    if price <= 0 or fraction <= 0:
        return 0
    raw = (capital * fraction) / price
    return int(raw // lot_size) * lot_size
