"""Prediction and model contracts.

The central design decision here is that **direction, certainty and
P(correct) are three different numbers** and this type refuses to let them be
conflated.

v1 collapsed them: it stored `predict_proba[:, 1]` — P(up) — in a field named
`Confidence`, then gated on `Confidence < 0.55`. Since `Prediction == 0` implies
P(up) < 0.5, a short call could never clear the gate. The system was long-only
by accident, and the only route to a SELL was bullish news lifting a bearish
call over the threshold.

So:

* ``p_up``            — direction. P(the up barrier is touched first).
* ``certainty``       — max(p_up, 1 - p_up). Conviction regardless of side.
                        Gates go here. Derived, and validated as derived.
* ``meta_probability``— P(this call is correct), from the meta-labelling model.
                        This is the correct Kelly input, not ``p_up``.

`ExplanationFact` stays numeric. Turning facts into English is the gateway's
job, so wording changes without redeploying the ML service and the narrative
layer is unit-testable against fixed facts.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

_CERTAINTY_TOLERANCE = 1e-9


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    """Declined. Either certainty or P(correct) failed its gate."""


class Direction(StrEnum):
    """Which way a feature pushed the call. For explanation display only."""

    SUPPORTS_UP = "supports_up"
    SUPPORTS_DOWN = "supports_down"


class BarrierOutcome(StrEnum):
    """Which triple-barrier was touched first.

    `UNRESOLVED` exists so that bars whose barriers have not yet resolved are a
    representable, explicit state. v1's bug was that unresolved rows silently
    became label 0 — `(NaN > 0).astype(int)` is `0`, and the `dropna` meant to
    remove them was a no-op. They then trained the model, entered the OOS
    metric, and were the exact rows the live trade plan read.
    """

    PROFIT_TAKE = "profit_take"
    STOP_LOSS = "stop_loss"
    TIME = "time"
    UNRESOLVED = "unresolved"

    @property
    def is_trainable(self) -> bool:
        """Whether a row with this outcome may enter training or evaluation."""
        return self is not BarrierOutcome.UNRESOLVED


class ExplanationFact(BaseModel):
    """One SHAP-derived fact. Structured, never prose."""

    model_config = {"frozen": True}

    feature: str
    display_name: str
    value: float
    display_value: str = Field(description="Pre-formatted, e.g. '+18.2%'.")
    shap: float
    direction: Direction
    rank: int = Field(ge=1)


class Prediction(BaseModel):
    """One call about one symbol at one instant."""

    model_config = {"frozen": True}

    symbol: str
    as_of: dt.datetime
    horizon_bars: int = Field(gt=0)

    p_up: float = Field(ge=0.0, le=1.0, description="Calibrated P(up barrier first).")
    certainty: float = Field(ge=0.5, le=1.0, description="max(p_up, 1 - p_up). Derived.")
    meta_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P(this call is correct), from the meta-labelling model. "
        "None when meta-labelling is disabled.",
    )

    side: Side
    size_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of capital. Half-Kelly x vol scaling x hard cap. "
        "Zero whenever side is FLAT.",
    )

    is_calibrated: bool = Field(
        description="False means p_up is a raw model score. Raw tree-ensemble "
        "scores are not probabilities and must not be shown as percentages."
    )
    facts: tuple[ExplanationFact, ...] = ()

    @model_validator(mode="after")
    def _certainty_is_derived_not_asserted(self) -> Prediction:
        expected = max(self.p_up, 1.0 - self.p_up)
        if abs(self.certainty - expected) > _CERTAINTY_TOLERANCE:
            raise ValueError(
                f"certainty {self.certainty} != max(p_up, 1-p_up) = {expected}. "
                "These drifting apart is the v1 confidence bug."
            )
        return self

    @model_validator(mode="after")
    def _side_agrees_with_probability(self) -> Prediction:
        if self.side is Side.FLAT:
            if self.size_fraction != 0.0:
                raise ValueError(f"side is FLAT but size_fraction is {self.size_fraction}")
            return self
        implied = Side.LONG if self.p_up > 0.5 else Side.SHORT
        if self.side is not implied:
            raise ValueError(
                f"side {self.side} contradicts p_up {self.p_up}; a non-flat call "
                "must point the way its probability points"
            )
        return self


class CPCVResult(BaseModel):
    """Combinatorial purged CV output: a distribution, not a point estimate.

    A strategy with mean Sharpe 0.4 and sd 0.1 is a different object from one
    with mean Sharpe 0.4 and sd 0.8, and single-path walk-forward cannot tell
    them apart.
    """

    model_config = {"frozen": True}

    n_paths: int = Field(gt=0)
    sharpe_mean: float
    sharpe_std: float = Field(ge=0.0)
    sharpe_p05: float
    sharpe_p50: float
    sharpe_p95: float


class ModelCard(BaseModel):
    """What the model is and how much it should be believed.

    Every field here is meant to be shown. A system that reports predictions
    without reporting whether they are distinguishable from chance is hiding the
    single most important number about itself.
    """

    model_config = {"frozen": True}

    model_id: str
    trained_at: dt.datetime
    n_samples: int = Field(gt=0)
    n_symbols: int = Field(gt=0)
    n_features: int = Field(gt=0)

    train_start: dt.datetime
    train_end: dt.datetime

    oos_accuracy: float = Field(ge=0.0, le=1.0)
    oos_auc: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0, description="Lower is better. 0.25 is a coin flip.")

    cpcv: CPCVResult | None = None
    deflated_sharpe: float | None = Field(
        default=None,
        description="Sharpe discounted for the number of configurations tried.",
    )

    permutation_p_value: float = Field(
        gt=0.0,
        le=1.0,
        description="Strictly > 0. With N permutations the minimum achievable "
        "p is 1/(N+1); reporting p=0 claims infinite significance.",
    )
    permutation_n: int = Field(gt=0)

    baseline_oos_auc: float | None = Field(
        default=None,
        description="ElasticNet baseline. If the stack does not beat a "
        "regularised linear model out of sample, that is the finding.",
    )

    trained_on_synthetic: bool = Field(
        description="True quarantines the model. A synthetic-trained artifact is "
        "useful for dev and CI but must never be the served model."
    )

    @property
    def is_significant(self) -> bool:
        return self.permutation_p_value < 0.05

    @property
    def beats_baseline(self) -> bool | None:
        if self.baseline_oos_auc is None:
            return None
        return self.oos_auc > self.baseline_oos_auc

    @model_validator(mode="after")
    def _p_value_respects_permutation_floor(self) -> ModelCard:
        floor = 1.0 / (self.permutation_n + 1)
        if self.permutation_p_value < floor - _CERTAINTY_TOLERANCE:
            raise ValueError(
                f"p={self.permutation_p_value} is below the floor {floor:.6f} "
                f"achievable with {self.permutation_n} permutations"
            )
        return self
