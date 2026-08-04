"""Data quality contracts.

Governing principle, carried over from the Indicant design:

    **Unknown quality fails. Fail-safe, not fail-open.**

Bad data is quarantined *with its evidence* — never silently dropped, never
silently used. `evidence` is non-optional on a failing rule for exactly that
reason: a rule that reports "failed" without saying what it saw cannot be
debugged six weeks later.
"""

from __future__ import annotations

import datetime as dt
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field, model_validator


class Tier(IntEnum):
    """Rule tiers, ordered by how early they fire and how fatal they are."""

    STRUCTURAL = 1
    """Schema, parseability, file-level sanity. Fatal — reject the whole file."""

    VALIDITY = 2
    """Per-row geometry: high >= low, close > 0. Fatal per row — quarantine it."""

    COMPLETENESS = 3
    """Was this a trading day? Are the expected symbols present?"""

    CONTINUITY = 4
    """prev_close(today) == close(yesterday) unless a corporate action explains it.

    This is the leak detector and the largest correctness risk in the ingest
    path. An unexplained mismatch means the adjustment pipeline is wrong, and it
    must be loud rather than smoothed over.
    """

    PLAUSIBILITY = 5
    """Statistical: circuit-limit breaches, 6-sigma moves, staleness, volume spikes."""

    CROSS_SOURCE = 6
    """Agreement with a second source where one is available. Never load-bearing."""


class Severity(StrEnum):
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Verdict(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    """The whole file is unusable."""


class RuleResult(BaseModel):
    """The outcome of one rule against one ingestion batch."""

    model_config = {"frozen": True}

    rule_id: str = Field(
        pattern=r"^T[1-6]\.[a-z_]+\.[a-z_]+$",
        description="Stable dotted id, e.g. 'T4.continuity.prev_close'. The tier "
        "prefix must match the `tier` field — enforced below.",
    )
    tier: Tier
    severity: Severity
    passed: bool
    affected_symbols: tuple[str, ...] = ()
    affected_rows: int = Field(default=0, ge=0)
    evidence: dict[str, object] = Field(
        default_factory=dict,
        description="Actual vs expected. Always populated on failure.",
    )

    @model_validator(mode="after")
    def _tier_matches_rule_id(self) -> RuleResult:
        declared = int(self.rule_id[1])
        if declared != int(self.tier):
            raise ValueError(
                f"rule_id {self.rule_id!r} declares tier {declared} "
                f"but tier field is {int(self.tier)}"
            )
        return self

    @model_validator(mode="after")
    def _failures_carry_evidence(self) -> RuleResult:
        if not self.passed and not self.evidence:
            raise ValueError(
                f"rule {self.rule_id} failed without evidence; a failure that "
                "cannot be explained cannot be fixed"
            )
        return self


class QualityRun(BaseModel):
    """One ingestion batch's full rule outcome."""

    model_config = {"frozen": True}

    run_id: str
    started_at: dt.datetime
    source_ref: str = Field(description="File name or endpoint call that produced the batch.")
    verdict: Verdict
    results: tuple[RuleResult, ...]

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def fatal_failures(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.failures if r.severity is Severity.FATAL)


class QualityScore(BaseModel):
    """Per-symbol quality, in [0, 1], and the eligibility decision it drives.

    `eligible` is stored rather than recomputed on read so that the reason a
    symbol was or was not offered on a given date stays auditable after the
    thresholds change.
    """

    model_config = {"frozen": True}

    symbol: str
    as_of: dt.date
    score: float = Field(ge=0.0, le=1.0)

    history_completeness: float = Field(ge=0.0, le=1.0)
    validity_clean_rate: float = Field(ge=0.0, le=1.0)
    continuity_clean_rate: float = Field(ge=0.0, le=1.0)
    liquidity_adequacy: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)

    eligible: bool
    ineligible_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ineligible_is_explained(self) -> QualityScore:
        if not self.eligible and not self.ineligible_reasons:
            raise ValueError(
                f"{self.symbol} marked ineligible with no reason; the frontend "
                "renders this as an explanation to the user"
            )
        return self
