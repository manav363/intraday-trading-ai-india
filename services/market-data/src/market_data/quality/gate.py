"""The quality gate.

    **Unknown quality fails. Fail-safe, not fail-open.**

Rows that fail a fatal rule are quarantined with the rule that condemned them.
Nothing is deleted — fix a bug, replay the quarantine.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

import pandas as pd
from intraday_contracts import IST, Interval, QualityRun, RuleResult, Severity, Verdict

from .rules import ROW_RULES, check_session_coverage, empty_mask


@dataclass(frozen=True, slots=True)
class GateOutcome:
    run: QualityRun
    clean: pd.DataFrame
    quarantined: pd.DataFrame
    """Failing rows, with a `failed_rule` column naming what condemned them."""

    @property
    def verdict(self) -> Verdict:
        return self.run.verdict


def run_gate(frame: pd.DataFrame, interval: Interval, *, source_ref: str = "") -> GateOutcome:
    """Apply every applicable rule and split the frame.

    An empty input is `REJECTED`, not `PASS`. An empty batch that reports clean
    is indistinguishable from a healthy quiet market, which is how a dead
    ingestion path stays invisible.
    """
    started = dt.datetime.now(tz=IST)
    run_id = uuid.uuid4().hex[:12]

    if frame.empty:
        return GateOutcome(
            run=QualityRun(
                run_id=run_id,
                started_at=started,
                source_ref=source_ref,
                verdict=Verdict.REJECTED,
                results=(
                    RuleResult(
                        rule_id="T1.structural.non_empty",
                        tier=1,
                        severity=Severity.FATAL,
                        passed=False,
                        evidence={"rows": 0},
                    ),
                ),
            ),
            clean=frame,
            quarantined=frame,
        )

    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    results: list[RuleResult] = []
    condemned = empty_mask(frame)
    reason = pd.Series([""] * len(frame), index=frame.index)

    for rule in ROW_RULES:
        result, mask = rule(frame)
        results.append(result)

        if result.severity is Severity.FATAL:
            newly = mask & ~condemned
            reason.loc[newly] = result.rule_id
            condemned = condemned | mask
        elif mask.any():
            # Non-fatal rules annotate but do not quarantine. A 6-sigma move is
            # often a real event, and dropping it would silently remove exactly
            # the observations a volatility model most needs.
            unlabelled = mask & ~condemned & (reason == "")
            reason.loc[unlabelled] = f"flagged:{result.rule_id}"

    results.append(check_session_coverage(frame, interval))

    clean = frame.loc[~condemned].copy()
    quarantined = frame.loc[condemned].copy()
    if not quarantined.empty:
        quarantined["failed_rule"] = reason.loc[condemned]

    return GateOutcome(
        run=QualityRun(
            run_id=run_id,
            started_at=started,
            source_ref=source_ref,
            verdict=_verdict(results),
            results=tuple(results),
        ),
        clean=clean,
        quarantined=quarantined,
    )


def _verdict(results: list[RuleResult]) -> Verdict:
    failures = [r for r in results if not r.passed]
    if not failures:
        return Verdict.PASS
    if any(r.severity is Severity.FATAL for r in failures):
        return Verdict.QUARANTINED
    if any(r.severity is Severity.ERROR for r in failures):
        return Verdict.PASS_WITH_WARNINGS
    return Verdict.PASS_WITH_WARNINGS
