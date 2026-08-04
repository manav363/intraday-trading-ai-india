"""Per-symbol quality score and the eligible universe it produces.

This is how "no symbol silently falls back to a bad answer" is actually
delivered. The frontend only ever offers symbols from the eligible universe, so
a user cannot ask for a ticker the system can't answer honestly. A thin-data
symbol is not a crash and not a fallback — it is an explicit, explained
"not enough history to be honest about this one".
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from intraday_contracts import Interval, QualityScore

MIN_SCORE = 0.85
MIN_SESSIONS = 20
MIN_MEDIAN_TURNOVER = 1_00_00_000.0  # ₹1 crore
MAX_STALE_DAYS = 5

WEIGHTS = {
    "history_completeness": 0.30,
    "validity_clean_rate": 0.25,
    "continuity_clean_rate": 0.15,
    "liquidity_adequacy": 0.20,
    "recency": 0.10,
}


def score_symbol(
    frame: pd.DataFrame,
    symbol: str,
    interval: Interval,
    as_of: dt.date,
    *,
    expected_sessions: int,
    quarantined_rows: int = 0,
) -> QualityScore:
    """Score one symbol's data quality over the window it covers."""
    rows = frame[frame["symbol"] == symbol]
    reasons: list[str] = []

    if rows.empty:
        return QualityScore(
            symbol=symbol,
            as_of=as_of,
            score=0.0,
            history_completeness=0.0,
            validity_clean_rate=0.0,
            continuity_clean_rate=0.0,
            liquidity_adequacy=0.0,
            recency=0.0,
            eligible=False,
            ineligible_reasons=("no data",),
        )

    sessions = rows["timestamp"].dt.date.nunique()
    expected_bars = expected_sessions * interval.bars_per_session

    history_completeness = min(1.0, len(rows) / expected_bars) if expected_bars else 0.0
    validity_clean_rate = (
        len(rows) / (len(rows) + quarantined_rows) if len(rows) + quarantined_rows else 0.0
    )

    # No intraday prev_close to check, so continuity here means "no missing
    # sessions in the middle of the window". Reported honestly rather than
    # hard-coded to 1.0, which would inflate the composite score.
    continuity_clean_rate = min(1.0, sessions / expected_sessions) if expected_sessions else 0.0

    typical = (rows["high"] + rows["low"] + rows["close"]) / 3.0
    daily_turnover = (typical * rows["volume"]).groupby(rows["timestamp"].dt.date).sum()
    median_turnover = float(daily_turnover.median()) if len(daily_turnover) else 0.0
    liquidity_adequacy = min(1.0, median_turnover / MIN_MEDIAN_TURNOVER)

    last_day = rows["timestamp"].dt.date.max()
    stale_days = (as_of - last_day).days
    recency = max(0.0, 1.0 - stale_days / MAX_STALE_DAYS)

    components = {
        "history_completeness": history_completeness,
        "validity_clean_rate": validity_clean_rate,
        "continuity_clean_rate": continuity_clean_rate,
        "liquidity_adequacy": liquidity_adequacy,
        "recency": recency,
    }
    score = sum(components[k] * w for k, w in WEIGHTS.items())

    if sessions < MIN_SESSIONS:
        reasons.append(f"only {sessions} sessions of history (need {MIN_SESSIONS})")
    if median_turnover < MIN_MEDIAN_TURNOVER:
        reasons.append(f"median daily turnover ₹{median_turnover:,.0f} below ₹1 crore")
    if stale_days > MAX_STALE_DAYS:
        reasons.append(f"no data for {stale_days} days")
    if score < MIN_SCORE:
        reasons.append(f"composite quality {score:.2f} below {MIN_SCORE}")

    return QualityScore(
        symbol=symbol,
        as_of=as_of,
        score=round(score, 4),
        history_completeness=round(history_completeness, 4),
        validity_clean_rate=round(validity_clean_rate, 4),
        continuity_clean_rate=round(continuity_clean_rate, 4),
        liquidity_adequacy=round(liquidity_adequacy, 4),
        recency=round(recency, 4),
        eligible=not reasons,
        ineligible_reasons=tuple(reasons),
    )


def eligible_universe(scores: list[QualityScore]) -> tuple[str, ...]:
    return tuple(sorted(s.symbol for s in scores if s.eligible))
