# ADR 0002 — yfinance is deleted; NSE public sources replace it

**Date:** 2026-08-04
**Status:** Accepted

## Context

v1's entire data layer was `yfinance`. Yahoo Finance data is personal-use-only
and `yfinance` is an unofficial client for it. A prior decision on a sibling
project (2026-07-25) established the rule: for a data source that cannot lawfully
be used, the only durable state is **not in the repo** — a config gate is a thing
someone flips later, usually under deadline, usually without re-reading the
licence.

The constraint for this project: free, lawful, no account, no signup, no cost.
It is a personal portfolio project, not a commercial product.

The complication: the sibling project's fix — NSE bhavcopy — is **end-of-day
only**, so it cannot feed an intraday system on its own.

## Decision

Three tiers, none of which require a key, an account, or money.

| Tier | Source | Role |
|---|---|---|
| 1 | **NSE bhavcopy** — official published file | EOD, point-in-time universe, corporate actions, adjustment factors |
| 2 | **NSE public charting endpoint** via [OpenChart](https://github.com/marketcalls/openchart) (MIT, no auth) | Intraday 1m/5m/15m bars |
| 3 | **Deterministic synthetic generator** | Seeded, session-aware bars so a cold clone and CI run with no network |

`yfinance` and `data_engine_v2.py` are **deleted**, not gated.

## Why this is a lawful position

NSE's own site terms permit use "by individuals for personal, non-commercial
use." A personal portfolio project satisfies that. This is a materially better
position than v1's: the problem with yfinance was a *third-party* personal-use-only
feed, and the sibling decision that removed it was driven by that feed sitting
inside a *commercial* product. Neither condition applies here.

The routing-through-the-Economic-Policy-Research-Department requirement in NSE's
data policy governs *subscribing to market data transmitted by NSE* — the
licensed feed product. It does not govern reading public web endpoints.

Guardrails that keep it that way, enforced in code and config:

- **No NSE data enters git at all** — not the lake, not test fixtures. `data/`
  is gitignored, and the test suite runs on a seeded synthetic generator rather
  than a committed slice of real bars. The original plan was to commit a small
  frozen slice as replay fixtures; that was dropped precisely because "a small
  amount of redistribution" is still redistribution, and a synthetic generator
  costs nothing and makes the question disappear.
- The `replay` provider therefore reads whatever the operator has already
  ingested **locally**, which is never committed.
- Nothing is redistributed. The lake is local.
- No commercial use, no paid product, no resale.

## The constraint that shapes the architecture

The public charting endpoint returns roughly a **60–90 day rolling window** for
minute bars. There is no deep one-shot backfill available at any price we are
paying.

Therefore **the lake is append-only and accumulates**: the worker appends each
session, and history grows past what any single API call can return. A design
that assumed a one-shot backfill would cap this project's history permanently at
three months.

## Consequences

- `MarketDataProvider` is a port with adapters `nse_charting`, `replay`,
  `synthetic`. A named-but-unbuilt provider **refuses to boot** — it never falls
  back to synthetic, because falling back is how fabricated data reaches a real
  screen while every log line reads healthy.
- Provenance (`source`) travels **on every row** and has no default. A bar of
  unknown origin cannot be constructed.
- A synthetic-trained model is quarantined and can never be the served model.
- **Stated plainly in the README:** 60–90 days of intraday history spans one,
  maybe two market regimes. The model cannot learn regime dependence from it
  until the lake has accumulated for months. That limitation is published rather
  than papered over.

## What this rules out

Tick data, order-book depth, and true order-flow imbalance — all of which need a
paid licensed feed. The microstructure features in `intelligence` are therefore
**OHLCV-derived estimators** (Corwin-Schultz, Roll, Amihud, tick-rule OFI), and
they are labelled as estimators, not as measured microstructure.
