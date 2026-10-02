<div align="center">

# Intraday Trading AI — NSE

**Intraday signal research for Indian equities, built to report its own uncertainty.**

[![CI](https://github.com/manav363/intraday-trading-ai-india/actions/workflows/ci.yml/badge.svg)](https://github.com/manav363/intraday-trading-ai-india/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Nuxt 4](https://img.shields.io/badge/Nuxt-4-00DC82?logo=nuxt&logoColor=white)](https://nuxt.com/)
[![Tests](https://img.shields.io/badge/tests-260%20passing-22c55e)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## Overview

A four-service research platform that generates intraday directional signals for
NSE equities and publishes the statistical evidence for — or against — those
signals alongside them.

It is a **research system**: it places no orders, connects to no broker, and
holds no position overnight. The deliverable is the evidence, not the trade.

The design goal is narrow and unusual: make the *unflattering* numbers the ones a
reader sees first. A permutation p-value sits in the interface on every screen; a
linear baseline is trained specifically to be beaten, and the comparison is
published whichever way it goes.

---

## Contents

- [Relationship to indicant](#relationship-to-indicant)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [Data sources](#data-sources)
- [Modelling pipeline](#modelling-pipeline)
- [Validation](#validation)
- [Getting started](#getting-started)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Relationship to indicant

[**indicant**](https://github.com/manav363/indicant) is the end-of-day sibling of this project. The two share a design philosophy (publish the significance test, refuse rather than guess) and a four-service layout, but they cover different problems:

| | This repo | [indicant](https://github.com/manav363/indicant) |
|---|---|---|
| Question | Which way will a stock move *within the session*? | How likely is a stock to be higher over the coming months? |
| Data | Intraday 1m / 5m / 15m bars from OpenChart, plus daily bhavcopy | Daily NSE bhavcopy archive |
| Distinct work | Session-confined triple-barrier labels, meta-labelling, volatility-scaled sizing | Point-in-time universe, six-tier quality gate, delisted-name retention |
| Interface | Nuxt 4, Vue 3, Three.js | React 18, TypeScript |

If you only read one, read indicant: it is the more complete account of the data and validation work.

---

## Design principles

| Principle | Implementation |
|---|---|
| **Report significance, not just accuracy** | The permutation p-value and its verdict render in the status bar on every screen, not on a separate page. |
| **Never claim impossible precision** | With *N* permutations the minimum achievable p-value is `1/(N+1)`. The `ModelCard` contract rejects any value below that floor, so `p = 0` cannot be reported. |
| **Publish the baseline comparison** | An ElasticNet is trained alongside the ensemble expressly to be beaten. When it wins, that result is shown. |
| **Refuse rather than guess** | With no trained model, prediction endpoints return `503` — never a neutral `0.5`, which the narrative layer would render as a genuine call about a real company. |
| **Costs enter the objective** | A round trip costs approximately 0.08% before slippage. Barrier widths are chosen against that floor rather than having costs subtracted afterwards. |
| **Distinguish absent from zero** | A missing value renders as an em-dash. Loading, empty, and failed are three separate UI states. |

---

## Architecture

```
                    web  ·  Nuxt 4 · Vue 3 · Three.js
                     │
                     │  HTTP/JSON — the only public surface
                     ▼
              ┌── gateway ──┐   composition · narrative · chart payloads
              │             │
     ┌────────▼───┐   ┌─────▼──────────┐
     │ market-data│   │  intelligence  │
     │ ingest     │   │  panel         │
     │ quality    │   │  labelling     │
     │ adjust     │   │  models        │
     │ store      │   │  validation    │
     └──────┬─────┘   └─────┬──────────┘
            │ writes        │ reads (read-only mount)
            ▼               │
        ┌───────────────────▼───┐
        │     PARQUET LAKE      │   market-data is the sole writer
        └───────────┬───────────┘
                    ▲
                worker  ·  scheduled append · retrain · drift
```

Bulk data never crosses the network. Serialising a million rows of OHLCV as JSON
to train a model costs minutes for no benefit — in production ML systems the
feature store *is* the interface between the data and modelling layers. HTTP
carries control-plane queries only.

Services share exactly one dependency: `packages/contracts`. The parquet
partition layout is published there as a contract with a schema-freeze test, so a
column or partition change breaks the build rather than integration.

See [ADR 0001](docs/adr/0001-service-architecture.md).

---

## Data sources

All sources are free, require no account or API key, and permit personal
non-commercial use.

| Tier | Source | Role |
|---|---|---|
| 1 | NSE bhavcopy (official published file) | End-of-day bars, corporate actions, symbol master, liquidity screen |
| 2 | [OpenChart](https://github.com/marketcalls/openchart) — NSE public charting API (MIT, no auth) | Intraday 1m / 5m / 15m bars |
| 3 | Deterministic synthetic generator | Offline tests and CI, with no network access |

**No NSE data is committed to this repository** — not the lake, and not test
fixtures. The test suite runs entirely on generated bars so that the
redistribution question does not arise. `yfinance` was removed rather than
feature-flagged; see [ADR 0002](docs/adr/0002-data-sources.md).

> **Architectural constraint.** The public endpoint serves a rolling **60–90 day**
> window and no deeper backfill is available without a commercial licence. The
> lake is therefore **append-only**: the worker appends each session, and stored
> history grows beyond what any single request can return.

---

## Modelling pipeline

| Layer | Component | Description |
|---|---|---|
| **L0** | Panel construction | Pooled cross-sectional panel; per-timestamp percentile ranks and sector-neutral z-scores |
| **L2** | Triple-barrier labelling | Volatility-scaled profit-take, stop-loss and time barriers; evaluated against bar high/low; confined to the session |
| **L3** | Sample weighting | Average uniqueness, correcting for overlapping labels |
| **L4** | Feature engineering | Technical indicators plus OHLCV-derived microstructure estimators |
| **L5** | Base learners | Gradient boosting, random forest, and an ElasticNet baseline; purged out-of-fold predictions |
| **L7** | Meta-labelling | A second model answering "given the primary predicted long, will *this* call be correct?" |
| **L8** | Calibration | Platt / isotonic scaling fitted on held-out scores |
| **L9** | Position sizing | Half-Kelly, volatility-scaled, capped at 10% |

Three quantities are kept strictly separate and are validated as such by the
`Prediction` contract:

- **`p_up`** — direction, `P(upper barrier touched first)`
- **`certainty`** — `max(p, 1−p)`; conviction irrespective of side, used for gating
- **`meta_probability`** — `P(this call is correct)`; the correct Kelly input

---

## Validation

| Method | Question answered |
|---|---|
| Purged K-fold, split by timestamp | Did the model train on future information? |
| Combinatorial purged CV (CPCV) | How stable is the result across backtest paths? |
| Deflated Sharpe ratio | How much of this is selection luck across configurations tried? |
| Permutation test (shuffled within folds) | Could this result arise by chance? |

Folds split on **timestamp**, so every symbol on a given bar is assigned to the
same fold — splitting by row would leak information across the cross-section.
Training rows whose labels resolve inside a test window are purged, and an
embargo removes rows immediately following it.

---

## Getting started

**Requirements:** Python 3.11+, Node 20+

**1. Install**

```bash
make install
```

**2. Start the API**

```bash
.venv/bin/python scripts/dev_gateway.py
```

**3. Start the interface**

```bash
npm run dev --prefix web
```

The terminal is served at `http://localhost:3000`; the API at
`http://localhost:8010`.

The development entry point trains on synthetic data — clearly flagged in the
interface — so that a fresh clone reaches a working screen rather than an empty
one. Synthetic-trained models are marked `trained_on_synthetic` and can never be
promoted to serve.

### Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

Only the web service publishes a port. The lake is bind-mounted, and
`intelligence` and `gateway` mount it read-only so the single-writer boundary is
enforced by the mount rather than by convention.

---

## Project layout

```
packages/
  contracts/          Shared types; the only cross-service dependency
services/
  market-data/        Ingest, quality gate, corporate actions, lake writes
  intelligence/       Panel, labelling, features, models, validation
  gateway/            Public API, composition, narrative templating
  worker/             Scheduled append, retraining, drift detection
web/                  Nuxt 4 · Vue 3 · Three.js
infra/                Docker Compose
docs/adr/             Architecture decision records
scripts/              Development entry points
```

---

## Testing

```bash
make test         # full suite, excluding network-marked tests
make lint         # ruff check and format verification
make imports      # assert every service imports standalone
```

`make imports` is deliberate: a shared virtual environment makes every service's
dependency list appear correct until a container starts and raises
`ModuleNotFoundError` at import time. CI runs it as an explicit step.

The suite emphasises properties that fail silently rather than loudly:

- **Leakage tests** assert that computing a feature over a longer series does not
  change values already computed on a shorter prefix.
- **Adversarial fixtures** provide one hand-built failing case per quality rule.
- **Contract tests** verify that invalid states cannot be constructed.

---

## Known limitations

These are stated plainly because they materially affect how results should be
read.

- **History is short.** A 60–90 day window spans one or two market regimes. The
  model cannot learn regime dependence until the lake has accumulated over a
  longer period.
- **Microstructure is estimated, not measured.** Genuine order-flow imbalance
  requires the trade-and-quote tape, and a genuine spread requires the order
  book. The features here are OHLCV-derived estimators and are named `*_est`
  where the distinction matters.
- **The NSE holiday calendar is maintained manually.** An unverified year raises
  rather than assuming the market was open. Each year must be added from NSE's
  published circular; several holidays follow lunar calendars and cannot be
  extrapolated.
- **No execution modelling.** There is no broker integration, no queue-position
  model, and no live data path.

---

## How this was built

This project was built with AI coding assistance (Claude Code). The reasoning behind the design is recorded in the [architecture decision records](docs/adr/), and the behaviour described here is covered by the test suite, which CI runs on every push and which also guards the published test count.

---

## License

Released under the [MIT License](LICENSE).

**Disclaimer.** This project is provided for research and educational purposes
only. It does not constitute investment advice, and no representation is made
regarding the profitability of any signal it produces. Signals are derived from
publicly available NSE data for personal, non-commercial use.
