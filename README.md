# Intraday Trading AI — NSE

Intraday signal research for Indian equities. Four services, a pooled
cross-sectional model, and a terminal that publishes its own p-value.

**It is a research system.** It places no orders, connects to no broker, and
holds nothing overnight. Its product is the evidence, not the signal.

---

## What makes it different from the usual ML-trading repo

Most of them report an accuracy and stop. This one is built so the
uncomfortable numbers are the ones you see first.

| | |
|---|---|
| **Publishes its own p-value** | The permutation result sits in the status bar on every screen. A `/model` page is a page nobody opens. |
| **Cannot report p = 0** | With N permutations the smallest achievable p-value is `1/(N+1)`. The contract rejects anything below that floor. |
| **Publishes when the baseline wins** | An ElasticNet is trained alongside the stack specifically to be beaten. When it isn't, that is shown. |
| **Refuses rather than guesses** | No trained model returns **503**, never a neutral 0.5 that the narrative layer would render as a real call about a real company. |
| **Costs are in the objective** | A round trip costs ~0.08% before slippage. A profit target below that is not a strategy. |

---

## Quick start

No API key, no account, no cost.

```bash
make install
```

```bash
.venv/bin/python scripts/dev_gateway.py
```

```bash
npm run dev --prefix web
```

The terminal is at `http://localhost:3000`. It boots on a synthetic model —
flagged as such in the status bar — so a cold clone lands on a working screen
instead of an honest but useless "no data".

---

## Architecture

```
web (Nuxt 4 · Vue 3 · Three.js)
        │ HTTP/JSON — the only public surface
   ┌────▼──── gateway ── composition · narrative · chart payloads
   │            │                        │
   │      market-data              intelligence
   │   ingest · quality gate      panel · labelling · models
   │   adjust · store             calibration · validation
   │            │                        │
   │            ▼   PARQUET LAKE         │
   │     (market-data = sole writer) ◄───┘ reads directly, mounted read-only
   │            ▲
   └────── worker ── scheduled append · retrain · drift
```

Bulk data never crosses the network. Moving a million rows of OHLCV as JSON to
train a model is minutes of serialisation for no benefit — in production ML
systems the feature store *is* the interface. HTTP carries control queries
only. See [ADR 0001](docs/adr/0001-service-architecture.md).

---

## Data — free, legal, no permission

| Tier | Source | Role |
|---|---|---|
| 1 | NSE bhavcopy (official) | EOD, corporate actions, symbol master, liquidity screen |
| 2 | [OpenChart](https://github.com/marketcalls/openchart) — NSE's public charting API (MIT, no auth) | Intraday 1m/5m/15m bars |
| 3 | Deterministic synthetic generator | Offline tests and CI, with zero network |

NSE's site terms permit personal, non-commercial use, which a portfolio project
satisfies. **No NSE data enters this repository** — not the lake, not test
fixtures. The suite runs on generated bars precisely so the redistribution
question never arises. `yfinance` was deleted rather than gated; see
[ADR 0002](docs/adr/0002-data-sources.md).

**The constraint that shapes everything:** the public endpoint serves a rolling
**60–90 day** window. There is no deep backfill at any price we are paying, so
the lake is **append-only** — the worker appends each session and history grows
past what any single call can return.

---

## The model

| Layer | What |
|---|---|
| **L0** | Pooled cross-sectional panel — per-timestamp percentile ranks, sector-neutral z-scores |
| **L2** | Triple-barrier labels: vol-scaled, checked on high/low, confined to the session |
| **L3** | Average-uniqueness sample weights |
| **L4** | Technical features + microstructure estimators |
| **L5** | Boosted trees, random forest, ElasticNet baseline — purged OOF |
| **L7** | Meta-labelling: "given the primary said long, will *this* long be right?" |
| **L8** | Platt / isotonic calibration on held-out scores |
| **L9** | Half-Kelly × volatility scaling × 10% cap |

Validation is purged K-fold **split by timestamp**, plus CPCV for a
*distribution* of Sharpes, a deflated Sharpe for multiple testing, and a
permutation test with labels shuffled within folds.

### What v1 got wrong

The rewrite exists because of five defects that produced plausible-looking
numbers rather than crashes:

1. **Labels were silently wrong.** `(Future_Return > 0).astype(int)` turned the
   unknowable last rows into label `0`, and the `dropna` meant to remove them
   was a verified no-op. Those rows trained the model, entered the
   out-of-sample metric, and were the exact rows the live plan read.
2. **Short calls were unreachable.** `Confidence` held P(up) while the gate was
   `< 0.55`, so a bearish call could never pass. The system was long-only by
   accident.
3. **VWAP never reset**, though its own comment said it did.
4. **One model per ticker on ~1,200 rows**, which forecloses a sub-1% effect
   before any modelling happens.
5. **The backtest checked stops on the close**, never the bar's high or low.

Each is now a test, and most are unrepresentable in the type system.

---

## Honest limitations

- **History is short.** 60–90 days spans one or two regimes. The model cannot
  learn regime dependence until the lake has accumulated for months.
- **Microstructure is estimated, not measured.** Real order-flow imbalance
  needs the tape; real spread needs the book. These are OHLCV-derived
  estimators, named `*_est` where the gap matters.
- **The NSE holiday calendar is maintained by hand.** An unverified year
  *raises* rather than assuming the market was open. Add each year from NSE's
  circular — several holidays follow lunar calendars and cannot be guessed.
- **No execution, no broker, no live data.**

---

## Development

```bash
make test
```

```bash
make lint && make imports
```

`make imports` asserts every service imports standalone — a shared virtualenv
makes every service's dependency list look correct right up until the container
starts.

---

## Licence & disclaimer

MIT. For research and education only — not investment advice. Signals are
generated from public NSE data for personal, non-commercial use.
