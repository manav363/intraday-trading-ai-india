# ADR 0001 — Four services, parquet lake for bulk, HTTP for control

**Date:** 2026-08-04
**Status:** Accepted

## Context

v1 was eight flat Python modules behind an `input()` prompt. Everything imported
everything; there was no boundary between fetching data, engineering features,
training, and deciding a trade. That is workable at 2,000 lines and stops being
workable immediately after.

The open question in splitting it up was how services talk to each other, given
that training reads on the order of 10⁶ rows of OHLCV.

## Options

- **A** — HTTP for everything, including bulk training reads.
- **B** — Shared parquet lake for bulk, HTTP for the control plane.
- **C** — gRPC + Arrow Flight for bulk, REST for control.

## Decision

**B.**

## Why

Moving a million rows of OHLCV as JSON in order to train a model is minutes of
serialisation for zero benefit. In production ML systems the feature store *is*
the interface between the data layer and the modelling layer — that is the
normal shape, not a shortcut.

A is also the exact failure mode that stalled the ai_trade integration phase,
where API contracts, container networking and auth all broke simultaneously.
There is no reason to re-create it for data that does not need to cross a
network.

C is genuinely fast, but adds a protobuf toolchain and a codegen step to every
contract change. That is real, permanent complexity for a solo project.

## Consequences

- `market-data` is the **single writer** to the lake. `intelligence` mounts it
  read-only — the boundary is enforced by the mount, not by a code-review
  comment.
- The partition layout is published in `packages/contracts/lake.py` as
  `LakePaths`, with a schema-freeze test. A column or partition change breaks
  both services at build time rather than at integration time.
- Only `gateway` is publicly routable. `market-data` and `intelligence` are
  internal.
- Services share exactly one thing: `packages/contracts`. No service imports
  another's internals.
- If bulk ever genuinely has to cross a machine boundary, the upgrade is Arrow
  Flight over the same contract — not REST.

## Services

| Service | Owns |
|---|---|
| `market-data` | Ingest, quality gate, corporate-action adjustment, lake writes, PIT universe |
| `intelligence` | Panel, labelling, features, models, calibration, validation, backtest |
| `gateway` | The only public API. Composition, narrative templating, chart payloads, cache |
| `worker` | Scheduled append-only ingest, retraining, drift detection |
