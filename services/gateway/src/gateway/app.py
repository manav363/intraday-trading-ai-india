"""The gateway: the only publicly routable service.

Composition, narrative templating, chart payloads, cache. It owns no data and
no model — it asks market-data and intelligence, and shapes the answer.

Two behaviours the frontend depends on:

* **503 when there is no model, never a neutral 0.5.** A 0.5 would reach the
  narrative layer and be rendered as a genuine call about a real company.
* **Loading, empty and failed are three different states.** Collapsing them is
  how a UI states "no data" as a fact while a request is still in flight.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from intraday_contracts import ErrorCode, ErrorEnvelope, Interval

from .narrative import explain, significance_banner
from .state import GatewayState, get_state

app = FastAPI(
    title="Intraday Trading AI — Gateway",
    version="2.0.0",
    description="The only public surface. Composition and narrative; owns no data.",
)

# The frontend is served from a different origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _error(code: ErrorCode, message: str, status: int, **detail: Any) -> HTTPException:
    envelope = ErrorEnvelope(code=code, message=message, detail=detail)
    return HTTPException(status_code=status, detail=envelope.model_dump(mode="json"))


@app.get("/health")
def health() -> dict[str, object]:
    state = get_state()
    return {
        "status": "ok",
        "lake_present": state.lake_present,
        "model_trained": state.model_card is not None,
        "checked_at": dt.datetime.now(tz=dt.UTC).isoformat(),
    }


@app.get("/api/status")
def status() -> dict[str, object]:
    """What the status bar renders.

    `model_trained` is explicit rather than inferred from a missing field, so
    the frontend can distinguish "still loading" from "genuinely untrained".
    """
    state = get_state()
    card = state.model_card

    return {
        "lake_present": state.lake_present,
        "model_trained": card is not None,
        "symbols_available": len(state.eligible_symbols),
        "significance": (
            significance_banner(card.permutation_p_value, card.permutation_n) if card else None
        ),
        "trained_on_synthetic": card.trained_on_synthetic if card else None,
    }


@app.get("/api/universe")
def universe() -> dict[str, object]:
    """Only symbols the system can answer for.

    A user cannot enter a ticker the system would have to guess about — a
    thin-data symbol is an explained absence, not a low-confidence answer.
    """
    state = get_state()
    return {
        "symbols": state.eligible_symbols,
        "excluded": state.excluded_symbols,
    }


@app.get("/api/model")
def model_card() -> dict[str, object]:
    state = get_state()
    if state.model_card is None:
        raise _error(
            ErrorCode.NO_MODEL,
            "No model has been trained. Predictions are unavailable.",
            503,
        )
    return state.model_card.model_dump(mode="json")


@app.get("/api/bars/{symbol}")
def bars(
    symbol: str,
    interval: Interval = Query(default=Interval.M15),
    limit: int = Query(default=200, ge=10, le=2000),
) -> dict[str, object]:
    """Chart-ready OHLCV."""
    state = get_state()
    frame = state.bars_for(symbol, interval, limit)

    if frame is None:
        raise _error(ErrorCode.UNKNOWN_SYMBOL, f"{symbol} is not in the lake.", 404, symbol=symbol)
    if frame.empty:
        # An empty range is a finding, not a failure — and it is reported as
        # such so the UI can say "no bars in this window" rather than "error".
        return {"symbol": symbol, "interval": interval.value, "bars": []}

    return {
        "symbol": symbol,
        "interval": interval.value,
        "bars": [
            {
                "t": row.timestamp.isoformat(),
                "o": float(row.open),
                "h": float(row.high),
                "l": float(row.low),
                "c": float(row.close),
                "v": float(row.volume),
            }
            for row in frame.itertuples(index=False)
        ],
    }


@app.get("/api/predict/{symbol}")
def predict(symbol: str) -> dict[str, object]:
    state = get_state()

    if state.model_card is None:
        raise _error(
            ErrorCode.NO_MODEL,
            "No model has been trained, so no prediction can be made. "
            "A neutral 0.5 would be rendered as a real call and is not returned.",
            503,
        )
    if symbol not in state.eligible_symbols:
        raise _error(
            ErrorCode.SYMBOL_NOT_ELIGIBLE,
            f"{symbol} does not have enough clean history to be answered honestly.",
            422,
            symbol=symbol,
            reasons=state.excluded_symbols.get(symbol, []),
        )

    prediction = state.predict(symbol)
    if prediction is None:
        raise _error(
            ErrorCode.INSUFFICIENT_HISTORY,
            f"{symbol} has no scorable bars in the current window.",
            422,
            symbol=symbol,
        )

    return {
        "prediction": prediction.model_dump(mode="json"),
        "narrative": explain(prediction),
    }


@app.get("/api/screen")
def screen(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    """Ranked calls across the eligible universe.

    Symbols the model cannot score are **absent**, not present with a
    placeholder. A placeholder in a ranked table reads as a real ranking.
    """
    state = get_state()
    if state.model_card is None:
        raise _error(ErrorCode.NO_MODEL, "No model has been trained.", 503)

    rows = state.screen(limit)
    return {
        "rows": [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "p_up": p.p_up,
                "certainty": p.certainty,
                "meta_probability": p.meta_probability,
                "size_fraction": p.size_fraction,
                "headline": explain(p)["headline"],
            }
            for p in rows
        ],
        "scored": len(rows),
        "universe_size": len(state.eligible_symbols),
    }


@app.get("/api/panel")
def panel_cube(
    limit_symbols: int = Query(default=20, ge=2, le=60),
    limit_times: int = Query(default=60, ge=5, le=300),
) -> dict[str, object]:
    """The cross-sectional cube the 3D view renders.

    `(time x symbol x feature-rank)` is genuinely three-dimensional data, which
    is why this one visualisation earns 3D rather than decorating with it.
    """
    state = get_state()
    cube = state.panel_cube(limit_symbols, limit_times)
    if cube is None:
        raise _error(ErrorCode.NO_DATA, "No panel has been built yet.", 503)
    return cube


__all__ = ["GatewayState", "app"]
