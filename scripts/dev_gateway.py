"""Boot the gateway against a freshly trained synthetic model.

The development entry point. It exists so `docker compose up` and a local run
land on a working terminal rather than an honest-but-useless "no lake, no
model" screen.

Synthetic by design here: the model it trains is flagged `trained_on_synthetic`
and the UI says so in the status bar. Nothing about that is hidden.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import uvicorn
from gateway.app import app
from gateway.state import build_state_from_training
from intelligence.labeling import BarrierConfig
from intelligence.panel import PanelConfig
from intelligence.pipeline import train
from intraday_contracts import IST, Interval
from market_data.calendar import NSECalendar
from market_data.providers import build_provider
from market_data.store import bars_to_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dev")

SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "LT",
    "KOTAKBANK", "AXISBANK",
]


def main() -> None:
    provider = build_provider("synthetic", seed=7)

    # Anchored to the last date the NSE holiday calendar has been VERIFIED
    # through, not to today.
    #
    # The calendar deliberately raises for an unverified year rather than
    # assuming every weekday was a trading day — that guard is the point, and
    # working around it by fabricating a holiday list would defeat it. Real
    # ingest will hit the same wall the day the year rolls over, and the fix is
    # to add the year from NSE's published circular (see
    # `market_data.calendar._HOLIDAYS`), which is a two-minute job done once a
    # year and cannot be guessed: several NSE holidays follow lunar calendars
    # and move by more than a fortnight.
    calendar = NSECalendar()
    end = dt.datetime(calendar.verified_through, 12, 20, 23, 59, tzinfo=IST)
    start = end - dt.timedelta(days=90)
    logger.info("using verified calendar window ending %s", end.date())

    logger.info("generating bars for %d symbols", len(SYMBOLS))
    bars = pd.concat(
        [bars_to_frame(provider.fetch_bars(s, Interval.M15, start, end)) for s in SYMBOLS],
        ignore_index=True,
    )
    logger.info("%d bars", len(bars))

    logger.info("training (this takes a moment)")
    result = train(
        bars,
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=5,
        n_permutations=10,
        n_trials=8,
    )
    print("\n" + result.summary() + "\n")

    from gateway.state import set_state

    set_state(build_state_from_training(bars, result, eligible=SYMBOLS))
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="warning")


if __name__ == "__main__":
    main()
