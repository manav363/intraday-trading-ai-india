"""Indian intraday trading costs.

**Costs belong in the objective, not subtracted at the end.** Research is blunt
that half a basis point per trade is lethal to a strategy turning over five
times a session — and an intraday model turns over far more than that. A
strategy evaluated gross and costed afterwards is not the strategy that would
have been traded.

Every rate below is a published statutory or exchange charge for the NSE equity
**intraday** segment as of 2026. Delivery trades are charged differently and are
out of scope: this system never holds overnight.

The one genuinely uncertain input is slippage, which depends on order size
relative to available liquidity. It is modelled explicitly rather than folded
into a single fudge factor, so it can be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- statutory and exchange charges (NSE equity intraday) ------------------

BROKERAGE_RATE = 0.0003
BROKERAGE_CAP = 20.0
"""Discount-broker standard: 0.03% or ₹20 per executed order, whichever is lower."""

STT_SELL_RATE = 0.00025
"""Securities Transaction Tax: 0.025%, **sell side only**, on intraday."""

EXCHANGE_TXN_RATE = 0.0000297
"""NSE transaction charge, both sides."""

SEBI_TURNOVER_RATE = 0.000001
"""₹10 per crore."""

STAMP_DUTY_RATE = 0.00003
"""0.003%, **buy side only**, intraday."""

GST_RATE = 0.18
"""On brokerage + exchange + SEBI charges. NOT on STT or stamp duty — those are
taxes and are not themselves taxable."""


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost model inputs.

    `slippage_bps` is the honest knob. It is a real, size-dependent quantity
    that OHLCV cannot measure, so it is exposed rather than buried.
    """

    brokerage_rate: float = BROKERAGE_RATE
    brokerage_cap: float = BROKERAGE_CAP
    slippage_bps: float = 2.0
    """Half-spread plus impact, per side, in basis points. 2bp is optimistic for
    a mid-cap and roughly right for a NIFTY-50 name in normal conditions."""

    include_statutory: bool = True


@dataclass(frozen=True, slots=True)
class TradeCost:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp_duty: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp_duty
            + self.gst
            + self.slippage
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange": self.exchange,
            "sebi": self.sebi,
            "stamp_duty": self.stamp_duty,
            "gst": self.gst,
            "slippage": self.slippage,
            "total": self.total,
        }


def round_trip_cost(
    entry_price: float,
    exit_price: float,
    quantity: float,
    *,
    config: CostConfig | None = None,
) -> TradeCost:
    """Full round-trip cost of one intraday position, in rupees.

    Charged on both legs where the statute charges both legs, and on one where
    it charges one — STT on the sell, stamp duty on the buy. Applying either to
    both sides roughly doubles that component.
    """
    config = config or CostConfig()

    buy_value = entry_price * quantity
    sell_value = exit_price * quantity
    total_value = buy_value + sell_value

    brokerage = min(buy_value * config.brokerage_rate, config.brokerage_cap) + min(
        sell_value * config.brokerage_rate, config.brokerage_cap
    )

    if config.include_statutory:
        stt = sell_value * STT_SELL_RATE
        exchange = total_value * EXCHANGE_TXN_RATE
        sebi = total_value * SEBI_TURNOVER_RATE
        stamp = buy_value * STAMP_DUTY_RATE
        gst = (brokerage + exchange + sebi) * GST_RATE
    else:
        stt = exchange = sebi = stamp = gst = 0.0

    slippage = total_value * (config.slippage_bps / 10_000.0)

    return TradeCost(
        brokerage=brokerage,
        stt=stt,
        exchange=exchange,
        sebi=sebi,
        stamp_duty=stamp,
        gst=gst,
        slippage=slippage,
    )


def cost_in_return_terms(
    entry_price: float,
    quantity: float,
    *,
    config: CostConfig | None = None,
) -> float:
    """Round-trip cost as a fraction of position value, assuming a flat exit.

    This is the number that decides whether a signal is tradable at all. If the
    triple-barrier profit target is 0.3% and the round trip costs 0.12%, forty
    percent of the edge is gone before the model is right about anything.
    """
    cost = round_trip_cost(entry_price, entry_price, quantity, config=config)
    return cost.total / (entry_price * quantity)


def breakeven_move(config: CostConfig | None = None, *, price: float = 1000.0) -> float:
    """The move required just to break even, as a fraction.

    Worth printing next to any barrier width. A profit target below this is not
    a strategy.
    """
    return cost_in_return_terms(price, 1000.0, config=config)


def apply_costs(
    trades: pd.DataFrame,
    *,
    config: CostConfig | None = None,
) -> pd.DataFrame:
    """Attach per-trade costs and net P&L to a trade ledger.

    Expects `entry_price`, `exit_price`, `quantity`, `direction` ('long'/'short')
    and `gross_pnl`.
    """
    config = config or CostConfig()
    if trades.empty:
        return trades.assign(cost=[], net_pnl=[])

    costs = [
        round_trip_cost(row.entry_price, row.exit_price, row.quantity, config=config).total
        for row in trades.itertuples(index=False)
    ]

    out = trades.copy()
    out["cost"] = costs
    out["net_pnl"] = out["gross_pnl"] - out["cost"]
    out["cost_pct_of_gross"] = np.where(
        out["gross_pnl"].abs() > 0, out["cost"] / out["gross_pnl"].abs(), np.nan
    )
    return out
