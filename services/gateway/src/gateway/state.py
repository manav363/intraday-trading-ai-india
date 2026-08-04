"""Gateway state: lake access, the trained model, and the derived views.

Everything expensive is memoised by `(symbols, as_of)`. Not as an optimisation
first — as a correctness habit. A sibling project's `screen()` called
`predict()` per symbol, and because the model is cross-sectional each call
rebuilt the entire universe panel; ranking 30 stocks built the same 30-symbol
panel 30 times. In the same function sat a bug of identical origin: a regime was
read from "the last row by date", which in a multi-symbol frame is whichever
symbol sorts last, so every stock reported the same regime. Losing track of
scope is how you both waste work and read the wrong row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
from intelligence.labeling import BarrierConfig
from intelligence.models import decide, predict_proba_up
from intelligence.panel import PanelConfig, build_panel
from intelligence.pipeline import TrainingResult
from intraday_contracts import Interval, ModelCard, Prediction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewayState:
    """What the gateway can currently answer.

    Every field is explicit, including the negative cases. `model_card is None`
    means "genuinely untrained", which the API reports as 503 — distinct from a
    request still in flight, which is the frontend's concern.
    """

    lake_root: Path
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    training: TrainingResult | None = None
    eligible_symbols: list[str] = field(default_factory=list)
    excluded_symbols: dict[str, list[str]] = field(default_factory=dict)
    barrier_config: BarrierConfig = field(default_factory=BarrierConfig)
    panel_config: PanelConfig = field(default_factory=lambda: PanelConfig(sector_neutral=False))

    _panel_cache: pd.DataFrame | None = None

    # -- status ---------------------------------------------------------

    @property
    def lake_present(self) -> bool:
        return not self.bars.empty

    @property
    def model_card(self) -> ModelCard | None:
        return self.training.card if self.training else None

    # -- bars -----------------------------------------------------------

    def bars_for(self, symbol: str, interval: Interval, limit: int) -> pd.DataFrame | None:
        if self.bars.empty or symbol not in set(self.bars["symbol"]):
            return None
        rows = self.bars[self.bars["symbol"] == symbol].sort_values("timestamp")
        return rows.tail(limit)

    # -- panel ----------------------------------------------------------

    def scored_panel(self) -> pd.DataFrame | None:
        """The feature panel, built once for the whole universe.

        Memoised because the model is cross-sectional: scoring one symbol needs
        every symbol's row for that bar, so building per-symbol would repeat the
        identical work once per symbol.
        """
        if self._panel_cache is not None:
            return self._panel_cache
        if self.bars.empty:
            return None

        self._panel_cache = build_panel(self.bars, self.panel_config)
        return self._panel_cache

    def _latest_rows(self) -> pd.DataFrame | None:
        """The most recent bar **per symbol**.

        Taking "the last row by date" from a multi-symbol frame returns
        whichever symbol sorts last — the exact bug that made every stock in a
        sibling project report the same regime.
        """
        panel = self.scored_panel()
        if panel is None or panel.empty:
            return None
        return panel.sort_values("timestamp").groupby("symbol", observed=True).tail(1)

    # -- prediction -----------------------------------------------------

    def predict(self, symbol: str) -> Prediction | None:
        latest = self._latest_rows()
        if latest is None or self.training is None:
            return None

        row = latest[latest["symbol"] == symbol]
        if row.empty:
            return None

        p_up = self._score(row)
        if p_up is None:
            return None

        meta = self._meta_probability(row, p_up)
        return decide(
            symbol=symbol,
            as_of=row.iloc[0]["timestamp"].to_pydatetime(),
            p_up=p_up,
            horizon_bars=self.barrier_config.horizon_bars,
            meta_probability=meta,
            volatility=float(row.iloc[0].get("vol_realised_10", 0.0)) or None,
            is_calibrated=self.training.calibrator is not None,
        )

    def _score(self, row: pd.DataFrame) -> float | None:
        if self.training is None:
            return None
        features = [f for f in self.training.feature_names if f in row.columns]
        if len(features) != len(self.training.feature_names):
            return None
        if row[features].isna().to_numpy().any():
            return None

        model = self.training.served_model
        if model is None:
            return None

        raw = float(predict_proba_up(model, row[features].to_numpy())[0])
        if self.training.calibrator is not None:
            raw = float(self.training.calibrator.transform([raw])[0])
        return raw

    def _meta_probability(self, row: pd.DataFrame, p_up: float) -> float | None:
        labeler = self.training.meta_labeler if self.training else None
        if labeler is None:
            return None

        probe = row.copy()
        probe["primary_p_up"] = p_up
        probe["primary_certainty"] = max(p_up, 1.0 - p_up)
        missing = [f for f in labeler.feature_names if f not in probe.columns]
        if missing:
            return None
        return float(labeler.predict_correct_probability(probe)[0])

    def screen(self, limit: int) -> list[Prediction]:
        """Ranked calls. Unscorable symbols are absent, never placeholders."""
        out: list[Prediction] = []
        for symbol in self.eligible_symbols:
            prediction = self.predict(symbol)
            if prediction is not None:
                out.append(prediction)

        out.sort(key=lambda p: p.certainty, reverse=True)
        return out[:limit]

    # -- 3D cube --------------------------------------------------------

    def panel_cube(self, limit_symbols: int, limit_times: int) -> dict[str, object] | None:
        """`(time x symbol x rank)` for the 3D view."""
        panel = self.scored_panel()
        if panel is None or panel.empty:
            return None

        ranked = [c for c in panel.columns if c.endswith("_pct")]
        if not ranked:
            return None

        feature = "rsi_14_pct" if "rsi_14_pct" in ranked else ranked[0]

        symbols = sorted(panel["symbol"].unique())[:limit_symbols]
        times = sorted(panel["timestamp"].unique())[-limit_times:]

        window = panel[panel["symbol"].isin(symbols) & panel["timestamp"].isin(times)]
        grid = window.pivot_table(
            index="timestamp", columns="symbol", values=feature, observed=True
        ).reindex(index=times, columns=symbols)

        return {
            "feature": feature,
            "available_features": ranked,
            "symbols": symbols,
            "timestamps": [pd.Timestamp(t).isoformat() for t in times],
            # None rather than 0.0 for a missing cell: 0.0 is the bottom of the
            # rank scale and would render as a real, extreme value.
            "values": [[None if pd.isna(v) else float(v) for v in row] for row in grid.to_numpy()],
        }


_STATE: GatewayState | None = None


def set_state(state: GatewayState) -> None:
    global _STATE
    _STATE = state
    get_state.cache_clear()


@lru_cache(maxsize=1)
def get_state() -> GatewayState:
    """The process-wide state.

    Falls back to an empty state rather than raising, so `/health` and
    `/api/status` still answer on a cold stack — which is exactly when someone
    needs to know what is missing.
    """
    if _STATE is not None:
        return _STATE
    return GatewayState(lake_root=Path("data/lake"))


def build_state_from_training(
    bars: pd.DataFrame,
    training: TrainingResult,
    *,
    lake_root: Path | str = "data/lake",
    served_model: object | None = None,
    eligible: list[str] | None = None,
) -> GatewayState:
    """Assemble state after a training run."""
    if served_model is not None:
        training.served_model = served_model

    symbols = eligible if eligible is not None else sorted(bars["symbol"].unique())
    state = GatewayState(
        lake_root=Path(lake_root),
        bars=bars,
        training=training,
        eligible_symbols=symbols,
    )
    logger.info("gateway state: %d symbols, model=%s", len(symbols), training.card.model_id)
    return state


__all__ = [
    "GatewayState",
    "build_state_from_training",
    "get_state",
    "set_state",
]
