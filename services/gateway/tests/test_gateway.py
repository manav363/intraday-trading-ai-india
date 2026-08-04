"""Gateway tests.

The refusal tests matter most: a system with no model must say so, not return a
neutral 0.5 that the narrative layer would render as a real call about a real
company.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gateway.app import app
from gateway.narrative import explain, headline, significance_banner
from gateway.state import GatewayState, build_state_from_training, set_state
from intelligence.labeling import BarrierConfig
from intelligence.panel import PanelConfig
from intelligence.pipeline import train
from intraday_contracts import IST, Direction, ExplanationFact, Interval, Prediction, Side
from market_data.providers import SyntheticProvider
from market_data.store import bars_to_frame

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
TS = dt.datetime(2025, 3, 3, 10, 15, tzinfo=IST)


def _prediction(**overrides) -> Prediction:
    p_up = overrides.pop("p_up", 0.72)
    base = {
        "symbol": "RELIANCE",
        "as_of": TS,
        "horizon_bars": 6,
        "p_up": p_up,
        "certainty": max(p_up, 1 - p_up),
        "side": Side.LONG if p_up > 0.5 else Side.SHORT,
        "size_fraction": 0.03,
        "is_calibrated": True,
    }
    return Prediction(**(base | overrides))


# ------------------------------------------------------------ narrative


def test_headline_states_direction_plainly() -> None:
    text = headline(_prediction(p_up=0.72))
    assert "RELIANCE" in text
    assert "higher" in text


def test_short_call_reads_as_lower_not_as_a_failed_long() -> None:
    text = headline(_prediction(p_up=0.2))
    assert "lower" in text


def test_every_probability_is_paired_with_failure_framing() -> None:
    """A probability shown alone reads as a promise."""
    payload = explain(_prediction(p_up=0.66))
    assert "66%" in payload["failure_framing"]
    assert "34 have gone the other way" in payload["failure_framing"]


def test_flat_call_is_stated_as_a_decision_not_an_absence() -> None:
    payload = explain(_prediction(p_up=0.51, side=Side.FLAT, size_fraction=0.0))
    assert "no call" in payload["headline"].lower()
    assert "itself a decision" in payload["failure_framing"]


def test_uncalibrated_scores_carry_a_caveat() -> None:
    """Raw tree scores are not probabilities and must not read as percentages."""
    payload = explain(_prediction(is_calibrated=False))
    assert payload["calibration_caveat"] is not None
    assert "ranking" in payload["calibration_caveat"]

    assert explain(_prediction(is_calibrated=True))["calibration_caveat"] is None


def test_meta_note_is_present_only_when_the_meta_model_ran() -> None:
    assert explain(_prediction())["meta_note"] is None
    with_meta = explain(_prediction(meta_probability=0.61))
    assert "61%" in with_meta["meta_note"]


def test_only_the_top_facts_are_narrated() -> None:
    """A list of forty SHAP values is not an explanation."""
    facts = tuple(
        ExplanationFact(
            feature=f"f{i}",
            display_name=f"Feature {i}",
            value=float(i),
            display_value=f"{i}",
            shap=0.1 / (i + 1),
            direction=Direction.SUPPORTS_UP,
            rank=i + 1,
        )
        for i in range(10)
    )
    payload = explain(_prediction(facts=facts))
    assert len(payload["reasons"]) == 3


def test_significance_banner_states_the_verdict() -> None:
    not_sig = significance_banner(0.0597, 200)
    assert not_sig["is_significant"] is False
    assert "NOT distinguishable" in not_sig["detail"]

    sig = significance_banner(0.001, 2000)
    assert sig["is_significant"] is True


# ----------------------------------------------------------------- API


@pytest.fixture
def cold_client() -> TestClient:
    """A stack with no lake and no model — the state someone debugs at 2am."""
    set_state(GatewayState(lake_root=Path("data/lake")))
    return TestClient(app)


@pytest.fixture(scope="module")
def warm_client() -> TestClient:
    provider = SyntheticProvider(seed=4)
    end = dt.datetime(2025, 4, 30, 23, 59, tzinfo=IST)
    start = end - dt.timedelta(days=50)
    bars = pd.concat(
        [bars_to_frame(provider.fetch_bars(s, Interval.M15, start, end)) for s in SYMBOLS],
        ignore_index=True,
    )

    result = train(
        bars,
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=4,
        n_permutations=5,
        compute_baseline=False,
    )
    set_state(build_state_from_training(bars, result, eligible=SYMBOLS))
    return TestClient(app)


def test_health_answers_on_a_cold_stack(cold_client: TestClient) -> None:
    """/health must work exactly when something is wrong."""
    body = cold_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["lake_present"] is False
    assert body["model_trained"] is False


def test_status_distinguishes_untrained_from_unknown(cold_client: TestClient) -> None:
    """`model_trained: false` is a finding. The frontend needs it to be
    explicit so it can tell 'untrained' from 'still loading'."""
    body = cold_client.get("/api/status").json()
    assert body["model_trained"] is False
    assert body["significance"] is None


def test_prediction_without_a_model_is_503_never_a_neutral_half(
    cold_client: TestClient,
) -> None:
    """A 0.5 would reach the narrative layer and be rendered as a genuine call
    about a real company."""
    response = cold_client.get("/api/predict/RELIANCE")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "no_model"


def test_screen_without_a_model_is_503(cold_client: TestClient) -> None:
    assert cold_client.get("/api/screen").status_code == 503


def test_model_card_without_a_model_is_503(cold_client: TestClient) -> None:
    assert cold_client.get("/api/model").status_code == 503


def test_status_reports_significance_when_trained(warm_client: TestClient) -> None:
    body = warm_client.get("/api/status").json()
    assert body["model_trained"] is True
    assert body["significance"]["verdict"] in {"significant", "NOT significant"}
    assert body["trained_on_synthetic"] is True


def test_bars_are_chart_ready(warm_client: TestClient) -> None:
    body = warm_client.get("/api/bars/AAA?limit=50").json()
    assert len(body["bars"]) == 50
    assert set(body["bars"][0]) == {"t", "o", "h", "l", "c", "v"}


def test_unknown_symbol_is_404_not_an_empty_chart(warm_client: TestClient) -> None:
    response = warm_client.get("/api/bars/NOPE")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_symbol"


def test_ineligible_symbol_explains_itself(warm_client: TestClient) -> None:
    response = warm_client.get("/api/predict/ZZZ")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "symbol_not_eligible"


def test_prediction_carries_narrative(warm_client: TestClient) -> None:
    body = warm_client.get("/api/predict/AAA").json()
    assert "headline" in body["narrative"]
    assert "failure_framing" in body["narrative"]
    assert 0.0 <= body["prediction"]["p_up"] <= 1.0


def test_screen_omits_unscorable_symbols_rather_than_placeholdering(
    warm_client: TestClient,
) -> None:
    """A placeholder in a ranked table reads as a real ranking."""
    body = warm_client.get("/api/screen?limit=20").json()
    assert body["scored"] <= body["universe_size"]
    for row in body["rows"]:
        assert row["p_up"] is not None


def test_panel_cube_is_three_dimensional(warm_client: TestClient) -> None:
    body = warm_client.get("/api/panel?limit_symbols=6&limit_times=20").json()
    assert len(body["symbols"]) <= 6
    assert len(body["timestamps"]) <= 20
    assert len(body["values"]) == len(body["timestamps"])
    assert len(body["values"][0]) == len(body["symbols"])


def test_panel_cube_uses_null_for_missing_cells(warm_client: TestClient) -> None:
    """0.0 is the bottom of the rank scale and would render as a real extreme."""
    body = warm_client.get("/api/panel?limit_symbols=6&limit_times=20").json()
    flat = [v for row in body["values"] for v in row]
    assert all(v is None or 0.0 <= v <= 1.0 for v in flat)
