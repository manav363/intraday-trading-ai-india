"""End-to-end training tests.

These assert the *wiring*, not the parts. A green suite proves the tested paths
work; it does not prove they are reachable from an entry point. A sibling
project had 208 passing tests over a registry integration that production could
never call, because the CLI flag that would have called it did not exist.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from intelligence.labeling import BarrierConfig
from intelligence.panel import PanelConfig
from intelligence.pipeline import InsufficientData, train
from intraday_contracts import IST, Interval
from market_data.providers import SyntheticProvider
from market_data.store import bars_to_frame

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def _bars(days: int = 25, symbols: list[str] = SYMBOLS, seed: int = 3) -> pd.DataFrame:
    provider = SyntheticProvider(seed=seed)
    end = dt.datetime(2025, 4, 30, 23, 59, tzinfo=IST)
    start = end - dt.timedelta(days=days * 2)

    frames = [
        bars_to_frame(provider.fetch_bars(symbol, Interval.M15, start, end)) for symbol in symbols
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trained():
    """One real training run, reused. Permutations are expensive."""
    return train(
        _bars(),
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=4,
        n_permutations=8,
        compute_baseline=True,
        seed=1,
    )


# ------------------------------------------------------------- wiring


def test_pipeline_runs_end_to_end(trained) -> None:
    assert trained.card.n_samples > 500
    assert trained.card.n_symbols == len(SYMBOLS)
    assert trained.card.n_features > 10


def test_every_reported_metric_is_out_of_sample(trained) -> None:
    card = trained.card
    assert 0.0 <= card.oos_accuracy <= 1.0
    assert 0.0 <= card.oos_auc <= 1.0
    assert 0.0 <= card.brier_score <= 1.0
    # Predictions exist only where a fold scored them.
    assert trained.oof_p_up.notna().sum() < len(trained.panel)


def test_model_card_is_contract_valid(trained) -> None:
    """Constructing the card is the assertion — it rejects a p-value below the
    permutation floor and a p-value of exactly zero."""
    assert trained.card.permutation_p_value > 0
    assert trained.card.permutation_p_value >= 1 / (trained.card.permutation_n + 1)


def test_synthetic_training_is_flagged_for_quarantine(trained) -> None:
    """A synthetic-trained model may exist for dev and CI, but it must never be
    servable, and the flag is what stops it."""
    assert trained.card.trained_on_synthetic is True


def test_unknown_provenance_counts_as_synthetic() -> None:
    """Fail-safe, not fail-open."""
    bars = _bars(days=25).drop(columns=["source"])
    result = train(
        bars,
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=4,
        n_permutations=0,
        compute_baseline=False,
    )
    assert result.card.trained_on_synthetic is True


def test_baseline_is_computed_and_compared(trained) -> None:
    """The ElasticNet exists to be beaten, and the comparison is published
    either way."""
    assert trained.card.baseline_oos_auc is not None
    assert trained.card.beats_baseline in (True, False)


def test_calibrator_and_meta_labeler_are_wired(trained) -> None:
    # Either they fitted, or the reason is recorded — never silently absent.
    if trained.calibrator is None:
        assert any("calibration skipped" in n for n in trained.notes)
    if trained.meta_labeler is None:
        assert any("meta-labelling skipped" in n for n in trained.notes)


def test_summary_states_significance_plainly(trained) -> None:
    text = trained.summary()
    assert "permutation p=" in text
    assert ("NOT significant" in text) or ("significant" in text)


def test_notes_record_what_could_not_be_established(trained) -> None:
    """A run that skipped a step must say so rather than omit the number."""
    assert isinstance(trained.notes, list)
    if not trained.card.is_significant:
        assert any("NOT distinguishable from chance" in n for n in trained.notes)


# ---------------------------------------------------------- refusals


def test_too_little_data_raises_rather_than_training_anyway() -> None:
    """A model fitted on 80 rows produces a number, and that number is worse
    than no number because it will be shown to someone."""
    with pytest.raises(InsufficientData):
        train(
            _bars(days=2, symbols=SYMBOLS[:3]),
            panel_config=PanelConfig(sector_neutral=False),
            n_permutations=0,
            compute_baseline=False,
        )


def test_empty_input_raises() -> None:
    with pytest.raises(InsufficientData, match="empty"):
        train(pd.DataFrame(), n_permutations=0, compute_baseline=False)


@pytest.mark.slow
def test_deflated_sharpe_falls_as_trials_rise() -> None:
    """Reporting n_trials honestly is the point; understating it inflates the
    result."""
    bars = _bars(days=25)
    once = train(
        bars,
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=4,
        n_permutations=0,
        compute_baseline=False,
        n_trials=1,
    )
    many = train(
        bars,
        panel_config=PanelConfig(sector_neutral=False),
        barrier_config=BarrierConfig(horizon_bars=6, vol_window=30),
        n_splits=4,
        n_permutations=0,
        compute_baseline=False,
        n_trials=500,
    )
    if once.card.deflated_sharpe and many.card.deflated_sharpe:
        assert many.card.deflated_sharpe <= once.card.deflated_sharpe
