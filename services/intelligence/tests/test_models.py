"""Model-stack tests.

The sizing tests pin v1's confidence bug at the layer where it did damage.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from intelligence.models import (
    available_boosters,
    binary_target,
    certainty_of,
    decide,
    fit_calibrator,
    fit_learner,
    fit_meta_labeler,
    kelly_fraction,
    meta_target,
    position_fraction,
    predict_proba_up,
    quantity_for,
)
from intelligence.models.sizing import SizingConfig
from intraday_contracts import IST, Side

TS = dt.datetime(2025, 3, 3, 10, 15, tzinfo=IST)


# ---------------------------------------------------------------- target


def test_binary_target_uses_realised_return_not_barrier_class() -> None:
    """Dropping time-barrier rows would keep only stretches where price moved
    far enough to touch a vol-scaled barrier, biasing training toward volatile
    periods — and the model would be confidently wrong on quiet days."""
    frame = pd.DataFrame({"label_return": [0.01, -0.02, 0.0001, -0.0001]})
    assert binary_target(frame).tolist() == [1, 0, 1, 0]


def test_binary_target_requires_labelling_first() -> None:
    with pytest.raises(KeyError, match="label_return"):
        binary_target(pd.DataFrame({"close": [1.0]}))


# --------------------------------------------------------------- sizing


def test_certainty_is_symmetric_around_a_half() -> None:
    """The one line v1 was missing."""
    assert certainty_of(0.8) == pytest.approx(0.8)
    assert certainty_of(0.2) == pytest.approx(0.8)
    assert certainty_of(0.5) == pytest.approx(0.5)


def test_a_confident_short_is_sized_not_declined() -> None:
    """v1's bug in the place it caused damage: gating on P(up) made every short
    fail the threshold, so the system was long-only by accident."""
    fraction = position_fraction(0.15)
    assert fraction > 0

    call = decide("RELIANCE", TS, 0.15, horizon_bars=8)
    assert call.side is Side.SHORT
    assert call.size_fraction > 0


def test_a_confident_long_is_sized() -> None:
    call = decide("RELIANCE", TS, 0.85, horizon_bars=8)
    assert call.side is Side.LONG
    assert call.size_fraction > 0


def test_an_uncertain_call_is_declined_either_way() -> None:
    for p in (0.52, 0.48):
        call = decide("RELIANCE", TS, p, horizon_bars=8)
        assert call.side is Side.FLAT
        assert call.size_fraction == 0.0


def test_meta_model_can_veto_a_confident_primary() -> None:
    """The whole point of L7: the primary is sure, the meta model is not."""
    sized = decide("RELIANCE", TS, 0.85, horizon_bars=8, meta_probability=0.8)
    vetoed = decide("RELIANCE", TS, 0.85, horizon_bars=8, meta_probability=0.2)

    assert sized.size_fraction > 0
    assert vetoed.side is Side.FLAT
    assert vetoed.size_fraction == 0.0


def test_size_is_capped() -> None:
    """Never full Kelly. A 50% loss needs a 100% gain to recover."""
    assert position_fraction(0.999, meta_probability=0.999) <= 0.10


def test_half_kelly_is_half_of_full_kelly() -> None:
    full = kelly_fraction(0.6, 1.0)
    half = position_fraction(
        0.6, meta_probability=0.6, config=SizingConfig(max_position_fraction=1.0)
    )
    assert half == pytest.approx(full * 0.5)


def test_kelly_declines_when_there_is_no_edge() -> None:
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(0.4, 1.0) == 0.0


def test_volatility_scaling_reduces_exposure() -> None:
    """Same conviction, twice the volatility, half the exposure."""
    calm = position_fraction(0.85, volatility=0.005)
    wild = position_fraction(0.85, volatility=0.04)
    assert wild < calm


def test_probability_is_clamped() -> None:
    call = decide("RELIANCE", TS, 1.7, horizon_bars=8)
    assert call.p_up == 1.0


def test_quantity_rounds_down() -> None:
    """Survival first — never round a position up."""
    assert quantity_for(100_000, price=1050.0, fraction=0.10) == 9
    assert quantity_for(100_000, price=0.0, fraction=0.10) == 0
    assert quantity_for(100_000, price=1050.0, fraction=0.0) == 0


def test_lot_size_is_respected() -> None:
    assert quantity_for(1_000_000, price=1000.0, fraction=0.10, lot_size=25) == 100


# --------------------------------------------------------------- learners


def _training_set(n: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    # A weak but real signal, which is what an honest equity dataset looks like.
    logit = 0.35 * X[:, 0] - 0.25 * X[:, 1] + rng.normal(scale=1.0, size=n)
    y = (logit > 0).astype(int)
    return X, y


@pytest.mark.parametrize("name", ["boosted", "forest", "elasticnet"])
def test_every_base_learner_fits_and_predicts(name: str) -> None:
    X, y = _training_set()
    result = fit_learner(name, X, y)
    proba = predict_proba_up(result.model, X)

    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_learner_raises() -> None:
    X, y = _training_set(100)
    with pytest.raises(KeyError, match="unknown learner"):
        fit_learner("magic", X, y)


def test_sample_weights_are_accepted_by_every_learner() -> None:
    """Uniqueness weights must actually reach the model, including through the
    ElasticNet pipeline, where they need routing by step name."""
    X, y = _training_set(400)
    w = np.linspace(0.5, 1.5, len(X))
    for name in ("boosted", "forest", "elasticnet"):
        assert fit_learner(name, X, y, sample_weight=w).n_train == len(X)


def test_single_class_fold_does_not_crash() -> None:
    """A degenerate fold is a real possibility on thin data; indexing [:, 1]
    blindly would raise."""
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = np.ones(100, dtype=int)
    result = fit_learner("forest", X, y)
    proba = predict_proba_up(result.model, X)
    assert np.allclose(proba, 1.0)


def test_optional_boosters_are_reported_not_assumed() -> None:
    """v1 printed 'XGBoost + LightGBM + RF' regardless of what was installed,
    and its pinned requirements contained neither."""
    status = available_boosters()
    assert set(status) == {"lightgbm", "xgboost"}
    assert all(isinstance(v, bool) for v in status.values())


# ---------------------------------------------------------- meta-labelling


def test_meta_target_asks_whether_the_primary_was_right() -> None:
    p_up = pd.Series([0.8, 0.8, 0.2, 0.2])
    realised = pd.Series([1, 0, 0, 1])
    assert meta_target(p_up, realised).tolist() == [1, 0, 1, 0]


def test_meta_labeler_refuses_too_few_out_of_fold_predictions() -> None:
    """Training on in-sample primary calls teaches it to trust an overfit
    model, and the whole stack becomes fiction."""
    frame = pd.DataFrame({"x": np.arange(50, dtype=float)})
    oof = pd.Series(np.full(50, 0.6))
    realised = pd.Series(np.ones(50, dtype=int))

    with pytest.raises(ValueError, match="out-of-fold"):
        fit_meta_labeler(frame, oof, realised)


def test_meta_labeler_learns_to_distrust_low_conviction_calls() -> None:
    rng = np.random.default_rng(2)
    n = 1500

    p_up = rng.uniform(0.3, 0.9, n)
    certainty = np.maximum(p_up, 1 - p_up)
    # Higher conviction really is more often right, which is what the meta
    # model should discover.
    correct = rng.random(n) < (0.35 + 0.5 * (certainty - 0.5) * 2)
    realised_up = np.where(p_up > 0.5, correct, ~correct).astype(int)

    frame = pd.DataFrame({"vol_ratio": rng.normal(size=n)})
    labeler = fit_meta_labeler(
        frame,
        pd.Series(p_up),
        pd.Series(realised_up),
        context_features=["vol_ratio"],
    )

    probe = pd.DataFrame(
        {
            "primary_p_up": [0.52, 0.95],
            "primary_certainty": [0.52, 0.95],
            "vol_ratio": [0.0, 0.0],
        }
    )
    scores = labeler.predict_correct_probability(probe)
    assert scores[1] > scores[0], "meta model should trust the high-conviction call more"


# ---------------------------------------------------------- calibration


def test_calibration_corrects_overconfidence() -> None:
    """Raw tree scores are not probabilities. A 0.9 score that is right 60% of
    the time must calibrate down, or Kelly allocates far too much."""
    rng = np.random.default_rng(5)
    n = 2000
    raw = pd.Series(rng.uniform(0.6, 0.99, n))
    # True accuracy is far below the raw score.
    outcomes = pd.Series((rng.random(n) < 0.6).astype(int))

    calibrator = fit_calibrator(raw, outcomes)
    calibrated = calibrator.transform(np.array([0.95]))

    assert calibrated[0] < 0.8


def test_calibration_needs_enough_points() -> None:
    with pytest.raises(ValueError, match="too few"):
        fit_calibrator(pd.Series([0.5] * 10), pd.Series([1] * 10))


def test_unknown_calibration_method_rejected() -> None:
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator(pd.Series([0.5] * 100), pd.Series([1, 0] * 50), method="magic")


def test_isotonic_calibration_is_available() -> None:
    rng = np.random.default_rng(7)
    raw = pd.Series(rng.uniform(0, 1, 500))
    outcomes = pd.Series((rng.random(500) < raw).astype(int))

    calibrator = fit_calibrator(raw, outcomes, method="isotonic")
    out = calibrator.transform(np.array([0.2, 0.8]))
    assert 0.0 <= out[0] <= 1.0
    assert out[1] > out[0]
