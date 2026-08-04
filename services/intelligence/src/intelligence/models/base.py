"""Base learners (L5) and the target they learn.

**The target.** Triple-barrier labelling emits +1 (profit-take first), -1
(stop-loss first) and 0 (time barrier). The primary model answers *side*, so the
binary target is the sign of the realised label return:

    y = 1 if label_return > 0 else 0

Using `label_return` rather than dropping time-barrier rows matters. Dropping
them would keep only observations where price moved far enough to touch a
volatility-scaled barrier, which biases the training set toward high-volatility
stretches — and a model trained only on days that moved will be confidently
wrong on days that don't.

**Learner choice.** Shallow beats deep here. The literature is consistent that
boosted trees are more reliable than deeper models at this signal-to-noise
ratio, so there are no LSTMs for show.

`sklearn.HistGradientBoostingClassifier` is the default booster rather than
LightGBM/XGBoost. It is as good at this scale and it removes a hard native
dependency — a sibling project shipped an image that built clean and died at
model load because LightGBM needs `libgomp1`, which `python:slim` lacks. The
optional boosters remain available and are reported when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class Learner(Protocol):
    """The narrow surface the stack needs from any model."""

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None): ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def binary_target(frame: pd.DataFrame) -> pd.Series:
    """Side target from the realised label return.

    Requires `label_return`, which only exists for rows whose barriers
    resolved — so an unresolved row cannot silently become a training example.
    """
    if "label_return" not in frame.columns:
        raise KeyError(
            "binary_target requires 'label_return'; label the panel with "
            "triple_barrier.label_panel first"
        )
    return (frame["label_return"] > 0).astype(int)


def build_boosted() -> Learner:
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=250,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE,
    )


def build_forest() -> Learner:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_elasticnet() -> Learner:
    """The honest baseline.

    This exists to be beaten. If the full stack does not beat a regularised
    linear model out of sample, that is the finding and it gets published —
    which is a stronger artifact than a fabricated edge.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                # `l1_ratio` alone selects elastic-net in current sklearn;
                # the old `penalty="elasticnet"` spelling is deprecated and is
                # removed in 1.10.
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.5,
                    C=0.1,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


BASE_LEARNERS = {
    "boosted": build_boosted,
    "forest": build_forest,
    "elasticnet": build_elasticnet,
}


@dataclass(frozen=True, slots=True)
class FitResult:
    name: str
    model: Learner
    n_train: int


def fit_learner(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> FitResult:
    if name not in BASE_LEARNERS:
        raise KeyError(f"unknown learner {name!r}; known: {', '.join(BASE_LEARNERS)}")

    model = BASE_LEARNERS[name]()

    if sample_weight is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        # A Pipeline routes fit params to a named step. Passing a bare
        # `sample_weight` raises, so the uniqueness weights would never reach
        # the estimator — which is worse than an error, because the model would
        # train fine and silently ignore the correction.
        model.fit(X, y, model__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)

    return FitResult(name=name, model=model, n_train=len(X))


def predict_proba_up(model: Learner, X: np.ndarray) -> np.ndarray:
    """P(class 1) as a flat array.

    A model fitted on a single-class fold has one probability column; returning
    a constant is correct there, and indexing `[:, 1]` blindly would raise.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        only_class = int(getattr(model, "classes_", np.array([0]))[0])
        return np.full(len(X), float(only_class))
    return proba[:, 1]


def available_boosters() -> dict[str, bool]:
    """Which optional boosters are importable.

    Reported rather than silently shrinking the ensemble. v1 printed
    "XGBoost + LightGBM + RF" regardless of what was installed, and its pinned
    requirements contained neither.
    """
    status = {}
    for package in ("lightgbm", "xgboost"):
        try:
            __import__(package)
            status[package] = True
        except ImportError:
            status[package] = False
    return status
