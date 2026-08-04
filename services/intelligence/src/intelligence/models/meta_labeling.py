"""Meta-labelling (L7) and probability calibration (L8).

**Meta-labelling** splits two jobs that v1 asked one model to do at once:

* primary model — "which way?"
* meta model — "given the primary just said LONG, will *this* LONG be right?"

The meta model finds no new trades. It declines bad ones. For a signal with a
marginal edge — which is every honest equity signal — that is the layer that
makes it usable, because precision improvements compound: declining 40% of the
worst calls helps Sharpe more than a small accuracy gain across all of them.

**The trap.** The meta model must be trained on the primary's *out-of-sample*
predictions. Trained on in-sample calls it learns to trust a model that was
overfit, and the whole construction is fiction. `fit_meta_labeler` therefore
takes OOF predictions and refuses obviously in-sample input.

**Calibration** exists because raw tree-ensemble scores are not probabilities. A
model whose 0.70 bucket is right 55% of the time is overconfident, and Kelly
sizing on that number allocates far too much. Platt scaling is fitted on a
held-out slice, never on the data the base model saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

MIN_META_SAMPLES = 200


@dataclass(frozen=True, slots=True)
class MetaLabeler:
    model: LogisticRegression
    feature_names: tuple[str, ...]
    n_train: int
    positive_rate: float

    def predict_correct_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """P(the primary's call is correct) for each row."""
        X = _meta_matrix(frame, self.feature_names)
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.full(len(frame), float(self.model.classes_[0]))
        return proba[:, 1]


def meta_target(primary_p_up: pd.Series, realised_up: pd.Series) -> pd.Series:
    """Was the primary right?

    1 when the primary's direction matched what happened, 0 otherwise. Note this
    is a question about the *model*, not about the market — which is why it is a
    better-posed and easier target than direction itself.
    """
    primary_says_up = primary_p_up > 0.5
    return (primary_says_up == realised_up.astype(bool)).astype(int)


def _meta_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    return frame[list(feature_names)].to_numpy(dtype=float)


def fit_meta_labeler(
    frame: pd.DataFrame,
    oof_p_up: pd.Series,
    realised_up: pd.Series,
    *,
    context_features: list[str] | None = None,
    sample_weight: pd.Series | None = None,
) -> MetaLabeler:
    """Train the meta model on the primary's out-of-fold calls.

    `oof_p_up` must come from `purged_oof_predictions`. In-sample primary
    predictions would teach the meta model to trust an overfit primary.
    """
    if oof_p_up.notna().sum() < MIN_META_SAMPLES:
        raise ValueError(
            f"only {int(oof_p_up.notna().sum())} out-of-fold predictions; "
            f"need at least {MIN_META_SAMPLES}. Training the meta model on "
            "in-sample primary calls makes the whole stack fiction."
        )

    usable = oof_p_up.notna() & realised_up.notna()
    work = frame.loc[usable].copy()

    # The primary's own conviction is the meta model's most important input:
    # it is being asked when to distrust that conviction.
    work["primary_p_up"] = oof_p_up.loc[usable].to_numpy()
    work["primary_certainty"] = np.maximum(work["primary_p_up"], 1.0 - work["primary_p_up"])

    feature_names = ("primary_p_up", "primary_certainty", *(context_features or []))
    feature_names = tuple(f for f in feature_names if f in work.columns)

    y = meta_target(work["primary_p_up"], realised_up.loc[usable])

    # Class imbalance is severe by construction: if the primary is 52%
    # accurate, the positive class is thin. Balanced weighting rather than
    # letting the model learn "always trust".
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5)
    model.fit(
        _meta_matrix(work, feature_names),
        y,
        sample_weight=sample_weight.loc[usable].to_numpy() if sample_weight is not None else None,
    )

    return MetaLabeler(
        model=model,
        feature_names=feature_names,
        n_train=int(usable.sum()),
        positive_rate=float(y.mean()),
    )


# ------------------------------------------------------------ calibration


@dataclass(frozen=True, slots=True)
class Calibrator:
    method: str
    model: object
    n_train: int

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if self.method == "platt":
            proba = self.model.predict_proba(scores)
            return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        return np.asarray(self.model.predict(scores.ravel()), dtype=float)


def fit_calibrator(
    raw_scores: pd.Series,
    outcomes: pd.Series,
    *,
    method: str = "platt",
) -> Calibrator:
    """Fit a calibration map on a **held-out** slice.

    Calibrating on data the base model trained on is itself an overfit: the
    scores there are already unrealistically good, so the map learned from them
    is wrong everywhere else.
    """
    if method not in {"platt", "isotonic"}:
        raise ValueError(f"unknown calibration method {method!r}")

    valid = raw_scores.notna() & outcomes.notna()
    x = raw_scores[valid].to_numpy(dtype=float)
    y = outcomes[valid].to_numpy(dtype=int)

    if len(x) < 50:
        raise ValueError(f"only {len(x)} points; too few to calibrate honestly")

    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(x.reshape(-1, 1), y)
    else:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(x, y)

    return Calibrator(method=method, model=model, n_train=len(x))
