"""End-to-end training: bars in, ModelCard out.

This module exists because a green test suite proves the tested paths work, not
that they are reachable from production. Every layer below is exercised by its
own tests; this is the one place they are wired together, and
`test_pipeline.py` asserts the wiring rather than the parts.

    bars
     -> L0  panel        pooled cross-section, per-timestamp ranks
     -> L2  labelling    triple barrier, vol-scaled, session-confined
     -> L3  weights      average uniqueness
     -> L5  base learner purged OOF predictions
     -> L7  meta-label   P(this call is correct)
     -> L8  calibration  Platt on held-out scores
     -> report           accuracy, AUC, Brier, permutation p, deflated Sharpe

Nothing here reports a number it did not compute out of sample.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from intraday_contracts import IST, ModelCard
from sklearn.metrics import accuracy_score, roc_auc_score

from .labeling import BarrierConfig, label_panel, panel_sample_weights, trainable
from .models import (
    Calibrator,
    MetaLabeler,
    binary_target,
    fit_calibrator,
    fit_learner,
    fit_meta_labeler,
    predict_proba_up,
)
from .panel import PanelConfig, build_panel, model_feature_names, panel_summary
from .validation import (
    PurgedTimeSeriesSplit,
    brier_score,
    deflated_sharpe_ratio,
    permutation_test,
    purged_oof_predictions,
    sharpe_ratio,
)
from .validation.statistics import shuffle_labels_within_folds

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 500
DEFAULT_PERMUTATIONS = 100


@dataclass(slots=True)
class TrainingResult:
    """Everything a training run produced, including what it failed to prove."""

    card: ModelCard
    panel: pd.DataFrame
    oof_p_up: pd.Series
    calibrator: Calibrator | None
    meta_labeler: MetaLabeler | None
    feature_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    served_model: object | None = None
    """The model used for live scoring, fitted on ALL data.

    Its in-sample performance is meaningless and is never reported. Every number
    on the `card` comes from the purged out-of-fold predictions instead. v1
    conflated these: it trained a final model on everything and then let that
    model's own predictions reach the backtest.
    """

    def summary(self) -> str:
        card = self.card
        verdict = "significant" if card.is_significant else "NOT significant"
        lines = [
            f"samples={card.n_samples}  symbols={card.n_symbols}  features={card.n_features}",
            f"OOS accuracy={card.oos_accuracy:.4f}  AUC={card.oos_auc:.4f}  "
            f"Brier={card.brier_score:.4f}",
            f"permutation p={card.permutation_p_value:.4f} ({verdict}, n={card.permutation_n})",
        ]
        if card.baseline_oos_auc is not None:
            beat = "beats" if card.beats_baseline else "does NOT beat"
            lines.append(f"baseline ElasticNet AUC={card.baseline_oos_auc:.4f} — stack {beat} it")
        if card.deflated_sharpe is not None:
            lines.append(f"deflated Sharpe P(SR>0)={card.deflated_sharpe:.4f}")
        lines.extend(self.notes)
        return "\n".join(lines)


class InsufficientData(RuntimeError):
    """Not enough usable rows to train honestly.

    Raised rather than training on whatever is available. A model fitted on 80
    rows produces a number, and that number is worse than no number because it
    will be shown to someone.
    """


def train(
    bars: pd.DataFrame,
    *,
    panel_config: PanelConfig | None = None,
    barrier_config: BarrierConfig | None = None,
    learner: str = "boosted",
    n_splits: int = 5,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_trials: int = 1,
    symbol_meta: pd.DataFrame | None = None,
    compute_baseline: bool = True,
    seed: int = 0,
) -> TrainingResult:
    """Run the whole stack and report what it actually established.

    `n_trials` is how many configurations have been tried across the project so
    far. It feeds the deflated Sharpe, and honestly reporting it is the point —
    understating it inflates the result.
    """
    panel_config = panel_config or PanelConfig()
    barrier_config = barrier_config or BarrierConfig()
    notes: list[str] = []

    panel = build_panel(bars, panel_config, symbol_meta=symbol_meta)
    if panel.empty:
        raise InsufficientData("panel is empty after feature construction")

    labelled = label_panel(panel, barrier_config)
    usable = trainable(labelled).reset_index(drop=True)

    if len(usable) < MIN_TRAINING_ROWS:
        raise InsufficientData(
            f"{len(usable)} trainable rows after labelling; need {MIN_TRAINING_ROWS}. "
            "Ingest more sessions — the lake accumulates."
        )

    features = model_feature_names(usable, panel_config)
    if not features:
        raise InsufficientData("no model features survived panel construction")

    usable = usable.dropna(subset=features).reset_index(drop=True)
    if len(usable) < MIN_TRAINING_ROWS:
        raise InsufficientData(f"{len(usable)} rows remain after dropping incomplete features")

    y = binary_target(usable)
    weights = panel_sample_weights(usable)
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits)

    def fit_predict(X_tr, y_tr, w_tr, X_te):
        result = fit_learner(learner, X_tr, y_tr, sample_weight=w_tr)
        return predict_proba_up(result.model, X_te)

    usable["_target"] = y
    oof = purged_oof_predictions(
        usable, features, "_target", fit_predict, splitter=splitter, weights=weights
    )

    scored = oof.notna()
    if scored.sum() < 100:
        raise InsufficientData(f"only {int(scored.sum())} out-of-fold predictions")

    oos_accuracy = accuracy_score(y[scored], (oof[scored] > 0.5).astype(int))
    try:
        oos_auc = roc_auc_score(y[scored], oof[scored])
    except ValueError:
        oos_auc = 0.5
        notes.append("AUC undefined (single class out of sample); reported as 0.5")

    calibrator = _try_calibrate(oof[scored], y[scored], notes)
    calibrated = (
        pd.Series(calibrator.transform(oof[scored].to_numpy()), index=oof[scored].index)
        if calibrator
        else oof[scored]
    )
    brier = brier_score(calibrated, y[scored])

    meta_labeler = _try_meta_label(usable[scored], oof[scored], y[scored], features, notes)

    baseline_auc = (
        _baseline_auc(usable, features, y, weights, splitter, notes) if compute_baseline else None
    )

    p_value, n_perm = _permutation_p_value(
        usable, features, y, weights, splitter, oos_auc, n_permutations, seed, notes
    )

    dsr = _deflated_sharpe(usable, oof, y, scored, n_trials, notes)

    card = ModelCard(
        model_id=f"{learner}-{dt.datetime.now(tz=IST):%Y%m%dT%H%M%S}",
        trained_at=dt.datetime.now(tz=IST),
        n_samples=len(usable),
        n_symbols=int(usable["symbol"].nunique()),
        n_features=len(features),
        train_start=usable["timestamp"].min().to_pydatetime(),
        train_end=usable["timestamp"].max().to_pydatetime(),
        oos_accuracy=float(oos_accuracy),
        oos_auc=float(oos_auc),
        brier_score=float(brier) if np.isfinite(brier) else 0.25,
        permutation_p_value=p_value,
        permutation_n=n_perm,
        baseline_oos_auc=baseline_auc,
        deflated_sharpe=dsr,
        trained_on_synthetic=_is_synthetic(bars),
    )

    # The served model is fitted on everything, for live scoring only. Nothing
    # on the card comes from it — every reported number above was produced from
    # purged out-of-fold predictions before this line runs.
    served = fit_learner(
        learner, usable[features].to_numpy(), y.to_numpy(), sample_weight=weights.to_numpy()
    ).model

    logger.info("panel: %s", panel_summary(panel))
    return TrainingResult(
        card=card,
        panel=usable,
        oof_p_up=oof,
        calibrator=calibrator,
        meta_labeler=meta_labeler,
        feature_names=features,
        notes=notes,
        served_model=served,
    )


# ------------------------------------------------------------- helpers


def _is_synthetic(bars: pd.DataFrame) -> bool:
    """Unknown provenance counts as synthetic. Fail-safe, not fail-open."""
    if "source" not in bars.columns:
        return True
    return bool((bars["source"] == "synthetic").any())


def _try_calibrate(scores: pd.Series, y: pd.Series, notes: list[str]) -> Calibrator | None:
    try:
        return fit_calibrator(scores, y)
    except ValueError as exc:
        notes.append(f"calibration skipped: {exc}")
        return None


def _try_meta_label(
    frame: pd.DataFrame,
    oof: pd.Series,
    y: pd.Series,
    features: list[str],
    notes: list[str],
) -> MetaLabeler | None:
    context = [f for f in ("vol_ratio_pct", "volume_intensity_pct") if f in frame.columns]
    try:
        return fit_meta_labeler(frame, oof, y, context_features=context)
    except ValueError as exc:
        notes.append(f"meta-labelling skipped: {exc}")
        return None


def _baseline_auc(
    frame: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    notes: list[str],
) -> float | None:
    """Purged OOF AUC of the ElasticNet baseline."""

    def fit_predict(X_tr, y_tr, w_tr, X_te):
        result = fit_learner("elasticnet", X_tr, y_tr, sample_weight=w_tr)
        return predict_proba_up(result.model, X_te)

    try:
        oof = purged_oof_predictions(
            frame, features, "_target", fit_predict, splitter=splitter, weights=weights
        )
        scored = oof.notna()
        return float(roc_auc_score(y[scored], oof[scored]))
    except ValueError as exc:
        notes.append(f"baseline skipped: {exc}")
        return None


def _permutation_p_value(
    frame: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    weights: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    observed_auc: float,
    n_permutations: int,
    seed: int,
    notes: list[str],
) -> tuple[float, int]:
    """Shuffle labels within folds and re-run, N times.

    Expensive by nature. Reduce `n_permutations` for a quick run, never to
    zero — a model without a null distribution has no claim to significance.
    """
    if n_permutations < 1:
        notes.append("permutation test skipped; p-value reported as 1.0")
        return 1.0, 1

    rng = np.random.default_rng(seed)
    fold_ids = pd.Series(0, index=frame.index)
    for split in splitter.split(frame["timestamp"], frame.get("t1")):
        fold_ids.iloc[split.test] = split.fold

    null_aucs: list[float] = []
    for _ in range(n_permutations):
        shuffled = shuffle_labels_within_folds(y, fold_ids, rng)
        work = frame.copy()
        work["_target"] = shuffled

        def fit_predict(X_tr, y_tr, w_tr, X_te):
            result = fit_learner("elasticnet", X_tr, y_tr, sample_weight=w_tr)
            return predict_proba_up(result.model, X_te)

        try:
            oof = purged_oof_predictions(
                work, features, "_target", fit_predict, splitter=splitter, weights=weights
            )
            scored = oof.notna()
            null_aucs.append(float(roc_auc_score(shuffled[scored], oof[scored])))
        except ValueError:
            continue

    if not null_aucs:
        notes.append("permutation test produced no valid nulls; p-value reported as 1.0")
        return 1.0, 1

    result = permutation_test(observed_auc, null_aucs)
    if not result.is_significant:
        notes.append(
            f"edge is NOT distinguishable from chance (p={result.p_value:.4f}). "
            "This is the honest result and it is published."
        )
    return result.p_value, result.n_permutations


def _deflated_sharpe(
    frame: pd.DataFrame,
    oof: pd.Series,
    y: pd.Series,
    scored: pd.Series,
    n_trials: int,
    notes: list[str],
) -> float | None:
    """Deflated Sharpe of a signal-following return series.

    Gross of costs and deliberately labelled as such: it measures whether the
    signal has directional information, not whether it is tradable. The cost
    model answers the second question.
    """
    try:
        direction = np.where(oof[scored] > 0.5, 1.0, -1.0)
        realised = frame.loc[scored, "label_return"].to_numpy()
        returns = pd.Series(direction * realised)

        per_bar_sharpe = returns.mean() / returns.std(ddof=1) if returns.std(ddof=1) > 0 else 0.0
        notes.append(
            f"gross-of-cost Sharpe {sharpe_ratio(returns):.3f} — directional "
            "information only; see the cost model for tradability"
        )
        return deflated_sharpe_ratio(float(per_bar_sharpe), len(returns), n_trials)
    except (KeyError, ValueError) as exc:
        notes.append(f"deflated Sharpe skipped: {exc}")
        return None
