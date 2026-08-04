"""Schema freeze.

The lake layout and the contract field sets are the interface between services.
A silent rename here is the failure mode that shows up as an integration bug
weeks later, so the shapes are pinned and any change has to be made
deliberately by editing this file.

Field *names* are frozen rather than full JSON Schema — full schema output
churns between pydantic point releases and produces noisy failures that teach
people to update the snapshot without reading it.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

from intraday_contracts import (
    BAR_COLUMNS,
    CORP_ACTION_COLUMNS,
    EOD_COLUMNS,
    QUALITY_SCORE_COLUMNS,
    SCHEMA_VERSION,
    UNIVERSE_COLUMNS,
    CPCVResult,
    ErrorEnvelope,
    ExplanationFact,
    Interval,
    LakePaths,
    ModelCard,
    OHLCVBar,
    Prediction,
    QualityRun,
    QualityScore,
    RuleResult,
    SymbolMeta,
    UniverseSnapshot,
)

EXPECTED_FIELDS: dict[type[BaseModel], set[str]] = {
    OHLCVBar: {
        "symbol",
        "timestamp",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    },
    SymbolMeta: {"symbol", "name", "isin", "sector", "listing_date", "delisting_date"},
    UniverseSnapshot: {"as_of", "index", "symbols"},
    RuleResult: {
        "rule_id",
        "tier",
        "severity",
        "passed",
        "affected_symbols",
        "affected_rows",
        "evidence",
    },
    QualityRun: {"run_id", "started_at", "source_ref", "verdict", "results"},
    QualityScore: {
        "symbol",
        "as_of",
        "score",
        "history_completeness",
        "validity_clean_rate",
        "continuity_clean_rate",
        "liquidity_adequacy",
        "recency",
        "eligible",
        "ineligible_reasons",
    },
    ExplanationFact: {
        "feature",
        "display_name",
        "value",
        "display_value",
        "shap",
        "direction",
        "rank",
    },
    Prediction: {
        "symbol",
        "as_of",
        "horizon_bars",
        "p_up",
        "certainty",
        "meta_probability",
        "side",
        "size_fraction",
        "is_calibrated",
        "facts",
    },
    CPCVResult: {
        "n_paths",
        "sharpe_mean",
        "sharpe_std",
        "sharpe_p05",
        "sharpe_p50",
        "sharpe_p95",
    },
    ModelCard: {
        "model_id",
        "trained_at",
        "n_samples",
        "n_symbols",
        "n_features",
        "train_start",
        "train_end",
        "oos_accuracy",
        "oos_auc",
        "brier_score",
        "cpcv",
        "deflated_sharpe",
        "permutation_p_value",
        "permutation_n",
        "baseline_oos_auc",
        "trained_on_synthetic",
    },
    ErrorEnvelope: {"code", "message", "detail"},
}


def test_model_fields_are_frozen() -> None:
    for model, expected in EXPECTED_FIELDS.items():
        actual = set(model.model_fields)
        assert actual == expected, (
            f"{model.__name__} fields changed: "
            f"added={actual - expected} removed={expected - actual}. "
            "This is a cross-service breaking change — update EXPECTED_FIELDS "
            "deliberately and bump SCHEMA_VERSION."
        )


def test_lake_columns_are_frozen() -> None:
    assert BAR_COLUMNS == (
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    )
    assert EOD_COLUMNS[:6] == ("date", "symbol", "series", "open", "high", "low")
    assert "delivery_pct" in EOD_COLUMNS
    assert "source" in EOD_COLUMNS
    assert CORP_ACTION_COLUMNS == (
        "ex_date",
        "symbol",
        "action_type",
        "ratio_from",
        "ratio_to",
        "adj_factor",
    )
    assert UNIVERSE_COLUMNS == ("as_of", "index", "symbol")
    assert "eligible" in QUALITY_SCORE_COLUMNS


def test_every_bar_column_exists_on_the_bar_model() -> None:
    """The lake columns and the wire model must not drift apart.

    `interval` is intentionally absent from BAR_COLUMNS — it is a partition key,
    not a column, so it is carried by the path rather than repeated a million
    times in the file.
    """
    model_fields = set(OHLCVBar.model_fields)
    assert set(BAR_COLUMNS) <= model_fields
    assert model_fields - set(BAR_COLUMNS) == {"interval"}


def test_partition_layout_is_frozen() -> None:
    paths = LakePaths("/lake")
    day = dt.date(2026, 3, 9)
    assert str(paths.bars_file(Interval.M5, "TCS", day)).endswith(
        "bars/5m/symbol=TCS/year=2026/month=03/2026-03-09.parquet"
    )


def test_schema_version_is_declared() -> None:
    assert SCHEMA_VERSION == "2.0.0"
