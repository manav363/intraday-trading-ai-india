"""worker: scheduled append-only ingest, retraining, and drift detection."""

from .jobs import JobReport, append_session, check_drift

__version__ = "2.0.0"

__all__ = ["JobReport", "append_session", "check_drift"]
