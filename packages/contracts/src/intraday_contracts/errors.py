"""Shared error envelope.

Every service returns failures in this shape so the gateway and the frontend
have exactly one error contract to handle.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Stable, machine-readable failure reasons.

    These are part of the API contract — the frontend branches on them. Renaming
    one is a breaking change and the schema-freeze test will say so.
    """

    # Input
    UNKNOWN_SYMBOL = "unknown_symbol"
    SYMBOL_NOT_ELIGIBLE = "symbol_not_eligible"
    INVALID_INTERVAL = "invalid_interval"
    INVALID_RANGE = "invalid_range"

    # State — the honest 503 family. See NO_MODEL in particular: a system with
    # no trained model must say so, never return a neutral 0.5 probability that
    # the narrative layer would render as a real call about a real company.
    NO_MODEL = "no_model"
    NO_DATA = "no_data"
    INSUFFICIENT_HISTORY = "insufficient_history"
    PROVIDER_UNAVAILABLE = "provider_unavailable"

    # Configuration — these are boot-time refusals, surfaced if they ever reach
    # a request path.
    PROVIDER_NOT_IMPLEMENTED = "provider_not_implemented"

    INTERNAL = "internal"


class ErrorEnvelope(BaseModel):
    """The single error shape crossing any service boundary."""

    model_config = {"frozen": True}

    code: ErrorCode
    message: str = Field(description="Human-readable, safe to display.")
    detail: dict[str, object] = Field(
        default_factory=dict,
        description="Structured context. Never contains secrets or stack traces.",
    )


class ContractViolation(Exception):
    """Raised when data crossing a boundary does not satisfy its contract.

    Deliberately not an ErrorEnvelope: a contract violation is a bug in our own
    code, not a condition to render for a user.
    """
