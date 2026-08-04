"""market-data configuration.

Validated at import so a misconfigured provider stops the process at boot rather
than at the first request.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .providers import KNOWN_PROVIDERS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MD_", env_file=".env", extra="ignore")

    provider: str = Field(
        default="synthetic",
        description="Which market data provider to use. Must be named explicitly; "
        "there is no implicit fallback.",
    )
    lake_root: Path = Field(default=Path("data/lake"))
    seed: int = Field(default=0, description="Synthetic provider seed.")

    default_interval: str = Field(default="15m")
    universe: tuple[str, ...] = Field(
        default=(
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "HINDUNILVR",
            "ITC",
            "SBIN",
            "BHARTIARTL",
            "LT",
        ),
        description="Default symbols to ingest. The point-in-time universe in the "
        "lake supersedes this once bhavcopy ingest has run.",
    )

    min_bars_per_symbol: int = Field(
        default=200,
        description="Below this a symbol is not modellable and is reported as "
        "insufficient history rather than scored with a low-confidence guess.",
    )

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in KNOWN_PROVIDERS:
            raise ValueError(
                f"MD_PROVIDER={v!r} is not a known provider "
                f"({', '.join(KNOWN_PROVIDERS)}). Refusing to start."
            )
        return v


settings = Settings()
