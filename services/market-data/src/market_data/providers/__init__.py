"""Provider registry.

`build_provider` is the only sanctioned way to construct a provider, because it
is where the no-fallback rule is enforced:

* an **unknown** name raises — no default is picked
* a **known but unbuildable** provider raises — it does not degrade
* `synthetic` must be asked for **by name**; nothing selects it implicitly
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    HistoryWindowExceeded,
    MarketDataProvider,
    ProviderError,
    ProviderHealth,
    ProviderNotImplemented,
    ProviderUnavailable,
)
from .nse_charting import NSEChartingProvider
from .replay import ReplayProvider
from .synthetic import SyntheticProvider

KNOWN_PROVIDERS = ("nse_charting", "replay", "synthetic")


def build_provider(
    name: str,
    *,
    lake_root: Path | str | None = None,
    seed: int = 0,
) -> MarketDataProvider:
    """Construct the named provider, or raise.

    There is deliberately no `default=` parameter. A caller that does not know
    which provider it wants has a configuration bug, and serving it synthetic
    bars would hide that bug behind plausible-looking numbers.
    """
    if name not in KNOWN_PROVIDERS:
        raise ProviderNotImplemented(
            f"unknown provider {name!r}. Known: {', '.join(KNOWN_PROVIDERS)}. "
            "There is no fallback by design."
        )

    if name == "nse_charting":
        return NSEChartingProvider()

    if name == "replay":
        if lake_root is None:
            raise ProviderNotImplemented("provider 'replay' requires lake_root")
        return ReplayProvider(lake_root)

    return SyntheticProvider(seed=seed)


__all__ = [
    "KNOWN_PROVIDERS",
    "HistoryWindowExceeded",
    "MarketDataProvider",
    "NSEChartingProvider",
    "ProviderError",
    "ProviderHealth",
    "ProviderNotImplemented",
    "ProviderUnavailable",
    "ReplayProvider",
    "SyntheticProvider",
    "build_provider",
]
