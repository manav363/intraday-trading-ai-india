"""The market data provider port.

One rule governs this module, carried over from ai_trade:

    **A named-but-unbuilt provider refuses to boot. It never falls back.**

Falling back to synthetic is how fabricated data reaches a real screen while
every log line reads healthy. If the configured provider cannot be constructed,
the process dies at startup with the reason — it does not quietly serve
generated bars.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from intraday_contracts import BarSource, Interval, OHLCVBar


class ProviderError(RuntimeError):
    """Base for provider failures."""


class ProviderNotImplemented(ProviderError):
    """The provider name is recognised but no adapter exists for it.

    Raised at construction, not at request time. A provider that cannot serve
    should stop the process, not return empty results that look like a quiet
    market.
    """


class ProviderUnavailable(ProviderError):
    """The provider exists but could not serve this request.

    Distinct from "returned no bars": an unavailable provider is an error, and
    an empty result for a valid range is a finding. Collapsing the two is how a
    dead data path stays invisible for a month.
    """


class HistoryWindowExceeded(ProviderError):
    """The requested range is deeper than the provider can serve.

    Not a warning. Silently truncating to the available window produces a
    dataset whose start date is a lie.
    """


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    name: str
    available: bool
    detail: str


class MarketDataProvider(ABC):
    """Fetches OHLCV bars for a symbol over a range.

    Implementations are responsible for returning bars that already satisfy the
    `OHLCVBar` contract — geometry, positive prices, timezone-aware timestamps,
    correct `source`. Rows that cannot satisfy it are the provider's problem to
    reject or repair, so that everything downstream can assume validity.
    """

    name: ClassVar[str]
    source: ClassVar[BarSource]

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[OHLCVBar]:
        """Return bars in ascending timestamp order.

        An empty list means "no bars in this range", which is a legitimate
        answer for a holiday or a suspended symbol. Failure raises
        `ProviderUnavailable`.
        """

    @abstractmethod
    def max_history_days(self, interval: Interval) -> int:
        """How far back this provider can serve at this interval.

        This is a first-class part of the port because the public NSE endpoint
        serves a rolling window of roughly 60–90 days for minute bars. A design
        that assumed unlimited backfill would silently cap this project's
        history at three months forever; instead the lake accumulates.
        """

    def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, available=True, detail="ok")

    def check_window(self, interval: Interval, start: dt.datetime, now: dt.datetime) -> None:
        """Raise if `start` is deeper than this provider can reach."""
        limit = self.max_history_days(interval)
        requested = (now - start).days
        if requested > limit:
            raise HistoryWindowExceeded(
                f"{self.name} serves at most {limit}d of {interval.value} bars; "
                f"{requested}d requested. Deeper history comes from the accumulated "
                f"lake, not from a single fetch."
            )
