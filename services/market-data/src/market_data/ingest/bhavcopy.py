"""NSE bhavcopy — the official published end-of-day file.

**Why this is much lighter here than in a long-horizon daily project.**

A sibling project backfills bhavcopy to 2006 and treats corporate-action
adjustment across twenty years as its largest correctness risk. That work is
load-bearing there because the model trains on two decades of daily bars, and
because a universe built from *today's* index makes every delisted company
invisible.

This project's intraday window is 60–90 days. Over that span:

* survivorship bias is small — very few names delist in a quarter, and the ones
  that do are visible in the daily files we do read;
* the legacy pre-2024 bhavcopy format never appears, so only the UDiFF format
  needs a parser rather than one fixture set per era.

So bhavcopy earns its place here for three specific things, not for deep
history:

1. **Corporate actions.** A split inside the window breaks price continuity and
   would otherwise look like a 50% crash to every feature and every label.
2. **The symbol master** — ISIN, series, and name changes.
3. **The liquidity screen** that decides the eligible universe, using turnover
   and delivery percentage.

`delivery_pct` in particular is a genuine institutional-participation signal
that comes free with the delivery file, and v1 never had it.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from dataclasses import dataclass

import httpx
import pandas as pd

_UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

# NSE rejects requests without a browser-ish UA. This is not evasion — the
# archive is a public download; the header just makes the CDN serve it.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# UDiFF column -> canonical name.
_UDIFF_COLUMNS = {
    "TradDt": "date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
    "ISIN": "isin",
}

EQUITY_SERIES = frozenset({"EQ", "BE", "BZ", "SM", "ST"})
"""Series that represent ordinary equity trading. Everything else (debt,
warrants, rights) is excluded from the universe."""


class BhavcopyUnavailable(RuntimeError):
    """The file could not be fetched or parsed."""


@dataclass(frozen=True, slots=True)
class BhavcopyFile:
    day: dt.date
    frame: pd.DataFrame
    source_ref: str


def bhavcopy_url(day: dt.date) -> str:
    return _UDIFF_URL.format(yyyymmdd=day.strftime("%Y%m%d"))


def fetch_bhavcopy(day: dt.date, *, timeout: float = 30.0) -> BhavcopyFile:
    """Download and parse one day's bhavcopy.

    Raises rather than returning an empty frame: a missing file for a day the
    calendar says was a trading day is a real finding, and returning empty would
    render as "nothing traded that day".
    """
    url = bhavcopy_url(day)
    try:
        response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise BhavcopyUnavailable(f"could not fetch bhavcopy for {day}: {exc}") from exc

    return BhavcopyFile(day=day, frame=parse_bhavcopy_zip(response.content), source_ref=url)


def parse_bhavcopy_zip(payload: bytes) -> pd.DataFrame:
    """Parse the zipped UDiFF CSV into canonical columns.

    Split out from the fetch so the parser is testable without network.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise BhavcopyUnavailable(f"no CSV inside archive; members={archive.namelist()}")
            with archive.open(names[0]) as handle:
                raw = pd.read_csv(handle)
    except zipfile.BadZipFile as exc:
        raise BhavcopyUnavailable(f"payload is not a zip archive: {exc}") from exc

    return normalise_udiff(raw)


def normalise_udiff(raw: pd.DataFrame) -> pd.DataFrame:
    """Map UDiFF columns to canonical ones and coerce dtypes."""
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})

    missing = [c for c in _UDIFF_COLUMNS if c not in raw.columns]
    if missing:
        raise BhavcopyUnavailable(
            f"bhavcopy missing expected UDiFF columns {missing}; saw {list(raw.columns)}"
        )

    df = raw[list(_UDIFF_COLUMNS)].rename(columns=_UDIFF_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()

    for col in ("open", "high", "low", "close", "prev_close", "volume", "turnover", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Delivery data lives in a separate file. Absent rather than zero — zero
    # would be read as "no delivery today", which is a different claim.
    df["delivery_qty"] = pd.NA
    df["delivery_pct"] = pd.NA
    df["source"] = "bhavcopy"

    return df


def equity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordinary equity series only."""
    return frame[frame["series"].isin(EQUITY_SERIES)].copy()


def detect_corporate_actions(
    today: pd.DataFrame,
    yesterday: pd.DataFrame,
    *,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Flag symbols whose `prev_close` disagrees with yesterday's `close`.

    This is the Tier-4 continuity check in its detection form. An unexplained
    mismatch means either a corporate action we do not have on record, or an
    adjustment bug — and both must be loud. Smoothing it over is how a 1:5 split
    enters a model as an 80% single-bar crash.

    Returns one row per suspect symbol with the implied ratio, which for a real
    split lands close to a simple fraction.
    """
    merged = today[["symbol", "prev_close"]].merge(
        yesterday[["symbol", "close"]],
        on="symbol",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["prev_close", "close"])
    merged = merged[merged["close"] > 0]

    merged["implied_ratio"] = merged["prev_close"] / merged["close"]
    drift = (merged["implied_ratio"] - 1.0).abs()

    suspects = merged[drift > tolerance].copy()
    suspects["adj_factor"] = suspects["implied_ratio"]
    return suspects.sort_values("implied_ratio")
