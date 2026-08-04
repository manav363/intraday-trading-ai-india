"""Ingestion: bhavcopy (EOD, reference data) and intraday bars via the provider port."""

from .bhavcopy import (
    EQUITY_SERIES,
    BhavcopyFile,
    BhavcopyUnavailable,
    bhavcopy_url,
    detect_corporate_actions,
    equity_rows,
    fetch_bhavcopy,
    normalise_udiff,
    parse_bhavcopy_zip,
)

__all__ = [
    "EQUITY_SERIES",
    "BhavcopyFile",
    "BhavcopyUnavailable",
    "bhavcopy_url",
    "detect_corporate_actions",
    "equity_rows",
    "fetch_bhavcopy",
    "normalise_udiff",
    "parse_bhavcopy_zip",
]
