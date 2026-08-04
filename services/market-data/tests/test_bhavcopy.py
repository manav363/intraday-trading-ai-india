from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd
import pytest
from market_data.ingest import (
    BhavcopyUnavailable,
    bhavcopy_url,
    detect_corporate_actions,
    equity_rows,
    normalise_udiff,
    parse_bhavcopy_zip,
)

UDIFF_HEADER = (
    "TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,"
    "PrvsClsgPric,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,ISIN"
)


def _udiff_csv(rows: list[str]) -> str:
    return "\n".join([UDIFF_HEADER, *rows])


def _zipped(csv_text: str, name: str = "BhavCopy.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(name, csv_text)
    return buf.getvalue()


def test_url_uses_udiff_layout() -> None:
    url = bhavcopy_url(dt.date(2026, 3, 9))
    assert url.endswith("BhavCopy_NSE_CM_0_0_0_20260309_F_0000.csv.zip")


def test_parses_and_renames_to_canonical_columns() -> None:
    payload = _zipped(
        _udiff_csv(
            [
                "2026-03-09,RELIANCE,EQ,1400,1425,1395,1420,1398,1000000,1.4e9,45000,INE002A01018",
                "2026-03-09,TCS,EQ,3900,3950,3880,3930,3905,500000,1.9e9,30000,INE467B01029",
            ]
        )
    )
    df = parse_bhavcopy_zip(payload)

    assert list(df.columns)[:5] == ["date", "symbol", "series", "open", "high"]
    assert df.loc[0, "symbol"] == "RELIANCE"
    assert df.loc[0, "close"] == 1420.0
    assert df.loc[0, "date"] == dt.date(2026, 3, 9)
    assert (df["source"] == "bhavcopy").all()


def test_delivery_columns_are_absent_not_zero() -> None:
    """Zero would be read as 'no delivery today', which is a different claim
    from 'we do not have the delivery file'."""
    payload = _zipped(
        _udiff_csv(
            ["2026-03-09,RELIANCE,EQ,1400,1425,1395,1420,1398,1000000,1.4e9,45000,INE002A01018"]
        )
    )
    df = parse_bhavcopy_zip(payload)
    assert df["delivery_pct"].isna().all()
    assert df["delivery_qty"].isna().all()


def test_missing_columns_raise_with_what_was_seen() -> None:
    payload = _zipped("TradDt,TckrSymb\n2026-03-09,RELIANCE")
    with pytest.raises(BhavcopyUnavailable, match="missing expected UDiFF columns"):
        parse_bhavcopy_zip(payload)


def test_non_zip_payload_raises() -> None:
    with pytest.raises(BhavcopyUnavailable, match="not a zip archive"):
        parse_bhavcopy_zip(b"<html>404</html>")


def test_archive_without_csv_raises() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("readme.txt", "nope")
    with pytest.raises(BhavcopyUnavailable, match="no CSV inside archive"):
        parse_bhavcopy_zip(buf.getvalue())


def test_equity_rows_filters_non_equity_series() -> None:
    df = normalise_udiff(
        pd.read_csv(
            io.StringIO(
                _udiff_csv(
                    [
                        "2026-03-09,RELIANCE,EQ,1400,1425,1395,1420,1398,1,1,1,INE002A01018",
                        "2026-03-09,SOMEBOND,N1,100,101,99,100,100,1,1,1,INE999A01011",
                        "2026-03-09,SMALLCO,SM,50,52,49,51,50,1,1,1,INE888A01011",
                    ]
                )
            )
        )
    )
    kept = equity_rows(df)
    assert set(kept["symbol"]) == {"RELIANCE", "SMALLCO"}


# ------------------------------------------------- corporate actions (T4)


def test_detects_a_split_as_a_continuity_break() -> None:
    """A 1:5 split shows up as prev_close being ~5x yesterday's close.

    Left undetected it enters every model as an 80% single-bar crash.
    """
    yesterday = pd.DataFrame({"symbol": ["SPLITCO", "CALMCO"], "close": [1000.0, 500.0]})
    today = pd.DataFrame({"symbol": ["SPLITCO", "CALMCO"], "prev_close": [5000.0, 501.0]})

    suspects = detect_corporate_actions(today, yesterday)

    assert list(suspects["symbol"]) == ["SPLITCO"]
    assert suspects.iloc[0]["implied_ratio"] == pytest.approx(5.0)


def test_ordinary_price_movement_is_not_flagged() -> None:
    yesterday = pd.DataFrame({"symbol": ["A", "B"], "close": [100.0, 200.0]})
    today = pd.DataFrame({"symbol": ["A", "B"], "prev_close": [100.0, 201.0]})
    assert detect_corporate_actions(today, yesterday).empty


def test_zero_close_rows_are_skipped_not_divided_by() -> None:
    yesterday = pd.DataFrame({"symbol": ["DEAD"], "close": [0.0]})
    today = pd.DataFrame({"symbol": ["DEAD"], "prev_close": [10.0]})
    assert detect_corporate_actions(today, yesterday).empty
