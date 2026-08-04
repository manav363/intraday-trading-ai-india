from __future__ import annotations

import datetime as dt

import pytest
from market_data.calendar import CalendarUnavailable, NSECalendar


@pytest.fixture
def cal() -> NSECalendar:
    return NSECalendar()


def test_weekend_is_not_a_trading_day(cal: NSECalendar) -> None:
    assert not cal.is_trading_day(dt.date(2025, 3, 15))  # Saturday
    assert not cal.is_trading_day(dt.date(2025, 3, 16))  # Sunday


def test_ordinary_weekday_is_a_trading_day(cal: NSECalendar) -> None:
    assert cal.is_trading_day(dt.date(2025, 3, 12))  # Wednesday


def test_listed_holiday_is_not_a_trading_day(cal: NSECalendar) -> None:
    assert not cal.is_trading_day(dt.date(2025, 12, 25))  # Christmas
    assert not cal.is_trading_day(dt.date(2025, 8, 15))  # Independence Day


def test_unverified_year_raises_rather_than_assuming_open(cal: NSECalendar) -> None:
    """The rule that matters: no calendar data means refuse, not guess.

    Assuming weekdays are trading days would manufacture a missing-data finding
    for every symbol on every holiday of that year.
    """
    with pytest.raises(CalendarUnavailable, match="no verified NSE holiday list"):
        cal.is_trading_day(dt.date(2031, 6, 10))


def test_extra_holidays_extend_coverage() -> None:
    cal = NSECalendar(extra_holidays={2031: frozenset({dt.date(2031, 6, 10)})})
    assert not cal.is_trading_day(dt.date(2031, 6, 10))
    assert cal.is_trading_day(dt.date(2031, 6, 11))


def test_trading_days_range_excludes_weekends_and_holidays(cal: NSECalendar) -> None:
    days = list(cal.trading_days(dt.date(2025, 12, 22), dt.date(2025, 12, 28)))
    assert dt.date(2025, 12, 25) not in days  # Christmas
    assert dt.date(2025, 12, 27) not in days  # Saturday
    assert dt.date(2025, 12, 22) in days
    assert len(days) == 4  # Mon, Tue, Wed(skip Thu), Fri


def test_previous_trading_day_skips_the_weekend(cal: NSECalendar) -> None:
    # Tue 2025-03-11 <- Mon 2025-03-10, an ordinary consecutive pair.
    assert cal.previous_trading_day(dt.date(2025, 3, 11)) == dt.date(2025, 3, 10)
    # Mon 2025-03-10 <- Fri 2025-03-07, stepping over Sat/Sun.
    assert cal.previous_trading_day(dt.date(2025, 3, 10)) == dt.date(2025, 3, 7)


def test_previous_trading_day_skips_a_holiday_adjacent_to_a_weekend(cal: NSECalendar) -> None:
    """Mon 2025-03-17 <- Thu 2025-03-13: Fri 14th is Holi, then Sat and Sun.

    The holiday and the weekend have to be skipped together; handling only one
    of them lands on a date the exchange never traded.
    """
    assert cal.previous_trading_day(dt.date(2025, 3, 17)) == dt.date(2025, 3, 13)
