"""NSE equity trading calendar.

Governing rule: **a year with no holiday list raises.** It does not fall back to
"weekdays are trading days". Assuming the market was open on Diwali silently
manufactures a missing-data finding for every symbol on that date, and the
completeness rules downstream would then flag a real problem that does not
exist.

The holiday list is maintained by hand from NSE's published circular. That is a
liability, so it is made explicit: `HOLIDAYS_VERIFIED_THROUGH` states how far
the data has actually been checked, and anything past it raises rather than
guessing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

# NSE trading holidays (equity segment). Sourced from NSE's annual holiday
# circular. Saturdays and Sundays are excluded separately and are not listed.
#
# ⚠ Maintenance: add each year from the published circular and bump
# HOLIDAYS_VERIFIED_THROUGH. Do not extrapolate — several NSE holidays follow
# lunar calendars and move by more than a fortnight year to year.
_HOLIDAYS: dict[int, frozenset[dt.date]] = {
    2024: frozenset(
        {
            dt.date(2024, 1, 26),  # Republic Day
            dt.date(2024, 3, 8),  # Mahashivratri
            dt.date(2024, 3, 25),  # Holi
            dt.date(2024, 3, 29),  # Good Friday
            dt.date(2024, 4, 11),  # Id-Ul-Fitr
            dt.date(2024, 4, 17),  # Ram Navami
            dt.date(2024, 5, 1),  # Maharashtra Day
            dt.date(2024, 6, 17),  # Bakri Id
            dt.date(2024, 7, 17),  # Muharram
            dt.date(2024, 8, 15),  # Independence Day
            dt.date(2024, 10, 2),  # Gandhi Jayanti
            dt.date(2024, 11, 1),  # Diwali Laxmi Pujan
            dt.date(2024, 11, 15),  # Gurunanak Jayanti
            dt.date(2024, 12, 25),  # Christmas
        }
    ),
    2025: frozenset(
        {
            dt.date(2025, 2, 26),  # Mahashivratri
            dt.date(2025, 3, 14),  # Holi
            dt.date(2025, 3, 31),  # Id-Ul-Fitr
            dt.date(2025, 4, 10),  # Mahavir Jayanti
            dt.date(2025, 4, 14),  # Ambedkar Jayanti
            dt.date(2025, 4, 18),  # Good Friday
            dt.date(2025, 5, 1),  # Maharashtra Day
            dt.date(2025, 8, 15),  # Independence Day
            dt.date(2025, 8, 27),  # Ganesh Chaturthi
            dt.date(2025, 10, 2),  # Gandhi Jayanti / Dussehra
            dt.date(2025, 10, 21),  # Diwali Laxmi Pujan
            dt.date(2025, 10, 22),  # Balipratipada
            dt.date(2025, 11, 5),  # Gurunanak Jayanti
            dt.date(2025, 12, 25),  # Christmas
        }
    ),
}

HOLIDAYS_VERIFIED_THROUGH = 2025
"""Last year whose holiday list has been checked against NSE's circular."""


class CalendarUnavailable(RuntimeError):
    """No verified holiday data for the requested year."""


class NSECalendar:
    """Trading-day arithmetic for the NSE equity segment."""

    def __init__(self, extra_holidays: dict[int, frozenset[dt.date]] | None = None) -> None:
        self._holidays = dict(_HOLIDAYS)
        if extra_holidays:
            for year, days in extra_holidays.items():
                self._holidays[year] = self._holidays.get(year, frozenset()) | days

    @property
    def verified_through(self) -> int:
        return max(self._holidays) if self._holidays else HOLIDAYS_VERIFIED_THROUGH

    def _require_year(self, year: int) -> frozenset[dt.date]:
        holidays = self._holidays.get(year)
        if holidays is None:
            raise CalendarUnavailable(
                f"no verified NSE holiday list for {year} (verified through "
                f"{self.verified_through}). Add it from NSE's published circular "
                f"rather than assuming every weekday was a trading day."
            )
        return holidays

    def is_trading_day(self, day: dt.date) -> bool:
        if day.weekday() >= 5:  # Saturday, Sunday
            return False
        return day not in self._require_year(day.year)

    def trading_days(self, start: dt.date, end: dt.date) -> Iterator[dt.date]:
        """Trading days in [start, end], inclusive, ascending."""
        day = start
        while day <= end:
            if self.is_trading_day(day):
                yield day
            day += dt.timedelta(days=1)

    def count_trading_days(self, start: dt.date, end: dt.date) -> int:
        return sum(1 for _ in self.trading_days(start, end))

    def previous_trading_day(self, day: dt.date) -> dt.date:
        cursor = day - dt.timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cursor):
                return cursor
            cursor -= dt.timedelta(days=1)
        raise CalendarUnavailable(f"no trading day within 30 days before {day}")
