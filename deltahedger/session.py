"""Exchange calendar and time-to-expiry helpers.

0DTE is unforgiving about the time axis: an option with four hours left and
one with forty minutes left price very differently, so time-to-expiry is
computed from an actual expiry timestamp in exchange-local time rather than
by counting days.

The holiday set is the CME equity-index full-holiday list (which matches the
NYSE list, Good Friday included).  Early closes are not modelled -- on a
half day the daily option still settles at its listed time, and the effect
on a 0DTE backtest is that a few afternoon bars simply do not exist in the
data, which the engine handles by skipping.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from .instruments import RiskSource
from .pricing import SECONDS_PER_YEAR


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth ``weekday`` (Mon=0) of a month; n=-1 for the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last_day = (date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed(day: date) -> date:
    """Shift a fixed-date holiday to the observed weekday."""
    if day.weekday() == 5:  # Saturday -> Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    """Full-day exchange holidays for US equity index products."""
    days = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        easter(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving (4th Thursday)
        _observed(date(year, 7, 4)),  # Independence Day
        _observed(date(year, 12, 25)),  # Christmas
    }
    if year >= 2021:  # Juneteenth became a market holiday in 2022; 2021 was ad hoc
        days.add(_observed(date(year, 6, 19)))
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays(day.year)


def next_trading_day(day: date) -> date:
    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


class SessionClock:
    """Turns wall-clock timestamps into expiries and time-to-expiry."""

    def __init__(self, source: RiskSource):
        self.source = source
        self.tz = ZoneInfo(source.timezone)
        self.expiry_time = _hhmm(source.option_expiry_time)
        self.open_time = _hhmm(source.session_open)
        self.close_time = _hhmm(source.session_close)

    def localize(self, moment: datetime) -> datetime:
        """Interpret a naive timestamp as exchange-local; convert an aware one."""
        if moment.tzinfo is None:
            return moment.replace(tzinfo=self.tz)
        return moment.astimezone(self.tz)

    def expiry_datetime(self, expiry_day: date) -> datetime:
        return datetime.combine(expiry_day, self.expiry_time, tzinfo=self.tz)

    def candidate_expiries(self, moment: datetime, max_days: int) -> list[date]:
        """Listed daily expiries from ``moment`` forward, soonest first.

        Today counts only while its settlement is still ahead of us -- once
        the 16:00 bell passes, the 0DTE series is gone and the next listed
        expiry is tomorrow's.
        """
        now = self.localize(moment)
        found: list[date] = []
        day = now.date()
        if not is_trading_day(day) or now >= self.expiry_datetime(day):
            day = next_trading_day(day)
        while (day - now.date()).days <= max_days:
            found.append(day)
            day = next_trading_day(day)
        return found

    def seconds_to_expiry(self, moment: datetime, expiry_day: date) -> float:
        now = self.localize(moment)
        return max((self.expiry_datetime(expiry_day) - now).total_seconds(), 0.0)

    def time_to_expiry(self, moment: datetime, expiry_day: date) -> float:
        """Time to expiry in years, on a wall-clock (365-day) basis."""
        return self.seconds_to_expiry(moment, expiry_day) / SECONDS_PER_YEAR

    def in_session(self, moment: datetime) -> bool:
        now = self.localize(moment)
        return (
            is_trading_day(now.date())
            and self.open_time <= now.time() <= self.close_time
        )

    def local_time(self, moment: datetime) -> time:
        return self.localize(moment).time()


def _hhmm(value: str) -> time:
    hour, minute = (int(p) for p in value.split(":")[:2])
    return time(hour, minute)
