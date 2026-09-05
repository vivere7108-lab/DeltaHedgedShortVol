"""Exchange calendar, expiry selection and time-to-expiry helpers.

Two different clocks live here and they must not be confused.

*Pricing* runs on wall-clock time.  An option with four hours left and one
with forty minutes left price very differently, so time-to-expiry is
computed from an actual expiry timestamp in exchange-local time rather than
by counting days.

*Tenor selection* runs on **trading days**.  "2 DTE" means two sessions
away, not 48 hours: a Friday expiry is 1 DTE from Thursday and also 1 DTE
from the following Monday's point of view only if you count calendar days,
which is wrong -- the weekend carries no session, no dealer hedging and
almost no decay in the terms that matter here.  ``trading_days_between``
counts sessions and is what every DTE in this codebase means.

The holiday set is the CME equity-index full-holiday list (which matches the
NYSE list, Good Friday included).  Early closes are not modelled -- on a
half day the daily option still settles at its listed time, and the effect
on the backtest is that a few afternoon bars simply do not exist in the
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


def trading_days_between(start: date, end: date) -> int:
    """Sessions in the half-open interval ``(start, end]``.

    This is the definition of DTE used everywhere in this codebase, and it
    is deliberately *not* a calendar-day count.  Counted this way an expiry
    on the same date is 0 DTE, the next session is 1, and a weekend or a
    holiday costs nothing::

        Thursday -> Friday                        1
        Friday   -> Monday                        1
        Thursday -> Monday (Friday a holiday)     1
        Wednesday -> the Wednesday after          5

    ``start`` itself is never counted whether or not it is a session, so a
    Saturday reads Monday's expiry as 1 DTE rather than 0.  Returns a
    negative count for an expiry in the past, which lets a caller tell "no
    longer listed" from "listed today".
    """
    if end == start:
        return 0
    if end < start:
        return -trading_days_between(end, start)
    count = 0
    day = start
    while day < end:
        day += timedelta(days=1)
        if is_trading_day(day):
            count += 1
    return count


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
        """Listed daily expiries out to ``max_days`` DTE, soonest first.

        ``max_days`` is counted in *trading* days (see
        ``trading_days_between``), so ``max_days=5`` asked on a Wednesday
        reaches the Wednesday after rather than stopping on Monday.

        Today counts only while its settlement is still ahead of us -- once
        the 16:00 bell passes, today's series is gone and the soonest listed
        expiry is the next session's.
        """
        now = self.localize(moment)
        found: list[date] = []
        day = now.date()
        if not is_trading_day(day) or now >= self.expiry_datetime(day):
            day = next_trading_day(day)
        while trading_days_between(now.date(), day) <= max_days:
            found.append(day)
            day = next_trading_day(day)
        return found

    def days_to_expiry(self, moment: datetime, expiry_day: date) -> int:
        """DTE of ``expiry_day`` as of ``moment``, in trading days."""
        return trading_days_between(self.localize(moment).date(), expiry_day)

    def gap_before(self, moment: datetime, expiry_day: date) -> bool:
        """Whether a weekend or holiday sits between ``moment`` and ``expiry_day``.

        True when any calendar day after today's and up to the expiry is not
        a session -- equivalently, when the calendar-day count and the
        trading-day count disagree.  A Friday reads Monday's expiry as
        across a gap; a Thursday reads Friday's as not, unless Friday is a
        holiday.  Today itself is never counted, so a moment on a Saturday
        still reads Monday as across a gap (the weekend is not over yet).
        """
        today = self.localize(moment).date()
        if expiry_day <= today:
            return False
        return (expiry_day - today).days != trading_days_between(today, expiry_day)

    def gap_after(self, moment: datetime) -> bool:
        """Whether the calendar day after ``moment``'s is not a session --
        i.e. today is the last session before a weekend or a holiday."""
        return not is_trading_day(self.localize(moment).date() + timedelta(days=1))

    def select_expiry(
        self,
        moment: datetime,
        min_days: int,
        max_days: int,
        prefer_days: tuple[int, int] | None = None,
        min_seconds_to_expiry: float = 0.0,
        hold_over_gaps: bool = True,
    ) -> date | None:
        """The listed expiry to trade, or ``None`` if none is eligible.

        Among the expiries whose DTE falls inside ``[min_days, max_days]``,
        the one closest to the ``prefer_days`` window is chosen.  Ties break
        toward the *longer* tenor.

        Two further filters decide what is eligible at all:

        * a series settling within ``min_seconds_to_expiry`` is skipped.
          This is what rolls a 0DTE policy into tomorrow's series in the
          last minutes of today's: today's is still listed, but it is
          inside the buffer, so the next one is the nearest eligible;
        * with ``hold_over_gaps`` off, a series on the far side of a
          weekend or holiday (``gap_before``) is skipped, so a Friday
          afternoon has nothing to roll into and stays flat.

        No fallback reaches outside the range: if nothing is eligible the
        answer is "do not trade".
        """
        in_range = []
        for expiry in self.candidate_expiries(moment, max_days):
            dte = self.days_to_expiry(moment, expiry)
            if dte < min_days:
                continue
            if self.seconds_to_expiry(moment, expiry) <= min_seconds_to_expiry:
                continue
            if not hold_over_gaps and self.gap_before(moment, expiry):
                continue
            in_range.append((dte, expiry))
        if not in_range:
            return None
        if prefer_days is None:
            return in_range[0][1]

        low, high = min(prefer_days), max(prefer_days)
        distance = lambda dte: max(low - dte, dte - high, 0)  # noqa: E731
        return min(in_range, key=lambda row: (distance(row[0]), -row[0]))[1]

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
