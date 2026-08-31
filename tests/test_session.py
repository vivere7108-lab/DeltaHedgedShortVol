from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from deltahedger.session import (
    SessionClock, easter, holidays, is_trading_day, next_trading_day,
)

NY = ZoneInfo("America/New_York")


class TestCalendar:
    @pytest.mark.parametrize("year,expected", [
        (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)),
    ])
    def test_easter(self, year, expected):
        assert easter(year) == expected

    def test_2025_holidays_are_the_published_set(self):
        assert holidays(2025) == frozenset({
            date(2025, 1, 1),   # New Year's Day
            date(2025, 1, 20),  # MLK
            date(2025, 2, 17),  # Presidents' Day
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 26),  # Memorial Day
            date(2025, 6, 19),  # Juneteenth
            date(2025, 7, 4),   # Independence Day
            date(2025, 9, 1),   # Labor Day
            date(2025, 11, 27),  # Thanksgiving
            date(2025, 12, 25),  # Christmas
        })

    def test_thanksgiving_is_the_fourth_thursday(self):
        assert date(2024, 11, 28) in holidays(2024)
        assert date(2026, 11, 26) in holidays(2026)

    def test_a_saturday_holiday_is_observed_on_friday(self):
        assert date(2026, 7, 3) in holidays(2026)  # July 4 2026 is a Saturday

    def test_a_sunday_holiday_is_observed_on_monday(self):
        assert date(2027, 7, 5) in holidays(2027)  # July 4 2027 is a Sunday

    @pytest.mark.parametrize("day,expected", [
        (date(2025, 6, 10), True),   # a Tuesday
        (date(2025, 6, 14), False),  # a Saturday
        (date(2025, 6, 15), False),  # a Sunday
        (date(2025, 4, 18), False),  # Good Friday
    ])
    def test_is_trading_day(self, day, expected):
        assert is_trading_day(day) is expected

    def test_next_trading_day_skips_the_weekend_and_the_holiday(self):
        assert next_trading_day(date(2025, 7, 3)) == date(2025, 7, 7)
        assert next_trading_day(date(2025, 6, 13)) == date(2025, 6, 16)


class TestSessionClock:
    @pytest.fixture
    def clock(self, es):
        return SessionClock(es)

    def test_todays_expiry_is_listed_before_the_bell(self, clock):
        moment = datetime(2025, 6, 10, 9, 35, tzinfo=NY)
        assert clock.candidate_expiries(moment, 1)[0] == date(2025, 6, 10)

    def test_todays_expiry_is_gone_after_settlement(self, clock):
        moment = datetime(2025, 6, 10, 16, 30, tzinfo=NY)
        assert clock.candidate_expiries(moment, 1)[0] == date(2025, 6, 11)

    def test_expiry_rolls_over_a_holiday(self, clock):
        moment = datetime(2025, 7, 3, 16, 30, tzinfo=NY)
        assert clock.candidate_expiries(moment, 5)[0] == date(2025, 7, 7)

    def test_time_to_expiry_counts_wall_clock_hours(self, clock):
        moment = datetime(2025, 6, 10, 9, 30, tzinfo=NY)
        seconds = clock.seconds_to_expiry(moment, date(2025, 6, 10))
        assert seconds == pytest.approx(6.5 * 3600)

    def test_time_to_expiry_is_never_negative(self, clock):
        moment = datetime(2025, 6, 10, 18, 0, tzinfo=NY)
        assert clock.seconds_to_expiry(moment, date(2025, 6, 10)) == 0.0

    def test_a_naive_timestamp_is_read_as_exchange_local(self, clock):
        localized = clock.localize(datetime(2025, 6, 10, 9, 35))
        assert localized.tzinfo is not None
        assert localized.hour == 9

    def test_a_utc_timestamp_is_converted(self, clock):
        localized = clock.localize(datetime(2025, 6, 10, 13, 35, tzinfo=ZoneInfo("UTC")))
        assert localized.hour == 9 and localized.minute == 35

    @pytest.mark.parametrize("moment,expected", [
        (datetime(2025, 6, 10, 9, 35, tzinfo=NY), True),
        (datetime(2025, 6, 10, 8, 0, tzinfo=NY), False),
        (datetime(2025, 6, 10, 17, 0, tzinfo=NY), False),
        (datetime(2025, 6, 14, 12, 0, tzinfo=NY), False),  # Saturday
    ])
    def test_in_session(self, clock, moment, expected):
        assert clock.in_session(moment) is expected

    def test_dst_does_not_shift_the_expiry_hour(self, clock):
        """Expiry is 16:00 local on both sides of the DST boundary."""
        for day in (date(2025, 1, 15), date(2025, 7, 15)):
            assert clock.expiry_datetime(day).hour == 16
