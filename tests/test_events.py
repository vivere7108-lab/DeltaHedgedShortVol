"""The event blackout calendar: parsing, the window, and the file format."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config, StrategyConfig
from deltahedger.events import EventCalendar, MarketEvent, parse_event, read_events_file

NY = ZoneInfo("America/New_York")
FOMC = datetime(2026, 9, 16, 14, 0, tzinfo=NY)


class TestParsing:
    @pytest.mark.parametrize("raw", [
        "2026-09-16 14:00 FOMC statement",
        "2026-09-16T14:00 FOMC statement",
        "2026-09-16 14:00:00 FOMC statement",
        "  2026-09-16 14:00   FOMC statement  ",
        {"at": "2026-09-16 14:00", "label": "FOMC statement"},
    ])
    def test_the_accepted_forms_all_read_the_same_event(self, raw):
        event = parse_event(raw, NY)
        assert event.at == FOMC
        assert event.label == "FOMC statement"

    def test_a_bare_timestamp_has_an_empty_label(self):
        assert parse_event("2026-09-16 14:00", NY).label == ""

    def test_naive_times_are_exchange_local(self):
        assert parse_event("2026-09-16 14:00", NY).at.tzinfo is not None
        assert parse_event("2026-09-16 14:00", NY).at.hour == 14

    def test_an_aware_time_is_converted(self):
        event = parse_event(datetime(2026, 9, 16, 18, 0, tzinfo=ZoneInfo("UTC")), NY)
        assert event.at == FOMC

    def test_garbage_is_rejected_rather_than_ignored(self):
        with pytest.raises(ValueError, match="cannot read an event"):
            parse_event("next wednesday afternoon", NY)

    def test_a_file_reads_one_event_per_line_and_skips_comments(self, tmp_path):
        path = tmp_path / "events.txt"
        path.write_text(
            "# FOMC\n\n2026-09-16 14:00 FOMC statement  # trailing comment\n"
            "2026-10-28 14:00 FOMC statement\n"
        )
        events = read_events_file(path, NY)
        assert [e.at.date().isoformat() for e in events] == ["2026-09-16", "2026-10-28"]
        assert events[0].label == "FOMC statement"

    def test_a_bad_line_names_the_file_and_line(self, tmp_path):
        path = tmp_path / "events.txt"
        path.write_text("2026-09-16 14:00 ok\nnot an event\n")
        with pytest.raises(ValueError, match="events.txt:2"):
            read_events_file(path, NY)

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_events_file(tmp_path / "nope.txt", NY)


class TestBlackout:
    @pytest.fixture
    def calendar(self):
        return EventCalendar(
            [MarketEvent(FOMC, "FOMC"), MarketEvent(FOMC + timedelta(days=42), "FOMC")],
            before=timedelta(minutes=15), after=timedelta(minutes=15),
        )

    @pytest.mark.parametrize("offset", [-15, -10, 0, 5, 15])
    def test_inside_the_window_names_the_event(self, calendar, offset):
        event = calendar.blackout(FOMC + timedelta(minutes=offset))
        assert event is not None and event.label == "FOMC"

    @pytest.mark.parametrize("offset", [-16, 16, -60 * 24, 60 * 24])
    def test_outside_the_window_is_clear(self, calendar, offset):
        assert calendar.blackout(FOMC + timedelta(minutes=offset)) is None

    def test_the_edges_are_inclusive(self, calendar):
        """A bar landing exactly on the boundary is treated as inside --
        the conservative side of a rounding question."""
        assert calendar.blackout(FOMC - timedelta(minutes=15)) is not None
        assert calendar.blackout(FOMC + timedelta(minutes=15)) is not None

    def test_asymmetric_windows(self):
        calendar = EventCalendar(
            [MarketEvent(FOMC)], before=timedelta(minutes=30), after=timedelta(minutes=5)
        )
        assert calendar.blackout(FOMC - timedelta(minutes=25)) is not None
        assert calendar.blackout(FOMC + timedelta(minutes=10)) is None

    def test_an_empty_calendar_never_blacks_out(self):
        assert EventCalendar.empty().blackout(FOMC) is None
        assert len(EventCalendar.empty()) == 0

    def test_next_event_and_upcoming(self, calendar):
        assert calendar.next_event(FOMC - timedelta(days=1)).at == FOMC
        assert calendar.next_event(FOMC + timedelta(minutes=16)).at == FOMC + timedelta(days=42)
        assert len(calendar.upcoming(FOMC - timedelta(days=1))) == 2
        assert calendar.upcoming(FOMC + timedelta(days=100)) == []


class TestFromConfig:
    def test_inline_events_and_a_file_are_merged(self, tmp_path):
        path = tmp_path / "events.txt"
        path.write_text("2026-10-28 14:00 FOMC statement\n")
        cfg = StrategyConfig(
            events=["2026-09-16 14:00 FOMC statement"], events_path=str(path),
            event_blackout_minutes_before=20, event_blackout_minutes_after=10,
        )
        calendar = EventCalendar.from_config(cfg, NY)
        assert len(calendar) == 2
        assert calendar.before == timedelta(minutes=20)
        assert calendar.after == timedelta(minutes=10)

    def test_the_default_config_has_no_events(self):
        """A plain ``Config()`` must not stand aside on dates a test happens
        to use; the shipped YAML files point at configs/events.txt."""
        assert len(Config().event_calendar()) == 0

    def test_the_shipped_calendar_loads(self):
        cfg = Config.from_yaml("configs/es_default.yaml")
        calendar = cfg.event_calendar()
        assert len(calendar) > 0
        assert all(e.at.hour == 14 for e in calendar.events)  # FOMC statements

    def test_the_blackout_lengths_are_validated(self):
        with pytest.raises(ValueError, match="event_blackout"):
            StrategyConfig(event_blackout_minutes_before=-1).validate()
