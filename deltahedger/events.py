"""Scheduled high-volatility events, and the blackout around each one.

An FOMC statement, a CPI print or a payrolls release is a known moment at
which the underlying can gap and implied vol reprices in one step.  A
delta-hedged straddle cannot hedge through a gap -- there is no path to
rebalance along -- so the position is taken off for a window either side
of the event and put back on afterwards if the regime still asks for it.

The calendar is *input*, not something this module knows.  Events are
listed in ``strategy.events`` (inline) or ``strategy.events_path`` (a text
file, one per line), in exchange-local time::

    2026-09-16 14:00 FOMC statement
    2026-10-28T14:00 FOMC statement      # ISO "T" separator is fine too

Everything after the timestamp is a label carried into the log.  Blank
lines and ``#`` comments are ignored in a file.  The shipped
``configs/events.txt`` lists the FOMC statement times; anything else
worth standing aside for -- CPI, NFP, an index rebalance -- goes in the
same list.  Nothing here downloads a calendar: a walk that quietly traded
through an event because a fetch failed would be worse than one that
needs its calendar maintained by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

_TIMESTAMP = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})[T ](\d{1,2}:\d{2})(?::\d{2})?\s*(.*?)\s*$"
)


@dataclass(frozen=True)
class MarketEvent:
    """One scheduled event, in exchange-local time."""

    at: datetime
    label: str = ""

    def __str__(self) -> str:
        when = self.at.strftime("%Y-%m-%d %H:%M")
        return f"{when} {self.label}".strip()


def parse_event(raw: Any, tz: ZoneInfo) -> MarketEvent:
    """One event from a config entry.

    Accepts ``"YYYY-MM-DD HH:MM label"`` (or with a ``T``), or a mapping
    with ``at`` and an optional ``label``.  Naive timestamps are read as
    exchange-local; aware ones are converted.
    """
    if isinstance(raw, MarketEvent):
        return raw
    if isinstance(raw, dict):
        at = raw.get("at") or raw.get("time") or raw.get("timestamp")
        if at is None:
            raise ValueError(f"an event mapping needs an 'at' key; got {raw!r}")
        label = str(raw.get("label", "") or "")
        return MarketEvent(_localize(_parse_when(at), tz), label)
    if isinstance(raw, datetime):
        return MarketEvent(_localize(raw, tz), "")
    if isinstance(raw, str):
        match = _TIMESTAMP.match(raw)
        if not match:
            raise ValueError(
                f"cannot read an event from {raw!r}; expected "
                "'YYYY-MM-DD HH:MM [label]'"
            )
        day, clock, label = match.groups()
        return MarketEvent(
            _localize(datetime.fromisoformat(f"{day}T{_pad(clock)}"), tz), label
        )
    raise TypeError(f"cannot read an event from {raw!r}")


def _pad(clock: str) -> str:
    hour, minute = clock.split(":")
    return f"{int(hour):02d}:{minute}"


def _parse_when(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    match = _TIMESTAMP.match(text)
    if match and not match.group(3):
        return datetime.fromisoformat(f"{match.group(1)}T{_pad(match.group(2))}")
    return datetime.fromisoformat(text)


def _localize(moment: datetime, tz: ZoneInfo) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=tz)
    return moment.astimezone(tz)


def read_events_file(path: str | Path, tz: ZoneInfo) -> list[MarketEvent]:
    """Events from a text file, one per line; ``#`` starts a comment."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no events file at {path}")
    events: list[MarketEvent] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            events.append(parse_event(text, tz))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from None
    return events


class EventCalendar:
    """The events, and the blackout window around each.

    ``blackout(moment)`` answers the only question the strategy asks: is
    this moment inside ``[event - before, event + after]`` for any event?
    Both edges are inclusive, so a bar that lands exactly on the boundary
    is treated as inside -- the conservative side of a rounding question.
    """

    def __init__(
        self,
        events: Iterable[MarketEvent],
        before: timedelta = timedelta(minutes=15),
        after: timedelta = timedelta(minutes=15),
    ):
        self.events: tuple[MarketEvent, ...] = tuple(sorted(events, key=lambda e: e.at))
        self.before = before
        self.after = after

    @classmethod
    def from_config(cls, cfg, tz: ZoneInfo) -> "EventCalendar":
        """Build from ``StrategyConfig``: the inline list plus the file."""
        events = [parse_event(raw, tz) for raw in (cfg.events or [])]
        if cfg.events_path:
            events.extend(read_events_file(cfg.events_path, tz))
        return cls(
            events,
            before=timedelta(minutes=cfg.event_blackout_minutes_before),
            after=timedelta(minutes=cfg.event_blackout_minutes_after),
        )

    @classmethod
    def empty(cls) -> "EventCalendar":
        return cls(())

    def __len__(self) -> int:
        return len(self.events)

    def window(self, event: MarketEvent) -> tuple[datetime, datetime]:
        return event.at - self.before, event.at + self.after

    def blackout(self, moment: datetime) -> MarketEvent | None:
        """The event whose blackout contains ``moment``, or ``None``."""
        for event in self.events:
            start, end = self.window(event)
            if start <= moment <= end:
                return event
            if start > moment:
                break  # sorted: nothing later can contain this moment
        return None

    def next_event(self, moment: datetime) -> MarketEvent | None:
        """The first event whose blackout has not yet ended."""
        for event in self.events:
            if self.window(event)[1] >= moment:
                return event
        return None

    def upcoming(self, moment: datetime, limit: int = 5) -> Sequence[MarketEvent]:
        return [e for e in self.events if e.at >= moment][:limit]
