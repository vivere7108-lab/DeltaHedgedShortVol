"""Append-only record of a live session.

A forward walk is only worth running if it produces evidence, and evidence
that lives in a process's memory is not evidence -- a crash, an OOM kill or
a careless ``systemctl restart`` takes the whole test with it.  This writes
every decision to disk as it happens, in JSON Lines, flushed per record.

Three files per session date, under ``live.journal_dir``:

``events-YYYY-MM-DD.jsonl``
    Entries, exits, hedges, skipped entries -- the decision log, one object
    per line, in the order they happened.
``fills-YYYY-MM-DD.jsonl``
    Every fill, so the record can be reconciled against the broker's own
    statement rather than trusted.
``bars-YYYY-MM-DD.jsonl``
    One snapshot per poll: the GEX read, the greeks, net delta, equity.
    This is the series you need to answer "was the band ever binding?" or
    "how long did the regime hold?" after the fact.

Appending rather than rewriting is deliberate: a restart mid-session adds to
the day's files instead of truncating them, so an interrupted forward walk
loses the position but not the history.  ``deltahedger report`` reads them
back into the same frames the backtest produces, so a paper session and a
backtest are analysed with one set of tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    """Make a value JSON-safe without losing what it meant."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class SessionJournal:
    """Writes a live session's decisions to disk as they are made."""

    def __init__(self, directory: str | Path, session_date: date | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._day = session_date
        self._written = {"events": 0, "fills": 0, "bars": 0}

    # -- paths -----------------------------------------------------------

    def _path(self, kind: str, moment: datetime) -> Path:
        day = self._day or moment.date()
        return self.directory / f"{kind}-{day.isoformat()}.jsonl"

    # -- writing ---------------------------------------------------------

    def _append(self, kind: str, moment: datetime, payload: dict) -> None:
        """One record, flushed. Never raises: losing the log must not stop
        the strategy, but it must be loud about having happened."""
        try:
            with self._path(kind, moment).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_plain(payload), separators=(",", ":")) + "\n")
                handle.flush()
            self._written[kind] = self._written.get(kind, 0) + 1
        except OSError as exc:
            log.error("could not write the %s journal (%s)", kind, exc)

    def record_event(self, event) -> None:
        self._append("events", event.timestamp, asdict(event))

    def record_fill(self, fill) -> None:
        self._append("fills", fill.timestamp, asdict(fill))

    def record_bar(self, state) -> None:
        self._append("bars", state.timestamp, asdict(state))

    def counts(self) -> dict[str, int]:
        return dict(self._written)


class JournallingStrategy:
    """Wraps a strategy so everything it decides is written down.

    A wrapper rather than a hook inside ``GexStraddleStrategy``: the strategy
    is shared with the backtest, which has no business touching a filesystem,
    and keeping the seam here means the live path gains persistence without
    the backtest gaining a way to differ from it.
    """

    def __init__(self, strategy, journal: SessionJournal):
        self.strategy = strategy
        self.journal = journal
        self._events_seen = 0
        self._fills_seen = 0

    def __getattr__(self, name: str):
        return getattr(self.strategy, name)

    def on_bar(self, bar, execution):
        state = self.strategy.on_bar(bar, execution)
        # Drain whatever the bar appended, so ordering on disk matches the
        # order things actually happened in.
        for event in self.strategy.events[self._events_seen:]:
            self.journal.record_event(event)
        self._events_seen = len(self.strategy.events)
        for fill in self.strategy.fills[self._fills_seen:]:
            self.journal.record_fill(fill)
        self._fills_seen = len(self.strategy.fills)
        self.journal.record_bar(state)
        return state


def read_journal(directory: str | Path, kind: str, day: date | None = None):
    """Read journal lines back as a DataFrame, for analysis after the fact.

    ``kind`` is "events", "fills" or "bars". With no ``day``, every session
    in the directory is concatenated, which is what you want at the end of a
    multi-day walk.
    """
    import pandas as pd

    directory = Path(directory)
    pattern = f"{kind}-{day.isoformat()}.jsonl" if day else f"{kind}-*.jsonl"
    paths = sorted(directory.glob(pattern))
    if not paths:
        return pd.DataFrame()

    rows: list[dict] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A half-written final line is what a hard kill leaves behind.
                # Skipping it is right; silently skipping it is not.
                log.warning("skipping malformed line %d of %s", number, path.name)
    frame = pd.DataFrame(rows)
    if not frame.empty and "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame
