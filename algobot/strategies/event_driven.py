"""Strategies driven by events (earnings, filings, news) rather than price alone."""
from __future__ import annotations

import os

import pandas as pd

from ..events import EventStore
from .base import Strategy, register


@register
class EventDrift(Strategy):
    """Follow through on company events.

    When an event of one of ``kinds`` lands, the strategy takes a position in
    the direction of the signal and holds it for ``hold`` bars:

    ``signal: reaction``  the stock's own move on the event bar, relative to
                          ``benchmark`` if one is set. A jump of more than
                          ``threshold`` buys, a drop sells.
    ``signal: value``     the event's own value (an earnings surprise, a
                          sentiment score), compared with ``threshold``.

    Events come from the CSV at ``events_path`` (see ``algobot events fetch``).
    An event only counts once the bar that closes after it has completed, so
    the entry happens on the following bar at the earliest. The file is re-read
    when it changes, so a live bot picks up newly collected events.
    """

    name = "event_drift"
    summary = "Trades the follow-through after earnings, filings or news events."
    PARAMS = {"events_path": None, "kinds": ("earnings",), "signal": "reaction",
              "threshold": 0.02, "hold": 20, "position_size": 0.1, "benchmark": None,
              "allow_short": True, "close": "16:00"}

    def setup(self):
        if self.signal not in ("reaction", "value"):
            raise ValueError("event_drift: signal must be 'reaction' or 'value'")
        if self.hold < 1 or self.position_size <= 0:
            raise ValueError("event_drift: hold and position_size must be positive")
        self.kinds = tuple(self.kinds)
        hours, minutes = (int(x) for x in str(self.close).split(":"))
        self._close = pd.Timedelta(hours=hours, minutes=minutes)
        self._store: EventStore | None = None
        self._mtime: float | None = None
        self._cut: dict[str, pd.Timestamp] = {}
        self._pos: dict[str, tuple[int, int]] = {}   # symbol -> (direction, bars left)
        self._load()

    def use_events(self, store: EventStore) -> "EventDrift":
        """Supply events directly instead of through ``events_path``."""
        self._store, self.events_path = store, None
        return self

    def _load(self) -> None:
        if not self.events_path:
            return
        try:
            mtime = os.path.getmtime(self.events_path)
        except OSError:
            if self._store is None:
                raise FileNotFoundError(f"event_drift: no events file at {self.events_path}")
            return
        if mtime != self._mtime:
            self._store, self._mtime = EventStore.from_csv(self.events_path), mtime

    def _cutoff(self, stamp) -> pd.Timestamp:
        stamp = pd.Timestamp(stamp)
        return stamp + self._close if stamp == stamp.normalize() else stamp

    def _bar_return(self, sym, h) -> float | None:
        if h.size(sym) < 2:
            return None
        c = h.close(sym, 2)
        return c[1] / c[0] - 1.0

    def _signal(self, sym, h, events) -> int:
        if self.signal == "value":
            score = events["value"].mean()
        else:
            score = self._bar_return(sym, h)
            bench = self.benchmark
            if score is not None and bench and bench in h and h.last_ts(bench) == h.last_ts(sym):
                score -= self._bar_return(bench, h) or 0.0
        if score is None or pd.isna(score) or abs(score) < self.threshold:
            return 0
        direction = 1 if score > 0 else -1
        return direction if direction > 0 or self.allow_short else 0

    def target_weights(self, h):
        self._load()
        out: dict[str, float] = {}
        for sym in self.universe(h):
            if sym == self.benchmark:
                continue
            out[sym] = 0.0
            if not h.size(sym):
                continue
            cut = self._cutoff(h.last_ts(sym))
            prev = self._cut.get(sym)
            direction, left = self._pos.get(sym, (0, 0))
            if prev is None or cut > prev:           # a new bar for this symbol
                self._cut[sym] = cut
                if left > 0:
                    left -= 1
                    if left == 0:
                        direction = 0
                if prev is not None and self._store is not None:
                    events = self._store.between(sym, prev, cut, self.kinds)
                    if len(events):
                        fresh = self._signal(sym, h, events)
                        if fresh:
                            direction, left = fresh, self.hold
                self._pos[sym] = (direction, left)
            out[sym] = direction * self.position_size
        gross = sum(abs(w) for w in out.values())
        if gross > 1.0:
            out = {s: w / gross for s, w in out.items()}
        return out
