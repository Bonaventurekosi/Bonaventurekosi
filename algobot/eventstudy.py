"""Event study: how do prices move around events?

For every event the study lines up the symbol's *abnormal* return (its return
minus the market's on the same day) in event time: day 0 is the first session
that closes after the event became public, so its close-to-close return is the
market's reaction. Averaging across events washes out everything unrelated and
leaves the typical effect of that kind of event.

Two questions it answers:

* Reaction: how much does the price move on day 0?
* Drift:    does it keep moving in the same direction afterwards? That is the
            part a strategy can actually trade, since day 0 has already
            happened by the time the event is known.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data.dataset import Dataset
from .events import EventStore

DEFAULT_WINDOWS = ((0, 0), (1, 5), (1, 20), (1, 60))


@dataclass
class EventStudy:
    events: pd.DataFrame     # one row per usable event, plus its session date
    paths: np.ndarray        # abnormal returns, shape (events, pre + post + 1)
    pre: int
    post: int
    benchmark: str
    normal_abs_move: float   # mean |abnormal return| on ordinary days

    def __len__(self) -> int:
        return len(self.events)

    def subset(self, mask) -> "EventStudy":
        mask = np.asarray(mask, dtype=bool)
        return EventStudy(self.events[mask].reset_index(drop=True), self.paths[mask],
                          self.pre, self.post, self.benchmark, self.normal_abs_move)

    # -- per-event numbers ---------------------------------------------------
    def car(self, lo: int, hi: int) -> np.ndarray:
        """Cumulative abnormal return over event days ``lo..hi`` (NaN if the
        window runs past the available prices)."""
        if lo < -self.pre or hi > self.post or lo > hi:
            raise ValueError(f"window ({lo}, {hi}) is outside -{self.pre}..{self.post}")
        return self.paths[:, self.pre + lo:self.pre + hi + 1].sum(axis=1)

    @property
    def reaction(self) -> np.ndarray:
        return self.car(0, 0)

    # -- grouping --------------------------------------------------------------
    def by_kind(self) -> pd.Series:
        return self.events["kind"].reset_index(drop=True)

    def by_reaction(self, threshold: float = 0.02) -> pd.Series:
        """Did the stock jump, drop, or barely move on the event day?"""
        r = self.reaction
        up, down = f"jumped over {threshold:.0%}", f"dropped over {threshold:.0%}"
        labels = np.where(r >= threshold, up, np.where(r <= -threshold, down, "little change"))
        return pd.Series(pd.Categorical(labels, [up, "little change", down], ordered=True))

    def by_value(self, positive: str = "beat", negative: str = "miss",
                 neutral: str = "in line") -> pd.Series:
        """Split on the sign of each event's value (surprise, sentiment)."""
        v = self.events["value"].to_numpy()
        labels = np.where(np.isnan(v), None,
                          np.where(v > 0, positive, np.where(v < 0, negative, neutral)))
        return pd.Series(pd.Categorical(labels, [positive, neutral, negative], ordered=True))

    # -- aggregates ------------------------------------------------------------
    def windows(self) -> tuple[tuple[int, int], ...]:
        """The standard windows, trimmed to what this study covers."""
        return tuple(dict.fromkeys((lo, min(hi, self.post)) for lo, hi in DEFAULT_WINDOWS
                                   if lo <= self.post))

    def summary(self, groups: pd.Series | None = None,
                windows: Sequence[tuple[int, int]] | None = None,
                min_dates: int = 10) -> pd.DataFrame:
        """Mean abnormal return per group and window, with a t-statistic and the
        share of events that were positive.

        Events that share a date are not independent (earnings season, or fifty
        headlines about one story), so the t-statistic is clustered by event
        date. |t| above about 2 is the usual bar for "unlikely to be chance".
        A window observed on fewer than ``min_dates`` distinct dates is reported
        as missing rather than as a number that means nothing.
        """
        if groups is None:
            groups = pd.Series(["all events"] * len(self))
        windows = self.windows() if windows is None else windows
        names = groups.cat.categories if hasattr(groups, "cat") else pd.unique(groups.dropna())
        dates = self.events["session"].to_numpy()
        rows = []
        for name in names:
            mask = (groups == name).to_numpy()
            if not mask.any():
                continue
            row: dict[str, object] = {"group": name, "events": int(mask.sum())}
            row["avg_abs_move_day0"] = float(np.nanmean(np.abs(self.reaction[mask])))
            for lo, hi in windows:
                key = f"day {lo}" if lo == hi else f"days {lo}..{hi}"
                values = self.car(lo, hi)[mask]
                ok = ~np.isnan(values)
                values, when = values[ok], dates[mask][ok]
                row[key], row[key + " t"], row[key + " up%"] = np.nan, np.nan, np.nan
                if len(values) < 2 or len(np.unique(when)) < min_dates:
                    continue
                mean = values.mean()
                by_date = pd.Series(values - mean).groupby(when).sum().to_numpy()
                row[key] = float(mean)
                row[key + " up%"] = float((values > 0).mean())
                clusters = len(by_date)
                if clusters < 2:
                    continue          # one date is one observation: no t-statistic
                var = float((by_date ** 2).sum()) * clusters / (clusters - 1) / len(values) ** 2
                se = math.sqrt(var)
                row[key + " t"] = (float(mean / se) if se > 0
                                   else math.copysign(math.inf, mean) if mean else 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def mean_path(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Average cumulative abnormal return in event time, zeroed at day -1."""
        paths = self.paths if mask is None else self.paths[mask]
        cum = np.nancumsum(np.nanmean(paths, axis=0)) if len(paths) else np.zeros(self.pre + self.post + 1)
        return cum - (cum[self.pre - 1] if self.pre > 0 else 0.0)

    def days(self) -> np.ndarray:
        return np.arange(-self.pre, self.post + 1)


def run_event_study(store: EventStore, dataset: Dataset, *, benchmark: str | None = None,
                    pre: int = 10, post: int = 60, close: str = "16:00") -> EventStudy:
    """Line up abnormal returns around every event in ``store``.

    ``benchmark`` is the symbol whose return counts as "the market" (it must be
    in the dataset); without one, the equal-weight average of the dataset is used.
    """
    closes = dataset.closes().ffill()
    rets = closes.pct_change(fill_method=None)
    if benchmark is not None:
        if benchmark not in closes.columns:
            raise ValueError(f"benchmark {benchmark!r} is not in the price data")
        market = rets[benchmark]
    else:
        market = rets.mean(axis=1, skipna=True)
    abnormal = rets.sub(market, axis=0)
    values = abnormal.to_numpy()
    column = {s: i for i, s in enumerate(closes.columns)}
    n = len(closes)

    frame = store.frame
    sessions = store.sessions(dataset.timeline, close)
    keep, paths = [], []
    width = pre + post + 1
    for row, (symbol, session) in enumerate(zip(frame["symbol"], sessions)):
        j = column.get(symbol)
        if j is None or symbol == benchmark or session < 1 or np.isnan(values[session, j]):
            continue
        path = np.full(width, np.nan)
        lo, hi = session - pre, session + post + 1
        src = values[max(lo, 1):min(hi, n), j]
        start = max(lo, 1) - lo
        path[start:start + len(src)] = src
        keep.append(row)
        paths.append(path)

    events = frame.iloc[keep].reset_index(drop=True)
    events["session"] = dataset.timeline[sessions[keep]] if keep else pd.DatetimeIndex([])
    # What an ordinary day looks like, for scale: all stock-days that are not event days.
    ordinary = np.abs(values[1:]).copy()
    for row in keep:
        ordinary[sessions[row] - 1, column[frame["symbol"].iat[row]]] = np.nan
    tradable = [j for s, j in column.items() if s != benchmark]
    normal = float(np.nanmean(ordinary[:, tradable])) if tradable else float("nan")
    return EventStudy(events=events,
                      paths=np.array(paths) if paths else np.empty((0, width)),
                      pre=pre, post=post,
                      benchmark=benchmark or "equal-weight universe",
                      normal_abs_move=normal)


def format_summary(table: pd.DataFrame) -> str:
    """Plain-text version of ``EventStudy.summary`` for the terminal."""
    if table.empty:
        return "  (no events)"
    windows = [c for c in table.columns if c.startswith("day") and not c.endswith((" t", " up%"))]
    head = f"  {'group':<26}{'events':>7}{'|move| d0':>11}" + "".join(f"{w:>20}" for w in windows)
    lines = [head]
    for _, row in table.iterrows():
        line = f"  {str(row['group']):<26}{int(row['events']):>7}{row['avg_abs_move_day0'] * 100:>10.2f}%"
        for w in windows:
            if pd.isna(row[w]):
                line += f"{'n/a':>20}"
            else:
                line += f"{row[w] * 100:>+10.2f}% (t{row[w + ' t']:>+5.1f})"
        lines.append(line)
    return "\n".join(lines)
