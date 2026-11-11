"""HTML report for an event study (shares the backtest report's page and charts)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.report import chart, page, table, tile
from .eventstudy import EventStudy

_SERIES = ["--series-1", "--series-2", "--series-3"]


def _cell(row: pd.Series, key: str) -> str:
    if key not in row or pd.isna(row[key]):
        return "n/a"
    return f"{row[key] * 100:+.2f}% (t {row[key + ' t']:+.1f})"


def _summary_table(summary: pd.DataFrame, first_header: str) -> str:
    windows = [c for c in summary.columns if c.startswith("day") and not c.endswith((" t", " up%"))]
    headers = [first_header, "Events", "Typical move, day 0"] + [w.capitalize() for w in windows]
    rows = [[str(r["group"]).replace("_", " "), f"{int(r['events']):,}",
             f"{r['avg_abs_move_day0'] * 100:.2f}%"] + [_cell(r, w) for w in windows]
            for _, r in summary.iterrows()]
    return table(headers, rows)


def _path_chart(cid: str, title: str, study: EventStudy, groups: pd.Series, min_events: int = 5) -> str:
    days = study.days()
    labels = [f"day {d:+d}" if d else "day 0 (event)" for d in days]
    step = 10 if study.post > 30 else 5
    ticks = [(i, f"{d:+d}" if d else "0") for i, d in enumerate(days) if d % step == 0]
    series = []
    for name in groups.cat.categories:
        mask = (groups == name).to_numpy()
        if mask.sum() >= min_events and len(series) < len(_SERIES):
            series.append((f"{name} ({int(mask.sum()):,} events)", _SERIES[len(series)],
                           study.mean_path(mask)))
    return chart(cid, title, labels, ticks, series, "signed") if series else ""


def render_event_report(study: EventStudy, title: str = "Event study", kind: str = "earnings",
                        threshold: float = 0.02) -> str:
    if not len(study):
        return page(title, "no usable events", '<section class="card">No events matched the price data.</section>')
    kinds = study.by_kind()
    focus = study.subset((kinds == kind).to_numpy()) if (kinds == kind).any() else study
    label = kind.replace("_", " ") if (kinds == kind).any() else "event"
    reaction = focus.by_reaction(threshold)
    post = f"days 1..{min(60, study.post)}"
    windows = study.windows()
    by_reaction = focus.summary(reaction, windows)

    def drift_tile(name: str, heading: str) -> str:
        row = by_reaction[by_reaction["group"] == name]
        key = [c for c in by_reaction.columns
               if c.startswith("days 1..") and not c.endswith((" t", " up%"))][-1]
        if row.empty or pd.isna(row[key].iat[0]):
            return tile(heading, "n/a")
        return tile(heading, f"{row[key].iat[0] * 100:+.2f}%",
                    f"vs market, {int(row['events'].iat[0]):,} events, t {row[key + ' t'].iat[0]:+.1f}")

    up, down = reaction.cat.categories[0], reaction.cat.categories[-1]
    sessions = study.events["session"]
    tiles = "".join([
        tile("Events analysed", f"{len(study):,}",
             f"{study.events['symbol'].nunique()} companies, {sessions.min().year}-{sessions.max().year}"),
        tile(f"Typical move on {label} day", f"{np.nanmean(np.abs(focus.reaction)) * 100:.2f}%",
             f"ordinary day {study.normal_abs_move * 100:.2f}%"),
        drift_tile(up, f"Next {post.split('..')[1]} days after a jump"),
        drift_tile(down, f"Next {post.split('..')[1]} days after a drop"),
    ])

    body = f'<div class="tiles">{tiles}</div>'
    body += _path_chart("reaction", f"{label.capitalize()}: average path versus the market, "
                                    f"split by the first day's move", focus, reaction)
    body += (f'<section class="card"><h2>{label.capitalize()} by first-day reaction</h2>'
             f'{_summary_table(by_reaction, "First-day move")}</section>')
    valued = focus.subset(~np.isnan(focus.events["value"].to_numpy()))
    if len(valued) >= 20:
        split = valued.by_value()
        body += _path_chart("surprise", f"{label.capitalize()}: average path by reported surprise",
                            valued, split)
        body += (f'<section class="card"><h2>{label.capitalize()} by reported surprise</h2>'
                 f'{_summary_table(valued.summary(split, windows), "Surprise")}</section>')
    if kinds.nunique() > 1:
        order = kinds.value_counts().index
        groups = pd.Series(pd.Categorical(kinds, order, ordered=True))
        body += (f'<section class="card"><h2>All event types</h2>'
                 f'{_summary_table(study.summary(groups, windows), "Event type")}</section>')
    days = study.days()
    paths = {str(name): focus.mean_path((reaction == name).to_numpy()) for name in reaction.cat.categories
             if (reaction == name).any()}
    rows = [[f"{d:+d}"] + [f"{p[i] * 100:+.2f}%" for p in paths.values()] for i, d in enumerate(days)]
    body += (f'<section class="card"><details><summary>Chart data as a table</summary>'
             f'{table(["Event day"] + list(paths), rows)}</details></section>')
    subtitle = (f"{len(study):,} events &middot; {sessions.min().date()} to {sessions.max().date()}"
                f" &middot; market = {study.benchmark}")
    return page(title, subtitle, body,
                "Returns are the stock's return minus the market's, averaged across events; day 0 is the "
                "first session that closed after the event became public. Day 0 itself cannot be traded "
                "on the news; only what follows can. A t value beyond about 2 in either direction suggests "
                "the average is unlikely to be chance (events sharing a date count as one observation). "
                "If the companies were picked because they are large today, every group's drift is lifted "
                "by hindsight, so compare the groups with each other. Past patterns can fade or reverse.")


def write_event_report(study: EventStudy, path: str | Path, title: str = "Event study",
                       kind: str = "earnings", threshold: float = 0.02) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_event_report(study, title, kind, threshold), encoding="utf-8")
    return path
