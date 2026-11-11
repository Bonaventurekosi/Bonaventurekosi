import json

import numpy as np
import pandas as pd
import pytest
import yaml

from algobot.backtest.engine import Backtester
from algobot.cli import main
from algobot.data import Dataset
from algobot.data.event_sources import (AlpacaNews, SecFilings, YahooNews, headline_sentiment,
                                        sec_kind)
from algobot.eventreport import render_event_report
from algobot.events import EventStore, session_cutoffs
from algobot.eventstudy import format_summary, run_event_study
from algobot.execution.orders import CostModel, ExecutionConfig
from algobot.pipeline import Pipeline
from algobot.strategies import create
from conftest import make_bars
from test_data_brokers import FakeResponse, FakeSession

DAYS = pd.date_range("2020-01-01", periods=320, freq="B")
MARKET = 100 * 1.0003 ** np.arange(len(DAYS))


def planted(events, jump=0.05, drift=0.003, drift_days=10):
    """A stock that tracks the market except around events: it moves ``jump``
    on the event day and keeps drifting the same way for ``drift_days``."""
    extra = np.zeros(len(DAYS))
    for day, sign in events:
        extra[day] += sign * jump
        extra[day + 1:day + 1 + drift_days] += sign * drift
    rets = MARKET[1:] / MARKET[:-1] - 1 + extra[1:]
    return np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)])


PLAN = {"AAA": [(50, 1), (120, -1), (200, 1)], "BBB": [(60, -1), (130, 1), (210, -1)],
        "CCC": [(70, 1), (140, 1), (220, -1)]}
FRAMES = {"MKT": make_bars(MARKET, start=DAYS[0]),
          **{s: make_bars(planted(ev), start=DAYS[0]) for s, ev in PLAN.items()}}
EVENT_ROWS = [{"symbol": s, "time": DAYS[d] + pd.Timedelta(hours=9), "kind": "earnings",
               "value": 4.0 * sign, "text": "results"} for s, ev in PLAN.items() for d, sign in ev]


# -- the store -------------------------------------------------------------------
def test_events_belong_to_the_first_session_that_closes_after_them():
    monday, friday = pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-05")
    timeline = pd.DatetimeIndex([friday, monday, monday + pd.Timedelta(days=1)])
    store = EventStore([
        {"symbol": "X", "time": "2024-01-05 08:00"},      # before the open: Friday
        {"symbol": "X", "time": "2024-01-05 16:00"},      # exactly at the close: still Friday
        {"symbol": "X", "time": "2024-01-05 16:30"},      # after the close: Monday
        {"symbol": "X", "time": "2024-01-06 12:00"},      # weekend: Monday
        {"symbol": "X", "time": "2024-01-09 15:59"},      # Tuesday
        {"symbol": "X", "time": "2024-01-09 18:00"},      # after the data ends
    ])
    assert store.sessions(timeline).tolist() == [0, 0, 1, 1, 2, -1]
    assert session_cutoffs(timeline)[0] == pd.Timestamp("2024-01-05 16:00")
    hourly = pd.date_range("2024-01-05 10:00", periods=3, freq="h")
    assert (session_cutoffs(hourly) == hourly).all()       # intraday bars: known at their own stamp


def test_store_converts_time_zones_dedupes_and_filters(tmp_path):
    store = EventStore([
        {"symbol": "A", "time": pd.Timestamp("2024-01-10 21:30", tz="UTC"), "kind": "earnings", "value": "2.5"},
        {"symbol": "A", "time": pd.Timestamp("2024-01-10 21:30", tz="UTC"), "kind": "earnings", "value": 2.5},
        {"symbol": "B", "time": pd.Timestamp("2024-07-10 20:30", tz="UTC"), "kind": "news", "text": "hello"},
        {"symbol": "C", "time": "not a date"},
    ])
    assert len(store) == 2 and store.symbols == ["A", "B"]
    assert store.frame["time"].tolist() == [pd.Timestamp("2024-01-10 16:30"), pd.Timestamp("2024-07-10 16:30")]
    assert store.counts() == {"earnings": 1, "news": 1} and store.frame["value"].iloc[0] == 2.5
    path = store.to_csv(tmp_path / "sub" / "e.csv")
    back = EventStore.from_csv(path)
    pd.testing.assert_frame_equal(back.frame, store.frame)
    assert len(store.merge(back)) == 2                                     # nothing doubled
    assert len(store.filter(kinds=["news"])) == 1 and len(store.filter(start="2024-06-01")) == 1
    got = store.between("A", "2024-01-10 16:00", "2024-01-10 16:30")
    assert len(got) == 1 and store.between("A", "2024-01-10 16:30", "2024-01-11").empty
    assert store.between("ZZZ", "2020", "2030").empty and len(EventStore()) == 0
    (tmp_path / "bad.csv").write_text("ticker,when\nA,2024-01-01\n")
    with pytest.raises(ValueError, match="symbol"):
        EventStore.from_csv(tmp_path / "bad.csv")


# -- collectors ------------------------------------------------------------------
def test_sec_item_classification():
    assert sec_kind("2.02,9.01") == "earnings" and sec_kind("7.01,2.02") == "earnings"
    assert sec_kind("5.02,9.01") == "executive_change" and sec_kind("8.01") == "other_event"
    assert sec_kind("1.01,5.02") == "material_agreement"       # the more material item wins
    assert sec_kind("9.01") == "filing" and sec_kind("") == "filing"


def test_sec_filings_fetch_and_time_conversion():
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    recent = {"form": ["8-K", "10-Q", "8-K", "8-K"],
              "acceptanceDateTime": ["2024-05-02T20:30:34.000Z", "2024-05-03T10:00:00.000Z",
                                     "2024-02-01T21:30:30.000Z", "2023-03-10T13:05:00.000Z"],
              "items": ["2.02,9.01", "", "2.02,9.01", "5.02"]}
    older = {"form": ["8-K", "8-K"], "items": ["2.02,9.01", "8.01"],
             "acceptanceDateTime": ["2015-01-27T21:30:00.000Z", "2011-06-01T12:00:00.000Z"]}
    submissions = {"filings": {"recent": recent, "files": [
        {"name": "CIK0000320193-submissions-001.json", "filingFrom": "1994-01-26", "filingTo": "2015-10-05"},
        {"name": "ancient.json", "filingFrom": "1980-01-01", "filingTo": "1993-12-31"}]}}
    session = FakeSession(tickers, submissions, older)
    store = SecFilings(session=session, pause=0).fetch(["AAPL"], start="2012-01-01")
    assert session.calls[1][1].endswith("CIK0000320193.json")
    assert session.calls[2][1].endswith("submissions-001.json") and len(session.calls) == 3
    assert "User-Agent" in session.calls[0][2]["headers"]
    earnings = store.filter(kinds=["earnings"]).frame["time"].tolist()
    # 4:30 pm New York in both summer and winter, whatever the UTC offset was.
    assert earnings == [pd.Timestamp("2015-01-27 16:30:00"), pd.Timestamp("2024-02-01 16:30:30"),
                        pd.Timestamp("2024-05-02 16:30:34")]
    assert store.counts() == {"earnings": 3, "executive_change": 1}      # 10-Q and pre-2012 dropped
    with pytest.raises(KeyError, match="no company"):
        SecFilings(session=FakeSession(tickers), pause=0).cik("NOPE")


def test_headline_sentiment():
    assert headline_sentiment("Acme beats estimates as profit surges") == 1.0
    assert headline_sentiment("Acme plunges after earnings miss and lawsuit") == -1.0
    assert headline_sentiment("Acme upgrade follows weak quarter") == 0.0
    assert headline_sentiment("Acme does not beat forecasts") == -1.0      # negation flips it
    assert headline_sentiment("Acme to hold annual meeting") == 0.0


def test_yahoo_news_parse_keeps_only_stories_about_the_symbol():
    payload = {"news": [
        {"title": "Acme surges on record sales", "providerPublishTime": 1704898800,
         "publisher": "Wire", "relatedTickers": ["ACME"]},
        {"title": "Markets wrap", "providerPublishTime": 1704898800, "relatedTickers": ["SPY"]},
        {"title": "", "providerPublishTime": 1704898800}]}
    store = YahooNews(session=FakeSession(payload)).fetch(["ACME"])
    assert len(store) == 1
    row = store.frame.iloc[0]
    assert (row["kind"], row["value"], row["source"]) == ("news", 1.0, "Wire")
    assert row["time"] == pd.Timestamp("2024-01-10 10:00")                # 15:00 UTC in New York


def test_alpaca_news_pages_and_splits_by_symbol():
    page1 = {"news": [{"headline": "A beats", "created_at": "2024-01-10T15:00:00Z",
                       "symbols": ["A", "B", "ZZ"], "source": "bz"}], "next_page_token": "t"}
    page2 = {"news": [{"headline": "B recall", "created_at": "2024-01-11T15:00:00Z",
                       "symbols": ["B"], "source": "bz"}], "next_page_token": None}
    session = FakeSession(page1, page2)
    store = AlpacaNews("k", "s", session=session).fetch(["A", "B"], start="2024-01-01")
    assert len(store) == 3 and store.symbols == ["A", "B"]
    assert session.calls[1][2]["params"]["page_token"] == "t"
    assert session.calls[0][2]["params"]["start"] == "2024-01-01T00:00:00Z"
    assert store.filter(symbols=["B"]).frame["value"].tolist() == [1.0, -1.0]


# -- the study -------------------------------------------------------------------
def study(rows=EVENT_ROWS, **kw):
    return run_event_study(EventStore(rows), Dataset(FRAMES), benchmark="MKT", **{"pre": 5, "post": 20, **kw})


def test_event_study_recovers_a_planted_effect():
    st = study()
    assert len(st) == 9 and st.benchmark == "MKT"
    up = st.events["value"].to_numpy() > 0
    np.testing.assert_allclose(st.reaction[up], 0.05, atol=1e-3)
    np.testing.assert_allclose(st.reaction[~up], -0.05, atol=1e-3)
    np.testing.assert_allclose(st.car(1, 10)[up], 0.03, atol=2e-3)         # the drift
    np.testing.assert_allclose(st.car(11, 20), 0.0, atol=1e-9)             # and nothing after it
    np.testing.assert_allclose(st.car(-5, -1), 0.0, atol=1e-9)             # nor before the event
    assert st.normal_abs_move < 0.002                                      # ordinary days are quiet
    path = st.mean_path(up)
    assert path[st.pre - 1] == 0.0 and path[st.pre] == pytest.approx(0.05, abs=1e-3)
    assert path[-1] == pytest.approx(0.08, abs=3e-3)


def test_after_close_events_count_from_the_next_session():
    morning = [{"symbol": "AAA", "time": DAYS[50] + pd.Timedelta(hours=9), "kind": "earnings"}]
    evening = [{"symbol": "AAA", "time": DAYS[49] + pd.Timedelta(hours=17), "kind": "earnings"}]
    a, b = study(morning), study(evening)
    assert a.events["session"].iloc[0] == b.events["session"].iloc[0] == DAYS[50]
    np.testing.assert_allclose(a.paths, b.paths)


def test_study_groups_and_statistics():
    st = study()
    reaction = st.by_reaction(0.02)
    assert reaction.value_counts().to_dict() == {"jumped over 2%": 5, "little change": 0, "dropped over 2%": 4}
    table = st.summary(reaction, windows=((0, 0), (1, 10)), min_dates=3)
    jumped = table[table["group"] == "jumped over 2%"].iloc[0]
    assert jumped["events"] == 5 and jumped["day 0"] == pytest.approx(0.05, abs=1e-3)
    assert jumped["days 1..10"] == pytest.approx(0.03, abs=2e-3) and jumped["days 1..10 up%"] == 1.0
    assert jumped["day 0 t"] > 50
    assert st.by_value().value_counts().to_dict() == {"beat": 5, "in line": 0, "miss": 4}
    # Too few distinct dates to say anything: reported as missing, not as a number.
    assert st.summary(reaction, min_dates=10)["day 0"].isna().all()
    assert "jumped over 2%" in format_summary(table) and "(no events)" in format_summary(table.iloc[0:0])
    with pytest.raises(ValueError, match="outside"):
        st.car(1, 99)


def test_same_day_events_are_one_observation():
    """Fifty headlines about one story must not look like fifty independent results."""
    one_day = [{"symbol": "AAA", "time": DAYS[50] + pd.Timedelta(hours=9, minutes=m), "kind": "news",
                "text": f"headline {m}"} for m in range(50)]
    spread = [{"symbol": s, "time": DAYS[d] + pd.Timedelta(hours=9), "kind": "news"}
              for s, ev in PLAN.items() for d, sign in ev if sign > 0]
    clustered = study(one_day).summary(min_dates=1)
    assert clustered["events"].iloc[0] == 50 and np.isnan(clustered["day 0 t"].iloc[0])
    assert clustered["day 0"].iloc[0] == pytest.approx(0.05, abs=1e-3)      # the average is still shown
    assert study(spread).summary(min_dates=1)["day 0 t"].iloc[0] > 50


def test_study_skips_what_it_cannot_measure():
    rows = [{"symbol": "MKT", "time": DAYS[50] + pd.Timedelta(hours=9)},        # the benchmark itself
            {"symbol": "ZZZ", "time": DAYS[50] + pd.Timedelta(hours=9)},        # no prices
            {"symbol": "AAA", "time": DAYS[0] + pd.Timedelta(hours=9)},         # no prior bar
            {"symbol": "AAA", "time": DAYS[-1] + pd.Timedelta(days=30)},        # after the data
            {"symbol": "AAA", "time": DAYS[310] + pd.Timedelta(hours=9)}]       # window runs off the end
    st = study(rows)
    assert len(st) == 1 and np.isnan(st.car(1, 20)[0]) and not np.isnan(st.reaction[0])
    assert len(study([])) == 0
    with pytest.raises(ValueError, match="benchmark"):
        run_event_study(EventStore(rows), Dataset(FRAMES), benchmark="NOPE")
    market_free = run_event_study(EventStore(EVENT_ROWS), Dataset(FRAMES))
    assert market_free.benchmark == "equal-weight universe" and len(market_free) == 9


def test_event_report_renders():
    html = render_event_report(study(pre=10), "Planted <b>", "earnings")
    assert html.count("<svg") == 1 and "day 0 (event)" in html and "&lt;b&gt;" in html
    assert "jumped over 2% (5 events)" in html
    assert "by reported surprise" not in html                               # too few valued events
    many = [dict(row, time=row["time"] + pd.Timedelta(days=7 * k), text=f"q{k}")
            for row in EVENT_ROWS for k in range(3)]
    assert "by reported surprise" in render_event_report(study(many, pre=10), "More", "earnings")
    assert "nan" not in html.lower().replace("financial", "")
    assert "no usable events" in render_event_report(study([]), "Empty")


# -- trading on events -----------------------------------------------------------
def drift_run(store=None, cut=None, **params):
    spec = {"type": "event_drift", "params": {"benchmark": "MKT", "hold": 5, "position_size": 0.2, **params}}
    strategy = create(spec).use_events(store or EventStore(EVENT_ROWS))
    frames = FRAMES if cut is None else {s: df.iloc[:cut] for s, df in FRAMES.items()}
    pipe = Pipeline(strategy, execution=ExecutionConfig(min_trade_weight=0.05))
    return Backtester(frames, pipe, CostModel(0, 0)).run()


def test_event_drift_enters_after_the_reaction_and_holds():
    res = drift_run()
    aaa = [f for f in res.fills if f.symbol == "AAA"]
    # Event on day 50 is known at that close; the entry is day 51's open, the exit 5 bars later.
    assert [(f.ts, np.sign(f.qty)) for f in aaa[:2]] == [(DAYS[51], 1.0), (DAYS[56], -1.0)]
    assert [(f.ts, np.sign(f.qty)) for f in aaa[2:4]] == [(DAYS[121], -1.0), (DAYS[126], 1.0)]   # short the drop
    assert not [f for f in res.fills if f.symbol == "MKT"]                  # the benchmark is never traded
    assert res.metrics["total_return"] > 0.02 and res.metrics["win_rate"] == 1.0
    assert res.weights["AAA"].iloc[53] == pytest.approx(0.2, abs=0.02)


def test_event_drift_options():
    long_only = drift_run(allow_short=False)
    assert all(f.qty > 0 for f in long_only.fills if f.ts in (DAYS[51], DAYS[121], DAYS[61]))
    assert not [f for f in long_only.fills if f.ts == DAYS[121]]
    assert not drift_run(threshold=0.10).fills                              # 5% moves are below the bar
    assert not drift_run(kinds=["news"]).fills                              # wrong event type
    by_value = drift_run(signal="value", threshold=3.0)                     # surprises are +/-4
    assert len(by_value.fills) == len(drift_run().fills)
    assert not drift_run(signal="value", threshold=5.0).fills
    with pytest.raises(ValueError, match="signal"):
        create({"type": "event_drift", "params": {"signal": "vibes"}})


def test_event_drift_cannot_see_the_future():
    full, part = drift_run(), drift_run(cut=125)
    np.testing.assert_allclose(part.equity.to_numpy(), full.equity.to_numpy()[:125], rtol=1e-12)
    # An event stamped after the close is not acted on until the next session has reacted.
    late = [{"symbol": "AAA", "time": DAYS[49] + pd.Timedelta(hours=17), "kind": "earnings"}]
    assert drift_run(EventStore(late)).fills[0].ts == DAYS[51]
    future = [{"symbol": "AAA", "time": DAYS[-1] + pd.Timedelta(days=9), "kind": "earnings"}]
    assert not drift_run(EventStore(future)).fills


def test_event_drift_reloads_its_file(tmp_path):
    import os
    path = tmp_path / "events.csv"
    EventStore(EVENT_ROWS[:1]).to_csv(path)
    strategy = create({"type": "event_drift", "params": {"events_path": str(path)}})
    assert len(strategy._store) == 1
    EventStore(EVENT_ROWS).to_csv(path)
    os.utime(path, (1, 1))                                                  # make the change visible
    strategy._load()
    assert len(strategy._store) == 9
    with pytest.raises(FileNotFoundError):
        create({"type": "event_drift", "params": {"events_path": str(tmp_path / "missing.csv")}})


# -- command line ----------------------------------------------------------------
@pytest.fixture
def event_cfg(tmp_path):
    for sym, df in FRAMES.items():
        df.to_csv(tmp_path / f"{sym}.csv", index_label="date")
    events = tmp_path / "events.csv"
    EventStore(EVENT_ROWS).to_csv(events)
    cfg = {"name": "event test", "data": {"source": "csv", "path": str(tmp_path), "symbols": list(FRAMES)},
           "events": {"path": str(events), "benchmark": "MKT", "pre": 5, "post": 20, "sources": ["sec"]},
           "strategy": {"type": "event_drift", "params": {"events_path": str(events), "benchmark": "MKT",
                                                          "hold": 5}}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path), cfg, tmp_path


def test_events_study_command(event_cfg, capsys):
    path, _, tmp = event_cfg
    assert main(["events", "study", "-c", path, "--report", str(tmp / "r.html")]) == 0
    out = capsys.readouterr().out
    assert "9 events, 3 symbols" in out and "market = MKT" in out and "split by the move" in out
    assert (tmp / "r.html").read_text().count("<svg") >= 1
    assert main(["backtest", "-c", path, "--json"]) == 0                    # and the strategy runs from config
    assert json.loads(capsys.readouterr().out)["metrics"]["fills"] > 0


def test_events_fetch_command(event_cfg, capsys, monkeypatch):
    path, cfg, tmp = event_cfg
    import algobot.cli as cli
    seen = {}

    class Source:
        def fetch(self, symbols, start=None):
            seen["symbols"] = list(symbols)
            return EventStore(EVENT_ROWS[:2] + [{"symbol": "AAA", "time": "2021-03-01 09:00", "kind": "news"}])

    monkeypatch.setattr(cli, "make_event_source", lambda name: Source())
    assert main(["events", "fetch", "-c", path]) == 0
    out = capsys.readouterr().out
    assert seen["symbols"] == ["AAA", "BBB", "CCC"]                         # the benchmark is not fetched
    assert "10 events (+1 new)" in out and "news" in out
    assert len(EventStore.from_csv(tmp / "events.csv")) == 10
    bad = tmp / "bad.yaml"
    bad.write_text(yaml.safe_dump({**cfg, "events": {"benchmark": "MKT"}}))
    with pytest.raises(SystemExit, match="events.path"):
        main(["events", "study", "-c", str(bad)])
