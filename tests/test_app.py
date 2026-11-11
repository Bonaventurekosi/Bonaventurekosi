import json
import threading
import time
from http.client import HTTPConnection

import numpy as np
import pandas as pd
import pytest
import yaml

from algobot import tasks
from algobot.app.paper import PaperSession, bar_seconds
from algobot.app.server import ApiError, App, make_server, to_json
from algobot.data import synthetic_universe
from algobot.live import ReplayProvider

DEMO = {
    "name": "Demo setup",
    "data": {"source": "synthetic", "symbols": ["AA", "BB", "CC"], "bars": 400, "seed": 2, "kind": "range"},
    "strategy": {"type": "ensemble", "params": {"children": [
        {"type": "bollinger", "weight": 2, "regime": "range"}, {"type": "donchian"}]}},
    "sizing": {"target_vol": 0.1},
    "risk": {"long_only": True, "max_position_weight": 0.5, "trailing_stop_pct": 0.1,
             "max_drawdown_pct": 0.2, "drawdown_halt_bars": 30},
    "backtest": {"initial_cash": 20000},
    "optimize": {"space": {"strategy.params.children.0.params.window": [10, 20]}},
    "walk_forward": {"train_bars": 150, "test_bars": 100},
}
HEADER = "# A test setup on simulated prices.\n#   python -m algobot backtest -c demo.yaml\n"


@pytest.fixture
def app(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "demo.yaml").write_text(HEADER + yaml.safe_dump(DEMO))
    (configs / "zeta.yaml").write_text(yaml.safe_dump({**DEMO, "name": "Another", "strategy": {"type": "tsmom"}}))
    (configs / "broken.yaml").write_text("strategy: {type: moonshot}\n")
    (configs / "not a setup!.yaml").write_text("x: 1\n")
    application = App(configs, tmp_path / "state")
    yield application
    application.close()


def finish(app, job, timeout=60):
    deadline = time.time() + timeout
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.02)
    return job


# -- setups ----------------------------------------------------------------------
def test_setups_are_listed_described_and_validated(app):
    listed = app.list_setups()
    assert [s["id"] for s in listed] == ["demo", "zeta", "broken"]          # demo first, broken last
    demo, _, broken = listed
    assert (demo["name"], demo["strategy"], demo["source"], demo["symbols"]) == ("Demo setup", "ensemble", "synthetic", 3)
    assert demo["blurb"] == "A test setup on simulated prices."              # the command line is left out
    assert not broken["ok"] and "unknown strategy" in broken["error"]
    outline = app.read_setup("demo")["outline"]
    assert outline["limits"] == [
        "Sizes positions to aim for about 10% yearly volatility",
        "Never more than 100% of the account invested",
        "Only buys; never bets on prices falling",
        "No single position above 50% of the account",
        "Sells a position that falls 10% from its best price",
        "Goes fully to cash if the account falls 20% from its peak, then waits 30 bars"]
    kids = outline["strategy"]["children"]
    assert kids[0] == {"type": "bollinger", "summary": kids[0]["summary"], "params": {}, "weight": 2, "regime": "range"}
    assert outline["has_search"] and outline["has_walk_forward"] and not outline["live_feed"]
    assert outline["initial_cash"] == 20000 and "error" in app.read_setup("broken")


def test_setup_names_cannot_escape_the_folder(app):
    for bad in ("../demo", "a/b", "", ".hidden", "x" * 80, "demo.yaml"):
        with pytest.raises(ApiError) as err:
            app.read_setup(bad)
        assert err.value.status == 400
    with pytest.raises(ApiError) as err:
        app.read_setup("missing")
    assert err.value.status == 404


def test_saving_checks_before_writing(app):
    path = app.configs / "demo.yaml"
    before = path.read_text()
    for bad, reason in (("strategy: {type: nope}", "unknown strategy"), ("name: [unclosed", ""),
                        (yaml.safe_dump({**DEMO, "risk": {"stop_los": 1}}), "unknown option"),
                        ("just text", "mapping"), ("   ", "empty")):
        with pytest.raises(ApiError) as err:
            app.save_setup("demo", bad)
        assert err.value.status in (400, 422) and reason in err.value.message
        assert path.read_text() == before                                    # nothing was written
    saved = app.save_setup("demo", yaml.safe_dump({**DEMO, "name": "Renamed"}))
    assert saved["outline"]["name"] == "Renamed" and "Renamed" in path.read_text()
    with pytest.raises(ApiError) as err:
        app.save_setup("demo", before, create=True)
    assert err.value.status == 409
    assert app.save_setup("copy-1", before, create=True)["id"] == "copy-1"
    names = {s["name"] for s in app.strategies()}
    assert {"event_drift", "ma_cross", "ensemble"} <= names
    json.dumps(app.strategies())                                             # everything is JSON-safe


def test_to_json_strips_what_json_cannot_hold():
    out = to_json({"a": np.float64("nan"), "b": [np.int64(3), float("inf")], "c": pd.Timestamp("2024-01-02"),
                   "d": np.array([1.5, np.nan]), 5: np.bool_(True)})
    assert out == {"a": None, "b": [3, None], "c": "2024-01-02 00:00:00", "d": [1.5, None], "5": True}
    json.dumps(out)


# -- jobs ------------------------------------------------------------------------
def test_backtest_job_returns_numbers_and_a_report(app):
    job = finish(app, app.submit("backtest", "demo"))
    assert job.status == "done", job.error
    r = job.result
    assert r["bars"] == 400 and r["metrics"]["fills"] > 0 and ["Sharpe", f"{r['metrics']['sharpe']:.2f}"] in r["rows"]
    assert job.report.count("<svg") == 2 and "Demo setup" in job.report
    public = job.public()
    assert public["has_report"] and public["status"] == "done" and json.dumps(public)


def test_other_jobs(app):
    ranking = finish(app, app.submit("tournament", "demo")).result
    assert ranking["rows"][0]["sharpe"] >= ranking["rows"][-1]["sharpe"] and len(ranking["rows"]) >= 18
    assert any(r["strategy"] == "(benchmark)" for r in ranking["rows"])
    plan = finish(app, app.submit("plan", "demo", {"cash": 5000, "positions": {"AA": 10, "BB": 0}})).result
    price = synthetic_universe(["AA", "BB", "CC"], bars=400, seed=2, kind="range")["AA"]["close"].iloc[-1]
    assert plan["equity"] == pytest.approx(5000 + 10 * price, abs=0.01)
    assert all(o["side"] in ("buy", "sell") and o["est_value"] > 0 for o in plan["orders"])
    replay = finish(app, app.submit("rehearse", "demo", {"bars": 25})).result
    assert replay["bars"] == 25 and len(replay["curve"]) == 26 and replay["start_cash"] == 20000
    assert replay["equity"] == pytest.approx(replay["curve"][-1][1])
    forward = finish(app, app.submit("walkforward", "demo"), 120).result
    assert len(forward["folds"]["rows"]) == 3 and forward["folds"]["columns"][2] == "window"


def test_job_failures_are_reported_not_raised(app):
    job = finish(app, app.submit("events_study", "demo"))
    assert job.status == "error" and "events.path" in job.error
    job = finish(app, app.submit("walkforward", "zeta"))             # search path does not fit the strategy
    assert job.status == "error" and "unknown parameter" in job.error
    plain = {k: v for k, v in DEMO.items() if k not in ("optimize", "walk_forward")}
    app.save_setup("plain", yaml.safe_dump(plain), create=True)
    job = finish(app, app.submit("walkforward", "plain"))
    assert job.status == "error" and "optimize.space" in job.error
    job = finish(app, app.submit("plan", "demo", {"positions": {"ZZZ": 1}}))
    assert job.status == "error" and "no price data" in job.error
    for kind in ("nope", "__init__", "../x", ""):
        with pytest.raises(ApiError):
            app.submit(kind, "demo")
    with pytest.raises(ApiError) as err:
        app.submit("backtest", "broken")
    assert err.value.status == 422
    with pytest.raises(ApiError):
        app.job("unknown")


def test_tasks_describe_and_event_settings():
    cfg = {"strategy": {"type": "tsmom"}, "data": {"source": "yahoo", "symbols": ["SPY", "QQQ"]},
           "risk": {"max_drawdown_pct": 0.1}, "events": {"path": "e.csv", "benchmark": "SPY"}}
    outline = tasks.describe(cfg)
    assert outline["live_feed"] and outline["has_events"] and outline["prices"] == "Yahoo Finance"
    assert outline["limits"][-1] == ("Goes fully to cash if the account falls 10% from its peak, "
                                     "and stays out until you resume it")
    assert tasks.event_settings(cfg)["symbols"] == ["QQQ"]
    with pytest.raises(ValueError, match="events.path"):
        tasks.event_settings({**cfg, "events": {}})
    with pytest.raises(ValueError, match="unknown option"):
        tasks.event_settings({**cfg, "events": {"path": "e.csv", "sorces": []}})
    with pytest.raises(ValueError, match="data.symbols"):
        tasks.symbols_of({"data": {}})


# -- paper trading ---------------------------------------------------------------
def test_paper_session_trades_new_bars_and_reports_status(app, tmp_path):
    frames = synthetic_universe(["AA", "BB", "CC"], bars=400, seed=2, kind="range")
    provider = ReplayProvider(frames, start=380)
    assert app.paper_status("demo") == {"setup": "demo", "running": False}
    with pytest.raises(ApiError) as err:
        app.paper_start("demo")                                 # simulated prices: nothing live to follow
    assert err.value.status == 422 and "no live feed" in err.value.message

    session = PaperSession("demo", DEMO, tmp_path / "state", provider, poll_seconds=3600)
    session.check_once()
    first = session.status()
    assert not first["running"] and first["checks"] == 1 and first["error"] is None
    assert first["last_traded"] == str(provider.timeline[379]) and first["equity"] is not None
    provider.advance()
    session.check_once()
    session.check_once()                                        # no new bar: nothing more happens
    status = session.status()
    assert status["checks"] == 3 and len(status["curve"]) == 2
    assert status["last_bar"] == str(provider.timeline[380])
    assert status["equity"] == pytest.approx(status["cash"] + sum(
        q * frames[s]["close"].iloc[380] for s, q in status["positions"].items()), abs=0.01)
    assert (tmp_path / "state" / "app-demo.json").exists()

    class Down:
        def fetch(self, symbols, bars=None):
            raise ConnectionError("feed is down")

    broken = PaperSession("demo", DEMO, tmp_path / "s2", Down(), poll_seconds=3600)
    broken.check_once()                                         # a failing feed is reported, not fatal
    assert "feed is down" in broken.status()["error"]


def test_paper_start_and_stop_through_the_app(app):
    frames = synthetic_universe(["AA", "BB", "CC"], bars=400, seed=2, kind="range")
    started = app.paper_start("demo", ReplayProvider(frames, start=390), poll_seconds=0.01)
    assert started["setup"] == "demo"
    deadline = time.time() + 20
    while app.paper_status("demo")["checks"] < 2 and time.time() < deadline:
        time.sleep(0.02)
    status = app.paper_status("demo")
    assert status["running"] and status["checks"] >= 2 and status["last_traded"]
    assert app.paper_start("demo")["started"] == status["started"]          # already running: same session
    assert app.paper_stop("demo")["running"] is False
    json.dumps(status)


def test_bar_length_is_inferred():
    assert bar_seconds({"data": {"interval": "1d"}}) == 86400
    assert bar_seconds({"data": {"timeframe": "4h"}}) == 14400
    assert bar_seconds({"data": {"timeframe": "1Day"}}) == 86400
    assert bar_seconds({"data": {"interval": "1d"}, "live": {"bar_seconds": 60}}) == 60
    assert bar_seconds({"data": {"interval": "7m"}}) is None


# -- over HTTP -------------------------------------------------------------------
@pytest.fixture
def web(app):
    server = make_server(app, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def call(method, path, body=None, headers=None, raw=None):
        conn = HTTPConnection("127.0.0.1", port, timeout=30)
        hdrs = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        kind = resp.getheader("Content-Type", "")
        return resp.status, (json.loads(payload) if kind.startswith("application/json") else payload.decode())

    yield call
    server.shutdown()
    server.server_close()


def test_http_pages_and_api(web):
    http = web
    status, page = http("GET", "/")
    assert status == 200 and "<title>algobot</title>" in page and "/static/app.js" in page
    assert http("GET", "/static/app.js")[0] == 200 and http("GET", "/static/app.css")[0] == 200
    status, setups = http("GET", "/api/setups")
    assert status == 200 and setups[0]["id"] == "demo"
    assert http("GET", "/api/setups/demo")[1]["outline"]["name"] == "Demo setup"
    assert http("GET", "/api/strategies")[0] == 200

    status, job = http("POST", "/api/jobs", {"kind": "backtest", "setup": "demo"})
    assert status == 202 and job["status"] == "running"
    for _ in range(600):
        status, job = http("GET", f"/api/jobs/{job['id']}")
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["has_report"] and job["result"]["bars"] == 400
    status, report = http("GET", f"/api/jobs/{job['id']}/report")
    assert status == 200 and "<svg" in report

    status, made = http("POST", "/api/setups", {"id": "fresh"})                # starts from the template
    assert status == 201 and made["outline"]["source"] == "yahoo"
    assert http("POST", "/api/setups", {"id": "fresh"})[0] == 409
    status, saved = http("PUT", "/api/setups/fresh", {"text": yaml.safe_dump({**DEMO, "name": "Edited"})})
    assert status == 200 and saved["outline"]["name"] == "Edited"
    status, err = http("PUT", "/api/setups/fresh", {"text": "strategy: {type: nope}"})
    assert status == 422 and "unknown strategy" in err["error"]
    assert http("GET", "/api/paper/demo")[1] == {"setup": "demo", "running": False}
    assert http("POST", "/api/paper/demo/start", {})[0] == 422


def test_http_refuses_what_it_should(web):
    http = web
    assert http("GET", "/api/nope")[0] == 404 and http("GET", "/elsewhere")[0] == 404
    assert http("GET", "/api/setups/..%2Fdemo")[0] in (400, 404)
    for path in ("/static/../server.py", "/static/..%2f..%2fcli.py", "/static/app.py", "/static/missing.js"):
        assert http("GET", path)[0] == 404
    # Another site's page, or a DNS-rebound host name, must not be able to drive the app.
    assert http("GET", "/api/setups", headers={"Host": "evil.example"})[0] == 403
    assert http("POST", "/api/jobs", {"kind": "backtest", "setup": "demo"},
                headers={"Origin": "https://evil.example"})[0] == 403
    assert http("POST", "/api/jobs", raw=b"kind=backtest&setup=demo",
                headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert http("POST", "/api/jobs", raw=b"{not json", headers={"Content-Type": "application/json"})[0] == 400
    assert http("POST", "/api/jobs", raw=b"[1]", headers={"Content-Type": "application/json"})[0] == 400
    assert http("POST", "/api/jobs", {"kind": "rm -rf", "setup": "demo"})[0] == 400
    assert http("GET", "/api/jobs/nothing/report")[0] == 404
    assert http("DELETE", "/api/setups/demo")[0] == 501            # no delete endpoint at all


def test_host_check_can_be_turned_off_for_a_trusted_proxy(app):
    server = make_server(app, "127.0.0.1", 0, check_host=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/api/setups", headers={"Host": "preview.example"})
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
