import json
from pathlib import Path

import pytest
import yaml

from algobot.backtest.report import render_report
from algobot.cli import main
from algobot.config import build_pipeline, load_config, run_backtest
from algobot.data import _check

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
BASE = {
    "name": "cli test",
    "data": {"source": "synthetic", "symbols": ["AA", "BB", "CC"], "bars": 500, "seed": 2,
             "kind": "range"},
    "strategy": {"type": "bollinger"},
    "risk": {"max_drawdown_pct": 0.3},
    "optimize": {"metric": "sharpe", "space": {"strategy.params.window": [10, 20]}},
    "walk_forward": {"train_bars": 200, "test_bars": 100},
    "live": {"cash": 50000},
}


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "bot.yaml"
    path.write_text(yaml.safe_dump(BASE))
    return str(path)


def write(tmp_path, **changes):
    path = tmp_path / "variant.json"
    path.write_text(json.dumps({**BASE, **changes}))
    return str(path)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    build_pipeline(cfg)
    _check(cfg["data"])
    assert cfg["data"]["symbols"]


def test_shipped_demo_config_runs():
    assert len(CONFIGS) >= 5
    res = run_backtest(load_config(next(p for p in CONFIGS if p.name == "demo.yaml")))
    assert len(res.equity) == 2520 and res.metrics["trades"] > 100


def test_strategies_command(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "ma_cross" in out and "entry_z = 2.0" in out and "ensemble" in out


def test_backtest_text_json_and_report(cfg_path, tmp_path, capsys):
    assert main(["backtest", "-c", cfg_path, "--bootstrap"]) == 0
    text = capsys.readouterr().out
    assert "Sharpe" in text and "benchmark" in text and "Bootstrap" in text
    report = tmp_path / "out" / "report.html"
    assert main(["backtest", "-c", cfg_path, "--json", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["bars"] == 500 and "sharpe" in payload["benchmark"]
    html = report.read_text()
    assert html.count("<svg") == 2 and "cli test" in html and "NaN" not in html and "nan%" not in html


def test_report_yearly_returns_compound_to_the_total(cfg_path):
    import re
    res = run_backtest(load_config(cfg_path))
    html = render_report(res)
    block = html.split("Calendar-year returns")[1].split("</table>")[0]
    cells = re.findall(r"<td>(\d{4})</td><td>([+-][\d.]+)%</td>", block)
    assert len(cells) >= 2
    total = 1.0
    for _, pct in cells:
        total *= 1 + float(pct) / 100
    assert total - 1 == pytest.approx(res.metrics["total_return"], abs=2e-3)


def test_report_escapes_and_handles_events(cfg_path):
    cfg = load_config(cfg_path)
    cfg["risk"] = {"trailing_stop_pct": 0.02}
    res = run_backtest(cfg)
    assert res.risk_events
    html = render_report(res, title="<script>alert(1)</script>")
    assert "<script>alert(1)" not in html and "&lt;script&gt;" in html
    assert "Risk events" in html and "trailing stop" in html


def test_optimize_and_walkforward_commands(cfg_path, capsys):
    assert main(["optimize", "-c", cfg_path, "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "strategy.params.window" in out and "2 candidates" in out
    assert main(["walkforward", "-c", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "3 folds, out-of-sample only" in out and "test_sharpe" in out


def test_tournament_command(cfg_path, capsys):
    assert main(["tournament", "-c", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "(benchmark)" in out and "bollinger" in out and "ml_logistic" in out
    assert out.count("\n") > 17


def test_plan_command(cfg_path, tmp_path, capsys):
    assert main(["plan", "-c", cfg_path, "--json", "--cash", "10000"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["equity"] == 10000.0 and all(o["side"] in ("buy", "sell") for o in empty["orders"])
    # Holding something the strategy does not want produces an exit order.
    held = tmp_path / "pos.json"
    held.write_text(json.dumps({"cash": 5000, "positions": {"AA": 7, "BB": {"qty": -3, "avg_price": 90}}}))
    assert main(["plan", "-c", cfg_path, "--positions", str(held), "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    by_symbol = {o["symbol"]: o for o in plan["orders"]}
    for sym, qty in (("AA", 7), ("BB", -3)):
        target = plan["targets"].get(sym, 0.0)
        if target == 0.0:
            assert by_symbol[sym]["qty"] == abs(qty) and by_symbol[sym]["reason"] == "exit"
            assert by_symbol[sym]["side"] == ("sell" if qty > 0 else "buy")
    assert main(["plan", "-c", cfg_path]) == 0
    assert "target weights" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"positions": {"ZZZ": 1}}))
    with pytest.raises(SystemExit, match="no price data"):
        main(["plan", "-c", cfg_path, "--positions", str(bad)])


def test_run_replay(cfg_path, capsys):
    assert main(["run", "-c", cfg_path, "--replay", "60"]) == 0
    assert "replayed 60 bars on the paper broker" in capsys.readouterr().out


def test_run_refuses_unsafe_or_impossible_setups(cfg_path, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no live feed"):
        main(["run", "-c", cfg_path, "--once"])
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    real = write(tmp_path, data={"source": "alpaca", "symbols": ["SPY"]},
                 live={"broker": "alpaca", "paper": False})
    with pytest.raises(SystemExit, match="REAL-MONEY"):
        main(["run", "-c", real, "--once"])
    monkeypatch.delenv("ALPACA_API_KEY")
    with pytest.raises(SystemExit, match="ALPACA_API_KEY"):
        main(["run", "-c", real, "--once", "--live"])
    typo = write(tmp_path, live={"brokr": "paper"})
    with pytest.raises(SystemExit, match="unknown option"):
        main(["run", "-c", typo, "--replay", "5"])


def test_config_errors_are_reported(tmp_path, capsys):
    assert main(["backtest", "-c", write(tmp_path, risks={})]) == 2
    assert "unknown config section" in capsys.readouterr().err
    assert main(["backtest", "-c", write(tmp_path, risk={"stop_loss": 0.1})]) == 2
    assert "unknown option" in capsys.readouterr().err
    assert main(["backtest", "-c", write(tmp_path, strategy={"type": "moon"})]) == 2
    assert "unknown strategy" in capsys.readouterr().err
    assert main(["backtest", "-c", str(tmp_path / "missing.yaml")]) == 2
    no_space = write(tmp_path, optimize={})
    with pytest.raises(SystemExit, match="optimize.space"):
        main(["optimize", "-c", no_space])
