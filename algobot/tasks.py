"""High-level operations shared by the command line and the app.

Each function takes a parsed config and returns plain data (dicts, lists,
numbers) that either front end can print or send as JSON.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .backtest.metrics import compute_metrics
from .config import build_costs, build_pipeline, load_dataset, run_backtest
from .data import Dataset, infer_periods_per_year, load_data
from .data.event_sources import make_event_source
from .events import EventStore
from .eventstudy import EventStudy, run_event_study
from .execution.brokers import PaperBroker
from .history import BarHistory
from .live import LiveRunner, ReplayProvider, feed
from .strategies import available
from .types import Position

EVENT_KEYS = {"path", "sources", "start", "benchmark", "kind", "pre", "post", "threshold", "symbols"}
_RANKED = ("total_return", "cagr", "sharpe", "max_drawdown", "trades", "fills")


def symbols_of(cfg: Mapping[str, Any]) -> list[str]:
    symbols = (cfg.get("data") or {}).get("symbols")
    if not symbols:
        raise ValueError("this needs data.symbols in the config")
    return list(symbols)


# -- describing a setup in plain language ------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:g}%"


def _strategy_outline(spec: Mapping[str, Any]) -> dict[str, Any]:
    kind = spec.get("type")
    cls = available().get(kind)
    params = dict(spec.get("params") or {})
    children = params.pop("children", None)
    out: dict[str, Any] = {"type": kind, "summary": cls.summary if cls else "unknown strategy",
                           "params": params}
    if "weight" in spec:
        out["weight"] = spec["weight"]
    if spec.get("regime", "any") != "any":
        out["regime"] = spec["regime"]
    if children:
        out["children"] = [_strategy_outline(child) for child in children]
    return out


def describe(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What a setup trades, how, and within which limits, in plain words."""
    data = dict(cfg.get("data") or {})
    source = data.get("source", "synthetic")
    where = {"synthetic": "built-in simulated markets (for trying things out, not real prices)",
             "csv": f"your own CSV files in {data.get('path', '?')}",
             "yahoo": "Yahoo Finance", "alpaca": "Alpaca market data",
             "ccxt": f"the {data.get('exchange', 'binance')} crypto exchange"}[source]
    sizing, risk, costs = (dict(cfg.get(k) or {}) for k in ("sizing", "risk", "costs"))
    limits = []
    if sizing.get("target_vol") is not None:
        limits.append(f"Sizes positions to aim for about {_pct(sizing['target_vol'])} yearly volatility")
    limits.append(f"Never more than {_pct(min(sizing.get('max_leverage', 1.0), risk.get('max_gross_leverage', 1.0)))} "
                  f"of the account invested")
    if risk.get("long_only"):
        limits.append("Only buys; never bets on prices falling")
    if risk.get("max_position_weight", 1.0) < 1.0:
        limits.append(f"No single position above {_pct(risk['max_position_weight'])} of the account")
    if risk.get("stop_loss_pct") is not None:
        limits.append(f"Sells a position that loses {_pct(risk['stop_loss_pct'])}")
    if risk.get("take_profit_pct") is not None:
        limits.append(f"Takes profit on a position that gains {_pct(risk['take_profit_pct'])}")
    if risk.get("trailing_stop_pct") is not None:
        limits.append(f"Sells a position that falls {_pct(risk['trailing_stop_pct'])} from its best price")
    if risk.get("max_drawdown_pct") is not None:
        pause = risk.get("drawdown_halt_bars")
        limits.append(f"Goes fully to cash if the account falls {_pct(risk['max_drawdown_pct'])} from its peak, "
                      + (f"then waits {pause} bars" if pause else "and stays out until you resume it"))
    if risk.get("daily_loss_limit_pct") is not None:
        limits.append(f"Stops for the day after losing {_pct(risk['daily_loss_limit_pct'])}")
    return {
        "name": cfg.get("name", ""),
        "source": source,
        "prices": where,
        "symbols": list(data.get("symbols") or []),
        "strategy": _strategy_outline(cfg["strategy"]),
        "limits": limits,
        "costs": f"Assumes {costs.get('commission_bps', 1.0):g} bp commission and "
                 f"{costs.get('slippage_bps', 2.0):g} bp slippage on every trade",
        "live_feed": source in ("yahoo", "alpaca", "ccxt"),
        "has_events": bool((cfg.get("events") or {}).get("path")),
        "has_search": bool((cfg.get("optimize") or {}).get("space")),
        "has_walk_forward": bool(cfg.get("walk_forward")),
        "initial_cash": float((cfg.get("backtest") or {}).get("initial_cash", 100_000.0)),
    }


# -- comparing strategies ------------------------------------------------------------
def tournament(cfg: Mapping[str, Any], dataset: Dataset | None = None) -> dict[str, Any]:
    """Every built-in strategy with default settings on this config's data,
    costs, sizing and risk limits, best risk-adjusted return first."""
    dataset = dataset if dataset is not None else load_dataset(cfg)
    rows, skipped = [], []
    for name, cls in available().items():
        if name == "ensemble":
            continue
        try:
            m = run_backtest({**cfg, "strategy": {"type": name}}, dataset).metrics
        except ValueError as exc:
            skipped.append({"strategy": name, "reason": str(exc)})
            continue
        rows.append({"strategy": name, "summary": cls.summary,
                     **{k: m.get(k, float("nan")) for k in _RANKED}})
    bm = compute_metrics(dataset.benchmark(), periods_per_year=dataset.periods_per_year)
    rows.append({"strategy": "(benchmark)", "summary": "Hold every symbol equally, the yardstick",
                 **{k: bm.get(k, float("nan")) for k in _RANKED[:4]}, "trades": 0, "fills": 0})
    rows.sort(key=lambda r: r["sharpe"] if not math.isnan(r["sharpe"]) else -math.inf, reverse=True)
    return {"rows": rows, "skipped": skipped, "bars": len(dataset),
            "start": str(dataset.timeline[0].date()), "end": str(dataset.timeline[-1].date())}


# -- what would it do now ------------------------------------------------------------
def make_plan(cfg: Mapping[str, Any], cash: float | None = None,
              positions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The orders that would move the given holdings to the strategy's current
    target. Sends nothing anywhere."""
    frames = Dataset(load_data(cfg.get("data") or {})).frames
    pipeline = build_pipeline(cfg)
    if pipeline.sizer.config.periods_per_year is None:
        longest = max(frames.values(), key=len)
        pipeline.sizer.config.periods_per_year = infer_periods_per_year(longest.index)
    history = BarHistory(frames.keys())
    feed(pipeline, history, frames, hold_last=True)
    prices = history.prices()
    held: dict[str, Position] = {}
    for sym, value in (positions or {}).items():
        if isinstance(value, Mapping):
            held[sym] = Position(float(value["qty"]), float(value.get("avg_price", 0.0)))
        elif float(value) != 0.0:
            held[sym] = Position(float(value), 0.0)
    missing = [s for s in held if s not in prices]
    if missing:
        raise ValueError(f"no price data for held symbol(s) {missing}; add them to data.symbols")
    if cash is None:
        cash = float((cfg.get("backtest") or {}).get("initial_cash", 100_000.0))
    equity = float(cash) + sum(p.qty * prices[s] for s, p in held.items())
    decision = pipeline.step(history, equity=equity, positions=held, prices=prices)
    stamp = pd.Timestamp(history.now)
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    return {
        "as_of": str(history.now), "equity": round(equity, 2), "cash": round(float(cash), 2),
        # A daily bar dated today may still be forming: prices and targets can move before the close.
        "bar_may_be_forming": bool(stamp == stamp.normalize() and stamp >= today),
        "targets": {s: round(w, 4) for s, w in decision.targets.items() if abs(w) > 1e-9},
        "orders": [{"symbol": o.symbol, "side": "buy" if o.qty > 0 else "sell",
                    "qty": round(abs(o.qty), 6), "est_value": round(abs(o.qty) * prices[o.symbol], 2),
                    "ref_price": round(prices[o.symbol], 4), "reason": o.reason}
                   for o in decision.orders],
    }


# -- rehearsing the live loop on recent history -----------------------------------------
def rehearse(cfg: Mapping[str, Any], bars: int = 60) -> dict[str, Any]:
    """Run the live trading loop over the last ``bars`` bars with a paper account."""
    live = dict(cfg.get("live") or {})
    frames = Dataset(load_data(cfg["data"])).frames
    total = len(Dataset(frames, clean=False).timeline)
    bars = max(1, min(int(bars), total - 1))
    provider = ReplayProvider(frames, start=total - bars)
    cash = float(live.get("cash", (cfg.get("backtest") or {}).get("initial_cash", 100_000.0)))
    broker = PaperBroker(cash, build_costs(cfg))
    runner = LiveRunner(build_pipeline(cfg), provider, broker, symbols_of(cfg),
                        history_bars=int(live.get("history_bars", 500)))
    curve = []

    def tick() -> None:
        if runner.tick() is not None:
            curve.append([str(runner.history.now), round(broker.account().equity, 2)])

    tick()
    while provider.advance():
        tick()
    account = broker.account()
    return {
        "bars": bars, "start_cash": cash, "equity": round(account.equity, 2),
        "cash": round(account.cash, 2), "curve": curve,
        "positions": {s: round(p.qty, 6) for s, p in broker.positions().items()},
        "orders": [{"time": str(f.ts), "symbol": f.symbol, "side": "buy" if f.qty > 0 else "sell",
                    "qty": round(abs(f.qty), 6), "price": round(f.price, 4), "reason": f.reason}
                   for f in broker.portfolio.fills],
    }


# -- events ----------------------------------------------------------------------------
def event_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ev = dict(cfg.get("events") or {})
    unknown = set(ev) - EVENT_KEYS
    if unknown:
        raise ValueError(f"events: unknown option(s) {sorted(unknown)}; valid: {sorted(EVENT_KEYS)}")
    if not ev.get("path"):
        raise ValueError("config needs events.path, the CSV file where events are kept")
    if not ev.get("symbols"):
        ev["symbols"] = [s for s in symbols_of(cfg) if s != ev.get("benchmark")]
    return ev


def collect_events(cfg: Mapping[str, Any], sources: Sequence[str] | None = None) -> dict[str, Any]:
    """Gather events from the configured sources into the events file."""
    ev = event_settings(cfg)
    path = Path(ev["path"])
    store = EventStore.from_csv(path) if path.exists() else EventStore()
    before = len(store)
    fetched = []
    for name in sources or ev.get("sources") or ["sec"]:
        fresh = make_event_source(name).fetch(ev["symbols"], start=ev.get("start"))
        store = store.merge(fresh)
        fetched.append({"source": name, "events": len(fresh)})
    store.to_csv(path)
    return {"path": str(path), "total": len(store), "new": len(store) - before,
            "symbols": len(ev["symbols"]), "fetched": fetched, "counts": store.counts()}


def study_events(cfg: Mapping[str, Any], kind: str | None = None) -> tuple[EventStudy, dict[str, Any]]:
    """Run the event study for a config. Returns the study and its settings."""
    ev = event_settings(cfg)
    if not Path(ev["path"]).exists():
        raise ValueError(f"no events collected yet at {ev['path']}; collect them first")
    study = run_event_study(EventStore.from_csv(ev["path"]), load_dataset(cfg),
                            benchmark=ev.get("benchmark"), pre=int(ev.get("pre", 10)),
                            post=int(ev.get("post", 60)))
    if not len(study):
        raise ValueError("no events fall inside the price data")
    return study, {"kind": kind or ev.get("kind", "earnings"),
                   "threshold": float(ev.get("threshold", 0.02))}
