"""Command line interface: ``python -m algobot <command> ...``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .backtest.metrics import bootstrap_sharpe, format_value
from .backtest.optimize import optimize, walk_forward
from .backtest.report import write_report
from .config import (_section, build_costs, build_pipeline, load_config, load_dataset,
                     run_backtest)
from .data import Dataset, infer_periods_per_year, load_data, make_provider
from .data.event_sources import make_event_source
from .eventreport import write_event_report
from .events import EventStore
from .eventstudy import format_summary, run_event_study
from .execution.brokers import AlpacaBroker, Broker, CCXTBroker, PaperBroker
from .history import BarHistory
from .live import LiveRunner, ReplayProvider, feed
from .strategies import available
from .types import Position

_LIVE_KEYS = {"broker", "cash", "paper", "exchange", "history_bars", "refresh_bars",
              "bar_seconds", "poll_seconds", "state_path", "max_order_value",
              "time_in_force"}


def _symbols(cfg: Mapping[str, Any]) -> list[str]:
    symbols = (cfg.get("data") or {}).get("symbols")
    if not symbols:
        raise SystemExit("this command needs data.symbols in the config")
    return list(symbols)


# -- commands ------------------------------------------------------------------
def cmd_strategies(args) -> int:
    for name, cls in available().items():
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("PARAMS", {}))
        print(f"{name}\n    {cls.summary}")
        for key, value in merged.items():
            if key != "children":
                print(f"      {key} = {value!r}")
    return 0


def cmd_backtest(args) -> int:
    cfg = load_config(args.config)
    result = run_backtest(cfg)
    if args.json:
        print(json.dumps({"metrics": result.metrics, "benchmark": result.benchmark_metrics,
                          "final_positions": result.final_positions}, indent=2, default=float))
    else:
        print(f"{cfg.get('name', args.config)}")
        print(result.summary())
        if args.bootstrap:
            boot = bootstrap_sharpe(result.equity, periods_per_year=result.periods_per_year)
            if boot:
                print(f"\n  Bootstrap (1,000 resamples): Sharpe 5%..95% = "
                      f"{boot['sharpe_p05']:.2f}..{boot['sharpe_p95']:.2f}, "
                      f"chance of a losing history {boot['prob_loss'] * 100:.0f}%")
        if result.risk_events:
            print(f"\n  {len(result.risk_events)} risk event(s); first: {result.risk_events[0]}")
    if args.report:
        path = write_report(result, args.report, str(cfg.get("name", "Backtest report")))
        print(f"\nreport written to {path}", file=sys.stderr)
    return 0


def _search_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    opt = dict(cfg.get("optimize") or {})
    if not opt.get("space"):
        raise SystemExit("config needs an optimize.space section, e.g.\n"
                         "  optimize:\n    space:\n      strategy.params.fast: [10, 20, 30]")
    unknown = set(opt) - {"space", "metric", "method", "n_iter", "seed", "min_trades"}
    if unknown:
        raise SystemExit(f"optimize: unknown option(s) {sorted(unknown)}")
    return opt


def cmd_optimize(args) -> int:
    cfg = load_config(args.config)
    opt = _search_settings(cfg)
    space = opt.pop("space")
    table = optimize(cfg, load_dataset(cfg), space, n_jobs=args.jobs, **opt)
    shown = table.head(args.top).drop(columns=["score"])
    if "trades" in shown:
        shown = shown.astype({"trades": "Int64"})
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(table)} candidates. These numbers are fitted to this data; "
          f"run 'walkforward' for an out-of-sample estimate.")
    return 0


def cmd_walkforward(args) -> int:
    cfg = load_config(args.config)
    opt = _search_settings(cfg)
    space = opt.pop("space")
    wf = dict(cfg.get("walk_forward") or {})
    unknown = set(wf) - {"train_bars", "test_bars", "warmup_bars"}
    if unknown or "train_bars" not in wf or "test_bars" not in wf:
        raise SystemExit("walk_forward needs train_bars and test_bars (optional warmup_bars)")
    result = walk_forward(cfg, load_dataset(cfg), space, n_jobs=args.jobs, **opt, **wf)
    print(result.summary())
    return 0


def cmd_tournament(args) -> int:
    cfg = load_config(args.config)
    dataset = load_dataset(cfg)
    rows = []
    for name in available():
        if name == "ensemble":
            continue
        trial = {**cfg, "strategy": {"type": name}}
        try:
            m = run_backtest(trial, dataset).metrics
        except ValueError as exc:
            print(f"  skipped {name}: {exc}", file=sys.stderr)
            continue
        rows.append({"strategy": name, **{k: m.get(k, float("nan")) for k in
                     ("total_return", "cagr", "sharpe", "max_drawdown", "trades")}})
    bench = dataset.benchmark()
    from .backtest.metrics import compute_metrics
    bm = compute_metrics(bench, periods_per_year=dataset.periods_per_year)
    rows.append({"strategy": "(benchmark)", **{k: bm.get(k, float("nan")) for k in
                 ("total_return", "cagr", "sharpe", "max_drawdown")}, "trades": 0})
    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    print(f"  {'strategy':<18}{'return':>10}{'CAGR':>9}{'Sharpe':>8}{'max DD':>9}{'trades':>8}")
    for r in rows:
        print(f"  {r['strategy']:<18}{format_value(r['total_return'], 'pct'):>10}"
              f"{format_value(r['cagr'], 'pct'):>9}{r['sharpe']:>8.2f}"
              f"{format_value(r['max_drawdown'], 'pct'):>9}{int(r['trades']):>8}")
    print("\nDefault parameters, same data, costs, sizing and risk settings for all.")
    return 0


def _read_positions(path: str | None, default_cash: float) -> tuple[float, dict[str, Position]]:
    if not path:
        return default_cash, {}
    with open(path) as fh:
        raw = json.load(fh)
    positions = {}
    for sym, value in (raw.get("positions") or {}).items():
        if isinstance(value, Mapping):
            positions[sym] = Position(float(value["qty"]), float(value.get("avg_price", 0.0)))
        else:
            positions[sym] = Position(float(value), 0.0)
    return float(raw.get("cash", default_cash)), positions


def cmd_plan(args) -> int:
    """What the bot would do right now, without sending anything anywhere."""
    cfg = load_config(args.config)
    frames = Dataset(load_data(cfg.get("data") or {})).frames
    pipeline = build_pipeline(cfg)
    if pipeline.sizer.config.periods_per_year is None:
        longest = max(frames.values(), key=len)
        pipeline.sizer.config.periods_per_year = infer_periods_per_year(longest.index)
    history = BarHistory(frames.keys())
    feed(pipeline, history, frames, hold_last=True)
    prices = history.prices()
    default_cash = float((cfg.get("backtest") or {}).get("initial_cash", 100_000.0))
    cash, positions = _read_positions(args.positions, default_cash)
    if args.cash is not None:
        cash = args.cash
    missing = [s for s in positions if s not in prices]
    if missing:
        raise SystemExit(f"no price data for held symbol(s) {missing}; add them to data.symbols")
    equity = cash + sum(p.qty * prices[s] for s, p in positions.items())
    decision = pipeline.step(history, equity=equity, positions=positions, prices=prices)
    plan = {
        "as_of": str(history.now), "equity": round(equity, 2),
        "targets": {s: round(w, 4) for s, w in decision.targets.items() if abs(w) > 1e-9},
        "orders": [{"symbol": o.symbol, "side": "buy" if o.qty > 0 else "sell",
                    "qty": round(abs(o.qty), 6), "est_value": round(abs(o.qty) * prices[o.symbol], 2),
                    "ref_price": round(prices[o.symbol], 4), "reason": o.reason}
                   for o in decision.orders],
    }
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    print(f"as of {plan['as_of']}  equity {equity:,.2f}")
    print("target weights:", plan["targets"] or "all cash")
    if not plan["orders"]:
        print("no orders: the book is already where the strategy wants it")
    for o in plan["orders"]:
        print(f"  {o['side'].upper():<4} {o['qty']:>14,.6f} {o['symbol']:<10} "
              f"~{o['est_value']:>12,.2f} @ {o['ref_price']:,.4f}  ({o['reason']})")
    return 0


_EVENT_KEYS = {"path", "sources", "start", "benchmark", "kind", "pre", "post", "threshold",
               "symbols"}


def _event_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ev = dict(cfg.get("events") or {})
    unknown = set(ev) - _EVENT_KEYS
    if unknown:
        raise SystemExit(f"events: unknown option(s) {sorted(unknown)}; valid: {sorted(_EVENT_KEYS)}")
    if not ev.get("path"):
        raise SystemExit("config needs events.path, the CSV file where events are kept")
    if not ev.get("symbols"):
        ev["symbols"] = [s for s in _symbols(cfg) if s != ev.get("benchmark")]
    return ev


def cmd_events_fetch(args) -> int:
    """Collect events from the configured sources into the events file."""
    cfg = load_config(args.config)
    ev = _event_settings(cfg)
    sources = args.source or ev.get("sources") or ["sec"]
    path = Path(ev["path"])
    store = EventStore.from_csv(path) if path.exists() else EventStore()
    before = len(store)
    for name in sources:
        fresh = make_event_source(name).fetch(ev["symbols"], start=ev.get("start"))
        store = store.merge(fresh)
        print(f"{name}: {len(fresh):,} events for {len(ev['symbols'])} symbols")
    store.to_csv(path)
    print(f"{path}: {len(store):,} events ({len(store) - before:+,} new)")
    for kind, count in store.counts().items():
        print(f"  {kind:<26}{count:>7,}")
    return 0


def cmd_events_study(args) -> int:
    """Measure how prices moved around the collected events."""
    cfg = load_config(args.config)
    ev = _event_settings(cfg)
    store = EventStore.from_csv(ev["path"])
    dataset = load_dataset(cfg)
    study = run_event_study(store, dataset, benchmark=ev.get("benchmark"),
                            pre=int(ev.get("pre", 10)), post=int(ev.get("post", 60)))
    if not len(study):
        raise SystemExit("no events fall inside the price data")
    kind = args.kind or ev.get("kind", "earnings")
    threshold = float(ev.get("threshold", 0.02))
    kinds = study.by_kind()
    print(f"{len(study):,} events, {study.events['symbol'].nunique()} symbols, "
          f"{study.events['session'].min().date()} to {study.events['session'].max().date()}; "
          f"market = {study.benchmark}; ordinary day move {study.normal_abs_move * 100:.2f}%")
    print("\nBy event type (average return versus the market, t-statistic in brackets):")
    order = kinds.value_counts().index
    print(format_summary(study.summary(pd.Series(pd.Categorical(kinds, order, ordered=True)))))
    if (kinds == kind).any():
        focus = study.subset((kinds == kind).to_numpy())
        print(f"\n{kind}, split by the move on the event day:")
        print(format_summary(focus.summary(focus.by_reaction(threshold))))
        valued = focus.subset(~focus.events["value"].isna().to_numpy())
        if len(valued) >= 20:
            print(f"\n{kind}, split by the reported surprise:")
            print(format_summary(valued.summary(valued.by_value())))
    if args.report:
        out = write_event_report(study, args.report, str(cfg.get("name", "Event study")),
                                 kind, threshold)
        print(f"\nreport written to {out}", file=sys.stderr)
    return 0


def _build_broker(live: Mapping[str, Any], symbols: list[str], cfg: Mapping[str, Any]) -> Broker:
    kind = live.get("broker", "paper")
    paper = bool(live.get("paper", True))
    if kind == "paper":
        return PaperBroker(float(live.get("cash", 100_000.0)), build_costs(cfg))
    if kind == "alpaca":
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise SystemExit("set ALPACA_API_KEY and ALPACA_API_SECRET in the environment")
        return AlpacaBroker(key, secret, paper=paper, symbols=symbols,
                            time_in_force=live.get("time_in_force", "day"))
    if kind == "ccxt":
        key, secret = os.environ.get("CCXT_API_KEY"), os.environ.get("CCXT_API_SECRET")
        if not key or not secret:
            raise SystemExit("set CCXT_API_KEY and CCXT_API_SECRET in the environment")
        return CCXTBroker.connect(live.get("exchange", "binance"), symbols, key, secret,
                                  sandbox=paper)
    raise SystemExit(f"live.broker must be paper, alpaca or ccxt, got {kind!r}")


def cmd_run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    live = dict(cfg.get("live") or {})
    unknown = set(live) - _LIVE_KEYS
    if unknown:
        raise SystemExit(f"live: unknown option(s) {sorted(unknown)}; valid: {sorted(_LIVE_KEYS)}")
    symbols = _symbols(cfg)
    pipeline = build_pipeline(cfg)
    common = dict(history_bars=int(live.get("history_bars", 500)),
                  refresh_bars=int(live.get("refresh_bars", 30)),
                  dry_run=args.dry_run, max_order_value=live.get("max_order_value"))

    if args.replay:
        frames = Dataset(load_data(cfg["data"])).frames
        total = len(Dataset(frames, clean=False).timeline)
        provider = ReplayProvider(frames, start=max(1, total - args.replay))
        broker = PaperBroker(float(live.get("cash", 100_000.0)), build_costs(cfg))
        runner = LiveRunner(pipeline, provider, broker, symbols, **common)
        runner.tick()
        while provider.advance():
            runner.tick()
        acct = broker.account()
        print(f"replayed {args.replay} bars on the paper broker: equity {acct.equity:,.2f}, "
              f"cash {acct.cash:,.2f}, positions "
              f"{ {s: round(p.qty, 4) for s, p in broker.positions().items()} }")
        return 0

    source = (cfg.get("data") or {}).get("source", "synthetic")
    if source in ("synthetic", "csv"):
        raise SystemExit(f"data.source {source!r} has no live feed; use yahoo, alpaca or ccxt, "
                         f"or add --replay N to rehearse on historical bars")
    broker = _build_broker(live, symbols, cfg)
    if broker.real_money and not args.live:
        raise SystemExit("this config points at a REAL-MONEY account. Re-run with --live if "
                         "that is intended, or set live.paper: true.")
    if broker.real_money:
        logging.warning("LIVE TRADING with real money on %s", broker.name)
    runner = LiveRunner(pipeline, make_provider(cfg["data"]), broker, symbols,
                        bar_seconds=live.get("bar_seconds"),
                        poll_seconds=float(live.get("poll_seconds", 300)),
                        state_path=live.get("state_path"), reset_halt=args.reset_halt,
                        **common)
    if args.once:
        runner.tick()
    else:
        runner.run(max_ticks=args.max_ticks)
    return 0


# -- entry point -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="algobot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("strategies", help="list built-in strategies and their parameters"
                   ).set_defaults(func=cmd_strategies)

    p = sub.add_parser("backtest", help="simulate a config on historical data")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--report", metavar="PATH", help="also write an HTML report")
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    p.add_argument("--bootstrap", action="store_true", help="add a resampling robustness check")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("optimize", help="search the parameters in optimize.space")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("walkforward", help="out-of-sample test: refit, trade forward, repeat")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("tournament", help="rank every built-in strategy on the config's data")
    p.add_argument("-c", "--config", required=True)
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("plan", help="print the orders the bot would place now (sends nothing)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--positions", metavar="JSON", help='{"cash": 1000, "positions": {"SPY": 3}}')
    p.add_argument("--cash", type=float)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("events", help="collect earnings/filing/news events and study their effect")
    actions = p.add_subparsers(dest="action", required=True)
    a = actions.add_parser("fetch", help="collect events into the file named by events.path")
    a.add_argument("-c", "--config", required=True)
    a.add_argument("--source", action="append", choices=["sec", "yahoo_news", "alpaca_news"],
                   help="override events.sources (repeatable)")
    a.set_defaults(func=cmd_events_fetch)
    a = actions.add_parser("study", help="measure how prices moved around the events")
    a.add_argument("-c", "--config", required=True)
    a.add_argument("--kind", help="event type to focus on (default: events.kind or earnings)")
    a.add_argument("--report", metavar="PATH", help="also write an HTML report")
    a.set_defaults(func=cmd_events_study)

    p = sub.add_parser("run", help="trade: paper by default, real money only with --live")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--once", action="store_true", help="process new bars once and exit (for cron)")
    p.add_argument("--dry-run", action="store_true", help="log orders instead of sending them")
    p.add_argument("--live", action="store_true", help="allow a real-money broker")
    p.add_argument("--replay", type=int, metavar="N",
                   help="rehearse on the last N historical bars with the paper broker")
    p.add_argument("--max-ticks", type=int)
    p.add_argument("--reset-halt", action="store_true",
                   help="resume after the max-drawdown breaker has tripped")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # output piped into something like `head`
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
