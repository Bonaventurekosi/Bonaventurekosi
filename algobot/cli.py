"""Command line interface: ``python -m algobot <command> ...``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Mapping

import pandas as pd

from . import tasks
from .backtest.metrics import bootstrap_sharpe, format_value
from .backtest.optimize import optimize, walk_forward
from .backtest.report import write_report
from .config import (build_costs, build_pipeline, load_config, load_dataset,
                     run_backtest)
from .data import make_provider
from .eventreport import write_event_report
from .eventstudy import format_summary
from .execution.brokers import AlpacaBroker, Broker, CCXTBroker, PaperBroker
from .live import LiveRunner
from .strategies import available

_LIVE_KEYS = {"broker", "cash", "paper", "exchange", "history_bars", "refresh_bars",
              "bar_seconds", "poll_seconds", "state_path", "max_order_value",
              "time_in_force"}


def _symbols(cfg: Mapping[str, Any]) -> list[str]:
    try:
        return tasks.symbols_of(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc))


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
    result = tasks.tournament(load_config(args.config))
    for skip in result["skipped"]:
        print(f"  skipped {skip['strategy']}: {skip['reason']}", file=sys.stderr)
    print(f"  {'strategy':<18}{'return':>10}{'CAGR':>9}{'Sharpe':>8}{'max DD':>9}{'trades':>8}")
    for r in result["rows"]:
        print(f"  {r['strategy']:<18}{format_value(r['total_return'], 'pct'):>10}"
              f"{format_value(r['cagr'], 'pct'):>9}{r['sharpe']:>8.2f}"
              f"{format_value(r['max_drawdown'], 'pct'):>9}{int(r['trades']):>8}")
    print("\nDefault parameters, same data, costs, sizing and risk settings for all.")
    return 0


def cmd_plan(args) -> int:
    """What the bot would do right now, without sending anything anywhere."""
    cfg = load_config(args.config)
    cash, positions = None, {}
    if args.positions:
        with open(args.positions) as fh:
            raw = json.load(fh)
        cash, positions = raw.get("cash"), raw.get("positions") or {}
    if args.cash is not None:
        cash = args.cash
    try:
        plan = tasks.make_plan(cfg, cash, positions)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    print(f"as of {plan['as_of']}  equity {plan['equity']:,.2f}")
    print("target weights:", plan["targets"] or "all cash")
    if not plan["orders"]:
        print("no orders: the book is already where the strategy wants it")
    for o in plan["orders"]:
        print(f"  {o['side'].upper():<4} {o['qty']:>14,.6f} {o['symbol']:<10} "
              f"~{o['est_value']:>12,.2f} @ {o['ref_price']:,.4f}  ({o['reason']})")
    return 0


def _event_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return tasks.event_settings(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc))


def cmd_events_fetch(args) -> int:
    """Collect events from the configured sources into the events file."""
    cfg = load_config(args.config)
    _event_settings(cfg)
    result = tasks.collect_events(cfg, args.source)
    for item in result["fetched"]:
        print(f"{item['source']}: {item['events']:,} events for {result['symbols']} symbols")
    print(f"{result['path']}: {result['total']:,} events ({result['new']:+,} new)")
    for kind, count in result["counts"].items():
        print(f"  {kind:<26}{count:>7,}")
    return 0


def cmd_events_study(args) -> int:
    """Measure how prices moved around the collected events."""
    cfg = load_config(args.config)
    _event_settings(cfg)
    try:
        study, opts = tasks.study_events(cfg, args.kind)
    except ValueError as exc:
        raise SystemExit(str(exc))
    kind, threshold = opts["kind"], opts["threshold"]
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
        result = tasks.rehearse(cfg, args.replay)
        print(f"replayed {result['bars']} bars on the paper broker: equity {result['equity']:,.2f}, "
              f"cash {result['cash']:,.2f}, positions "
              f"{ {s: round(q, 4) for s, q in result['positions'].items()} }")
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


def cmd_app(args) -> int:
    """Open the browser app."""
    from .app import serve
    serve(args.configs, args.state, args.host, args.port, open_browser=not args.no_browser,
          check_host=not args.no_host_check)
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

    p = sub.add_parser("app", help="open the browser app (research and paper trading only)")
    p.add_argument("--configs", default="configs", help="folder of setups (default: configs)")
    p.add_argument("--state", default="state", help="folder for paper-trading state")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="address to listen on; the default is this computer only")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--no-host-check", action="store_true",
                   help="accept requests addressed to any host name (only behind a proxy you trust)")
    p.set_defaults(func=cmd_app)

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
