"""Parameter search and walk-forward validation.

Parameters are addressed by dotted paths into the config, for example
``strategy.params.fast`` or ``risk.trailing_stop_pct``, so anything in a
config can be searched over, including the children of an ensemble
(``strategy.params.children.0.params.window``).
"""
from __future__ import annotations

import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import run_backtest, with_overrides
from ..data.dataset import Dataset
from .metrics import compute_metrics

_REPORTED = ("sharpe", "total_return", "cagr", "max_drawdown", "calmar", "sortino",
             "trades", "turnover")


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the listed values."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def sample_space(space: Mapping[str, Any], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Random draws. A value is either a list of choices or a range such as
    ``{"low": 5, "high": 50, "type": "int"}`` (types: int, float, log)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        point = {}
        for key, spec in space.items():
            if isinstance(spec, Mapping):
                lo, hi, kind = spec["low"], spec["high"], spec.get("type", "float")
                if kind == "int":
                    point[key] = int(rng.integers(lo, hi + 1))
                elif kind == "log":
                    point[key] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
                else:
                    point[key] = float(rng.uniform(lo, hi))
            else:
                point[key] = spec[int(rng.integers(0, len(spec)))]
        draws.append(point)
    return draws


def _evaluate(job) -> dict[str, Any]:
    cfg, overrides, dataset, trade_start = job
    row: dict[str, Any] = dict(overrides)
    try:
        metrics = run_backtest(with_overrides(cfg, overrides), dataset, trade_start).metrics
        row.update({k: metrics.get(k, float("nan")) for k in _REPORTED})
    except ValueError as exc:  # an invalid combination, e.g. fast >= slow
        row["error"] = str(exc)
    return row


def optimize(cfg: Mapping[str, Any], dataset: Dataset, space: Mapping[str, Any], *,
             metric: str = "sharpe", method: str = "grid", n_iter: int = 50,
             seed: int = 0, n_jobs: int = 1, min_trades: int = 0,
             trade_start: Any = None) -> pd.DataFrame:
    """Backtest every candidate and return them best-first by ``metric``.

    The best row of this table is fitted to the data it was searched on. Treat
    its numbers as an upper bound and use ``walk_forward`` for an honest one.
    """
    if method == "grid":
        candidates = expand_grid(space)
    elif method == "random":
        candidates = sample_space(space, n_iter, seed)
    else:
        raise ValueError("optimize: method must be 'grid' or 'random'")
    if not candidates:
        raise ValueError("optimize: empty search space")
    jobs = [(cfg, c, dataset, trade_start) for c in candidates]
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_evaluate, jobs))
    else:
        rows = [_evaluate(job) for job in jobs]
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"optimize: no candidate produced metric {metric!r}: "
                         f"{table.get('error', pd.Series(dtype=str)).dropna().unique().tolist()}")
    score = table[metric].astype(float)
    if min_trades:
        score = score.where(table["trades"] >= min_trades)
    table["score"] = score.replace([np.inf, -np.inf], np.nan)
    return table.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def best_params(table: pd.DataFrame, space: Mapping[str, Any]) -> dict[str, Any]:
    if table.empty or pd.isna(table.loc[0, "score"]):
        raise ValueError("no valid candidate")
    # Read column by column: a whole row would be upcast and turn ints into floats.
    best = {k: table[k].iloc[0] for k in space}
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    equity: pd.Series
    metrics: dict[str, float]
    periods_per_year: float

    def summary(self) -> str:
        from .metrics import format_metrics
        return (f"{len(self.folds)} folds, out-of-sample only\n"
                f"{format_metrics(self.metrics)}\n\n{self.folds.to_string(index=False)}")


def walk_forward(cfg: Mapping[str, Any], dataset: Dataset, space: Mapping[str, Any], *,
                 train_bars: int, test_bars: int, metric: str = "sharpe",
                 method: str = "grid", n_iter: int = 50, seed: int = 0,
                 n_jobs: int = 1, min_trades: int = 0,
                 warmup_bars: int | None = None) -> WalkForwardResult:
    """Repeatedly fit on ``train_bars``, then trade the next ``test_bars`` with
    the winning parameters. Only the test periods are stitched into the result,
    so every bar of the equity curve was traded with parameters chosen before it.
    """
    n = len(dataset)
    if train_bars + test_bars > n:
        raise ValueError(f"walk_forward: need at least {train_bars + test_bars} bars, have {n}")
    warmup = train_bars if warmup_bars is None else warmup_bars
    ppy = dataset.periods_per_year
    initial = float((cfg.get("backtest") or {}).get("initial_cash", 100_000.0))
    rows, pieces, trades, fees = [], [], [], 0.0
    start = 0
    while start + train_bars < n:
        lo = start + train_bars
        hi = min(n, lo + test_bars)
        if hi - lo < min(20, test_bars):
            break
        table = optimize(cfg, dataset.slice(start, lo), space, metric=metric,
                         method=method, n_iter=n_iter, seed=seed + start,
                         n_jobs=n_jobs, min_trades=min_trades)
        params = best_params(table, space)
        result = run_backtest(with_overrides(cfg, params),
                              dataset.slice(max(0, lo - warmup), hi),
                              trade_start=dataset.timeline[lo])
        # Each fold starts flat, so its first bar carries a zero return.
        pieces.append(result.equity.pct_change().fillna(0.0))
        trades.extend(result.trades)
        fees += result.metrics.get("fees", 0.0)
        rows.append({"test_start": dataset.timeline[lo].date(),
                     "test_end": dataset.timeline[hi - 1].date(), **params,
                     f"train_{metric}": round(float(table.loc[0, "score"]), 3),
                     f"test_{metric}": round(result.metrics.get(metric, float("nan")), 3),
                     "test_return": round(result.metrics.get("total_return", 0.0), 4)})
        start += test_bars
    equity = initial * (1.0 + pd.concat(pieces)).cumprod()
    equity.name = "equity"
    return WalkForwardResult(
        folds=pd.DataFrame(rows), equity=equity,
        metrics=compute_metrics(equity, periods_per_year=ppy, trades=trades, fees=fees),
        periods_per_year=ppy)
