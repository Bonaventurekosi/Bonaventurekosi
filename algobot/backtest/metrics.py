"""Performance statistics for an equity curve and its trades."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..types import Trade


def drawdown(equity: np.ndarray) -> np.ndarray:
    return equity / np.maximum.accumulate(equity) - 1.0


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask.tolist():
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def compute_metrics(
    equity: pd.Series,
    *,
    periods_per_year: float = 252.0,
    trades: Sequence[Trade] = (),
    traded_notional: float = 0.0,
    exposure: pd.Series | None = None,
    fees: float = 0.0,
    fills: int | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """``fees`` is everything trading cost: commission, slippage and financing.
    ``trades`` counts round trips (a position opened and later fully closed);
    ``fills`` counts every executed order, including partial adjustments."""
    e = np.asarray(equity, dtype=float)
    out: dict[str, float] = {"bars": float(len(e))}
    if len(e) < 2 or e[0] <= 0:
        return out
    ppy = periods_per_year
    rets = e[1:] / e[:-1] - 1.0
    years = (len(e) - 1) / ppy
    excess = rets - risk_free / ppy
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = math.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    dd = drawdown(e)

    out["total_return"] = e[-1] / e[0] - 1.0
    out["cagr"] = (e[-1] / e[0]) ** (1.0 / years) - 1.0 if e[-1] > 0 else -1.0
    out["volatility"] = sd * math.sqrt(ppy)
    out["sharpe"] = excess.mean() / sd * math.sqrt(ppy) if sd > 0 else 0.0
    out["sortino"] = excess.mean() / downside * math.sqrt(ppy) if downside > 0 else 0.0
    out["max_drawdown"] = float(dd.min())
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else 0.0
    out["max_drawdown_bars"] = float(_longest_run(dd < 0))
    tail = np.percentile(rets, 5)
    out["var_95"] = float(-tail)
    out["cvar_95"] = float(-rets[rets <= tail].mean())
    out["best_bar"] = float(rets.max())
    out["worst_bar"] = float(rets.min())

    pnls = np.array([t.pnl for t in trades], dtype=float)
    if fills is not None:
        out["fills"] = float(fills)
    out["trades"] = float(len(pnls))
    if len(pnls):
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        out["win_rate"] = len(wins) / len(pnls)
        out["profit_factor"] = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        out["avg_trade"] = float(pnls.mean())
        if len(wins) and len(losses) and losses.mean() < 0:
            out["payoff_ratio"] = float(wins.mean() / -losses.mean())
    out["fees"] = float(fees)
    if years > 0 and traded_notional:
        out["turnover"] = traded_notional / float(e.mean()) / years
    if exposure is not None and len(exposure):
        out["avg_exposure"] = float(np.mean(exposure))
    return out


def bootstrap_sharpe(equity: pd.Series, *, periods_per_year: float = 252.0,
                     samples: int = 1000, block: int = 20, seed: int = 0,
                     ) -> dict[str, float]:
    """Block-bootstrap the bar returns to see how fragile the result is.

    Returns the 5th/50th/95th percentile Sharpe ratio and the share of
    resampled histories that lost money.
    """
    e = np.asarray(equity, dtype=float)
    rets = e[1:] / e[:-1] - 1.0
    n = len(rets)
    if n < block * 2:
        return {}
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block)
    sharpes, losses = [], 0
    for _ in range(samples):
        starts = rng.integers(0, n - block + 1, size=blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        r = rets[idx]
        sd = r.std(ddof=1)
        sharpes.append(r.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0)
        losses += np.prod(1.0 + r) < 1.0
    p5, p50, p95 = np.percentile(sharpes, [5, 50, 95])
    return {"sharpe_p05": float(p5), "sharpe_p50": float(p50),
            "sharpe_p95": float(p95), "prob_loss": losses / samples}


_FORMATS = [
    ("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"),
    ("volatility", "Volatility (ann.)", "pct"), ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"),
    ("calmar", "Calmar", "num"), ("max_drawdown_bars", "Longest drawdown (bars)", "int"),
    ("var_95", "VaR 95% (per bar)", "pct"), ("cvar_95", "CVaR 95% (per bar)", "pct"),
    ("fills", "Orders filled", "int"),
    ("trades", "Round-trip trades", "int"), ("win_rate", "Win rate", "pct"),
    ("profit_factor", "Profit factor", "num"), ("payoff_ratio", "Avg win / avg loss", "num"),
    ("turnover", "Turnover (x equity / yr)", "num"), ("avg_exposure", "Avg gross exposure", "pct"),
    ("fees", "Costs (fees + slippage)", "money"),
]


def format_value(value: float, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "money":
        return f"{value:,.2f}"
    return "inf" if math.isinf(value) else f"{value:,.2f}"


def metric_rows(metrics: dict[str, float]) -> list[tuple[str, str]]:
    return [(label, format_value(metrics[key], kind))
            for key, label, kind in _FORMATS if key in metrics]


def format_metrics(metrics: dict[str, float], benchmark: dict[str, float] | None = None) -> str:
    lines = []
    for key, label, kind in _FORMATS:
        if key not in metrics:
            continue
        line = f"  {label:<26}{format_value(metrics[key], kind):>14}"
        if benchmark and key in benchmark:
            line += f"{format_value(benchmark[key], kind):>14}"
        lines.append(line)
    return "\n".join(lines)
