"""Technical indicators on plain numpy arrays.

Every function is causal: element ``i`` of the output depends only on inputs
``[0..i]``. Outputs have the same length as the input and are NaN until enough
data exists, unless stated otherwise.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = [
    "sma", "ema", "ema_last", "wma", "rma", "rolling_std", "rolling_max", "rolling_min",
    "zscore", "roc", "rsi", "macd", "bollinger", "true_range", "atr", "adx",
    "donchian", "stochastic", "keltner", "supertrend", "obv", "vwap",
    "efficiency_ratio", "realized_vol", "hurst", "rolling_beta",
]


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, n)


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


# -- moving averages ---------------------------------------------------------
def sma(x, n: int) -> np.ndarray:
    x = _f(x)
    out = _nan(len(x))
    if n > 0 and len(x) >= n:
        out[n - 1:] = _windows(x, n).mean(axis=1)
    return out


def ema(x, n: float, alpha: float | None = None) -> np.ndarray:
    """Exponential MA seeded with the first value (pandas ``adjust=False``)."""
    x = _f(x)
    if not len(x):
        return x.copy()
    a = 2.0 / (n + 1.0) if alpha is None else alpha
    acc = x[0]
    res = []
    for v in x.tolist():
        acc += a * (v - acc)
        res.append(acc)
    return np.array(res)


@lru_cache(maxsize=512)
def _ema_weights(alpha: float, size: int) -> np.ndarray:
    w = alpha * (1.0 - alpha) ** np.arange(size - 1, -1, -1.0)
    w[0] = (1.0 - alpha) ** (size - 1)
    return w


def ema_last(x, n: float) -> float:
    """``ema(x, n)[-1]`` without the loop: one dot product with cached weights."""
    x = _f(x)
    return float(_ema_weights(2.0 / (n + 1.0), len(x)) @ x)


def wma(x, n: int) -> np.ndarray:
    """Linearly weighted MA; the newest bar carries weight ``n``."""
    x = _f(x)
    out = _nan(len(x))
    if n > 0 and len(x) >= n:
        w = np.arange(1, n + 1, dtype=float)
        out[n - 1:] = _windows(x, n) @ w / w.sum()
    return out


def rma(x, n: int) -> np.ndarray:
    """Wilder's smoothing: SMA seed over the first ``n`` valid values."""
    x = _f(x)
    out = _nan(len(x))
    valid = np.flatnonzero(~np.isnan(x))
    if not len(valid):
        return out
    start = int(valid[0])
    if len(x) - start < n:
        return out
    acc = float(x[start:start + n].mean())
    out[start + n - 1] = acc
    k = 1.0 / n
    res = []
    for v in x[start + n:].tolist():
        acc += (v - acc) * k
        res.append(acc)
    out[start + n:] = res
    return out


# -- rolling statistics ------------------------------------------------------
def rolling_std(x, n: int, ddof: int = 0) -> np.ndarray:
    x = _f(x)
    out = _nan(len(x))
    if n > ddof and len(x) >= n:
        out[n - 1:] = _windows(x, n).std(axis=1, ddof=ddof)
    return out


def rolling_max(x, n: int) -> np.ndarray:
    x = _f(x)
    out = _nan(len(x))
    if n > 0 and len(x) >= n:
        out[n - 1:] = _windows(x, n).max(axis=1)
    return out


def rolling_min(x, n: int) -> np.ndarray:
    x = _f(x)
    out = _nan(len(x))
    if n > 0 and len(x) >= n:
        out[n - 1:] = _windows(x, n).min(axis=1)
    return out


def zscore(x, n: int) -> np.ndarray:
    x = _f(x)
    sd = rolling_std(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - sma(x, n)) / sd
    return np.where(sd == 0, 0.0, z)


def roc(x, n: int) -> np.ndarray:
    """Rate of change: ``x[i] / x[i-n] - 1``."""
    x = _f(x)
    out = _nan(len(x))
    if 0 < n < len(x):
        out[n:] = x[n:] / x[:-n] - 1.0
    return out


def realized_vol(x, n: int, periods_per_year: float = 252.0) -> np.ndarray:
    """Annualised rolling volatility of log returns."""
    x = _f(x)
    out = _nan(len(x))
    if len(x) > n > 1:
        out[1:] = rolling_std(np.diff(np.log(x)), n, ddof=1)
    return out * math.sqrt(periods_per_year)


def rolling_beta(y, x, n: int) -> np.ndarray:
    """Rolling OLS slope of ``y`` on ``x``."""
    y, x = _f(y), _f(x)
    out = _nan(len(x))
    if n > 1 and len(x) >= n:
        xw, yw = _windows(x, n), _windows(y, n)
        xm = xw.mean(axis=1, keepdims=True)
        ym = yw.mean(axis=1, keepdims=True)
        var = ((xw - xm) ** 2).mean(axis=1)
        cov = ((xw - xm) * (yw - ym)).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = np.where(var > 0, cov / var, np.nan)
    return out


# -- oscillators -------------------------------------------------------------
def rsi(x, n: int = 14) -> np.ndarray:
    x = _f(x)
    out = _nan(len(x))
    if len(x) <= n:
        return out
    d = np.diff(x)
    up = rma(np.maximum(d, 0.0), n)
    dn = rma(np.maximum(-d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r = np.where(dn == 0, np.where(up == 0, 50.0, 100.0), r)
    r[np.isnan(up)] = np.nan
    out[1:] = r
    return out


def macd(x, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns ``(macd_line, signal_line, histogram)``."""
    x = _f(x)
    line = ema(x, fast) - ema(x, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(high, low, close, n: int = 14, d: int = 3):
    """Returns ``(%K, %D)``."""
    hh, ll = rolling_max(high, n), rolling_min(low, n)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (_f(close) - ll) / rng
    k = np.where(rng == 0, 50.0, k)
    k[np.isnan(hh)] = np.nan
    dline = _nan(len(k))
    if len(k) >= n - 1 + d:
        dline[n - 1 + d - 1:] = _windows(k[n - 1:], d).mean(axis=1)
    return k, dline


# -- bands and channels ------------------------------------------------------
def bollinger(x, n: int = 20, k: float = 2.0):
    """Returns ``(mid, upper, lower)``."""
    mid, sd = sma(x, n), rolling_std(x, n)
    return mid, mid + k * sd, mid - k * sd


def donchian(high, low, n: int):
    """Returns ``(upper, lower)`` including the current bar."""
    return rolling_max(high, n), rolling_min(low, n)


def keltner(high, low, close, n: int = 20, atr_n: int = 14, mult: float = 2.0):
    """Returns ``(mid, upper, lower)``."""
    mid = ema(close, n)
    a = atr(high, low, close, atr_n)
    return mid, mid + mult * a, mid - mult * a


# -- volatility and trend strength -------------------------------------------
def true_range(high, low, close) -> np.ndarray:
    high, low, close = _f(high), _f(low), _f(close)
    tr = high - low
    if len(tr) > 1:
        prev = close[:-1]
        tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(high[1:] - prev),
                                                np.abs(low[1:] - prev)))
    return tr


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return rma(true_range(high, low, close), n)


def adx(high, low, close, n: int = 14):
    """Returns ``(adx, +DI, -DI)``."""
    high, low, close = _f(high), _f(low), _f(close)
    size = len(close)
    if size < 2:
        return _nan(size), _nan(size), _nan(size)
    up = np.diff(high)
    down = -np.diff(low)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = rma(true_range(high, low, close)[1:], n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * rma(plus_dm, n) / tr
        mdi = 100.0 * rma(minus_dm, n) / tr
        total = pdi + mdi
        dx = np.where(total > 0, 100.0 * np.abs(pdi - mdi) / total, 0.0)
    dx[np.isnan(total)] = np.nan
    pad = np.array([np.nan])
    return (np.concatenate([pad, rma(dx, n)]), np.concatenate([pad, pdi]),
            np.concatenate([pad, mdi]))


def supertrend(high, low, close, n: int = 10, mult: float = 3.0):
    """Returns ``(line, direction)`` with direction +1 (up) or -1 (down)."""
    high, low, close = _f(high), _f(low), _f(close)
    size = len(close)
    line, direction = _nan(size), np.zeros(size)
    a = atr(high, low, close, n)
    mid = (high + low) / 2.0
    ub, lb = (mid + mult * a).tolist(), (mid - mult * a).tolist()
    c = close.tolist()
    fu = fl = float("nan")
    d = 1
    started = False
    for i in range(size):
        if math.isnan(ub[i]):
            continue
        if not started:
            fu, fl, started = ub[i], lb[i], True
            d = 1 if c[i] >= mid[i] else -1
        else:
            prev_fu, prev_fl = fu, fl
            fu = ub[i] if (ub[i] < prev_fu or c[i - 1] > prev_fu) else prev_fu
            fl = lb[i] if (lb[i] > prev_fl or c[i - 1] < prev_fl) else prev_fl
            if c[i] > prev_fu:
                d = 1
            elif c[i] < prev_fl:
                d = -1
        direction[i] = d
        line[i] = fl if d == 1 else fu
    return line, direction


def efficiency_ratio(x, n: int) -> np.ndarray:
    """Kaufman efficiency ratio: net move divided by path length, in [0, 1]."""
    x = _f(x)
    out = _nan(len(x))
    if 0 < n < len(x):
        net = np.abs(x[n:] - x[:-n])
        path = _windows(np.abs(np.diff(x)), n).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n:] = np.where(path > 0, net / path, 0.0)
    return out


def hurst(x, max_lag: int = 20) -> float:
    """Hurst exponent estimate of one window (0.5 random, >0.5 trending)."""
    x = np.log(_f(x))
    if len(x) < max_lag * 2:
        return float("nan")
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return float("nan")
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


# -- volume ------------------------------------------------------------------
def obv(close, volume) -> np.ndarray:
    close, volume = _f(close), _f(volume)
    out = np.zeros(len(close))
    if len(close) > 1:
        out[1:] = np.cumsum(np.sign(np.diff(close)) * volume[1:])
    return out


def vwap(high, low, close, volume, n: int) -> np.ndarray:
    """Rolling volume-weighted average of the typical price."""
    tp = (_f(high) + _f(low) + _f(close)) / 3.0
    volume = _f(volume)
    out = _nan(len(tp))
    if n > 0 and len(tp) >= n:
        num = _windows(tp * volume, n).sum(axis=1)
        den = _windows(volume, n).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = np.where(den > 0, num / den, np.nan)
    return out
