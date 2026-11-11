"""Market data: datasets, synthetic generators, file and network loaders."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .dataset import Dataset, clean_frame, infer_periods_per_year
from .loaders import AlpacaData, CCXTData, YahooData, load_csv, load_csv_dir
from .synthetic import synthetic_universe

__all__ = ["Dataset", "clean_frame", "infer_periods_per_year", "load_data",
           "make_provider", "synthetic_universe", "load_csv", "load_csv_dir",
           "YahooData", "AlpacaData", "CCXTData"]

_COMMON = {"source", "symbols", "start", "end", "cache", "cache_hours"}
_OPTIONS = {
    "synthetic": {"bars", "seed", "kind", "freq", "pairs"},
    "csv": {"path", "adjust"},
    "yahoo": {"interval", "range"},
    "alpaca": {"timeframe", "feed", "lookback_days"},
    "ccxt": {"exchange", "timeframe", "limit"},
}


def _check(spec: Mapping[str, Any]) -> str:
    source = spec.get("source", "synthetic")
    if source not in _OPTIONS:
        raise ValueError(f"data.source must be one of {sorted(_OPTIONS)}, got {source!r}")
    unknown = set(spec) - _COMMON - _OPTIONS[source]
    if unknown:
        raise ValueError(f"data: unknown option(s) {sorted(unknown)} for source {source!r}")
    return source


def make_provider(spec: Mapping[str, Any]):
    """A network provider with ``fetch(symbols, bars)`` for the given data spec."""
    source = _check(spec)
    opts = {k: spec[k] for k in _OPTIONS[source] if k in spec}
    if source == "yahoo":
        return YahooData(**opts)
    if source == "alpaca":
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_API_SECRET")
        return AlpacaData(key, secret, **opts)
    if source == "ccxt":
        return CCXTData(**opts)
    raise ValueError(f"data source {source!r} is not a live provider")


def _fetch_cached(spec: Mapping[str, Any], source: str, symbols) -> dict[str, pd.DataFrame]:
    """Download bars, reusing files under ``data.cache`` that are newer than
    ``data.cache_hours`` (default 12). Without ``cache`` nothing is stored."""
    if not spec.get("cache"):
        return make_provider(spec).fetch(symbols)
    options = sorted((k, str(spec[k])) for k in _OPTIONS[source] if k in spec)
    tag = hashlib.sha256(repr(options).encode()).hexdigest()[:10]
    folder = Path(spec["cache"]) / f"{source}-{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    fresh_after = time.time() - float(spec.get("cache_hours", 12)) * 3600
    paths = {s: folder / f"{s.replace('/', '-')}.csv" for s in symbols}
    stale = [s for s, p in paths.items() if not p.exists() or p.stat().st_mtime < fresh_after]
    if stale:
        for sym, df in make_provider(spec).fetch(stale).items():
            df.to_csv(paths[sym], index_label="date")
    return {s: load_csv(p, adjust=False) for s, p in paths.items()}


def load_data(spec: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Bars for a config's ``data`` section, as ``{symbol: DataFrame}``."""
    source = _check(spec)
    symbols = spec.get("symbols")
    if source == "synthetic":
        if not symbols:
            raise ValueError("data.symbols is required for synthetic data")
        opts = {k: spec[k] for k in _OPTIONS[source] if k in spec}
        if "start" in spec:
            opts["start"] = str(spec["start"])
        frames = synthetic_universe(symbols, **opts)
    elif source == "csv":
        frames = load_csv_dir(spec["path"], symbols, spec.get("adjust", True))
    else:
        if not symbols:
            raise ValueError(f"data.symbols is required for source {source!r}")
        frames = _fetch_cached(spec, source, symbols)
    if source != "synthetic" and spec.get("start") is not None:
        frames = {s: df.loc[str(spec["start"]):] for s, df in frames.items()}
    if spec.get("end") is not None:
        frames = {s: df.loc[:str(spec["end"])] for s, df in frames.items()}
    return frames
