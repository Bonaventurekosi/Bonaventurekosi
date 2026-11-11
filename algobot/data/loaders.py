"""Loading bars from files and from market-data services.

The network providers (Yahoo, Alpaca, CCXT) share one method,
``fetch(symbols, bars) -> {symbol: DataFrame}``, so the same object can
download history for a backtest and be polled for fresh bars by the live
runner. All timestamps are naive UTC.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .dataset import clean_frame

_DATE_COLUMNS = ("date", "datetime", "timestamp", "time")


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now(dt.timezone.utc)).tz_localize(None)


# -- files -------------------------------------------------------------------
def load_csv(path: str | Path, adjust: bool = True) -> pd.DataFrame:
    """Read one symbol's bars. Needs a date column and a close column; an
    "adj close" column, if present, is used to back-adjust the prices."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in _DATE_COLUMNS if c in df.columns), df.columns[0])
    idx = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.drop(columns=[date_col])
    df.index = pd.DatetimeIndex(idx)
    adj = next((c for c in ("adj close", "adj_close", "adjclose") if c in df.columns), None)
    if adj and adjust and "close" in df.columns:
        factor = pd.to_numeric(df[adj], errors="coerce") / pd.to_numeric(df["close"], errors="coerce")
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * factor
    return clean_frame(df)


def load_csv_dir(path: str | Path, symbols: Sequence[str] | None = None,
                 adjust: bool = True) -> dict[str, pd.DataFrame]:
    """Read ``<path>/<SYMBOL>.csv`` for each symbol (all CSVs when None)."""
    root = Path(path)
    if symbols is None:
        files = {p.stem: p for p in sorted(root.glob("*.csv"))}
    else:
        files = {s: root / f"{s.replace('/', '-')}.csv" for s in symbols}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing or not files:
        raise FileNotFoundError(f"missing CSV file(s): {missing or root}")
    return {s: load_csv(p, adjust) for s, p in files.items()}


# -- Yahoo Finance -----------------------------------------------------------
class YahooData:
    """Daily or intraday bars from Yahoo's public chart endpoint (no key)."""

    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, interval: str = "1d", range: str = "10y", session: Any = None,
                 timeout: float = 20.0):
        self.interval, self.range, self.timeout = interval, range, timeout
        self._session = session

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "Mozilla/5.0 (compatible; algobot)"
        return self._session

    def fetch(self, symbols: Sequence[str], bars: int | None = None) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in symbols:
            resp = self._http().get(
                self.URL.format(symbol=sym),
                params={"interval": self.interval, "range": self.range,
                        "includeAdjustedClose": "true"},
                timeout=self.timeout)
            resp.raise_for_status()
            df = self.parse(resp.json(), daily=not self.interval.endswith(("m", "h")))
            out[sym] = df.iloc[-bars:] if bars else df
        return out

    @staticmethod
    def parse(payload: Mapping, daily: bool = True) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise RuntimeError(f"Yahoo error: {chart['error']}")
        result = (chart.get("result") or [None])[0]
        if not result or not result.get("timestamp"):
            raise RuntimeError("Yahoo returned no bars")
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
                          index=pd.to_datetime(result["timestamp"], unit="s"))
        adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        if adj is not None:
            factor = pd.Series(adj, index=df.index, dtype=float) / df["close"]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * factor
        if daily:
            df.index = df.index.normalize()
        return clean_frame(df)


# -- Alpaca ------------------------------------------------------------------
class AlpacaData:
    """Bars from Alpaca's market-data API. Symbols containing "/" are crypto."""

    BASE = "https://data.alpaca.markets"

    def __init__(self, key: str, secret: str, timeframe: str = "1Day",
                 feed: str = "iex", lookback_days: int = 3650, session: Any = None,
                 timeout: float = 20.0):
        self.timeframe, self.feed, self.lookback_days = timeframe, feed, lookback_days
        self.timeout = timeout
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._session = session

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _pages(self, url: str, params: dict) -> dict[str, list]:
        merged: dict[str, list] = {}
        while True:
            resp = self._http().get(url, params=params, headers=self._headers,
                                    timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
            for sym, rows in (body.get("bars") or {}).items():
                merged.setdefault(sym, []).extend(rows)
            token = body.get("next_page_token")
            if not token:
                return merged
            params = {**params, "page_token": token}

    def fetch(self, symbols: Sequence[str], bars: int | None = None) -> dict[str, pd.DataFrame]:
        start = (_utc_now() - pd.Timedelta(days=self.lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        base = {"timeframe": self.timeframe, "start": start, "limit": 10000}
        raw: dict[str, list] = {}
        stocks = [s for s in symbols if "/" not in s]
        crypto = [s for s in symbols if "/" in s]
        if stocks:
            raw.update(self._pages(f"{self.BASE}/v2/stocks/bars",
                                   {**base, "symbols": ",".join(stocks),
                                    "adjustment": "all", "feed": self.feed}))
        if crypto:
            raw.update(self._pages(f"{self.BASE}/v1beta3/crypto/us/bars",
                                   {**base, "symbols": ",".join(crypto)}))
        out = {}
        for sym in symbols:
            if not raw.get(sym):
                raise RuntimeError(f"Alpaca returned no bars for {sym}")
            df = self.parse(raw[sym], daily=self.timeframe.lower().endswith("day"))
            out[sym] = df.iloc[-bars:] if bars else df
        return out

    @staticmethod
    def parse(rows: Sequence[Mapping], daily: bool = True) -> pd.DataFrame:
        df = pd.DataFrame({"open": [r["o"] for r in rows], "high": [r["h"] for r in rows],
                           "low": [r["l"] for r in rows], "close": [r["c"] for r in rows],
                           "volume": [r["v"] for r in rows]},
                          index=pd.to_datetime([r["t"] for r in rows], utc=True).tz_localize(None))
        if daily:
            df.index = df.index.normalize()
        return clean_frame(df)


# -- CCXT (crypto exchanges) ---------------------------------------------------
class CCXTData:
    """Bars from any exchange supported by the optional ``ccxt`` package."""

    def __init__(self, exchange: Any = "binance", timeframe: str = "1d", limit: int = 1000):
        if isinstance(exchange, str):
            import ccxt  # optional dependency
            exchange = getattr(ccxt, exchange)({"enableRateLimit": True})
        self.exchange, self.timeframe, self.limit = exchange, timeframe, limit

    def fetch(self, symbols: Sequence[str], bars: int | None = None) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in symbols:
            rows = self.exchange.fetch_ohlcv(sym, self.timeframe, limit=bars or self.limit)
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df.pop("ts"), unit="ms")
            out[sym] = clean_frame(df)
        return out
