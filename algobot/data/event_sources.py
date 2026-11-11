"""Collectors that turn outside data into an ``EventStore``.

``SecFilings``  company announcements from the SEC's EDGAR database: every 8-K
                (earnings releases, executive changes, deals, ...) with the
                second it became public. Free, official, goes back decades.
``YahooNews``   the latest headlines per symbol, scored for sentiment. Yahoo
                only serves recent items, so run it on a schedule to build up
                history.
``AlpacaNews``  historical headlines (2015 onward) for Alpaca account holders.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ..events import EventStore

# 8-K item numbers, most material first: the first match names the event.
SEC_ITEMS = [
    ("1.03", "bankruptcy"), ("4.02", "restatement"), ("3.01", "delisting_notice"),
    ("1.05", "cyber_incident"), ("2.06", "impairment"), ("2.05", "restructuring"),
    ("4.01", "auditor_change"), ("2.01", "acquisition_or_disposal"),
    ("1.01", "material_agreement"), ("1.02", "agreement_terminated"),
    ("5.02", "executive_change"), ("2.03", "new_debt"), ("3.02", "share_issuance"),
    ("5.07", "shareholder_vote"), ("5.03", "bylaw_change"), ("7.01", "disclosure"),
    ("8.01", "other_event"),
]


def sec_kind(items: str) -> str:
    """Event kind for an 8-K's comma-separated item list."""
    present = {i.strip() for i in str(items).split(",") if i.strip()}
    if "2.02" in present:
        return "earnings"
    for code, name in SEC_ITEMS:
        if code in present:
            return name
    return "filing"


class SecFilings:
    """8-K filings from EDGAR as events.

    The SEC asks automated clients to identify themselves and to stay under ten
    requests a second. Set ``SEC_USER_AGENT`` (for example your name and email)
    to identify yourself; a generic client name is sent otherwise.
    """

    TICKERS = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS = "https://data.sec.gov/submissions/"

    def __init__(self, user_agent: str | None = None, session: Any = None,
                 pause: float = 0.15, timeout: float = 30.0, max_files: int = 60):
        self.user_agent = (user_agent or os.environ.get("SEC_USER_AGENT")
                           or "algobot/0.1 (automated research client)")
        self.pause, self.timeout, self.max_files = pause, timeout, max_files
        self._session = session
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str) -> Any:
        if self._session is None:
            import requests
            self._session = requests.Session()
        resp = self._session.get(url, timeout=self.timeout,
                                 headers={"User-Agent": self.user_agent,
                                          "Accept-Encoding": "gzip, deflate"})
        resp.raise_for_status()
        if self.pause:
            time.sleep(self.pause)
        return resp.json()

    def cik(self, symbol: str) -> int:
        if self._ciks is None:
            listing = self._get(self.TICKERS)
            self._ciks = {str(row["ticker"]).upper(): int(row["cik_str"])
                          for row in listing.values()}
        key = symbol.upper().replace(".", "-")
        if key not in self._ciks:
            raise KeyError(f"SEC has no company for ticker {symbol!r}")
        return self._ciks[key]

    @staticmethod
    def parse(symbol: str, block: Mapping[str, Sequence], forms: Sequence[str] = ("8-K",)) -> list[dict]:
        """Events from one block of EDGAR's column-oriented filing list."""
        rows = []
        for form, accepted, items in zip(block["form"], block["acceptanceDateTime"],
                                         block.get("items") or [""] * len(block["form"])):
            if form not in forms or not accepted:
                continue
            rows.append({"symbol": symbol,
                         "time": pd.Timestamp(accepted),   # UTC; converted on load
                         "kind": sec_kind(items) if form == "8-K" else form.lower(),
                         "text": f"{form} items {items}" if items else form,
                         "source": "sec"})
        return rows

    def fetch(self, symbols: Sequence[str], start: Any = None,
              forms: Sequence[str] = ("8-K",)) -> EventStore:
        start = pd.Timestamp(start) if start is not None else None
        rows: list[dict] = []
        for symbol in symbols:
            cik = self.cik(symbol)
            data = self._get(f"{self.SUBMISSIONS}CIK{cik:010d}.json")
            filings = data.get("filings", {})
            rows += self.parse(symbol, filings.get("recent", {"form": [], "acceptanceDateTime": []}),
                               forms)
            # Busy filers overflow into extra files; read those that reach back to ``start``.
            older = [f for f in filings.get("files", [])
                     if start is None or pd.Timestamp(f["filingTo"]) >= start]
            for extra in sorted(older, key=lambda f: f["filingTo"], reverse=True)[:self.max_files]:
                rows += self.parse(symbol, self._get(self.SUBMISSIONS + extra["name"]), forms)
        store = EventStore(rows)
        return store.filter(start=start) if start is not None else store


# -- headline sentiment ----------------------------------------------------------
_POSITIVE = set("""
beat beats beating topped tops surge surges surged soar soars soared jump jumps jumped rally
rallies rallied gain gains gained rise rises rose climb climbs climbed record upgrade upgrades
upgraded outperform raise raises raised boost boosts boosted strong stronger strength growth
grows grew profit profits profitable win wins won approval approved approves breakthrough
bullish optimistic expands expansion buyback dividend exceeds exceeded accelerates rebound
rebounds recovered recovery upbeat momentum milestone launches partnership awarded
""".split())
_NEGATIVE = set("""
miss misses missed plunge plunges plunged tumble tumbles tumbled slump slumps slumped drop
drops dropped fall falls fell slide slides slid sink sinks sank decline declines declined
downgrade downgrades downgraded underperform cut cuts cutting slash slashes slashed weak
weaker weakness loss losses lawsuit sued sues probe investigation recall recalls fraud
bankruptcy bankrupt layoffs layoff warns warning warned bearish concern concerns fears crash
crashes halts halted delay delays delayed fine fined penalty default selloff slowdown
disappoints disappointing disappointed shortfall resigns resignation breach hack scandal
""".split())
_NEGATORS = {"no", "not", "never", "without", "fails", "failed", "unable"}


def headline_sentiment(text: str) -> float:
    """Crude word-list score in [-1, 1]; 0 when nothing recognisable is found.

    This is a baseline, not a language model: it cannot read context or
    sarcasm. Swap in a better scorer through the ``scorer`` argument.
    """
    words = re.findall(r"[a-z']+", str(text).lower())
    pos = neg = 0
    for i, word in enumerate(words):
        flipped = i > 0 and words[i - 1] in _NEGATORS
        if word in _POSITIVE:
            neg, pos = (neg + 1, pos) if flipped else (neg, pos + 1)
        elif word in _NEGATIVE:
            neg, pos = (neg, pos + 1) if flipped else (neg + 1, pos)
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


class YahooNews:
    """Latest headlines per symbol from Yahoo's public search endpoint."""

    URL = "https://query1.finance.yahoo.com/v1/finance/search"

    def __init__(self, count: int = 50, session: Any = None, timeout: float = 20.0,
                 scorer: Callable[[str], float] = headline_sentiment):
        self.count, self.timeout, self.scorer = min(count, 50), timeout, scorer
        self._session = session

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "Mozilla/5.0 (compatible; algobot)"
        return self._session

    def parse(self, symbol: str, payload: Mapping) -> list[dict]:
        rows = []
        for item in payload.get("news") or []:
            title, stamp = item.get("title"), item.get("providerPublishTime")
            if not title or not stamp:
                continue
            related = item.get("relatedTickers")
            if related and symbol not in related:
                continue  # a general story that merely matched the search
            rows.append({"symbol": symbol, "time": pd.Timestamp(stamp, unit="s", tz="UTC"),
                         "kind": "news", "value": self.scorer(title), "text": title,
                         "source": item.get("publisher", "yahoo")})
        return rows

    def fetch(self, symbols: Sequence[str], start: Any = None) -> EventStore:
        rows: list[dict] = []
        for symbol in symbols:
            resp = self._http().get(self.URL, timeout=self.timeout,
                                    params={"q": symbol, "newsCount": self.count, "quotesCount": 0})
            resp.raise_for_status()
            rows += self.parse(symbol, resp.json())
        return EventStore(rows)


class AlpacaNews:
    """Historical headlines from Alpaca's news API (needs API keys)."""

    URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, key: str, secret: str, session: Any = None, timeout: float = 20.0,
                 scorer: Callable[[str], float] = headline_sentiment, max_pages: int = 2000):
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._session, self.timeout, self.scorer, self.max_pages = session, timeout, scorer, max_pages

    def fetch(self, symbols: Sequence[str], start: Any = None) -> EventStore:
        if self._session is None:
            import requests
            self._session = requests.Session()
        wanted = set(symbols)
        params: dict[str, Any] = {"symbols": ",".join(symbols), "limit": 50, "sort": "asc"}
        if start is not None:
            params["start"] = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows: list[dict] = []
        for _ in range(self.max_pages):
            resp = self._session.get(self.URL, params=params, headers=self._headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
            for item in body.get("news") or []:
                for symbol in wanted.intersection(item.get("symbols") or []):
                    rows.append({"symbol": symbol, "time": pd.Timestamp(item["created_at"]),
                                 "kind": "news", "value": self.scorer(item.get("headline", "")),
                                 "text": item.get("headline", ""),
                                 "source": item.get("source", "alpaca")})
            token = body.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}
        return EventStore(rows)


def make_event_source(name: str):
    """Collector for a config's ``events.sources`` entry."""
    if name == "sec":
        return SecFilings()
    if name == "yahoo_news":
        return YahooNews()
    if name == "alpaca_news":
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_API_SECRET")
        return AlpacaNews(key, secret)
    raise ValueError(f"unknown event source {name!r}; choose sec, yahoo_news or alpaca_news")
