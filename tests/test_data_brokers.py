import numpy as np
import pandas as pd
import pytest

from algobot.data import (AlpacaData, CCXTData, Dataset, YahooData, clean_frame,
                          infer_periods_per_year, load_csv_dir, load_data, make_provider,
                          synthetic_universe)
from algobot.execution.brokers import AlpacaBroker, CCXTBroker, PaperBroker
from algobot.execution.orders import CostModel
from algobot.types import Order


# -- cleaning and files ----------------------------------------------------------
def test_clean_frame_repairs_bad_bars():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04", "2024-01-05",
                          "2024-01-08"])
    raw = pd.DataFrame({"Open": [10, 9, 9.5, None, 11, 12], "HIGH": [10.5, 9, 9.6, 12, 10, 13],
                        "Low": [9.8, 9, 9.4, 11, 12, -1], " Close ": [10.2, 9, 9.5, 11.5, 11.5, 12.5],
                        "Volume": [100, 50, 60, 70, None, 90]}, index=idx)
    df = clean_frame(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert list(df.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))
    assert df.loc["2024-01-02", "close"] == 9.5                      # last duplicate wins
    row = df.loc["2024-01-05"]                                       # high < low in the source
    assert row["high"] == 11.5 and row["low"] == 11.0 and row["volume"] == 0.0
    assert df.index.unit == "ns"
    assert "Open" in raw.columns                                     # caller's frame untouched


def test_clean_frame_close_only_and_timezones():
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="h", tz="America/New_York")
    df = clean_frame(pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx))
    assert df.index.tz is None and df.index[0] == pd.Timestamp("2024-01-02 14:30")
    assert (df["open"] == df["close"]).all() and (df["volume"] == 0).all()
    with pytest.raises(ValueError):
        clean_frame(pd.DataFrame({"price": [1.0]}, index=idx[:1]))


def test_periods_per_year():
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=1000, freq="B")) == pytest.approx(261, abs=1)
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=1000, freq="D")) == pytest.approx(365.25, abs=1)
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=5000, freq="h")) == pytest.approx(8766, abs=5)


def test_csv_round_trip_with_adjusted_close(tmp_path):
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Open": [10, 11, 12, 13], "High": [11, 12, 13, 14],
                  "Low": [9, 10, 11, 12], "Close": [10, 11, 12, 13], "Adj Close": [5, 5.5, 6, 6.5],
                  "Volume": [1, 2, 3, 4]}).to_csv(tmp_path / "SPY.csv", index=False)
    pd.DataFrame({"timestamp": idx, "close": [1.0, 2.0, 3.0, 4.0]}).to_csv(tmp_path / "BTC-USD.csv", index=False)
    frames = load_csv_dir(tmp_path)
    assert set(frames) == {"SPY", "BTC-USD"}
    assert frames["SPY"]["close"].tolist() == [5.0, 5.5, 6.0, 6.5]   # split/dividend adjusted
    assert frames["SPY"]["open"].iloc[0] == 5.0 and frames["SPY"].index[0] == idx[0]
    assert load_csv_dir(tmp_path, adjust=False)["SPY"]["close"].iloc[0] == 10.0
    via_cfg = load_data({"source": "csv", "path": str(tmp_path), "symbols": ["BTC/USD"],
                         "start": "2024-01-03"})
    assert via_cfg["BTC/USD"]["close"].tolist() == [2.0, 3.0, 4.0]
    with pytest.raises(FileNotFoundError):
        load_csv_dir(tmp_path, ["MISSING"])


def test_synthetic_is_seeded_and_structured():
    a = synthetic_universe(["A", "B"], bars=300, seed=1)
    b = synthetic_universe(["A", "B"], bars=300, seed=1)
    c = synthetic_universe(["A", "B"], bars=300, seed=2)
    assert a["A"].equals(b["A"]) and not a["A"].equals(c["A"])
    df = a["B"]
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all() and (df["low"] > 0).all()
    pair = synthetic_universe(["A", "B"], bars=2000, seed=4, kind="gbm", pairs=[["A", "B"]])
    spread = np.log(pair["B"]["close"]) - 0.8 * np.log(pair["A"]["close"])
    assert spread.std() < 0.1 * np.log(pair["A"]["close"]).std() + 0.1   # tied together
    with pytest.raises(ValueError):
        synthetic_universe(["A"], kind="nope")
    with pytest.raises(ValueError):
        load_data({"source": "synthetic", "symbols": ["A"], "typo": 1})
    with pytest.raises(ValueError):
        make_provider({"source": "csv", "path": "."})


# -- network providers, against recorded-shape payloads --------------------------
class FakeResponse:
    def __init__(self, payload):
        self.payload, self.content = payload, b"x"

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads, self.calls, self.headers = list(payloads), [], {}

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return FakeResponse(self.payloads.pop(0))

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return FakeResponse(self.payloads.pop(0))


YAHOO = {"chart": {"error": None, "result": [{
    "timestamp": [1704205800, 1704292200, 1704378600],
    "indicators": {"quote": [{"open": [100.0, 102.0, None], "high": [103.0, 104.0, None],
                              "low": [99.0, 101.0, None], "close": [102.0, 103.0, None],
                              "volume": [1000, 1100, None]}],
                   "adjclose": [{"adjclose": [51.0, 51.5, None]}]}}]}}


def test_yahoo_parse_and_fetch():
    session = FakeSession(YAHOO)
    out = YahooData(session=session).fetch(["SPY"])
    df = out["SPY"]
    assert list(df.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))   # null bar dropped
    assert df["close"].tolist() == [51.0, 51.5] and df["open"].iloc[0] == pytest.approx(50.0)
    method, url, kw = session.calls[0]
    assert url.endswith("/chart/SPY") and kw["params"]["interval"] == "1d"
    with pytest.raises(RuntimeError):
        YahooData.parse({"chart": {"error": {"code": "Not Found"}, "result": None}})
    intraday = YahooData.parse(YAHOO, daily=False)
    assert intraday.index[0] == pd.Timestamp("2024-01-02 14:30")


def test_alpaca_data_paginates_and_routes_crypto():
    bar = lambda t, c: {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 10}
    session = FakeSession(
        {"bars": {"SPY": [bar("2024-01-02T05:00:00Z", 100.0)]}, "next_page_token": "abc"},
        {"bars": {"SPY": [bar("2024-01-03T05:00:00Z", 101.0)]}, "next_page_token": None},
        {"bars": {"BTC/USD": [bar("2024-01-02T00:00:00Z", 40000.0)]}})
    out = AlpacaData("k", "s", session=session).fetch(["SPY", "BTC/USD"])
    assert out["SPY"]["close"].tolist() == [100.0, 101.0]
    assert out["SPY"].index[1] == pd.Timestamp("2024-01-03")
    assert session.calls[1][2]["params"]["page_token"] == "abc"
    assert "/v2/stocks/bars" in session.calls[0][1] and "/crypto/" in session.calls[2][1]
    assert session.calls[0][2]["headers"]["APCA-API-KEY-ID"] == "k"
    with pytest.raises(RuntimeError):
        AlpacaData("k", "s", session=FakeSession({"bars": {}})).fetch(["SPY"])


class FakeExchange:
    def __init__(self):
        self.orders, self.balance = [], {"USDT": 1000.0, "BTC": 0.5, "ETH": 0.0}

    def fetch_ohlcv(self, symbol, timeframe, limit=None):
        return [[1704153600000, 1.0, 2.0, 0.5, 1.5, 10.0], [1704240000000, 1.5, 2.5, 1.0, 2.0, 11.0]]

    def fetch_balance(self):
        return {"total": dict(self.balance)}

    def create_order(self, symbol, kind, side, amount):
        self.orders.append((symbol, kind, side, amount))
        return {"id": "42", "status": "closed"}


def test_ccxt_data_and_broker():
    ex = FakeExchange()
    df = CCXTData(ex).fetch(["BTC/USDT"])["BTC/USDT"]
    assert df["close"].tolist() == [1.5, 2.0] and df.index[0] == pd.Timestamp("2024-01-02")
    broker = CCXTBroker(ex, ["BTC/USDT", "ETH/USDT"])
    broker.update_prices({"BTC/USDT": 40000.0, "ETH/USDT": 2000.0})
    acct = broker.account()
    assert acct.cash == 1000.0 and acct.equity == pytest.approx(21000.0)
    assert {s: p.qty for s, p in broker.positions().items()} == {"BTC/USDT": 0.5}
    assert broker.submit(Order("ETH/USDT", -0.25), 2000.0) == {"status": "closed", "id": "42"}
    assert ex.orders == [("ETH/USDT", "market", "sell", 0.25)]
    assert not broker.real_money and CCXTBroker(ex, ["BTC/USDT"], sandbox=False).real_money
    with pytest.raises(ValueError):
        CCXTBroker(ex, ["BTC/USDT", "ETH/BTC"])


# -- brokers ---------------------------------------------------------------------
def test_paper_broker_fills_and_persists():
    broker = PaperBroker(1000.0, CostModel(commission_bps=10, slippage_bps=10))
    status = broker.submit(Order("X", 5.0), 100.0, now="t1")
    assert status["price"] == pytest.approx(100.1) and status["commission"] == pytest.approx(0.5005, abs=1e-4)
    broker.update_prices({"X": 110.0})
    acct = broker.account()
    assert acct.cash == pytest.approx(1000.0 - 500.5 - 0.5005)
    assert acct.equity == pytest.approx(acct.cash + 550.0)
    clone = PaperBroker(1.0)
    clone.load_state(broker.state())
    clone.update_prices({"X": 110.0})
    assert clone.account().equity == pytest.approx(acct.equity)
    assert clone.positions()["X"].avg_price == pytest.approx(100.1)
    assert not broker.real_money


def test_alpaca_broker_requests():
    session = FakeSession(
        {"equity": "10500.25", "cash": "2500.00"},
        [{"symbol": "SPY", "qty": "3", "side": "long", "avg_entry_price": "400.5"},
         {"symbol": "TSLA", "qty": "2", "side": "short", "avg_entry_price": "250"},
         {"symbol": "BTCUSD", "qty": "0.1", "side": "long", "avg_entry_price": "40000"}],
        {"id": "abc", "status": "accepted"}, {"id": "def", "status": "accepted"},
        {"is_open": True})
    broker = AlpacaBroker("k", "s", symbols=["SPY", "BTC/USD"], session=session)
    acct = broker.account()
    assert acct.equity == 10500.25 and acct.cash == 2500.0
    pos = broker.positions()
    assert pos["SPY"].qty == 3 and pos["TSLA"].qty == -2 and pos["BTC/USD"].qty == 0.1
    assert pos["SPY"].avg_price == 400.5
    assert broker.submit(Order("SPY", -1.5), 400.0) == {"status": "accepted", "id": "abc"}
    broker.submit(Order("BTC/USD", 0.000123456), 40000.0)
    assert broker.market_open() is True
    stock, crypto = session.calls[2], session.calls[3]
    assert stock[0] == "POST" and stock[1] == "https://paper-api.alpaca.markets/v2/orders"
    assert stock[2]["json"] == {"symbol": "SPY", "qty": "1.5", "side": "sell", "type": "market",
                                "time_in_force": "day"}
    assert crypto[2]["json"]["qty"] == "0.000123456" and crypto[2]["json"]["time_in_force"] == "gtc"
    assert stock[2]["headers"]["APCA-API-SECRET-KEY"] == "s"
    assert not broker.real_money
    live = AlpacaBroker("k", "s", paper=False, session=FakeSession())
    assert live.real_money and live.base == "https://api.alpaca.markets"
