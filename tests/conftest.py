import numpy as np
import pandas as pd
import pytest

from algobot.data import Dataset, synthetic_universe
from algobot.strategies.base import Strategy


def make_bars(closes, opens=None, start="2020-01-01", freq="B", spread=0.0, volume=1e6):
    """OHLCV frame from a close path. Opens default to the previous close."""
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) * (1 + spread),
        "low": np.minimum(opens, closes) * (1 - spread),
        "close": closes,
        "volume": volume,
    }, index=idx)


class Fixed(Strategy):
    """Test double: always asks for the same weights (not registered)."""

    name = "fixed"
    PARAMS = {"weights": None}

    def target_weights(self, h):
        return dict(self.weights)


@pytest.fixture(scope="session")
def mixed():
    return Dataset(synthetic_universe(["AAA", "BBB", "CCC", "DDD"], bars=700, seed=3,
                                      pairs=[["AAA", "DDD"]]))


@pytest.fixture(scope="session")
def ranging():
    return Dataset(synthetic_universe(["R1", "R2", "R3"], bars=900, seed=11, kind="range"))


@pytest.fixture(scope="session")
def trending():
    return Dataset(synthetic_universe(["T1", "T2", "T3"], bars=1500, seed=1, kind="trend"))
