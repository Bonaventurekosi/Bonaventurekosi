import numpy as np
import pandas as pd
import pytest

from algobot import indicators as ind

rng = np.random.default_rng(0)
X = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
H = X * (1 + np.abs(rng.normal(0, 0.004, 400)))
L = X * (1 - np.abs(rng.normal(0, 0.004, 400)))
V = rng.uniform(1e5, 1e6, 400)
S = pd.Series(X)


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_moving_averages_match_pandas():
    close(ind.sma(X, 20), S.rolling(20).mean())
    close(ind.ema(X, 20), S.ewm(span=20, adjust=False).mean())
    close(ind.rolling_std(X, 20), S.rolling(20).std(ddof=0))
    close(ind.rolling_max(X, 20), S.rolling(20).max())
    close(ind.rolling_min(X, 20), S.rolling(20).min())
    w = np.arange(1, 11)
    close(ind.wma(X, 10), S.rolling(10).apply(lambda v: (v * w).sum() / w.sum(), raw=True))


@pytest.mark.parametrize("n", [5, 20, 100])
def test_ema_last_equals_full_ema(n):
    for size in (1, 2, n, 300):
        assert ind.ema_last(X[:size], n) == pytest.approx(ind.ema(X[:size], n)[-1], rel=1e-12)


def test_rsi_matches_wilder_reference():
    n = 14
    d = np.diff(X)
    gain, loss = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    ref = np.full(len(X), np.nan)
    ref[n] = 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
        ref[i + 1] = 100 - 100 / (1 + ag / al)
    close(ind.rsi(X, n), ref)


def test_rsi_extremes():
    assert ind.rsi(np.arange(1.0, 40.0), 14)[-1] == 100.0
    assert ind.rsi(np.arange(40.0, 1.0, -1.0), 14)[-1] == 0.0
    assert ind.rsi(np.full(40, 5.0), 14)[-1] == 50.0


def test_macd_and_bollinger_identities():
    line, sig, hist = ind.macd(X)
    close(line, ind.ema(X, 12) - ind.ema(X, 26))
    close(hist, line - sig)
    mid, up, lo = ind.bollinger(X, 20, 2.0)
    close(up - mid, mid - lo)
    close(ind.zscore(X, 20), (X - mid) / ((up - mid) / 2.0))


def test_true_range_and_atr():
    tr = ind.true_range(H, L, X)
    assert tr[0] == pytest.approx(H[0] - L[0])
    i = 57
    assert tr[i] == pytest.approx(max(H[i] - L[i], abs(H[i] - X[i - 1]), abs(L[i] - X[i - 1])))
    a = ind.atr(H, L, X, 14)
    assert np.isnan(a[12]) and a[13] == pytest.approx(tr[:14].mean())
    assert a[14] == pytest.approx((a[13] * 13 + tr[14]) / 14)


def test_adx_bounds_and_direction():
    adx, pdi, mdi = ind.adx(H, L, X, 14)
    ok = ~np.isnan(adx)
    assert ok.sum() > 300 and (adx[ok] >= 0).all() and (adx[ok] <= 100).all()
    up = np.linspace(100, 200, 120)
    adx_u, pdi_u, mdi_u = ind.adx(up * 1.001, up * 0.999, up, 14)
    assert adx_u[-1] > 60 and pdi_u[-1] > mdi_u[-1]


def test_supertrend_direction_follows_trend():
    up = np.linspace(100, 200, 150)
    assert ind.supertrend(up * 1.002, up * 0.998, up, 10, 3.0)[1][-1] == 1
    dn = up[::-1]
    assert ind.supertrend(dn * 1.002, dn * 0.998, dn, 10, 3.0)[1][-1] == -1


def test_channels_stochastic_volume():
    up, lo = ind.donchian(H, L, 20)
    assert up[-1] == H[-20:].max() and lo[-1] == L[-20:].min()
    k, d = ind.stochastic(H, L, X, 14, 3)
    assert k[-1] == pytest.approx(100 * (X[-1] - L[-14:].min()) / (H[-14:].max() - L[-14:].min()))
    assert d[-1] == pytest.approx(k[-3:].mean())
    assert ind.obv(X, V)[-1] == pytest.approx((np.sign(np.diff(X)) * V[1:]).sum())
    tp = (H + L + X) / 3
    assert ind.vwap(H, L, X, V, 10)[-1] == pytest.approx((tp[-10:] * V[-10:]).sum() / V[-10:].sum())


def test_efficiency_ratio_hurst_beta_vol():
    assert ind.efficiency_ratio(np.arange(1.0, 50.0), 10)[-1] == pytest.approx(1.0)
    zigzag = 100 + np.tile([0.0, 1.0], 30)
    assert ind.efficiency_ratio(zigzag, 10)[-1] == pytest.approx(0.0)
    walk = np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.01, 4000)))
    assert 0.4 < ind.hurst(walk) < 0.6
    y = 2.5 * X + 3.0
    assert ind.rolling_beta(y, X, 30)[-1] == pytest.approx(2.5)
    lr = np.diff(np.log(X))
    assert ind.realized_vol(X, 20)[-1] == pytest.approx(lr[-20:].std(ddof=1) * np.sqrt(252))


def test_indicators_are_causal():
    """Output at i must not change when later data is appended."""
    funcs = [lambda x: ind.sma(x, 10), lambda x: ind.ema(x, 10), lambda x: ind.rsi(x, 14),
             lambda x: ind.zscore(x, 20), lambda x: ind.macd(x)[2],
             lambda x: ind.efficiency_ratio(x, 10), lambda x: ind.roc(x, 5)]
    for f in funcs:
        close(f(X[:200]), f(X)[:200])
    close(ind.supertrend(H[:200], L[:200], X[:200])[1], ind.supertrend(H, L, X)[1][:200])
    close(ind.adx(H[:200], L[:200], X[:200])[0], ind.adx(H, L, X)[0][:200])


def test_short_inputs_do_not_crash():
    tiny = np.array([1.0, 2.0])
    for f in (lambda: ind.sma(tiny, 5), lambda: ind.rsi(tiny, 14), lambda: ind.atr(tiny, tiny, tiny, 14),
              lambda: ind.adx(tiny, tiny, tiny, 14), lambda: ind.zscore(tiny, 5),
              lambda: ind.supertrend(tiny, tiny, tiny), lambda: ind.stochastic(tiny, tiny, tiny)):
        out = f()
        arr = out[0] if isinstance(out, tuple) else out
        assert len(arr) == 2
