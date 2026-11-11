# algobot

A multi-strategy algorithmic trading framework in Python. One pipeline takes you
from an idea to a backtest, a parameter search, an out-of-sample test, paper
trading and, if you choose, a live account. Seventeen strategies ship with it,
they can be blended into ensembles that adapt to market regime and recent
performance, and every stage uses the same code, so what you backtest is what
trades.

It works on stocks, ETFs and crypto, on any bar size, long-only or long/short.

> Trading involves risk of loss. Nothing here is financial advice, and a good
> backtest is not a promise. Start on paper.

## Quick start

```bash
pip install -e .            # or: pip install numpy pandas pyyaml requests

python -m algobot backtest -c configs/demo.yaml --report reports/demo.html
python -m algobot tournament -c configs/demo.yaml      # rank every strategy
python -m algobot walkforward -c configs/demo.yaml     # honest out-of-sample test
python -m algobot run -c configs/demo.yaml --replay 60 # rehearse the live loop
```

`configs/demo.yaml` runs on built-in synthetic markets, so it needs no data
vendor, account or network. The synthetic data exists to exercise the machinery;
its results say nothing about real markets.

## How it works

```
bars ──> strategy ──> sizing ──> risk ──> orders ──> broker
         target       scale to   caps,    diff vs    backtest sim,
         weights      a vol      stops,   current    paper, Alpaca
         per symbol   target     breakers positions  or CCXT
```

A strategy answers one question each bar: what fraction of capital should sit in
each symbol? Everything downstream is shared. That single contract is what makes
the strategies interchangeable and combinable.

### Strategies

| Name | Style | What it does |
|---|---|---|
| `ma_cross` | Trend | Fast vs slow moving average (EMA or SMA) |
| `macd` | Trend | Sign of the MACD histogram |
| `donchian` | Trend | Turtle-style channel breakout with a shorter exit channel |
| `supertrend` | Trend | Rides the ATR-based Supertrend line |
| `tsmom` | Trend | Sign of past returns blended over several horizons |
| `keltner_breakout` | Volatility breakout | Enters beyond an ATR band, exits at the EMA |
| `bollinger` | Mean reversion | Fades moves beyond the bands, exits near the mean |
| `rsi_reversion` | Mean reversion | Buys short-term oversold dips in an uptrend |
| `pairs` | Statistical arbitrage | Trades the spread of two co-moving symbols, hedged |
| `grid` | Market making style | Buys each step down, sells each step up |
| `dca` | Accumulation | Scheduled buying, heavier on dips |
| `xs_momentum` | Cross-sectional | Long the strongest symbols, optionally short the weakest |
| `dual_momentum` | Allocation | Strongest assets while their trend is up, else a safe asset |
| `inverse_vol` | Allocation | Naive risk parity |
| `buy_and_hold` | Baseline | Constant weights |
| `ml_logistic` | Machine learning | Online logistic regression on price features, trained walk-forward |
| `ensemble` | Meta | Blends any of the above (see below) |

`python -m algobot strategies` lists every parameter and its default.

### Ensembles

An `ensemble` runs several strategies side by side and nets their books. Capital
moves between them by `method`:

- `static` keeps fixed weights.
- `inverse_vol` gives more to the children whose returns have been calmer.
- `adaptive` gives more to the children that have been working recently.

Each child can also carry `regime: trend` or `regime: range`, in which case it
only trades a symbol while that symbol's ADX says the market is in that state.
That is how the demo runs breakout strategies in trends and mean reversion in
ranges without either fighting the other. Ensembles can be nested.

### Risk controls

Sizing can target a portfolio volatility (`sizing.target_vol`) using the recent
covariance of the holdings, under a hard leverage cap. The risk manager then has
the last word and can only reduce exposure:

| Setting | Effect |
|---|---|
| `max_position_weight`, `max_gross_leverage`, `long_only` | Hard limits on the book |
| `stop_loss_pct`, `take_profit_pct`, `trailing_stop_pct` | Per-position exits, then a cooldown |
| `max_drawdown_pct` | Flattens everything and halts, for `drawdown_halt_bars` or until you resume |
| `daily_loss_limit_pct` | Flattens for the rest of the day |

Stops are checked on bar closes and executed on the next bar, identically in
backtests and live.

## Commands

| Command | Purpose |
|---|---|
| `strategies` | List strategies and parameters |
| `backtest -c CFG [--report out.html] [--bootstrap] [--json]` | Simulate; optional HTML report and resampling check |
| `optimize -c CFG [--jobs N]` | Grid or random search over `optimize.space` |
| `walkforward -c CFG [--jobs N]` | Refit on a window, trade the next one, repeat; reports out-of-sample only |
| `tournament -c CFG` | Every built-in strategy on the same data, ranked |
| `plan -c CFG [--positions P.json]` | Print the orders the bot would place now; sends nothing |
| `run -c CFG [--once] [--dry-run] [--replay N] [--live]` | Trade on a live feed (paper unless told otherwise) |

Any value in a config can be searched by its dotted path, including ensemble
children: `strategy.params.children.0.params.entry: [20, 40, 60]`.

## Configuration

A config is one YAML (or JSON) file. Unknown keys are errors, so typos do not
silently fall back to defaults. `configs/` has a working example of each setup:

| File | Data | Shows |
|---|---|---|
| `demo.yaml` | Synthetic | Regime-gated adaptive ensemble, optimisation, walk-forward |
| `etf_trend_yahoo.yaml` | Yahoo Finance | Long-only ETF trend and momentum allocation |
| `alpaca_paper.yaml` | Alpaca | Paper trading stocks with stops and breakers |
| `crypto_ccxt.yaml` | Any CCXT exchange | 4-hour crypto grid plus breakout, sandbox trading |
| `csv_pairs.yaml` | Your CSV files | Pairs trading with borrow costs |

Data sources: `synthetic`, `csv` (one `SYMBOL.csv` per symbol with date, open,
high, low, close, volume), `yahoo`, `alpaca`, `ccxt`.

## Backtest honesty

- Orders decided on one bar fill at the **next bar's open**, moved against you by
  the slippage setting, plus commission. A strategy can never trade at a price it
  has already seen.
- Strategies only receive a history object that contains nothing after "now".
  The test suite re-runs every strategy on truncated data and requires the
  results to be identical bar for bar.
- Reported costs include commission, slippage and financing. "Round-trip
  trades" counts positions opened and later fully closed; "Orders filled" counts
  every execution, which is the number to watch for allocation strategies that
  adjust positions without ever exiting.
- A strategy holds nothing until it has enough history (up to a year for the
  slowest ones), while the benchmark is invested from the first bar. Over short
  tests that gap flatters the benchmark; `walkforward` compares warmed-up
  periods only.
- `optimize` output is fitted to the data it searched, and says so. Use
  `walkforward` for a number you can lean on, and `--bootstrap` to see how wide
  the uncertainty is.
- Not modelled: intrabar stop fills, partial fills, market impact beyond the
  fixed slippage, dividends and splits other than through adjusted prices, and
  exchange outages.
- Volatility targeting and pairs trading assume the symbols share bar
  timestamps. Mixing calendars (say, stocks and round-the-clock crypto) in one
  config will skew both.

## Trading for real

Work up in this order, and do not skip steps:

1. `backtest` and `walkforward` on real historical data for your symbols.
2. `run --replay N` to watch the live loop on recent history.
3. `run --dry-run --once` against the live feed: logs orders, sends nothing.
4. `run` with `live.paper: true` (Alpaca paper account or exchange sandbox).
5. Real money needs **both** `live.paper: false` in the config **and** `--live`
   on the command line. Either alone is refused.

Credentials come from the environment only: `ALPACA_API_KEY` /
`ALPACA_API_SECRET`, or `CCXT_API_KEY` / `CCXT_API_SECRET`.

Things to know before pointing it at an account:

- **Use a dedicated account.** The bot only trades the symbols in its config and
  ignores other holdings, but it sizes positions from total account equity and
  cancels the account's open orders before placing new ones.
- **State survives restarts.** With `live.state_path` set, the bot saves its risk
  state and a snapshot of the strategy after every traded bar. A restarted bot
  continues exactly as if it had never stopped, and never trades the same bar
  twice. A tripped drawdown breaker also survives; resume with `--reset-halt`.
  Changing the strategy config discards the snapshot and rebuilds state from
  history. Keep the state directory private: snapshots are loaded with `pickle`.
- **First start warms up on history.** Strategies are replayed over
  `live.history_bars` bars so indicators are ready. For `dca` set `start`, and
  for `grid` set `anchor_price` or `anchor_window`, so they do not treat the
  start of that history as the day they began.
- `live.max_order_value` caps any single order; `live.bar_seconds` makes the bot
  wait for a bar to finish before acting on it.

### Other brokers

`plan` is broker-neutral. Give it your holdings and it prints the trades that
would bring you to the strategy's target, without sending anything:

```bash
echo '{"cash": 5000, "positions": {"SPY": 4, "TLT": 10}}' > positions.json
python -m algobot plan -c configs/etf_trend_yahoo.yaml --positions positions.json --json
```

Place those by hand, or have another tool place them, on any broker.

## Writing a strategy

```python
from algobot.strategies import SignalStrategy, register

@register
class GapFade(SignalStrategy):
    name = "gap_fade"
    summary = "Fades large overnight gaps."
    PARAMS = {"threshold": 0.02}

    def min_bars(self):
        return 2

    def signal(self, sym, h):           # h holds bars up to now, nothing later
        gap = h.open(sym, 1)[-1] / h.close(sym, 2)[0] - 1
        return -1.0 if gap > self.threshold else 1.0 if gap < -self.threshold else 0.0
```

Import the module before loading a config and `gap_fade` is usable everywhere:
alone, inside an ensemble, in the optimiser and live. For portfolio-level logic
subclass `Strategy` and return a `{symbol: weight}` dict from `target_weights`.

## Layout

```
algobot/
  strategies/    the strategy library and registry
  indicators.py  26 causal indicators on numpy arrays
  pipeline.py    strategy -> sizing -> risk -> orders (shared by every mode)
  sizing.py, risk.py, portfolio.py
  backtest/      engine, metrics, optimiser, walk-forward, HTML report
  execution/     cost model, order generation, paper / Alpaca / CCXT brokers
  data/          dataset, synthetic markets, CSV / Yahoo / Alpaca / CCXT loaders
  live.py        polling runner with restart-safe state
  cli.py
configs/         example configurations
tests/           the test suite
```

## Tests and what they cover

```bash
pip install pytest && python -m pytest
```

The suite checks indicators against reference implementations, accounting to the
cent, order timing, every risk control, each strategy's behaviour on markets
with known structure, the no-lookahead property for all strategies, that live
decisions match backtest decisions, and that a restarted bot matches one that
never stopped.

The Yahoo data adapter has been run against the live service. The Alpaca and
CCXT adapters are tested against stand-in responses in the shape those services
document, not against the services themselves. Run them on paper and check the
first orders yourself before trusting them.

Developed and tested on Python 3.13 with pandas 3 and numpy 2.
