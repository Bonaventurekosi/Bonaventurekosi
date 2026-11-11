# algobot

A multi-strategy algorithmic trading framework in Python. One pipeline takes you
from an idea to a backtest, a parameter search, an out-of-sample test, paper
trading and, if you choose, a live account. Eighteen strategies ship with it,
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

Or use it from your browser:

```bash
python -m algobot app          # opens http://127.0.0.1:8765
```

`configs/demo.yaml` runs on built-in synthetic markets, so it needs no data
vendor, account or network. The synthetic data exists to exercise the machinery;
its results say nothing about real markets.

## The app

`python -m algobot app` starts a small web app on your own computer and opens
it in the browser. Each file in `configs/` appears as a setup, and for each one
you can:

| Tab | What it does |
|---|---|
| Overview | Explains the setup in plain words: what it trades, how it decides, its safety limits |
| Backtest | Runs the backtest and shows the full report; an out-of-sample test where the setup defines one |
| Compare strategies | Ranks every built-in strategy on the setup's symbols against simply holding them |
| Today's orders | You enter cash and holdings; it lists the trades that reach the strategy's target |
| News & earnings | Collects events and runs the event study (setups with an `events` section) |
| Paper trading | Rehearses the live loop on recent bars, or paper trades live prices in the background |
| Edit | Change the setup's file with validation before saving, or save it as a copy |

Things worth knowing:

- **It cannot place a real order.** The app only researches and paper trades.
  Real-money trading exists only on the command line, behind `--live`.
- **Live paper trading runs while the app runs.** It checks for a new completed
  bar every few minutes, trades a simulated account, and saves its state under
  `state/`, so restarting the app continues where it stopped.
- **It listens on your computer only** (127.0.0.1) and refuses requests that do
  not come from its own page, so other websites cannot drive it. `--host` and
  `--no-host-check` loosen that; use them only behind a proxy you trust.
- It needs nothing beyond the bot's own dependencies: no Node, no build step.

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
| `event_drift` | Event-driven | Trades the follow-through after earnings, filings or news |
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
| `events fetch -c CFG` | Collect earnings, filing and news events into a file |
| `events study -c CFG [--report out.html]` | Measure how prices moved around those events |
| `run -c CFG [--once] [--dry-run] [--replay N] [--live]` | Trade on a live feed (paper unless told otherwise) |
| `app [--port N] [--configs DIR]` | Open the browser app |

Any value in a config can be searched by its dotted path, including ensemble
children: `strategy.params.children.0.params.entry: [20, 40, 60]`.

## Configuration

A config is one YAML (or JSON) file. Unknown keys are errors, so typos do not
silently fall back to defaults. `configs/` has a working example of each setup:

| File | Data | Shows |
|---|---|---|
| `demo.yaml` | Synthetic | Regime-gated adaptive ensemble, optimisation, walk-forward |
| `etf_trend_yahoo.yaml` | Yahoo Finance | Long-only ETF trend and momentum allocation |
| `etf_growth_yahoo.yaml` | Yahoo Finance | Higher-return, deeper-drawdown rotation into the 4 strongest ETFs |
| `earnings_events.yaml` | Yahoo + SEC | Earnings and announcement study, and a strategy that trades it |
| `alpaca_paper.yaml` | Alpaca | Paper trading stocks with stops and breakers |
| `crypto_ccxt.yaml` | Any CCXT exchange | 4-hour crypto grid plus breakout, sandbox trading |
| `csv_pairs.yaml` | Your CSV files | Pairs trading with borrow costs |

Data sources: `synthetic`, `csv` (one `SYMBOL.csv` per symbol with date, open,
high, low, close, volume), `yahoo`, `alpaca`, `ccxt`. Add `cache: data/cache` to
a network source to reuse downloads for `cache_hours` (default 12) instead of
fetching on every command; throttled Yahoo requests are retried automatically.

## Events: earnings, filings and news

Prices move on information, so the bot can collect it and measure its effect.

```bash
python -m algobot events fetch -c configs/earnings_events.yaml
python -m algobot events study -c configs/earnings_events.yaml --report reports/events.html
```

`events fetch` gathers events into one CSV (`events.path`) from these sources:

| Source | What it gives | Cost |
|---|---|---|
| `sec` | Every 8-K filing from the SEC's EDGAR database, timestamped to the second and classified: earnings releases, executive changes, deals, restructurings and more. Goes back decades. | Free |
| `yahoo_news` | The latest headlines per symbol with a simple word-list sentiment score. Yahoo only serves the last few hours to days, so run it on a schedule to build history. | Free |
| `alpaca_news` | Headlines back to 2015, same scoring. | Free Alpaca account |

Any other feed works too: write a CSV with `symbol` and `time` columns (plus
optional `kind`, `value`, `text`, `source`) and point `events.path` at it. That
is how analyst-estimate surprises or a paid news feed plug in; put the surprise
or sentiment in `value`.

`events study` lines up each stock's return *minus the market's* around every
event and averages across events. It reports the reaction on day 0, the drift
over the following 5, 20 and 60 days, split by how the stock first reacted (and
by the event's `value` when it has one), with a table for every event type.

How to read it, and how not to:

- **Day 0 is not tradable.** It is the first session that closes after the event
  became public; by the time the event is known, that move has happened. Only
  the drift afterwards can be traded, which is what `event_drift` does.
- **Timing rule.** An event belongs to the first bar that closes at or after its
  timestamp, so a 4:30 pm earnings release counts from the next session. The
  strategy obeys the same rule and is tested for it.
- **Significance.** The t-statistics treat all events on one date as a single
  observation, so fifty headlines about one story cannot pass for fifty
  results. Windows seen on fewer than ten dates are shown as n/a.
- **Survivorship.** A universe of today's large companies outperformed the
  market by construction, which lifts every group's drift. Compare groups with
  each other (jumped versus dropped) rather than reading one row alone.

The `event_drift` strategy reads the same file. On each event it follows either
the stock's own first-day move (`signal: reaction`) or the event's `value`
(`signal: value`) for `hold` bars. It re-reads the file when it changes, so a
live bot picks up events collected by a scheduled `events fetch`.

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
  data/          dataset, synthetic markets, CSV / Yahoo / Alpaca / CCXT loaders,
                 event collectors (SEC filings, Yahoo and Alpaca headlines)
  events.py      timestamped events and the rule for when they can be acted on
  eventstudy.py  price reaction and drift around events; eventreport.py renders it
  live.py        polling runner with restart-safe state
  tasks.py       operations shared by the command line and the app
  app/           the browser app: JSON API, background paper trading, front end
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

The Yahoo price and headline adapters and the SEC filings adapter have been run
against the live services. The Alpaca and CCXT adapters are tested against
stand-in responses in the shape those services document, not against the
services themselves. Run them on paper and check the first orders yourself
before trusting them.

Developed and tested on Python 3.13 with pandas 3 and numpy 2.
