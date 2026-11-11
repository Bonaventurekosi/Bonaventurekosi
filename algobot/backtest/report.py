"""Self-contained HTML report for a backtest (no external assets)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import _FORMATS, drawdown, format_value

W, H = 920, 300
ML, MR, MT, MB = 64, 92, 14, 30
MAX_POINTS = 1500


def _ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw)) * mag
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + 1e-9 * span:
        out.append(round(v, 10))
        v += step
    return out


def _money_axis(hi: float):
    if hi >= 1e6:
        return lambda v: f"{v / 1e6:.1f}M"
    if hi >= 1e4:
        return lambda v: f"{v / 1e3:.0f}K"
    return lambda v: f"{v:,.0f}"


def _date_ticks(index: pd.DatetimeIndex) -> list[tuple[int, str]]:
    for freq, fmt in (("5YS", "%Y"), ("2YS", "%Y"), ("YS", "%Y"), ("QS", "%b %Y"),
                      ("MS", "%b %Y"), ("W", "%d %b"), ("D", "%d %b")):
        marks = pd.date_range(index[0], index[-1], freq=freq)
        if 3 <= len(marks) <= 9:
            break
    return [(int(index.searchsorted(m)), m.strftime(fmt)) for m in marks
            if index.searchsorted(m) < len(index)]


def _chart(cid: str, title: str, index: pd.DatetimeIndex, series: list[tuple[str, str, np.ndarray]],
           kind: str, area: bool = False) -> str:
    """One line chart as static SVG plus the data its hover layer needs."""
    n = len(index)
    lo = min(float(v.min()) for _, _, v in series)
    hi = max(float(v.max()) for _, _, v in series)
    pad = (hi - lo) * 0.06 or abs(hi) * 0.06 or 1.0
    if kind == "pct":           # drawdown: zero is the ceiling
        hi, lo = 0.0, lo - pad
        ticks = _ticks(lo, hi)
        whole = all(abs(t * 100 - round(t * 100)) < 1e-6 for t in ticks)
        label = lambda v: "0%" if abs(v) < 1e-12 else f"{v * 100:.{0 if whole else 1}f}%"
    else:
        lo, hi = lo - pad, hi + pad
        ticks = _ticks(lo, hi)
        label = _money_axis(hi)
    pw, ph = W - ML - MR, H - MT - MB
    x = lambda i: ML + pw * i / max(n - 1, 1)
    y = lambda v: MT + ph * (1.0 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" tabindex="0" aria-label="{html.escape(title)}">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{ML - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{label(t)}</text>')
    base = y(0.0) if kind == "pct" else H - MB
    parts.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{base:.1f}" y2="{base:.1f}"/>')
    for i, text in _date_ticks(index):
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{H - 8}" text-anchor="middle">{text}</text>')

    ends = []
    for name, var, values in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values.tolist()))
        if area:
            parts.append(f'<polygon class="area" style="fill:var({var})" '
                         f'points="{x(0):.1f},{base:.1f} {pts} {x(n - 1):.1f},{base:.1f}"/>')
        parts.append(f'<polyline class="line" style="stroke:var({var})" points="{pts}"/>')
        ends.append((y(float(values[-1])), var, float(values[-1])))
    # Direct end labels: skip the later series' label if it would collide.
    placed: list[float] = []
    for ey, var, value in ends:
        parts.append(f'<circle class="dot" style="fill:var({var})" cx="{x(n - 1):.1f}" cy="{ey:.1f}" r="4"/>')
        if all(abs(ey - other) >= 15 for other in placed):
            text = f"{value * 100:.1f}%" if kind == "pct" else f"{value:,.0f}"
            parts.append(f'<text class="end" x="{x(n - 1) + 9:.1f}" y="{ey + 4:.1f}">{text}</text>')
            placed.append(ey)
    parts.append(f'<line class="xh" y1="{MT}" y2="{H - MB}" visibility="hidden"/>')
    parts.append('</svg>')

    stamp = "%Y-%m-%d" if index.normalize().is_unique else "%Y-%m-%d %H:%M"
    payload = {"dates": [d.strftime(stamp) for d in index],
               "kind": kind, "lo": lo, "hi": hi,
               "geom": {"w": W, "h": H, "ml": ML, "mr": MR, "mt": MT, "mb": MB},
               "series": [{"label": name, "var": var, "values": [round(float(v), 6) for v in values]}
                          for name, var, values in series]}
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var({var})"></i>{html.escape(name)}</span>'
            for name, var, _ in series) + "</div>"
    return (f'<section class="card"><h2>{html.escape(title)}</h2>{legend}'
            f'<div class="chart" id="{cid}">{"".join(parts)}<div class="tip" hidden></div></div>'
            f'<script type="application/json" data-for="{cid}">{json.dumps(payload)}</script></section>')


def _tile(label: str, value: str, note: str = "") -> str:
    note = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="tile"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>{note}</div>')


def _table(headers: list[str], rows: list[list[str]], text: bool = False) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                   for row in rows)
    cls = ' class="text"' if text else ""
    return f"<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _when(ts) -> str:
    ts = pd.Timestamp(ts)
    return str(ts.date()) if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M")


def render_report(result, title: str = "Backtest report") -> str:
    eq, bench = result.equity, result.benchmark.reindex(result.equity.index).ffill()
    step = max(1, math.ceil(len(eq) / MAX_POINTS))
    keep = sorted(set(range(0, len(eq), step)) | {len(eq) - 1})
    idx = eq.index[keep]
    e, b = eq.to_numpy()[keep], bench.to_numpy()[keep]
    dd = drawdown(eq.to_numpy())[keep]
    m, bm = result.metrics, result.benchmark_metrics

    def fmt(key, kind, source=m):
        return format_value(source[key], kind) if key in source else "n/a"

    tiles = "".join([
        _tile("Total return", fmt("total_return", "pct"), f"benchmark {fmt('total_return', 'pct', bm)}"),
        _tile("CAGR", fmt("cagr", "pct"), f"benchmark {fmt('cagr', 'pct', bm)}"),
        _tile("Sharpe ratio", fmt("sharpe", "num"), f"benchmark {fmt('sharpe', 'num', bm)}"),
        _tile("Max drawdown", fmt("max_drawdown", "pct"), f"benchmark {fmt('max_drawdown', 'pct', bm)}"),
        _tile("Volatility", fmt("volatility", "pct"), "annualised"),
        _tile("Round-trip trades", fmt("trades", "int"), f"win rate {fmt('win_rate', 'pct')}"),
    ])
    charts = (_chart("equity", "Account value", idx,
                     [("Strategy", "--series-1", e), ("Benchmark (equal-weight buy and hold)", "--series-2", b)],
                     "money")
              + _chart("drawdown", "Strategy drawdown from peak", idx,
                       [("Strategy", "--series-1", dd)], "pct", area=True))

    stats = _table(["Metric", "Strategy", "Benchmark"],
                   [[label, format_value(m[key], kind),
                     format_value(bm[key], kind) if key in bm else ""]
                    for key, label, kind in _FORMATS if key in m])
    # Year-end to year-end, so no bar's return falls between two rows.
    ends = pd.DataFrame({"s": eq, "b": bench}).groupby(eq.index.year).last()
    starts = ends.shift(1)
    starts.iloc[0] = [eq.iloc[0], bench.iloc[0]]
    yearly = ends / starts - 1.0
    years = _table(["Year", "Strategy", "Benchmark"],
                   [[str(year), f"{row.s * 100:+.2f}%", f"{row.b * 100:+.2f}%"]
                    for year, row in yearly.iterrows()])
    monthly = pd.DataFrame({"s": eq, "b": bench, "d": drawdown(eq.to_numpy())}).groupby(
        [eq.index.year, eq.index.month]).last()
    months = _table(["Month end", "Strategy", "Benchmark", "Drawdown"],
                    [[f"{y}-{mo:02d}", f"{r.s:,.0f}", f"{r.b:,.0f}", f"{r.d * 100:.1f}%"]
                     for (y, mo), r in monthly.iterrows()])
    events = ""
    if result.risk_events:
        events = ('<section class="card"><h2>Risk events</h2>'
                  + _table(["When", "Event", "Detail"],
                           [[_when(ts), kind.replace("_", " "), detail]
                            for ts, kind, detail in result.risk_events[:200]], text=True)
                  + "</section>")
    period = f"{eq.index[0].date()} to {eq.index[-1].date()} &middot; {len(eq):,} bars"
    return (_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__PERIOD__", period)
            .replace("__TILES__", tiles).replace("__CHARTS__", charts)
            .replace("__STATS__", stats).replace("__YEARS__", years)
            .replace("__MONTHS__", months).replace("__EVENTS__", events))


def write_report(result, path: str | Path, title: str = "Backtest report") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result, title), encoding="utf-8")
    return path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 2px}
h2{font-size:14px;font-weight:600;margin:0 0 8px}
.sub{color:var(--text-secondary);margin:0 0 18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px}
.tile{padding:12px 14px}
.tile .label{color:var(--text-secondary);font-size:12px}
.tile .value{font-size:24px;font-weight:600;line-height:1.25}
.tile .note{color:var(--muted);font-size:12px}
.card{padding:14px 16px;margin-bottom:12px;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--text-secondary);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:2px;margin-right:6px;vertical-align:middle}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto;min-width:520px;outline:none}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.end{fill:var(--text-primary);font-size:11px;font-weight:600;font-variant-numeric:tabular-nums}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.area{opacity:.10}
.dot{stroke:var(--surface-1);stroke-width:2}
.xh{stroke:var(--axis);stroke-width:1}
.tip{position:absolute;top:8px;pointer-events:none;background:var(--surface-1);
border:1px solid var(--border);border-radius:6px;padding:6px 9px;font-size:12px;white-space:nowrap;
box-shadow:0 2px 8px rgba(0,0,0,.12)}
.tip .when{color:var(--text-secondary);margin-bottom:2px}
.tip .row{display:flex;align-items:center;gap:6px}
.tip .row i{width:12px;height:2px;display:inline-block}
.tip .row b{font-variant-numeric:tabular-nums}
.tip .row span{color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:5px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
table.text th,table.text td{text-align:left}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
details summary{cursor:pointer;color:var(--text-secondary)}
.foot{color:var(--muted);font-size:12px;margin-top:16px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">__PERIOD__</p>
<div class="tiles">__TILES__</div>
__CHARTS__
<div class="cols">
<section class="card"><h2>Statistics</h2>__STATS__</section>
<section class="card"><h2>Calendar-year returns</h2>__YEARS__</section>
</div>
__EVENTS__
<section class="card"><details><summary>Chart data as a table (month ends)</summary>__MONTHS__</details></section>
<p class="foot">Simulated results. Fills are modelled at the next bar's open with slippage and
commission; real trading will differ. Past performance does not predict future returns.</p>
</main>
<script>
document.querySelectorAll('script[data-for]').forEach(function (tag) {
  var d = JSON.parse(tag.textContent), box = document.getElementById(tag.dataset.for);
  var svg = box.querySelector('svg'), tip = box.querySelector('.tip'), xh = svg.querySelector('.xh');
  var g = d.geom, n = d.dates.length, cur = n - 1;
  function fmt(v) {
    return d.kind === 'pct' ? (v * 100).toFixed(2) + '%'
      : v.toLocaleString(undefined, {maximumFractionDigits: 0});
  }
  function show(i) {
    cur = Math.max(0, Math.min(n - 1, i));
    var px = g.ml + (g.w - g.ml - g.mr) * cur / Math.max(n - 1, 1);
    xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.setAttribute('visibility', 'visible');
    tip.textContent = '';
    var when = document.createElement('div');
    when.className = 'when'; when.textContent = d.dates[cur]; tip.appendChild(when);
    d.series.forEach(function (s) {
      var row = document.createElement('div'), key = document.createElement('i');
      var val = document.createElement('b'), name = document.createElement('span');
      row.className = 'row'; key.style.background = 'var(' + s.var + ')';
      val.textContent = fmt(s.values[cur]); name.textContent = s.label;
      row.appendChild(key); row.appendChild(val); row.appendChild(name); tip.appendChild(row);
    });
    tip.hidden = false;
    var r = svg.getBoundingClientRect(), left = px / g.w * r.width;
    tip.style.left = (left + 12 + tip.offsetWidth > r.width ? left - 12 - tip.offsetWidth : left + 12) + 'px';
  }
  function hide() { tip.hidden = true; xh.setAttribute('visibility', 'hidden'); }
  svg.addEventListener('pointermove', function (ev) {
    var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * g.w;
    show(Math.round((px - g.ml) / (g.w - g.ml - g.mr) * (n - 1)));
  });
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('focus', function () { show(cur); });
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', function (ev) {
    var jump = ev.shiftKey ? Math.ceil(n / 20) : 1;
    if (ev.key === 'ArrowLeft') { show(cur - jump); ev.preventDefault(); }
    if (ev.key === 'ArrowRight') { show(cur + jump); ev.preventDefault(); }
  });
});
</script></body></html>
"""
