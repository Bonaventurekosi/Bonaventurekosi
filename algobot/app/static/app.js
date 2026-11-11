'use strict';

const main = document.getElementById('main');
const state = { setups: [], id: null, setup: null, tab: 'overview', results: {}, strategies: null,
                paper: null, paperTimer: null };

// ---------- small helpers ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'value') el.value = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (method !== 'GET') {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body || {});
  }
  const res = await fetch('/api' + path, opts);
  let data = {};
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) throw new Error(data.error || 'Request failed (' + res.status + ')');
  return data;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const pct = (x, d = 2) => x == null ? 'n/a' : (x * 100).toFixed(d) + '%';
const signedPct = (x, d = 2) => x == null ? 'n/a' : (x > 0 ? '+' : '') + (x * 100).toFixed(d) + '%';
const num = (x, d = 2) => x == null ? 'n/a' : x.toFixed(d);
const whole = x => x == null ? 'n/a' : Math.round(x).toLocaleString();
const money = x => x == null ? 'n/a'
  : x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const qty = x => x.toLocaleString(undefined, { maximumFractionDigits: 6 });
const res = () => (state.results[state.id] = state.results[state.id] || {});
const slug = s => s.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);

function table(headers, rows, opts = {}) {
  const left = opts.left == null ? 1 : opts.left;
  const cls = i => (i < left ? 'left' : null);
  return h('div', { class: 'table-wrap' }, h('table', null,
    h('thead', null, h('tr', null, headers.map((t, i) => h('th', { class: cls(i), text: t })))),
    h('tbody', null, rows.map((row, r) => h('tr', { class: opts.mark && opts.mark(r) ? 'mark' : null },
      row.map((cell, i) => h('td', { class: cls(i) }, cell)))))));
}

function tiles(items) {
  return h('div', { class: 'tiles' }, items.map(([label, value, note]) => h('div', { class: 'tile' },
    h('div', { class: 'label', text: label }), h('div', { class: 'value', text: value }),
    note ? h('div', { class: 'note', text: note }) : null)));
}

function reportFrame(jobId, title) {
  const frame = h('iframe', { class: 'report', title, src: '/api/jobs/' + jobId + '/report' });
  frame.addEventListener('load', () => {
    try { frame.style.height = (frame.contentDocument.documentElement.scrollHeight + 8) + 'px'; }
    catch (e) { /* keep the default height */ }
  });
  return frame;
}

// ---------- jobs ----------
async function runJob(kind, params, slot, label) {
  const key = state.id, store = res();
  store[slot] = { status: 'running', label };
  render();
  try {
    let job = await api('POST', '/jobs', { kind, setup: key, params: params || {} });
    while (job.status === 'running') {
      await sleep(600);
      job = await api('GET', '/jobs/' + job.id);
      const el = document.getElementById('run-' + slot);
      if (el && state.id === key) el.textContent = label + '… ' + Math.round(job.seconds) + 's';
    }
    store[slot] = job.status === 'done' ? { status: 'done', job } : { status: 'error', error: job.error };
  } catch (e) {
    store[slot] = { status: 'error', error: e.message };
  }
  if (state.id === key) render();
}

const busy = slot => (res()[slot] || {}).status === 'running';

function slotView(slot, done) {
  const s = res()[slot];
  if (!s) return null;
  if (s.status === 'running') {
    return h('div', { class: 'status', role: 'status' }, h('span', { class: 'spinner' }),
      h('span', { id: 'run-' + slot, text: s.label + '…' }));
  }
  if (s.status === 'error') return h('div', { class: 'notice error', role: 'alert', text: s.error });
  return done(s.job.result, s.job);
}

// ---------- sidebar ----------
function renderSide() {
  const nav = document.getElementById('setups');
  nav.replaceChildren(...state.setups.map(s => h('a', {
    href: '#/' + s.id, class: s.ok ? null : 'bad', 'aria-current': s.id === state.id ? 'page' : null,
  }, s.name, h('small', { text: s.ok ? s.strategy.replace(/_/g, ' ') + ' · ' + s.source : 'needs fixing' }))));
}

async function loadSetups() {
  state.setups = await api('GET', '/setups');
  renderSide();
}

function setupNewForm() {
  const form = document.getElementById('new-form'), button = document.getElementById('new-setup');
  const input = document.getElementById('new-name'), error = document.getElementById('new-error');
  const toggle = show => { form.hidden = !show; button.hidden = show; error.hidden = true; if (show) input.focus(); };
  button.addEventListener('click', () => toggle(true));
  document.getElementById('new-cancel').addEventListener('click', () => toggle(false));
  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    const id = slug(input.value);
    if (!id) { error.textContent = 'Give it a name using letters or digits.'; error.hidden = false; return; }
    try {
      await api('POST', '/setups', { id });
      input.value = '';
      toggle(false);
      await loadSetups();
      location.hash = '#/' + id + '/edit';
    } catch (e) { error.textContent = e.message; error.hidden = false; }
  });
}

// ---------- routing ----------
function route() {
  const [, id, tab] = location.hash.split('/');
  if (!id) {
    const first = state.setups.find(s => s.id === 'demo' && s.ok) || state.setups.find(s => s.ok) || state.setups[0];
    if (first) location.hash = '#/' + first.id;
    else main.replaceChildren(h('p', { class: 'lead', text: 'No setups yet. Create one with “New setup”.' }));
    return;
  }
  if (id !== state.id) openSetup(id, tab);
  else { state.tab = tab || 'overview'; enterTab(); render(); }
}

async function openSetup(id, tab) {
  stopPaperPolling();
  state.id = id; state.tab = tab || 'overview'; state.setup = null; state.paper = null;
  renderSide();
  main.replaceChildren(h('p', { class: 'muted', text: 'Loading…' }));
  try { state.setup = await api('GET', '/setups/' + encodeURIComponent(id)); }
  catch (e) { state.setup = { id, text: '', error: e.message }; }
  if (state.id !== id) return;
  enterTab();
  render();
}

function tabsFor(setup) {
  const o = setup.outline;
  if (!o) return [['edit', 'Edit']];
  const tabs = [['overview', 'Overview'], ['backtest', 'Backtest'], ['compare', 'Compare strategies'],
                ['today', 'Today’s orders']];
  if (o.has_events) tabs.push(['events', 'News & earnings']);
  tabs.push(['paper', 'Paper trading'], ['edit', 'Edit']);
  return tabs;
}

function enterTab() {
  stopPaperPolling();
  if (state.tab === 'paper' && state.setup && state.setup.outline) refreshPaper(true);
  if (state.tab === 'edit' && !state.strategies) {
    api('GET', '/strategies').then(list => { state.strategies = list; if (state.tab === 'edit') render(); })
      .catch(() => {});
  }
}

// ---------- page ----------
function render() {
  const setup = state.setup;
  if (!setup) return;
  const o = setup.outline, tabs = tabsFor(setup);
  if (!tabs.some(([key]) => key === state.tab)) state.tab = tabs[0][0];
  const head = h('div', { class: 'head' },
    h('h1', { text: o ? (o.name || setup.id) : setup.id }),
    setup.blurb ? h('p', { text: setup.blurb }) : null,
    o ? h('div', { class: 'chips' },
      h('span', { class: 'chip', text: o.symbols.length + ' symbols' }),
      h('span', { class: 'chip', text: 'prices: ' + o.source }),
      h('span', { class: 'chip', text: 'strategy: ' + o.strategy.type.replace(/_/g, ' ') })) : null);
  const bar = h('div', { class: 'tabs', role: 'tablist' }, tabs.map(([key, label]) => h('button', {
    role: 'tab', 'aria-selected': String(key === state.tab), text: label,
    onclick: () => { location.hash = '#/' + state.id + '/' + key; },
  })));
  const views = { overview, backtest, compare, today, events, paper, edit };
  main.replaceChildren(head, bar, h('section', { role: 'tabpanel' }, views[state.tab](setup, o)));
}

// ---------- overview ----------
function strategyTree(node) {
  const params = Object.entries(node.params || {}).map(([k, v]) => k.replace(/_/g, ' ') + ' ' + JSON.stringify(v));
  const tags = [];
  if (node.weight != null) tags.push('weight ' + node.weight);
  if (node.regime) tags.push('only in ' + (node.regime === 'trend' ? 'trending' : 'sideways') + ' markets');
  return h('li', null, h('b', { text: node.type.replace(/_/g, ' ') }), ' — ' + node.summary,
    tags.length ? h('div', { class: 'params', text: tags.join(' · ') }) : null,
    params.length ? h('div', { class: 'params', text: params.join(', ') }) : null,
    node.children ? h('ul', { class: 'tree' }, node.children.map(strategyTree)) : null);
}

function overview(setup, o) {
  return h('div', { class: 'stack' },
    h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h2', { text: 'What it trades' }),
        h('div', { class: 'chips' }, o.symbols.slice(0, 80).map(s => h('span', { class: 'chip', text: s }))),
        h('p', { class: 'muted', style: 'margin-top:10px', text: 'Prices from ' + o.prices + '.' })),
      h('div', { class: 'card' }, h('h2', { text: 'Safety limits' }),
        h('ul', { class: 'plain' }, o.limits.map(t => h('li', { text: t }))),
        h('p', { class: 'muted', style: 'margin-top:10px', text: o.costs + '.' }))),
    h('div', { class: 'card' }, h('h2', { text: 'How it decides' }),
      h('ul', { class: 'tree' }, strategyTree(o.strategy))),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', text: 'Run a backtest',
        onclick: () => { location.hash = '#/' + state.id + '/backtest'; if (!res().backtest) startBacktest(); } }),
      h('button', { class: 'btn', text: 'See today’s orders',
        onclick: () => { location.hash = '#/' + state.id + '/today'; } })));
}

// ---------- backtest ----------
const startBacktest = () => runJob('backtest', {}, 'backtest', 'Running the backtest');

function metricTable(rows) {
  return table(['Measure', 'Value'], rows.map(([label, value]) => [label, value]));
}

function backtest(setup, o) {
  const canTest = o.has_search && o.has_walk_forward;
  return h('div', { class: 'stack' },
    h('p', { class: 'lead', text: 'Replays this setup over its price history. Orders fill at the next bar’s open '
      + 'and pay the assumed costs, so the strategy never trades at a price it has already seen.' }),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', text: 'Run backtest', disabled: busy('backtest'), onclick: startBacktest }),
      canTest ? h('button', { class: 'btn', text: 'Out-of-sample test', disabled: busy('walkforward'),
        title: 'Repeatedly tunes on past data, then trades the following period it has not seen',
        onclick: () => runJob('walkforward', {}, 'walkforward', 'Tuning and testing forward (this can take a minute)') }) : null),
    slotView('walkforward', r => h('div', { class: 'card' },
      h('h2', { text: 'Out-of-sample test, ' + r.start + ' to ' + r.end }),
      h('p', { class: 'muted', text: 'Every period below was traded with settings chosen only from earlier data. '
        + 'This is the more trustworthy number.' }),
      h('div', { class: 'cols' }, metricTable(r.rows),
        table(r.folds.columns.map(c => c.replace(/_/g, ' ')), r.folds.rows, { left: 2 })))),
    slotView('backtest', (r, job) => h('div', null,
      r.bankrupt ? h('div', { class: 'notice error', text: 'The account was wiped out in this backtest.' }) : null,
      h('p', { class: 'muted', text: '“Benchmark” in the report means simply holding every symbol in equal parts. '
        + 'Hover the charts for values.' }),
      reportFrame(job.id, 'Backtest report'))));
}

// ---------- compare ----------
function compare() {
  return h('div', { class: 'stack' },
    h('p', { class: 'lead', text: 'Runs every built-in strategy with its default settings on this setup’s symbols, '
      + 'costs and safety limits, and ranks them by return per unit of risk.' }),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', text: 'Compare all strategies',
      disabled: busy('compare'), onclick: () => runJob('tournament', {}, 'compare', 'Testing each strategy') })),
    slotView('compare', r => h('div', { class: 'card' },
      h('h2', { text: r.start + ' to ' + r.end + ' · ' + whole(r.bars) + ' bars' }),
      table(['Strategy', 'Total return', 'Per year', 'Sharpe', 'Worst drop', 'Orders'],
        r.rows.map(x => [h('span', null, x.strategy.replace(/_/g, ' '), h('small', { text: x.summary })),
          pct(x.total_return), pct(x.cagr), num(x.sharpe), pct(x.max_drawdown), whole(x.fills)]),
        { mark: i => r.rows[i].strategy === '(benchmark)' }),
      h('p', { class: 'muted', style: 'margin-top:10px',
        text: 'The highlighted row is simply holding every symbol. A strategy below it did not earn its keep here.' }))));
}

// ---------- today's orders ----------
function today(setup, o) {
  const store = res();
  const form = store.planForm = store.planForm || { cash: o.initial_cash, rows: [] };
  const rerender = () => render();
  const holdings = form.rows.map((row, i) => h('div', { class: 'row' },
    h('select', { 'aria-label': 'Symbol', onchange: ev => { row.symbol = ev.target.value; } },
      o.symbols.map(s => h('option', { value: s, selected: s === row.symbol, text: s }))),
    h('input', { type: 'number', step: 'any', 'aria-label': 'Quantity held', placeholder: 'quantity', value: row.qty,
      oninput: ev => { row.qty = ev.target.value; } }),
    h('button', { class: 'btn small', text: 'Remove', onclick: () => { form.rows.splice(i, 1); rerender(); } })));
  const submit = () => {
    const positions = {};
    for (const row of form.rows) {
      const q = parseFloat(row.qty);
      if (row.symbol && isFinite(q) && q !== 0) positions[row.symbol] = (positions[row.symbol] || 0) + q;
    }
    runJob('plan', { cash: parseFloat(form.cash) || 0, positions }, 'plan', 'Working out the orders');
  };
  return h('div', { class: 'stack' },
    h('p', { class: 'lead', text: 'Tell it what you hold and it lists the trades that would bring you to the '
      + 'strategy’s current target. Nothing is sent to any broker; place them yourself if you choose to.' }),
    h('div', { class: 'card' },
      h('label', { class: 'field' }, 'Cash available',
        h('input', { type: 'number', step: 'any', min: '0', value: form.cash, oninput: ev => { form.cash = ev.target.value; } })),
      h('h2', { style: 'margin-top:14px', text: 'What you already hold' }),
      holdings.length ? h('div', { class: 'stack', style: 'gap:6px;margin-bottom:8px' }, holdings)
        : h('p', { class: 'muted', text: 'Nothing yet (starting from cash).' }),
      h('div', { class: 'row' },
        h('button', { class: 'btn small', text: 'Add a holding',
          onclick: () => { form.rows.push({ symbol: o.symbols[0], qty: '' }); rerender(); } }),
        h('button', { class: 'btn primary', text: 'Show today’s orders', disabled: busy('plan'), onclick: submit }))),
    slotView('plan', r => h('div', { class: 'card' },
      h('h2', { text: 'As of ' + r.as_of.slice(0, 10) + ' · account value ' + money(r.equity) }),
      r.orders.length ? table(['Action', 'Symbol', 'Quantity', 'About', 'At price'],
        r.orders.map(x => [x.side === 'buy' ? 'Buy' : 'Sell', x.symbol, qty(x.qty), money(x.est_value), money(x.ref_price)]),
        { left: 2 })
        : h('p', { text: 'No orders: your holdings already match what the strategy wants.' }),
      h('h2', { style: 'margin-top:14px', text: 'Target mix' }),
      Object.keys(r.targets).length
        ? table(['Symbol', 'Share of account'], Object.entries(r.targets).sort((a, b) => b[1] - a[1])
            .map(([s, w]) => [s, pct(w, 1)]))
        : h('p', { class: 'muted', text: 'All cash right now.' }))));
}

// ---------- events ----------
function events() {
  return h('div', { class: 'stack' },
    h('p', { class: 'lead', text: 'Collects earnings releases, company announcements and headlines for this setup’s '
      + 'symbols, then measures how prices moved around them.' }),
    h('div', { class: 'row' },
      h('button', { class: 'btn', text: 'Collect latest events', disabled: busy('eventsFetch'),
        onclick: () => runJob('events_fetch', {}, 'eventsFetch', 'Collecting events (a minute or two the first time)') }),
      h('button', { class: 'btn primary', text: 'Run the study', disabled: busy('eventsStudy'),
        onclick: () => runJob('events_study', {}, 'eventsStudy', 'Measuring price reactions') })),
    slotView('eventsFetch', r => h('div', { class: 'card' },
      h('h2', { text: whole(r.total) + ' events on file (' + (r.new >= 0 ? '+' : '') + whole(r.new) + ' new)' }),
      table(['Type', 'Events'], Object.entries(r.counts).map(([k, v]) => [k.replace(/_/g, ' '), whole(v)])))),
    slotView('eventsStudy', (r, job) => reportFrame(job.id, 'Event study report')));
}

// ---------- paper trading ----------
function stopPaperPolling() {
  if (state.paperTimer) { clearInterval(state.paperTimer); state.paperTimer = null; }
}

async function refreshPaper(first) {
  const id = state.id;
  try { state.paper = await api('GET', '/paper/' + encodeURIComponent(id)); }
  catch (e) { state.paper = { running: false, error: e.message }; }
  if (state.id !== id || state.tab !== 'paper') return;
  const card = document.getElementById('live-card');
  if (card) card.replaceWith(liveCard(state.setup.outline));
  if (state.paper.running && !state.paperTimer) state.paperTimer = setInterval(refreshPaper, 4000);
  if (!state.paper.running) stopPaperPolling();
}

async function paperAction(action) {
  try { state.paper = await api('POST', '/paper/' + encodeURIComponent(state.id) + '/' + action); }
  catch (e) { state.paper = { running: false, error: e.message }; }
  refreshPaper();
}

function ordersTable(orders) {
  return table(['When', 'Action', 'Symbol', 'Quantity', 'Price'],
    orders.slice(0, 100).map(x => [x.time.slice(0, 16), x.side === 'buy' ? 'Buy' : 'Sell', x.symbol, qty(x.qty), money(x.price)]),
    { left: 3 });
}

function accountTiles(equity, startCash, cash, positions) {
  const pnl = equity == null ? null : equity - startCash;
  return tiles([
    ['Account value', money(equity), 'started with ' + money(startCash)],
    ['Profit or loss', pnl == null ? 'n/a' : (pnl >= 0 ? '+' : '') + money(pnl), pnl == null ? '' : signedPct(pnl / startCash)],
    ['Cash', money(cash), Object.keys(positions).length + ' open positions'],
  ]);
}

function liveCard(o) {
  const p = state.paper;
  const card = kids => h('div', { class: 'card', id: 'live-card' }, h('h2', { text: 'Live paper trading' }), kids);
  if (!o.live_feed) {
    return card(h('p', { class: 'muted', text: 'This setup uses ' + o.prices + ', which has no live feed. '
      + 'Use a setup with Yahoo, Alpaca or exchange prices to paper trade live, or run a rehearsal above.' }));
  }
  if (!p) return card(h('div', { class: 'status' }, h('span', { class: 'spinner' }), 'Checking…'));
  const head = h('div', { class: 'row', style: 'margin-bottom:10px' },
    h('span', { class: 'badge' + (p.running ? ' on' : ''), text: p.running ? 'Running' : 'Stopped' }),
    p.running ? h('button', { class: 'btn small', text: 'Stop', onclick: () => paperAction('stop') })
      : h('button', { class: 'btn primary small', text: 'Start paper trading', onclick: () => paperAction('start') }));
  const kids = [h('p', { class: 'lead', text: 'Trades a simulated account on real prices as each new bar completes. '
    + 'It keeps going only while this app is running, and picks up where it left off when restarted.' }), head];
  if (p.error) kids.push(h('div', { class: 'notice error', role: 'alert', text: p.error }));
  if (p.started) {
    kids.push(accountTiles(p.equity, p.start_cash, p.cash, p.positions));
    kids.push(h('p', { class: 'muted', text: 'Last checked ' + (p.last_check || 'not yet') + ' · latest completed bar '
      + (p.last_bar ? p.last_bar.slice(0, 16) : 'none yet') + ' · checks every ' + Math.round(p.poll_seconds / 60) + ' min' }));
    if (Object.keys(p.positions).length) {
      kids.push(table(['Symbol', 'Quantity'], Object.entries(p.positions).map(([s, q]) => [s, qty(q)])));
    }
    if (p.orders.length) kids.push(h('h2', { style: 'margin-top:14px', text: 'Recent orders' }), ordersTable(p.orders));
  }
  return card(kids);
}

function paper(setup, o) {
  const store = res();
  if (store.rehearseBars == null) store.rehearseBars = 60;
  return h('div', { class: 'stack' },
    h('div', { class: 'card' }, h('h2', { text: 'Rehearsal' }),
      h('p', { class: 'lead', text: 'Runs the live trading loop over the most recent bars as if they were arriving now, '
        + 'with a simulated account. A quick way to watch what the bot would have done lately.' }),
      h('div', { class: 'row' },
        h('label', { class: 'field' }, 'Bars to replay',
          h('input', { type: 'number', min: '1', max: '2000', value: store.rehearseBars,
            oninput: ev => { store.rehearseBars = ev.target.value; } })),
        h('button', { class: 'btn primary', style: 'align-self:flex-end', text: 'Run rehearsal', disabled: busy('rehearse'),
          onclick: () => runJob('rehearse', { bars: parseInt(store.rehearseBars, 10) || 60 }, 'rehearse', 'Replaying') })),
      slotView('rehearse', r => h('div', { style: 'margin-top:12px' },
        accountTiles(r.equity, r.start_cash, r.cash, r.positions),
        r.orders.length ? h('div', null, h('h2', { text: r.orders.length + ' orders over ' + r.bars + ' bars' }),
          ordersTable(r.orders.slice().reverse()))
          : h('p', { class: 'muted', text: 'No orders in this stretch.' })))),
    liveCard(o));
}

// ---------- edit ----------
function edit(setup) {
  const store = res();
  if (store.draft == null) store.draft = setup.text;
  const note = store.editNote;
  const save = async (id, create) => {
    try {
      const saved = create ? await api('POST', '/setups', { id, text: store.draft })
        : await api('PUT', '/setups/' + encodeURIComponent(id), { text: store.draft });
      await loadSetups();
      if (create) { state.results[id] = { editNote: ['ok', 'Saved as a new setup.'] }; location.hash = '#/' + id + '/edit'; return; }
      state.results[state.id] = { draft: saved.text, editNote: ['ok', 'Saved. Earlier results were cleared because the setup changed.'] };
      state.setup = saved;
    } catch (e) { store.editNote = ['error', e.message]; }
    renderSide();
    render();
  };
  const copyName = h('input', { 'aria-label': 'Name for the copy', placeholder: 'name for a copy', maxlength: '40' });
  return h('div', { class: 'stack' },
    setup.error ? h('div', { class: 'notice error', role: 'alert', text: 'This setup has a problem: ' + setup.error }) : null,
    note ? h('div', { class: 'notice ' + note[0], role: note[0] === 'error' ? 'alert' : 'status', text: note[1] }) : null,
    h('p', { class: 'lead', text: 'The setup is a plain text file. Change symbols, the strategy or the limits, then save; '
      + 'it is checked before anything is written.' }),
    h('textarea', { class: 'editor', spellcheck: 'false', 'aria-label': 'Setup file', value: store.draft,
      oninput: ev => { store.draft = ev.target.value; } }),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', text: 'Save changes', onclick: () => save(state.id, false) }),
      copyName,
      h('button', { class: 'btn', text: 'Save as a copy', onclick: () => {
        const id = slug(copyName.value);
        if (!id) { store.editNote = ['error', 'Type a name for the copy first.']; render(); return; }
        save(id, true);
      } })),
    h('details', { class: 'card' }, h('summary', { text: 'Strategy reference' }),
      state.strategies ? table(['Strategy', 'Settings and defaults'], state.strategies.map(s => [
        h('span', null, h('code', { text: s.name }), h('small', { text: s.summary })),
        h('span', { class: 'mono', style: 'white-space:normal;font-size:12px',
          text: Object.entries(s.params).map(([k, v]) => k + ': ' + JSON.stringify(v)).join(', ') })]), { left: 2 })
        : h('p', { class: 'muted', text: 'Loading…' })));
}

// ---------- start ----------
window.addEventListener('hashchange', route);
setupNewForm();
loadSetups().then(route).catch(e => {
  main.replaceChildren(h('div', { class: 'notice error', role: 'alert', text: 'Could not load setups: ' + e.message }));
});
