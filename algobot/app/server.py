"""The app's web server: a small JSON API over the bot plus the browser front end.

Standard library only. It listens on this computer alone by default and has no
way to place a real order: every action is research or paper trading.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import yaml

from .. import tasks
from ..backtest.metrics import metric_rows
from ..backtest.optimize import walk_forward
from ..backtest.report import render_report
from ..config import check_config, load_dataset, run_backtest
from ..eventreport import render_event_report
from ..strategies import available
from .paper import PaperSession

STATIC = Path(__file__).parent / "static"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".js": "text/javascript; charset=utf-8"}
MAX_BODY = 1_000_000

STARTER = """\
# A starting point: edit the symbols and the strategy, then press Save.
name: My setup

data:
  source: yahoo              # free daily prices
  cache: data/cache
  symbols: [SPY, QQQ, TLT, GLD]
  range: 10y
  interval: 1d

strategy:
  type: tsmom                # see the strategy reference below for the others
  params:
    allow_short: false

sizing:
  target_vol: 0.10
  max_leverage: 1.0

risk:
  long_only: true
  max_position_weight: 0.40
  max_drawdown_pct: 0.20
  drawdown_halt_bars: 40

costs:
  commission_bps: 0
  slippage_bps: 3
"""


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def to_json(value: Any) -> Any:
    """Make results JSON-safe: plain types only, and no NaN or infinity."""
    if isinstance(value, Mapping):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass
class Job:
    id: str
    kind: str
    setup: str
    status: str = "running"
    result: Any = None
    report: str | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "setup": self.setup, "status": self.status,
                "result": self.result, "error": self.error, "has_report": self.report is not None,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


class App:
    """Everything the server does, independent of HTTP (and directly testable)."""

    def __init__(self, configs: str | Path = "configs", state: str | Path = "state",
                 workers: int = 2):
        self.configs = Path(configs)
        self.configs.mkdir(parents=True, exist_ok=True)
        self.state = Path(state)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._paper: dict[str, PaperSession] = {}
        self._lock = threading.Lock()

    # -- setups --------------------------------------------------------------
    def _path(self, setup: str) -> Path:
        if not _ID.match(setup or ""):
            raise ApiError(400, "setup names use letters, digits, - and _ only")
        return self.configs / f"{setup}.yaml"

    def _read(self, setup: str) -> tuple[str, dict[str, Any]]:
        path = self._path(setup)
        if not path.exists():
            raise ApiError(404, f"no setup called {setup!r}")
        text = path.read_text(encoding="utf-8")
        try:
            cfg = yaml.safe_load(text)
            check_config(cfg)
        except (yaml.YAMLError, ValueError, TypeError, AttributeError) as exc:
            raise ApiError(422, f"{setup}: {exc}")
        return text, cfg

    @staticmethod
    def _blurb(text: str) -> str:
        """The setup's own leading comment, minus command examples."""
        lines = []
        for line in text.splitlines():
            if not line.startswith("#"):
                break
            body = line.lstrip("#").rstrip()
            if body.startswith("  ") or "python -m" in body or not body.strip():
                if lines:
                    break
                continue
            lines.append(body.strip())
        return " ".join(lines)

    def list_setups(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.configs.glob("*.yaml")):
            setup = path.stem
            if not _ID.match(setup):
                continue
            item: dict[str, Any] = {"id": setup, "name": setup, "ok": True}
            try:
                text, cfg = self._read(setup)
                outline = tasks.describe(cfg)
                item.update(name=outline["name"] or setup, blurb=self._blurb(text),
                            source=outline["source"], symbols=len(outline["symbols"]),
                            strategy=outline["strategy"]["type"])
            except ApiError as exc:
                item.update(ok=False, error=exc.message)
            out.append(item)
        # The demo first (it needs nothing to run), then alphabetically by name.
        out.sort(key=lambda item: (item["id"] != "demo", not item["ok"], str(item["name"]).lower()))
        return out

    def read_setup(self, setup: str) -> dict[str, Any]:
        path = self._path(setup)
        if not path.exists():
            raise ApiError(404, f"no setup called {setup!r}")
        text = path.read_text(encoding="utf-8")
        out: dict[str, Any] = {"id": setup, "text": text, "blurb": self._blurb(text)}
        try:
            out["outline"] = tasks.describe(self._read(setup)[1])
        except ApiError as exc:
            out["error"] = exc.message
        return out

    def save_setup(self, setup: str, text: str, create: bool = False) -> dict[str, Any]:
        path = self._path(setup)
        if create and path.exists():
            raise ApiError(409, f"a setup called {setup!r} already exists")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "the setup is empty")
        try:
            cfg = yaml.safe_load(text)
            check_config(cfg)
        except (yaml.YAMLError, ValueError, TypeError, AttributeError) as exc:
            raise ApiError(422, str(exc))
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return self.read_setup(setup)

    @staticmethod
    def strategies() -> list[dict[str, Any]]:
        out = []
        for name, cls in available().items():
            params: dict[str, Any] = {}
            for klass in reversed(cls.__mro__):
                params.update(vars(klass).get("PARAMS", {}))
            params.pop("children", None)
            out.append({"name": name, "summary": cls.summary, "params": to_json(params)})
        return out

    # -- jobs ----------------------------------------------------------------
    def submit(self, kind: str, setup: str, params: Mapping[str, Any] | None = None) -> Job:
        runner = getattr(self, f"_job_{kind}", None) if re.fullmatch(r"[a-z_]+", kind or "") else None
        if runner is None:
            raise ApiError(400, f"unknown job {kind!r}")
        _, cfg = self._read(setup)
        job = Job(uuid.uuid4().hex[:12], kind, setup)
        with self._lock:
            self._jobs[job.id] = job
            for old in list(self._jobs)[:-40]:          # keep memory bounded
                if self._jobs[old].status != "running":
                    del self._jobs[old]
        self._pool.submit(self._run, job, runner, cfg, dict(params or {}))
        return job

    def _run(self, job: Job, runner: Callable, cfg: dict, params: dict) -> None:
        try:
            result = runner(cfg, params)
            if isinstance(result, tuple):
                result, job.report = result
            job.result, job.status = to_json(result), "done"
        except Exception as exc:        # shown to the user as the job's error
            job.error, job.status = f"{exc}" or type(exc).__name__, "error"
        job.finished = time.time()

    def job(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ApiError(404, "that result is no longer available; run it again")
        return job

    @staticmethod
    def _job_backtest(cfg, params):
        result = run_backtest(cfg)
        m, bm = result.metrics, result.benchmark_metrics
        return ({"metrics": m, "benchmark": bm, "rows": metric_rows(m),
                 "start": str(result.equity.index[0].date()), "end": str(result.equity.index[-1].date()),
                 "bars": len(result.equity), "risk_events": len(result.risk_events),
                 "bankrupt": result.bankrupt},
                render_report(result, str(cfg.get("name", "Backtest"))))

    @staticmethod
    def _job_walkforward(cfg, params):
        opt, wf = dict(cfg.get("optimize") or {}), dict(cfg.get("walk_forward") or {})
        if not opt.get("space") or "train_bars" not in wf or "test_bars" not in wf:
            raise ValueError("this setup has no optimize.space and walk_forward settings to test with")
        space = opt.pop("space")
        result = walk_forward(cfg, load_dataset(cfg), space, **opt, **wf)
        folds = result.folds.copy()
        folds.columns = [c.split(".")[-1] if c.startswith(("strategy.", "risk.", "sizing.")) else c
                         for c in folds.columns]
        return {"metrics": result.metrics, "rows": metric_rows(result.metrics),
                "folds": {"columns": list(folds.columns),
                          "rows": [[str(v) for v in row] for row in folds.to_numpy().tolist()]},
                "start": str(result.equity.index[0].date()), "end": str(result.equity.index[-1].date())}

    @staticmethod
    def _job_tournament(cfg, params):
        return tasks.tournament(cfg)

    @staticmethod
    def _job_plan(cfg, params):
        return tasks.make_plan(cfg, params.get("cash"), params.get("positions") or {})

    @staticmethod
    def _job_rehearse(cfg, params):
        return tasks.rehearse(cfg, int(params.get("bars", 60)))

    @staticmethod
    def _job_events_fetch(cfg, params):
        return tasks.collect_events(cfg)

    @staticmethod
    def _job_events_study(cfg, params):
        study, opts = tasks.study_events(cfg, params.get("kind"))
        sessions = study.events["session"]
        return ({"events": len(study), "symbols": int(study.events["symbol"].nunique()),
                 "start": str(sessions.min().date()), "end": str(sessions.max().date()),
                 "market": study.benchmark, "normal_move": study.normal_abs_move,
                 "counts": study.events["kind"].value_counts().to_dict()},
                render_event_report(study, str(cfg.get("name", "Event study")),
                                    opts["kind"], opts["threshold"]))

    # -- paper trading ---------------------------------------------------------
    def paper_status(self, setup: str) -> dict[str, Any]:
        self._path(setup)
        session = self._paper.get(setup)
        return to_json(session.status()) if session else {"setup": setup, "running": False}

    def paper_start(self, setup: str, provider: Any = None,
                    poll_seconds: float | None = None) -> dict[str, Any]:
        _, cfg = self._read(setup)
        with self._lock:
            current = self._paper.get(setup)
            if current is not None and current.running:
                return to_json(current.status())
            if provider is None and not tasks.describe(cfg)["live_feed"]:
                raise ApiError(422, "this setup's data has no live feed; use Yahoo, Alpaca or an "
                                    "exchange as the data source, or run a rehearsal instead")
            try:
                session = PaperSession(setup, cfg, self.state, provider, poll_seconds)
            except (ValueError, RuntimeError, ImportError) as exc:
                raise ApiError(422, str(exc))
            self._paper[setup] = session
            session.start()
        return to_json(session.status())

    def paper_stop(self, setup: str) -> dict[str, Any]:
        session = self._paper.get(setup)
        if session is not None:
            session.stop()
        return self.paper_status(setup)

    def close(self) -> None:
        for session in self._paper.values():
            session.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    app: App
    check_host = True
    server_version = "algobot"

    def log_message(self, *args) -> None:      # keep the terminal quiet
        pass

    # -- plumbing ------------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(to_json(payload)).encode(), "application/json")

    def _guard(self, writing: bool) -> None:
        """Refuse requests that did not come from this app's own page.

        Without this, any website open in the same browser could talk to the
        app behind the user's back (DNS rebinding, cross-site requests).
        """
        if not self.check_host:
            return
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ApiError(403, "this app only answers on localhost")
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            raise ApiError(403, "cross-site request refused")
        if writing and not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise ApiError(415, "send JSON")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ApiError(400, "invalid JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "send a JSON object")
        return data

    def _handle(self, method: str) -> None:
        try:
            self._guard(writing=method != "GET")
            path = urlsplit(self.path).path.rstrip("/") or "/"
            parts = [p for p in path.split("/") if p]
            if method == "GET" and (path == "/" or parts[:1] == ["static"]):
                return self._static("index.html" if path == "/" else "/".join(parts[1:]))
            if parts[:1] != ["api"]:
                raise ApiError(404, "not found")
            self._api(method, parts[1:])
        except ApiError as exc:
            self._json({"error": exc.message}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:      # never leak a traceback to the browser
            self._json({"error": f"unexpected error: {exc}"}, 500)

    def _static(self, name: str) -> None:
        target = (STATIC / name).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file() \
                or target.suffix not in _TYPES:
            raise ApiError(404, "not found")
        self._send(200, target.read_bytes(), _TYPES[target.suffix])

    def _api(self, method: str, parts: list[str]) -> None:
        app = self.app
        match (method, *parts):
            case ("GET", "setups"):
                self._json(app.list_setups())
            case ("POST", "setups"):
                body = self._body()
                self._json(app.save_setup(str(body.get("id", "")), body.get("text") or STARTER,
                                          create=True), 201)
            case ("GET", "setups", setup):
                self._json(app.read_setup(setup))
            case ("PUT", "setups", setup):
                self._json(app.save_setup(setup, self._body().get("text", "")))
            case ("GET", "strategies"):
                self._json(app.strategies())
            case ("POST", "jobs"):
                body = self._body()
                job = app.submit(str(body.get("kind", "")), str(body.get("setup", "")),
                                 body.get("params") if isinstance(body.get("params"), dict) else {})
                self._json(job.public(), 202)
            case ("GET", "jobs", job_id):
                self._json(app.job(job_id).public())
            case ("GET", "jobs", job_id, "report"):
                job = app.job(job_id)
                if job.report is None:
                    raise ApiError(404, "this result has no report")
                self._send(200, job.report.encode(), _TYPES[".html"])
            case ("GET", "paper", setup):
                self._json(app.paper_status(setup))
            case ("POST", "paper", setup, "start"):
                self._body()
                self._json(app.paper_start(setup))
            case ("POST", "paper", setup, "stop"):
                self._body()
                self._json(app.paper_stop(setup))
            case _:
                raise ApiError(404, "not found")

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PUT(self) -> None:
        self._handle("PUT")


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765,
                check_host: bool = True) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app, "check_host": check_host})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(configs: str | Path = "configs", state: str | Path = "state", host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True, check_host: bool = True) -> None:
    app = App(configs, state)
    server = make_server(app, host, port, check_host)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{server.server_address[1]}/"
    print(f"algobot app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        app.close()
        server.server_close()
