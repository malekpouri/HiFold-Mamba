"""HiFold-Mamba monitor: a read-only web dashboard for training runs, plus report generation.

  python -m hifold.monitor --runs /work/runs --programme configs/programme.yaml --reports /work/reports

Standard library only (plus PyYAML). It never starts, stops or changes runs, and never reads test-set
results. Set HIFOLD_DASH_TOKEN to require a token (open the page once as /?token=<token>).
"""

import argparse
import hmac
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hifold.monitor import report
from hifold.monitor.report import SEALED
from hifold.monitor.runs import list_runs, load_programme, programme_status, seed_summary, summarize

STATIC = Path(__file__).parent / "static"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
_gpu_cache = {"t": 0.0, "v": None}
_gpu_lock = threading.Lock()


def gpu_status():
    """nvidia-smi summary, cached for 5 s. None if no GPU is visible."""
    with _gpu_lock:
        if time.time() - _gpu_cache["t"] < 5:
            return _gpu_cache["v"]
        q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            v = []
            for line in out.strip().splitlines():
                f = [x.strip() for x in line.split(",")]
                num = lambda s: float(s) if re.fullmatch(r"[\d.]+", s) else None
                v.append({"name": f[0], "util": num(f[1]), "mem_used": num(f[2]), "mem_total": num(f[3]),
                          "temp": num(f[4]), "power": num(f[5]), "power_limit": num(f[6])})
            v = v or None
        except (OSError, subprocess.SubprocessError, IndexError):
            v = None
        _gpu_cache.update(t=time.time(), v=v)
        return v


class Handler(BaseHTTPRequestHandler):
    server_version = "hifold-monitor"
    args = None

    def log_message(self, fmt, *a):      # quiet: only errors
        pass

    # -- helpers
    def _send(self, code, body, ctype="application/json", extra=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False, default=str)
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self, query):
        token = os.environ.get("HIFOLD_DASH_TOKEN")
        if not token:
            return True, ()
        given = (query.get("token") or [None])[0]
        if given and hmac.compare_digest(given, token):
            return True, (("Set-Cookie", f"hifold_token={token}; HttpOnly; SameSite=Strict; Path=/"),)
        c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        ok = "hifold_token" in c and hmac.compare_digest(c["hifold_token"].value, token)
        return ok, ()

    def _route(self, method):
        u = urlparse(self.path)
        ok, extra = self._authorized(parse_qs(u.query))
        if not ok:
            return self._send(401, "Unauthorized: open /?token=<token> once.", "text/plain; charset=utf-8")
        a, p = self.args, u.path
        try:
            if method == "GET" and p == "/":
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8", extra)
            if method == "GET" and p == "/api/overview":
                runs = list_runs(a.runs)
                try:
                    prog, prog_err = load_programme(a.programme), None
                except ValueError as e:
                    prog, prog_err = [], str(e)
                tasks, unmatched = programme_status(prog, runs)
                du = shutil.disk_usage(a.runs if Path(a.runs).exists() else "/")
                return self._send(200, _clean({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename, "sealed": SEALED,
                    "runs": runs, "tasks": tasks, "unmatched": unmatched, "seeds": seed_summary(runs),
                    "gpu": gpu_status(), "disk": {"free_gb": du.free / 2**30, "total_gb": du.total / 2**30},
                    "runs_root": str(a.runs), "programme_error": prog_err}))
            if method == "GET" and p.startswith("/api/run/"):
                name = p.rsplit("/", 1)[1]
                d = Path(a.runs) / name
                if not SAFE.match(name) or not (d / "config.yaml").exists():
                    return self._send(404, {"error": "no such run"})
                return self._send(200, _clean(summarize(d, with_series=True)))
            if method == "GET" and p == "/api/reports":
                rd = Path(a.reports)
                files = sorted(rd.glob("report-*.*"), reverse=True) if rd.is_dir() else []
                return self._send(200, [{"name": f.name, "size": f.stat().st_size} for f in files])
            if method == "POST" and p == "/api/report":
                if self.headers.get("X-Requested-With") != "hifold":     # blocks cross-site form posts
                    return self._send(403, {"error": "forbidden"})
                return self._send(200, report.build(a.runs, a.programme, a.reports, gpu_status()))
            if method == "GET" and p.startswith("/reports/"):
                name = p.rsplit("/", 1)[1]
                f = Path(a.reports) / name
                if not SAFE.match(name) or not f.is_file():
                    return self._send(404, "not found", "text/plain")
                ctype = {".html": "text/html; charset=utf-8", ".md": "text/markdown; charset=utf-8",
                         ".csv": "text/csv; charset=utf-8"}.get(f.suffix, "application/octet-stream")
                disp = () if f.suffix == ".html" else (("Content-Disposition", f'attachment; filename="{name}"'),)
                return self._send(200, f.read_bytes(), ctype, disp)
            return self._send(404, {"error": "not found"})
        except Exception as e:             # keep the dashboard alive; show the reason
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def _clean(o):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--programme", default="configs/programme.yaml")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    Handler.args = ap.parse_args()
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # docker stop: exit at once
    srv = ThreadingHTTPServer((Handler.args.host, Handler.args.port), Handler)
    print(f"HiFold monitor on http://{Handler.args.host}:{Handler.args.port}  runs={Handler.args.runs}  "
          f"token={'on' if os.environ.get('HIFOLD_DASH_TOKEN') else 'off'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
