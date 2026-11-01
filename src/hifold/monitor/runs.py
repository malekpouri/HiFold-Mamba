"""Read-only summaries of training runs for the monitor dashboard and reports.

A run directory (written by hifold.train) holds config.yaml, provenance.json, metrics.jsonl and,
since commit-level monitoring was added, status.json. Only VALIDATION metrics are ever read here:
the test set stays sealed until the confirmatory runs are finished (protocol / OSF registration).
No torch import, so the dashboard starts instantly and uses no GPU memory.
"""

import json
import os
import re
import statistics
import time
from pathlib import Path

import yaml

RUN_RE = re.compile(r"^(?P<name>.+)-(?P<hash>[0-9a-f]{10})-s(?P<seed>\d+)$")
METRICS = ("scc_hicrep", "scc_benchmark", "pcc")
STALE_MIN = float(os.environ.get("HIFOLD_STALE_MIN", "60"))   # no log line for this long -> "stalled"


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def read_metrics(path):
    """Train and eval records of the LAST attempt (a restart appends from step 0 again).

    Returns (train_records, eval_records, n_attempts, gpu_seconds_all_attempts)."""
    attempts, cur, last_step = [], [], -1          # last_step: last TRAINING step of this attempt
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return [], [], 0, 0.0
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue          # a half-written last line while the run is writing
        if not isinstance(r, dict) or "step" not in r:
            continue
        if not r.get("eval"):
            if r["step"] <= last_step:      # training restarted from scratch (train.py appends)
                attempts.append(cur)
                cur = []
            last_step = r["step"]
        cur.append(r)
    attempts.append(cur)
    attempts = [a for a in attempts if a]
    gpu_s = sum(max((r.get("sec", 0) for r in a), default=0) for a in attempts)
    last = attempts[-1] if attempts else []
    return ([r for r in last if not r.get("eval")], [r for r in last if r.get("eval")], len(attempts), gpu_s)


def sec_per_step(train):
    """Median seconds per step over the most recent logging intervals."""
    t = train[-21:]
    d = [(b["sec"] - a["sec"]) / (b["step"] - a["step"]) for a, b in zip(t, t[1:])
         if "sec" in a and "sec" in b and b["step"] > a["step"]]
    return statistics.median(d) if d else None


def summarize(run, now=None, with_series=False):
    run = Path(run)
    now = now or time.time()
    m = RUN_RE.match(run.name)
    cfg = {}
    try:
        cfg = yaml.safe_load((run / "config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        pass
    prov = _read_json(run / "provenance.json") or {}
    status = _read_json(run / "status.json") or {}
    train, evals, attempts, gpu_s = read_metrics(run / "metrics.jsonl")
    total = int(cfg.get("steps") or status.get("steps") or 0)
    step = max([r["step"] for r in train + evals], default=0)
    sps = sec_per_step(train)
    mpath = run / "metrics.jsonl"
    last_write = mpath.stat().st_mtime if mpath.exists() else (run.stat().st_mtime if run.exists() else None)
    idle_min = (now - last_write) / 60 if last_write else None

    select = cfg.get("select_metric", "scc_hicrep")
    best = None
    for e in evals:               # same rule as train.py: first strictly greater score is kept
        if e.get(select) is not None and (best is None or e[select] > best[select]):
            best = e

    state = status.get("state")
    if state is None:             # runs from before status.json existed
        state = "finished" if total and any(e["step"] >= total for e in evals) else "running"
    if state == "running" and idle_min is not None and idle_min > STALE_MIN:
        state = "stalled"         # process probably died without writing status (kill -9, power, OOM killer)

    eta_s = (total - step) * sps if state == "running" and sps and total else None
    out = {
        "run": run.name,
        "name": m["name"] if m else cfg.get("name", run.name),
        "hash": m["hash"] if m else None,
        "seed": int(m["seed"]) if m else cfg.get("seed"),
        "model_type": cfg.get("model_type"),
        "state": state,
        "step": step,
        "steps": total,
        "progress": step / total if total else None,
        "sec_per_step": sps,
        "eta_s": eta_s,
        "gpu_hours": gpu_s / 3600,
        "attempts": attempts,
        "idle_min": idle_min,
        "last_loss": train[-1].get("loss") if train else None,
        "peak_gb": max((r.get("peak_gb", 0) for r in train), default=None),
        "struct_from": (cfg.get("struct") or {}).get("start_step"),
        "select_metric": select,
        "best": {k: best.get(k) for k in ("step", *METRICS)} if best else None,
        "last_eval": {k: evals[-1].get(k) for k in ("step", *METRICS)} if evals else None,
        "n_evals": len(evals),
        "commit": prov.get("commit"),
        "device": prov.get("device"),
        "started": prov.get("started"),
        "status_time": status.get("time"),
        "error": status.get("error"),
        "traceback": status.get("traceback"),
        "host": status.get("host"),
    }
    if with_series:
        out["series"] = {
            "train": [[r["step"], r.get("loss"), r.get("contact"), r.get("ins"), r.get("bnd"), r.get("lr")]
                      for r in train],
            "eval": [[e["step"], *(e.get(k) for k in METRICS)] for e in evals],
        }
        out["config"] = cfg
        out["provenance"] = prov
    return out


def list_runs(root, now=None):
    root = Path(root)
    if not root.is_dir():
        return []
    runs = [summarize(d, now) for d in sorted(root.iterdir()) if d.is_dir() and (d / "config.yaml").exists()]
    return sorted(runs, key=lambda r: (r["state"] != "running", r["name"], r["seed"] or 0))


def load_programme(path):
    """Planned tasks. A missing file means no programme; a malformed one raises (shown on the page)."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    try:
        return (yaml.safe_load(text) or {}).get("tasks", [])
    except yaml.YAMLError as e:
        raise ValueError(f"{path} is not valid YAML: {e}") from None


def programme_status(tasks, runs):
    """Match runs to the planned tasks. A run matches at most one task (first match wins)."""
    used, out = set(), []
    for t in tasks:
        if "name" in t:
            mine = [r for r in runs if r["name"] == t["name"] and r["seed"] in t.get("seeds", [r["seed"]])]
            expected = len(t.get("seeds", [])) or None
        else:
            mine = [r for r in runs if r["name"].startswith(t.get("name_prefix", "\0"))]
            expected = t.get("expected")
        mine = [r for r in mine if r["run"] not in used]
        used.update(r["run"] for r in mine)
        done = [r for r in mine if r["state"] == "finished"]
        est = t.get("est_gpu_h_total") or ((t.get("est_gpu_h") or 0) * (expected or 0)) or None
        names = {(r["name"], r["seed"]) for r in mine}
        out.append({
            "id": t.get("id"), "label": t.get("label"), "hypothesis": t.get("hypothesis"),
            "config": t.get("config"), "expected": expected,
            "finished": len(done), "running": sum(r["state"] == "running" for r in mine),
            "problem": sum(r["state"] in ("failed", "stalled") for r in mine),
            "duplicates": len(mine) - len(names),      # same name+seed under different config hashes
            "runs": [r["run"] for r in mine],
            "est_gpu_h": est, "used_gpu_h": sum(r["gpu_hours"] for r in mine),
        })
    unmatched = [r["run"] for r in runs if r["run"] not in used]
    return out, unmatched


def seed_summary(runs, metric="scc_hicrep"):
    """Mean and SD over seeds of the selected-checkpoint validation score, per run name + config hash."""
    groups = {}
    for r in runs:
        if r["state"] == "finished" and r["best"] and r["best"].get(metric) is not None:
            groups.setdefault((r["name"], r["hash"]), []).append(r)
    rows = []
    for (name, h), rs in sorted(groups.items()):
        row = {"name": name, "hash": h, "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs)}
        for k in METRICS:
            v = [r["best"][k] for r in rs if r["best"].get(k) is not None]
            row[k] = statistics.mean(v) if v else None
            row[k + "_sd"] = statistics.stdev(v) if len(v) > 1 else None
        rows.append(row)
    return rows
