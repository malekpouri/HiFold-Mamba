"""Monitor dashboard: run parsing, status inference, programme matching, report, HTTP API."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
import yaml

from hifold.monitor import report
from hifold.monitor.runs import list_runs, programme_status, read_metrics, seed_summary, summarize


def _run(root, name, seed, steps_done, total=1000, status=None, idle_min=0, restart=False, h="0123456789"):
    d = root / f"{name}-{h}-s{seed}"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"name": name, "seed": seed, "steps": total,
                                                   "select_metric": "scc_hicrep"}))
    recs = [{"step": s, "loss": 1.0, "sec": s * 0.5} for s in (100, 200)] if restart else []
    for s in range(100, steps_done + 1, 100):
        recs.append({"step": s, "loss": 1.0 / s, "contact": 1.0 / s, "sec": s * 0.5})
        if s % 500 == 0:
            recs.append({"step": s, "eval": True, "scc_hicrep": s / 2000, "scc_benchmark": s / 4000, "pcc": 0.3})
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"step": 9')  # torn line
    if status:
        (d / "status.json").write_text(json.dumps(status))
    t = time.time() - idle_min * 60
    os.utime(d / "metrics.jsonl", (t, t))
    return d


def test_restart_counts_attempts_and_keeps_last(tmp_path):
    d = _run(tmp_path, "a", 0, 600, restart=True)
    train, evals, attempts, gpu_s = read_metrics(d / "metrics.jsonl")
    assert attempts == 2 and train[0]["step"] == 100 and train[-1]["step"] == 600
    assert gpu_s == pytest.approx(100 + 300)          # both attempts' wall-clock time counted


def test_state_inference(tmp_path):
    fin = summarize(_run(tmp_path, "f", 0, 1000))                       # old run, no status.json
    run = summarize(_run(tmp_path, "r", 0, 400, status={"state": "running"}))
    stl = summarize(_run(tmp_path, "s", 0, 400, status={"state": "running"}, idle_min=500))
    bad = summarize(_run(tmp_path, "x", 0, 300, status={"state": "failed", "error": "OOM"}))
    assert (fin["state"], run["state"], stl["state"], bad["state"]) == ("finished", "running", "stalled", "failed")
    assert run["sec_per_step"] == pytest.approx(0.5) and run["eta_s"] == pytest.approx(600 * 0.5)
    assert stl["eta_s"] is None and bad["error"] == "OOM"
    assert fin["best"]["step"] == 1000 and fin["best"]["scc_hicrep"] == pytest.approx(0.5)


def test_programme_matching_and_seed_summary(tmp_path):
    for s in (0, 1):
        _run(tmp_path, "hifold_paper", s, 1000)
    _run(tmp_path, "hifold_paper", 7, 1000)                             # seed outside the plan
    _run(tmp_path, "lambda_pilot_3", 0, 500, status={"state": "running"})
    _run(tmp_path, "hifold_paper", 0, 1000, h="aaaaaaaaaa")             # same name+seed, other config
    runs = list_runs(tmp_path)
    tasks, unmatched = programme_status([
        {"id": "base", "name": "hifold_paper", "seeds": [0, 1, 2], "est_gpu_h": 2},
        {"id": "lam", "name_prefix": "lambda_pilot", "expected": 9}], runs)
    base, lam = tasks
    assert (base["finished"], base["expected"], base["duplicates"], base["est_gpu_h"]) == (3, 3, 1, 6)
    assert (lam["running"], lam["expected"]) == (1, 9)
    assert unmatched == ["hifold_paper-0123456789-s7"]
    rows = {r["hash"]: r for r in seed_summary(runs)}
    assert rows["0123456789"]["n_seeds"] == 3 and rows["aaaaaaaaaa"]["scc_hicrep_sd"] is None


def test_report_files(tmp_path):
    _run(tmp_path / "runs", "hifold_paper", 0, 1000)
    _run(tmp_path / "runs", "akita_repro", 0, 300, status={"state": "failed", "error": "boom"})
    (tmp_path / "p.yaml").write_text("tasks:\n  - {id: base, label: 'Base: x', name: hifold_paper, seeds: [0]}\n")
    out = report.build(tmp_path / "runs", tmp_path / "p.yaml", tmp_path / "rep")
    html = (tmp_path / "rep" / out["html"]).read_text()
    md = (tmp_path / "rep" / out["md"]).read_text()
    assert "VALIDATION" in html and "VALIDATION" in md and "boom" in md and "<svg" in html
    assert (tmp_path / "rep" / out["csv"]).read_text().count("\n") == 3


def test_http_api(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer
    from hifold.monitor.__main__ import Handler

    _run(tmp_path / "runs", "hifold_paper", 0, 1000)
    (tmp_path / "p.yaml").write_text("tasks: [ {label: bad: yaml} ]")   # malformed: must be reported
    Handler.args = type("A", (), {"runs": str(tmp_path / "runs"), "programme": str(tmp_path / "p.yaml"),
                                  "reports": str(tmp_path / "rep")})
    monkeypatch.setenv("HIFOLD_DASH_TOKEN", "s3cret")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path, **kw):
        return urllib.request.urlopen(urllib.request.Request(base + path, **kw), timeout=10)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/api/overview")
        assert e.value.code == 401
        ov = json.load(get("/api/overview?token=s3cret"))
        assert ov["runs"][0]["state"] == "finished" and "not valid YAML" in ov["programme_error"]
        hdr = {"Cookie": "hifold_token=s3cret"}
        assert json.load(get("/api/run/hifold_paper-0123456789-s0", headers=hdr))["series"]["eval"]
        for bad in ("/api/run/..%2F..%2Fetc", "/reports/..%2Fp.yaml"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(bad, headers=hdr)
            assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:                 # cross-site POST without header
            get("/api/report", method="POST", headers=hdr)
        assert e.value.code == 403
        post = {**hdr, "X-Requested-With": "hifold"}
        with pytest.raises(urllib.error.HTTPError) as e:                 # no report from a broken plan
            get("/api/report", method="POST", headers=post)
        assert e.value.code == 500 and b"not valid YAML" in e.value.read()
        (tmp_path / "p.yaml").write_text("tasks: []\n")
        r = json.load(get("/api/report", method="POST", headers=post))
        assert get("/reports/" + r["csv"], headers=hdr).status == 200
        assert b"HiFold Monitor" in get("/", headers=hdr).read()
    finally:
        srv.shutdown()
