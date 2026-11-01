"""Progress reports: one self-contained HTML file (printable to PDF), a Markdown file and a CSV.

Validation metrics only. The report says so on its first line, so it cannot be mistaken for test results.
"""

import csv
import html
import io
import time
from pathlib import Path

from hifold.monitor.runs import METRICS, list_runs, load_programme, programme_status, seed_summary, summarize

SEALED = ("All scores are on the VALIDATION split. The test set is sealed until the pre-registered "
          "confirmatory runs are complete; no trained model has been scored on it.")


def _f(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def _dur(s):
    if s is None:
        return "–"
    h = s / 3600
    return f"{h / 24:.1f} d" if h >= 48 else f"{h:.1f} h"


def _svg_line(points, w=520, h=150, ylab=""):
    """Minimal static line chart (for the report only). points: {label: [(x, y), ...]}."""
    series = {k: [(x, y) for x, y in v if y is not None] for k, v in points.items()}
    allp = [p for v in series.values() for p in v]
    if len(allp) < 2:
        return ""
    x0, x1 = min(p[0] for p in allp), max(p[0] for p in allp)
    y0, y1 = min(p[1] for p in allp), max(p[1] for p in allp)
    if y1 == y0:
        y1 = y0 + 1e-9
    if x1 == x0:
        x1 = x0 + 1
    L, R, T, B = 46, 10, 8, 22
    sx = lambda x: L + (x - x0) / (x1 - x0) * (w - L - R)
    sy = lambda y: T + (1 - (y - y0) / (y1 - y0)) * (h - T - B)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(ylab)}">']
    for frac in (0, 0.5, 1):
        y = y0 + frac * (y1 - y0)
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>'
                   f'<text x="{L - 4}" y="{sy(y) + 3:.1f}" text-anchor="end" class="ax">{y:.3g}</text>')
    out.append(f'<text x="{L}" y="{h - 6}" class="ax">step {x0:,}</text>'
               f'<text x="{w - R}" y="{h - 6}" text-anchor="end" class="ax">{x1:,}</text>')
    for i, (k, v) in enumerate(series.items()):
        if v:
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in v)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % 3]}" stroke-width="2"/>')
    out.append("</svg>")
    legend = " ".join(f'<span class="key"><i style="background:{colors[i % 3]}"></i>{html.escape(k)}</span>'
                      for i, k in enumerate(series))
    return f'<figure>{"".join(out)}<figcaption>{html.escape(ylab)} {legend}</figcaption></figure>'


def _smooth(rows, k=10):
    out = []
    for i in range(len(rows)):
        win = [r[1] for r in rows[max(0, i - k + 1):i + 1] if r[1] is not None]
        out.append((rows[i][0], sum(win) / len(win) if win else None))
    return out


def build(runs_root, programme_path, out_dir, gpu=None):
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    runs = list_runs(runs_root, now)
    tasks, unmatched = programme_status(load_programme(programme_path), runs)
    seeds = seed_summary(runs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"report-{time.strftime('%Y%m%d-%H%M', time.localtime(now))}"

    # ---- CSV: one row per run
    buf = io.StringIO()
    cols = ["run", "name", "hash", "seed", "state", "step", "steps", "progress", "gpu_hours", "sec_per_step",
            "attempts", "best_step", *(f"best_{k}" for k in METRICS), "last_step",
            *(f"last_{k}" for k in METRICS), "device", "commit", "started", "error"]
    wr = csv.writer(buf)
    wr.writerow(cols)
    for r in runs:
        b, le = r["best"] or {}, r["last_eval"] or {}
        wr.writerow([r["run"], r["name"], r["hash"], r["seed"], r["state"], r["step"], r["steps"],
                     _f(r["progress"], 4), _f(r["gpu_hours"], 2), _f(r["sec_per_step"], 3), r["attempts"],
                     b.get("step"), *(_f(b.get(k), 4) for k in METRICS), le.get("step"),
                     *(_f(le.get(k), 4) for k in METRICS), r["device"], r["commit"], r["started"], r["error"]])
    base.with_suffix(".csv").write_text(buf.getvalue())

    # ---- Markdown
    n_exp = sum(t["expected"] or 0 for t in tasks)
    n_done = sum(min(t["finished"], t["expected"] or t["finished"]) for t in tasks)
    used = sum(r["gpu_hours"] for r in runs)
    md = [f"# HiFold-Mamba progress report, {stamp}", "", f"> {SEALED}", "",
          f"**Programme:** {n_done} of {n_exp} planned runs finished; {sum(r['state'] == 'running' for r in runs)} "
          f"running; {sum(r['state'] in ('failed', 'stalled') for r in runs)} failed or stalled. "
          f"Wall-clock GPU time used: {used:.1f} h.", ""]
    if gpu:
        md += ["**GPU:** " + "; ".join(f"{g['name']}: {g['util']}% busy, {g['mem_used']}/{g['mem_total']} MiB, "
                                       f"{g['temp']} °C" for g in gpu), ""]
    md += ["## Planned tasks", "", "| Task | Hypothesis | Finished | Running | Problems | GPU-h used / estimated |",
           "|---|---|---|---|---|---|"]
    for t in tasks:
        exp = t["expected"] if t["expected"] is not None else "?"
        cfgnote = "" if t["config"] else " (configs not written yet)"
        md.append(f"| {t['label']}{cfgnote} | {t['hypothesis']} | {t['finished']} / {exp} | {t['running']} | "
                  f"{t['problem']}{' + duplicates' if t['duplicates'] else ''} | "
                  f"{t['used_gpu_h']:.1f} / {_f(t['est_gpu_h'], 0)} |")
    md += ["", "## Runs", "", "| Run | State | Progress | Time left | Best validation SCC (HiCRep) at step | "
           "SCC (benchmark) | PCC |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        b = r["best"] or {}
        prog = f"{r['step']:,} / {r['steps']:,}" if r["steps"] else f"{r['step']:,}"
        md.append(f"| `{r['run']}` | {r['state']} | {prog} | {_dur(r['eta_s'])} | {_f(b.get('scc_hicrep'))} "
                  f"@ {b.get('step', '–')} | {_f(b.get('scc_benchmark'))} | {_f(b.get('pcc'))} |")
    if seeds:
        md += ["", "## Finished runs: mean ± SD over seeds (selected checkpoints, validation)", "",
               "| Model | Config hash | Seeds | SCC (HiCRep) | SCC (benchmark) | PCC |", "|---|---|---|---|---|---|"]
        for s in seeds:
            md.append(f"| {s['name']} | `{s['hash']}` | {','.join(map(str, s['seeds']))} | "
                      + " | ".join(f"{_f(s[k])} ± {_f(s[k + '_sd'])}" for k in METRICS) + " |")
        md += ["", "Validation means are descriptive only. The pre-registered inference (block bootstrap, "
               "intersection–union test) is run on the test set after the programme is complete."]
    problems = [r for r in runs if r["state"] in ("failed", "stalled")]
    if problems:
        md += ["", "## Problems", ""]
        for r in problems:
            md.append(f"- `{r['run']}`: {r['state']}"
                      + (f", {r['error']}" if r["error"] else f", no log line for {r['idle_min']:.0f} min"))
    if unmatched:
        md += ["", "Runs not in the programme: " + ", ".join(f"`{u}`" for u in unmatched)]
    base.with_suffix(".md").write_text("\n".join(md) + "\n")

    # ---- HTML (self-contained, printable; per-run charts)
    e = html.escape
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>HiFold progress {e(stamp)}</title>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{REPORT_CSS}</style></head><body>",
         f"<h1>HiFold-Mamba progress report</h1><p class='muted'>{e(stamp)}</p>",
         f"<p class='seal'>🔒 {e(SEALED)}</p>",
         f"<p><b>Programme:</b> {n_done} of {n_exp} planned runs finished · "
         f"{sum(r['state'] == 'running' for r in runs)} running · "
         f"{len(problems)} failed or stalled · wall-clock GPU time used {used:.1f} h</p>"]
    if gpu:
        h.append("<p><b>GPU:</b> " + "; ".join(
            e(f"{g['name']}: {g['util']}% busy, {g['mem_used']}/{g['mem_total']} MiB, {g['temp']} °C") for g in gpu)
            + "</p>")
    h.append("<h2>Planned tasks</h2><table><tr><th>Task</th><th>Hypothesis</th><th>Finished</th><th>Running</th>"
             "<th>Problems</th><th>GPU-h used / est.</th></tr>")
    for t in tasks:
        exp = t["expected"] if t["expected"] is not None else "?"
        note = "" if t["config"] else " <span class='muted'>(configs not written yet)</span>"
        h.append(f"<tr><td>{e(t['label'])}{note}</td><td>{e(str(t['hypothesis']))}</td>"
                 f"<td class='n'>{t['finished']} / {exp}</td><td class='n'>{t['running']}</td>"
                 f"<td class='n'>{t['problem']}{' + dup.' if t['duplicates'] else ''}</td>"
                 f"<td class='n'>{t['used_gpu_h']:.1f} / {_f(t['est_gpu_h'], 0)}</td></tr>")
    h.append("</table><h2>Runs</h2><table><tr><th>Run</th><th>State</th><th>Progress</th><th>Time left</th>"
             "<th>Best SCC (HiCRep)</th><th>at step</th><th>SCC (benchmark)</th><th>PCC</th></tr>")
    for r in runs:
        b = r["best"] or {}
        prog = f"{100 * r['progress']:.0f}%" if r["progress"] is not None else "–"
        h.append(f"<tr><td><code>{e(r['run'])}</code></td><td>{STATE_ICON.get(r['state'], '')} {e(r['state'])}</td>"
                 f"<td class='n'>{prog}</td><td class='n'>{_dur(r['eta_s'])}</td>"
                 f"<td class='n'>{_f(b.get('scc_hicrep'))}</td><td class='n'>{b.get('step', '–')}</td>"
                 f"<td class='n'>{_f(b.get('scc_benchmark'))}</td><td class='n'>{_f(b.get('pcc'))}</td></tr>")
    h.append("</table>")
    if seeds:
        h.append("<h2>Finished runs: mean ± SD over seeds</h2><p class='muted'>Selected checkpoints, validation "
                 "split. Descriptive only: the pre-registered inference runs on the test set after the programme."
                 "</p><table><tr><th>Model</th><th>Config hash</th><th>Seeds</th><th>SCC (HiCRep)</th>"
                 "<th>SCC (benchmark)</th><th>PCC</th></tr>")
        for s in seeds:
            h.append(f"<tr><td>{e(s['name'])}</td><td><code>{s['hash']}</code></td>"
                     f"<td>{','.join(map(str, s['seeds']))}</td>"
                     + "".join(f"<td class='n'>{_f(s[k])} ± {_f(s[k + '_sd'])}</td>" for k in METRICS) + "</tr>")
        h.append("</table>")
    if problems:
        h.append("<h2>Problems</h2><ul>")
        for r in problems:
            why = r["error"] or f"no log line for {r['idle_min']:.0f} min"
            h.append(f"<li><code>{e(r['run'])}</code>: {e(r['state'])}, {e(why)}</li>")
        h.append("</ul>")
    h.append("<h2>Run details</h2>")
    for r in runs:
        d = summarize(Path(runs_root) / r["run"], now, with_series=True)
        tr, ev = d["series"]["train"], d["series"]["eval"]
        h.append(f"<section><h3><code>{e(r['run'])}</code></h3><p class='muted'>{e(r['state'])} · "
                 f"step {r['step']:,} / {r['steps']:,} · {r['gpu_hours']:.1f} h · "
                 f"{_f(r['sec_per_step'], 2)} s/step · {e(str(r['device']))} · commit "
                 f"<code>{e(str(r['commit'])[:12])}</code> · started {e(str(r['started']))}"
                 + (f" · {r['attempts']} attempts" if r["attempts"] > 1 else "") + "</p><div class='charts'>")
        h.append(_svg_line({"training loss (moving average of 10 logs)": _smooth([(x[0], x[1]) for x in tr])},
                           ylab="Training loss"))
        h.append(_svg_line({LABEL[k]: [(x[0], x[i + 1]) for x in ev] for i, k in enumerate(METRICS)},
                           ylab="Validation (EMA weights)"))
        h.append("</div></section>")
    h.append("</body></html>")
    base.with_suffix(".html").write_text("\n".join(h))
    return {"html": base.with_suffix(".html").name, "md": base.with_suffix(".md").name,
            "csv": base.with_suffix(".csv").name}


LABEL = {"scc_hicrep": "SCC (HiCRep)", "scc_benchmark": "SCC (benchmark code)", "pcc": "PCC"}
STATE_ICON = {"running": "▶", "finished": "✔", "failed": "✖", "stalled": "⚠"}

REPORT_CSS = """
:root{--bg:#fff;--fg:#1d2329;--muted:#5b6670;--line:#d9dee3;--seal:#eef4fb}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);
max-width:1100px;margin:24px auto;padding:0 16px}
h1{margin-bottom:0}h2{margin-top:28px;border-bottom:1px solid var(--line)}
.muted{color:var(--muted)}.seal{background:var(--seal);padding:8px 12px;border-radius:6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
td.n{text-align:right}code{font-size:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
figure{margin:0}figcaption{font-size:12px;color:var(--muted)}
.key{margin-left:8px}.key i{display:inline-block;width:10px;height:3px;margin-right:4px;vertical-align:middle}
svg .grid{stroke:var(--line)}svg .ax{font-size:10px;fill:var(--muted)}
section{break-inside:avoid}
@media(max-width:700px){.charts{grid-template-columns:1fr}}
@media print{body{margin:0}}
"""
