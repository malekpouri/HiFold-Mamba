"""End-to-end smoke test of the training loop on a tiny synthetic cache (CPU)."""

import json

import numpy as np
import pytest

from hifold import train as T

L = 2**15
N_TRIU = 99_681


def _fake_cache(root, n=4, lines=("A", "B")):
    rng = np.random.default_rng(0)
    iu = np.triu_indices(448, k=2)
    dist = (iu[1] - iu[0]).astype(np.float32)
    for split in ("train", "valid"):
        d = root / split
        d.mkdir(parents=True)
        seq = np.lib.format.open_memmap(d / "seq.u8.npy", mode="w+", dtype=np.uint8, shape=(n, L))
        tgt = np.lib.format.open_memmap(d / "target.f16.npy", mode="w+", dtype=np.float16, shape=(n, N_TRIU, len(lines)))
        for i in range(n):
            seq[i] = rng.integers(0, 4, L)
            for k in range(len(lines)):
                # learnable part depends on distance + cell only (tests mechanics, not memorisation);
                # small fixed noise makes every distance stratum non-constant, so SCC is defined
                tgt[i, :, k] = np.clip(-0.002 * dist + 0.3 * (k + 1) * np.sin(dist / 20)
                                       + 0.05 * rng.normal(size=dist.shape), -2, 2)
        seq.flush()
        tgt.flush()
        (d / "windows.tsv").write_text("".join(f"chr1\t{i * L}\t{(i + 1) * L}\n" for i in range(n)))
        (d / "meta.json").write_text(json.dumps({"lines": list(lines), "n": n}))


def _run(tmp_path, cell_dist_bias, steps=60):
    cfg = {
        "name": f"smoke{int(cell_dist_bias)}", "seed": 0, "device": "cpu", "run_root": str(tmp_path / "runs"),
        "model": dict(seq_len=L, tower_channels=[8, 8, 8, 8, 16, 16], d_model=16, n_blocks=3, d_state=4,
                      d_pair=8, cell_dim=4, n_cells=2, checkpoint=True, cell_dist_bias=cell_dist_bias),
        "data": {"root": str(tmp_path / "cache"), "workers": 0, "max_shift": 11},
        "batch_size": 2, "grad_accum": 1, "lr": 2e-2, "warmup": 5, "steps": steps, "ema_decay": 0.9,
        "weight_decay": 0.0, "log_every": 2, "eval_every": steps // 2,
        "struct": {"start_step": steps // 2, "ins": 0.1, "w": 10},
    }
    run, best = T.train(cfg)
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return run, best, recs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_training_loop_runs_and_learns(tmp_path):
    """Mechanics: loss falls, stage-2 insulation loss engages, eval + checkpoint + provenance are written.

    Also documents why `cell_dist_bias` exists: under an identical budget, the per-cell distance bias
    fits the cell-specific distance profile that the shared b_|i-j| cannot express.
    (Absolute thresholds are avoided: with 60 steps the cosine schedule ends before convergence.)
    """
    _fake_cache(tmp_path / "cache")
    run, best, recs = _run(tmp_path, cell_dist_bias=True)
    losses = [r["contact"] for r in recs if "contact" in r]
    final_cell = np.mean(losses[-3:])
    assert final_cell < 0.1 * np.mean(losses[:3])
    assert any("ins" in r for r in recs), "structure loss never switched on"
    evals = [r for r in recs if r.get("eval")]
    assert len(evals) == 2 and all(np.isfinite(e["scc_hicrep"]) for e in evals)
    assert (run / "best.pt").exists() and (run / "provenance.json").exists() and np.isfinite(best)

    _, _, recs_shared = _run(tmp_path, cell_dist_bias=False)
    final_shared = np.mean([r["contact"] for r in recs_shared if "contact" in r][-3:])
    assert final_cell < 0.7 * final_shared
