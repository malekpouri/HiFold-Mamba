"""Score a model or a non-learned baseline on a cached split; write one row per (window, line).

Output columns: model, seed, window, chrom, start, end, line, scc_benchmark, scc_hicrep, pcc.
These per-window files feed the block bootstrap (hifold.eval.stats).

  python scripts/evaluate.py checkpoint RUN_DIR/best.pt SPLIT OUT_TSV
  python scripts/evaluate.py global_mean CACHE_DIR SPLIT OUT_TSV      # mean training map per line

The global cell-line mean is a non-oracle reference (not in B) and the minimum bar of the §7
stopping rule: the mean of that line's training maps, the same prediction for every window.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from hifold.data.dataset import ContactCache
from hifold.eval import metrics as M


def write(rows, out_tsv, ds):
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("model\tseed\twindow\tchrom\tstart\tend\tline\tscc_benchmark\tscc_hicrep\tpcc\n")
        for model, seed, i, line, a, b, c in rows:
            ch, s, e = ds.window(i)
            fh.write(f"{model}\t{seed}\t{i}\t{ch}\t{s}\t{e}\t{line}\t{a:.6f}\t{b:.6f}\t{c:.6f}\n")
    arr = np.array([r[4:] for r in rows], float)
    print(f"{rows[0][0]}: SCC_benchmark {np.nanmean(arr[:, 0]):.4f}  SCC_hicrep {np.nanmean(arr[:, 1]):.4f}  "
          f"PCC {np.nanmean(arr[:, 2]):.4f}  ({len(rows)} window-lines) -> {out_tsv}")


def score(pred_vec, tgt_vec):
    P, T = M.triu_to_matrix(pred_vec), M.triu_to_matrix(tgt_vec)
    return M.scc_benchmark(P, T), M.scc_hicrep(P, T), M.pcc_window(pred_vec, tgt_vec)


def global_mean(cache, split, out_tsv):
    train = ContactCache(cache, "train")
    names = train.meta["lines"]
    mean = np.zeros((train.tgt.shape[1], len(names)))
    for i in range(len(train.tgt)):
        mean += np.asarray(train.tgt[i], np.float64)
    mean /= len(train.tgt)
    ds = ContactCache(cache, split)
    rows = []
    for i in range(len(ds.tgt)):
        t = np.asarray(ds.tgt[i], np.float64)
        for k, name in enumerate(names):
            rows.append(("global_mean", 0, i, name, *score(mean[:, k], t[:, k])))
    write(rows, out_tsv, ds)


@torch.no_grad()
def checkpoint(ckpt_path, split, out_tsv):
    from hifold.model.akita import Akita, AkitaConfig
    from hifold.model.hifold import HiFold, HiFoldConfig
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    akita = cfg.get("model_type") == "akita"
    model = (Akita(AkitaConfig(**cfg["model"])) if akita else HiFold(HiFoldConfig(**cfg["model"])))
    model.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    d = cfg["data"]
    ds = ContactCache(d["root"], split, d.get("lines"), d.get("cell_ids"), 0)
    rows = []
    for i in range(len(ds)):
        x, y, cells, idx = ds[i]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = model(x[None].to(dev), cells[None].to(dev)).float().cpu().numpy()[0]
        for k, line in enumerate(d.get("lines") or ds.meta["lines"]):
            rows.append((cfg["name"], cfg["seed"], int(idx), line,
                         *score(M.matrix_to_triu(pred[k]), y[:, k].numpy())))
    write(rows, out_tsv, ds)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"global_mean": global_mean, "checkpoint": checkpoint}[mode](*args)
