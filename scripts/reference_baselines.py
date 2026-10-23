"""Measured-map reference baselines on a cached split (protocol §4; oracle-style, never in B).

locus_shared: prediction for line k at window i = mean over the other lines' measured maps at i.
Scores per (window, line): SCC (benchmark and HiCRep variants), per-window PCC. Writes a TSV.

Usage: python scripts/reference_baselines.py CACHE_DIR SPLIT OUT_TSV
"""

import sys
from pathlib import Path

import numpy as np

from hifold.data.dataset import ContactCache
from hifold.eval import metrics as M


def main(cache, split, out_tsv):
    ds = ContactCache(cache, split)
    names = ds.meta["lines"]
    rows = []
    for i in range(len(ds.tgt)):
        t = np.asarray(ds.tgt[i], dtype=np.float64)            # (99681, L)
        for k, name in enumerate(names):
            pred = np.delete(t, k, axis=1).mean(1)
            P, T = M.triu_to_matrix(pred), M.triu_to_matrix(t[:, k])
            rows.append((i, name, M.scc_benchmark(P, T), M.scc_hicrep(P, T), M.pcc_window(pred, t[:, k])))
        if i % 50 == 0:
            print(f"{i}/{len(ds.tgt)}", flush=True)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("window\tchrom\tstart\tend\tline\tscc_benchmark\tscc_hicrep\tpcc\n")
        for i, name, a, b, c in rows:
            ch, s, e = ds.window(i)
            fh.write(f"{i}\t{ch}\t{s}\t{e}\t{name}\t{a:.6f}\t{b:.6f}\t{c:.6f}\n")
    arr = np.array([(a, b, c) for _, _, a, b, c in rows])
    print(f"locus-shared mean, {split}: SCC_benchmark {np.nanmean(arr[:, 0]):.4f}  "
          f"SCC_hicrep {np.nanmean(arr[:, 1]):.4f}  PCC {np.nanmean(arr[:, 2]):.4f}")
    for name in names:
        sel = np.array([r[1] == name for r in rows])
        print(f"  {name:8s} SCC_bench {np.nanmean(arr[sel, 0]):.4f}  SCC_hicrep {np.nanmean(arr[sel, 1]):.4f}  "
              f"PCC {np.nanmean(arr[sel, 2]):.4f}")
    print(f"Spearman(SCC_bench, SCC_hicrep) over windows: "
          f"{__import__('scipy.stats').stats.spearmanr(arr[:, 0], arr[:, 1], nan_policy='omit')[0]:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
