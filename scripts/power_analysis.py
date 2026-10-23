"""Stage-1 power analysis for the H1 block-bootstrap test (protocol §3), from pilot reference scores.

  python scripts/power_analysis.py PILOT_TSV OUT_TSV [metric]

PILOT_TSV is a per-(window, line) score file from scripts/evaluate.py (no trained model: the locus-shared
reference). It supplies the realistic between-block spread of window scores on the test windows. Two
paired models are simulated as block means with that spread, a between-model correlation rho and a true
difference delta. Success = mean Delta >= 0.01 and 97.5 % one-sided bound > 0, as pre-registered.
"""

import sys

import numpy as np
import pandas as pd

from hifold.eval.stats import block_ids

MARGIN = 0.01


def main(pilot, out, metric="scc_hicrep", n_sim=2000, n_boot=2000, seed=0):
    d = pd.read_csv(pilot, sep="\t")
    w = d.groupby(["window", "chrom", "start", "end"], as_index=False)[metric].mean()   # mean over lines
    blocks = block_ids(list(zip(w.chrom, w.start, w.end)))
    bm = w.groupby(blocks)[metric].mean().to_numpy()
    n, sd = len(bm), bm.std(ddof=1)
    print(f"{len(w)} windows -> {n} blocks of 1 Mb; between-block SD of {metric} = {sd:.4f}")
    rng = np.random.default_rng(seed)
    rows = []
    for rho in (0.8, 0.9, 0.95, 0.98):
        sd_delta = sd * np.sqrt(2 * (1 - rho))
        for delta in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
            ok = 0
            for _ in range(n_sim):
                x = rng.normal(delta, sd_delta, n)
                boot = x[rng.integers(0, n, (n_boot, n))].mean(1)
                ok += (x.mean() >= MARGIN) and (np.percentile(boot, 2.5) > 0)
            rows.append(dict(rho=rho, sd_delta_block=round(sd_delta, 4), true_delta=delta, p_success=ok / n_sim))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).assign(n_blocks=n, sd_block=round(sd, 4), metric=metric).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
