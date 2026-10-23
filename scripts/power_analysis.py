"""Stage-1 power analysis for the H1 block-bootstrap test (protocol §3), from pilot reference scores.

  python scripts/power_analysis.py PILOT_TSV OUT_TSV [metric] [n_baselines] [rule]

rule = "max": Delta_b = ours_b - max_m baseline_m,b (original per-block-best rule);
rule = "iut": intersection-union, success only if the H1 rule holds against EVERY baseline separately.

PILOT_TSV is a per-(window, line) score file from scripts/evaluate.py (no trained model: the locus-shared
reference). It supplies the realistic between-block spread of window scores on the test windows. Our model
and n_baselines baselines are simulated as equicorrelated block means (correlation rho, that spread); ours
has a true advantage delta over every baseline (worst case: all baselines equally good), and Delta_b uses
the per-block maximum over the baselines, exactly as pre-registered. Success = mean Delta >= 0.01 and 97.5 % one-sided bound > 0, as pre-registered.
"""

import sys

import numpy as np
import pandas as pd

from hifold.eval.stats import block_ids

MARGIN = 0.01


def success(x, rng, n_boot):
    boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return (x.mean() >= MARGIN) and (np.percentile(boot, 2.5) > 0)


def main(pilot, out, metric="scc_hicrep", n_baselines=1, rule="max", n_sim=2000, n_boot=2000, seed=0):
    n_baselines = int(n_baselines)
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
                # shared block effect cancels in the difference; only the idiosyncratic parts matter
                e = rng.normal(0, sd * np.sqrt(1 - rho), (n_baselines + 1, n))
                if rule == "max":
                    ok += success(delta + e[0] - e[1:].max(0), rng, n_boot)
                else:
                    ok += all(success(delta + e[0] - e[k], rng, n_boot) for k in range(1, n_baselines + 1))
            rows.append(dict(rule=rule, n_baselines=n_baselines, rho=rho, sd_delta_block=round(sd_delta, 4), true_delta=delta, p_success=ok / n_sim))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).assign(n_blocks=n, sd_block=round(sd, 4), metric=metric).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
