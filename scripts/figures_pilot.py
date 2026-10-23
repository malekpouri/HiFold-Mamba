"""Stage-1 pilot figure (paper Fig. 2) from logged reference scores and the power simulation.

  python scripts/figures_pilot.py OUT_PREFIX
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LINES = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
# DNALongBench (Cheng et al. 2025, Nat Commun 16:10108), Table 4, Expert Model (Akita), SCC
AKITA_PUBLISHED = {"HFF": 0.258, "H1hESC": 0.247, "GM12878": 0.227, "IMR90": 0.210, "HCT116": 0.210}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]   # ordinal: rho 0.8 -> 0.98
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def main(out):
    ls = pd.read_csv("results/reference/locus_shared_test.tsv", sep="\t")
    gm = pd.read_csv("results/reference/global_mean_test.tsv", sep="\t")
    pw = pd.read_csv("results/pilot/power_h1.tsv", sep="\t")
    iut = pd.read_csv("results/pilot/power_h1_iut3.tsv", sep="\t")
    mx = pd.read_csv("results/pilot/power_h1_max3.tsv", sep="\t")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))

    a = ax[0]
    x = np.arange(len(LINES))
    series = [("Locus-shared mean (measured maps; reference)", ls.groupby("line").scc_benchmark.mean(), BLUE),
              ("Akita, published (DNALongBench)", pd.Series(AKITA_PUBLISHED), ORANGE),
              ("Global cell-line mean", gm.groupby("line").scc_benchmark.mean(), AQUA)]
    for k, (lab, s, c) in enumerate(series):
        a.bar(x + (k - 1) * 0.26, [s[l] for l in LINES], 0.24, color=c, label=lab, edgecolor="white", linewidth=0.8)
    a.set_xticks(x, ["HFF", "H1", "GM12878", "IMR-90", "HCT116"], rotation=35, ha="right", fontsize=7)
    a.set_ylabel("SCC (benchmark code)")
    a.grid(axis="y", color=GRID, linewidth=0.6); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    a.set_title("a  Test-set references", loc="left", fontsize=8.5, color=INK)

    b = ax[1]
    rho = spearmanr(ls.scc_benchmark, ls.scc_hicrep, nan_policy="omit").statistic
    b.scatter(ls.scc_hicrep, ls.scc_benchmark, s=4, color=BLUE, alpha=0.35, linewidths=0)
    b.set_xlabel("SCC (HiCRep weights)"); b.set_ylabel("SCC (benchmark code)")
    b.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\nn = {len(ls)} window–lines", transform=b.transAxes,
           va="top", fontsize=6.5, color=INK)
    b.grid(color=GRID, linewidth=0.6); b.set_axisbelow(True)
    b.set_title("b  Two SCC weightings disagree", loc="left", fontsize=8.5, color=INK)

    c = ax[2]
    r = 0.95
    for lab, d, col in [("one baseline", pw, RAMP[0]), ("each of 3 (IUT; adopted)", iut, RAMP[3]),
                        ("per-block max of 3", mx, ORANGE)]:
        t = d[d.rho == r]
        c.plot(t.true_delta, t.p_success, color=col, lw=2, marker="o", ms=3.5, label=lab)
    c.axhline(0.9, color=MUTED, lw=0.8, ls="--"); c.axvline(0.01, color=MUTED, lw=0.8, ls=":")
    c.set_xlabel("True ΔSCC vs. every baseline"); c.set_ylabel("P(H1 success)")
    c.legend(frameon=False, fontsize=6.5, title=f"comparison rule (ρ = {r})", title_fontsize=6.5,
             loc="upper left", bbox_to_anchor=(1.0, 1.0))
    c.grid(color=GRID, linewidth=0.6); c.set_axisbelow(True)
    c.set_title(f"c  Power, {int(pw.n_blocks[0])} blocks", loc="left", fontsize=8.5, color=INK)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print("rho", round(rho, 3))


if __name__ == "__main__":
    main(sys.argv[1])
