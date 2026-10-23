"""Paper Fig. 3: H2 positive control on validation windows (results/pilot/structure_control_valid.tsv).

  python scripts/figure_structure_control.py OUT_PREFIX
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINES = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
LABEL = {"HFF": "HFF", "H1hESC": "H1", "GM12878": "GM12878", "IMR90": "IMR-90", "HCT116": "HCT116"}
COL = {"HFF": "#2a78d6", "H1hESC": "#eb6834", "GM12878": "#1baf7a", "IMR90": "#eda100", "HCT116": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})


def main(out):
    d = pd.read_csv("results/pilot/structure_control_valid.tsv", sep="\t")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    b = d[(d.kind == "boundary")]
    for line in LINES:
        for sc, ls in (("observed", "-"), ("locus_shared", ":")):
            t = b[(b.line == line) & (b.scorer == sc)].sort_values("w")
            a.plot(t.w * 2, t.auroc, ls, color=COL[line], lw=2 if sc == "observed" else 1.4, marker="o" if sc == "observed" else None,
                   ms=3.5, label=LABEL[line] if sc == "observed" else None)
    a.axvline(80, color=MUTED, lw=0.8, ls="--")
    a.text(83, 0.935, "adopted: w = 40 bins", fontsize=6.5, color=MUTED, va="top")
    a.set_xscale("log", base=2); a.set_xticks([20, 40, 80, 160], ["20", "40", "80", "160"])
    a.set_xlabel("insulation half-window (kb)"); a.set_ylabel("boundary AUROC")
    a.set_ylim(0.48, 0.95); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5,
             title="solid: line's own map · dotted: locus-shared reference", title_fontsize=6.5)
    a.set_title("a  Independent 4DN boundaries", loc="left", fontsize=8.5, color=INK)
    c = ax[1]
    l = d[d.kind == "loop"]
    lines = [x for x in LINES if x in set(l.line)]
    x = np.arange(len(lines))
    for k, (sc, col) in enumerate((("observed", "#2a78d6"), ("locus_shared", "#9ec5f4"))):
        v = [l[(l.line == ln) & (l.scorer == sc)].auroc.iloc[0] for ln in lines]
        c.bar(x + (k - 0.5) * 0.36, v, 0.34, color=col, edgecolor="white", lw=0.8,
              label="line's own map" if sc == "observed" else "locus-shared reference")
    c.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    c.set_xticks(x, [LABEL[ln] for ln in lines]); c.set_ylim(0.4, 1.0); c.set_ylabel("loop AUROC (PET ≥ 10)")
    c.grid(axis="y", color=GRID, lw=0.6); c.set_axisbelow(True)
    c.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    c.set_title("b  Independent CTCF ChIA-PET loops", loc="left", fontsize=8.5, color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
