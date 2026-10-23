"""Paper Fig. 1: HiFold-Mamba schematic with a real target map (validation window) and its insulation profile.

  python scripts/figure_schematic.py CACHE_DIR OUT_PREFIX
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hifold.data.dataset import ContactCache
from hifold.eval.metrics import triu_to_matrix
from hifold.structure import boundary_strength_t, insulation

INK, MUTED, BLUE, ORANGE, FILL, FILL2 = "#1f1f1e", "#6b6a64", "#2a78d6", "#eb6834", "#eef4fc", "#fdf0ea"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, body, fc=FILL, ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=7.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.12, body, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.4)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=1.0))


def main(cache, out):
    ds = ContactCache(cache, "valid")
    i = 7
    t = triu_to_matrix(np.asarray(ds.tgt[i][:, 0], np.float64))
    ins = insulation(torch.as_tensor(t)[None], 20)[0]
    bs = boundary_strength_t(ins, 5).numpy()

    fig = plt.figure(figsize=(7.4, 3.6))
    ax = fig.add_axes([0, 0, 0.70, 1]); ax.set_xlim(-0.01, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    W, H, Y, G = 0.18, 0.36, 0.56, 0.025
    xs = [k * (W + G) + 0.005 for k in range(5)]
    box(ax, xs[0], Y, W, H, "Input", "1 Mb DNA\none-hot\nhg38")
    box(ax, xs[1], Y, W, H, "Conv tower", "11 RC conv\nblocks, pool 2\n→ 512 bins")
    box(ax, xs[2], Y, W, H, "Bi-Mamba ×8", "RC-shared\nscans\n+ FiLM(cell)")
    box(ax, xs[3], Y, W, H, "Mixer", "3 depths ×\n4 scales\n(1–8 bins)")
    box(ax, xs[4], Y, W, H, "Pair head", "biaffine\n+ b(c,|i−j|)\n→ 448²")
    for k in range(4):
        arrow(ax, (xs[k] + W, Y + H / 2), (xs[k + 1], Y + H / 2))
    box(ax, 0.005, 0.03, 0.48, 0.45, "Objective", "L_contact: masked MSE on log O/E\n"
        "+ λ_ins L_ins + λ_bnd L_bnd\n(last 25 % of steps)\n"
        "insulation of g(C), g = exp(2 tanh(·/2)),\nsame operator on Ĉ and C", fc=FILL2, ec=ORANGE)
    box(ax, 0.51, 0.03, 0.485, 0.45, "By construction", "f(RC x, c) = rot180 f(x, c)\n"
        "(unit test, fp32)\nĈ symmetric\ncell line enters via FiLM\n+ distance bias only", fc="#f4f4f1", ec=MUTED)

    a2 = fig.add_axes([0.73, 0.36, 0.25, 0.58])
    a2.imshow(t, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    a2.set_xticks([]); a2.set_yticks([])
    ch, s, e = ds.window(i)
    a2.set_title(f"Target (HFF, validation)\n{ch}:{s/1e6:.2f}–{e/1e6:.2f} Mb", fontsize=7, color=INK)
    a3 = fig.add_axes([0.73, 0.08, 0.25, 0.20])
    a3.plot(np.arange(len(bs)), bs, color=ORANGE, lw=1.2)
    a3.set_ylabel("dip depth", fontsize=6.5, color=INK); a3.set_xlim(0, len(bs)); a3.tick_params(labelsize=6)
    a3.set_xlabel("bin (2,048 bp)", fontsize=6.5)
    for sp in ("top", "right"):
        a3.spines[sp].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:])
