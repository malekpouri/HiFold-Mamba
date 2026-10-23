"""H2 positive control on VALIDATION windows (no model): do the true benchmark maps recover the
independent boundaries and loops under the exact Stage 2 pipeline?

  python scripts/pilot_structure.py CACHE_DIR G8_DIR OUT_TSV [split]

For each training line with annotations, scores come from (a) the observed map of that line ("ceiling")
and (b) the locus-shared reference (mean of the other four lines' observed maps). Boundaries: s_i of the
map, deduplicated over overlapping windows, AUROC / AP per line, for w in {10, 20, 40, 80}. Loops: map value
at ChIA-PET pairs with PET >= 10 vs. distance-matched negatives (5 per positive), AUROC / AP per line.
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hifold.data.dataset import ContactCache
from hifold.eval import structure_eval as E
from hifold.eval.metrics import triu_to_matrix
from hifold.structure import boundary_strength_t, insulation

LINES = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
N_OUT, CROP, BIN = 448, 32, 2048
MIN_SEP, PET_MIN = 10, 10           # 20 kb, PET count >= 10


def read_boundaries(g8, line):
    f = next((Path(g8) / "boundaries").glob(f"{line}_*.bed.gz"))
    d = pd.read_csv(f, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
    return list(d.itertuples(index=False, name=None))


def read_loops(g8, line):
    fs = list((Path(g8) / "loops").glob(f"{line}_*.bedpe.gz"))
    if not fs:
        return None
    d = pd.read_csv(fs[0], sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5, 6])
    d = d[(d[0] == d[3]) & (d[6] >= PET_MIN)]
    return pd.DataFrame({"chrom": d[0], "a": (d[1] + d[2]) // 2, "b": (d[4] + d[5]) // 2})


def main(cache, g8, out, split="valid"):
    ds = ContactCache(cache, split)
    names = ds.meta["lines"]
    wins = [ds.window(i) for i in range(len(ds.tgt))]
    maps = {}                                       # window -> (n_lines, 448, 448)
    for i in range(len(ds.tgt)):
        t = np.asarray(ds.tgt[i], np.float64)
        maps[i] = np.stack([triu_to_matrix(t[:, k]) for k in range(len(names))])
    rows = []
    for k, line in enumerate(names):
        others = [j for j in range(len(names)) if j != k]
        bnd = read_boundaries(g8, line)
        for w in (10, 20, 40, 80):
            keys, sc_obs, sc_ref = [], [], []
            for i, (ch, s, _) in enumerate(wins):
                starts = E.output_bin_starts(s)
                obs = torch.as_tensor(maps[i][k])
                ref = torch.as_tensor(maps[i][others].mean(0))
                b_obs = boundary_strength_t(insulation(obs[None], w)[0], 5).numpy()
                b_ref = boundary_strength_t(insulation(ref[None], w)[0], 5).numpy()
                for j in range(w + 5, N_OUT - w - 5):
                    keys.append((ch, int(starts[j])))
                    sc_obs.append(b_obs[j]); sc_ref.append(b_ref[j])
            uk, m_obs = E.aggregate(keys, sc_obs)
            uk2, m_ref = E.aggregate(keys, sc_ref)
            assert uk == uk2
            lab = E.label_bins(uk, bnd)
            for name, sc in (("observed", m_obs), ("locus_shared", m_ref)):
                one = np.ones(len(sc))
                rows.append(dict(line=line, kind="boundary", w=w, scorer=name, n_items=len(sc), n_pos=int(lab.sum()),
                                 auroc=E.weighted_auroc(sc, lab, one), ap=E.weighted_ap(sc, lab, one),
                                 base_rate=lab.mean()))
                print(rows[-1], flush=True)
        loops = read_loops(g8, line)
        if loops is None:
            continue
        rng = np.random.default_rng(0)
        pos_keys, pos_obs, pos_ref, cand = [], [], [], []
        by_chrom = {c: g for c, g in loops.groupby("chrom")}
        for i, (ch, s, _) in enumerate(wins):
            lo, hi = s + CROP * BIN, s + (CROP + N_OUT) * BIN
            g = by_chrom.get(ch)
            obs, ref = maps[i][k], maps[i][others].mean(0)
            if g is not None:
                g = g[(g.a >= lo) & (g.b < hi)]
                ia, ib = (g.a.to_numpy() - lo) // BIN, (g.b.to_numpy() - lo) // BIN
                ia, ib = np.minimum(ia, ib), np.maximum(ia, ib)
                ok = ib - ia >= MIN_SEP
                for x, y in zip(ia[ok], ib[ok]):
                    pos_keys.append((ch, int(lo + x * BIN), int(lo + y * BIN)))
                    pos_obs.append(obs[x, y]); pos_ref.append(ref[x, y])
            # candidate negatives: random pairs in this window (separation >= MIN_SEP)
            x = rng.integers(0, N_OUT, 3000); y = rng.integers(0, N_OUT, 3000)
            x, y = np.minimum(x, y), np.maximum(x, y)
            ok = y - x >= MIN_SEP
            for a, b in zip(x[ok], y[ok]):
                cand.append(((ch, int(lo + a * BIN), int(lo + b * BIN)), obs[a, b], ref[a, b]))
        pk, p_obs = E.aggregate(pos_keys, pos_obs)
        _, p_ref = E.aggregate(pos_keys, pos_ref)
        pset = set(pk)
        cand = [c for c in cand if c[0] not in pset]
        ck, c_obs = E.aggregate([c[0] for c in cand], [c[1] for c in cand])
        _, c_ref = E.aggregate([c[0] for c in cand], [c[2] for c in cand])
        pos_pairs = np.array([[k_[1], k_[2]] for k_ in pk], float)
        cand_pairs = np.array([[k_[1], k_[2]] for k_ in ck], float)
        neg = E.distance_matched_negatives(pos_pairs, cand_pairs, n_per_pos=5, seed=0)
        lab = np.r_[np.ones(len(pk), bool), np.zeros(len(neg), bool)]
        for name, ps, cs in (("observed", p_obs, c_obs), ("locus_shared", p_ref, c_ref)):
            sc = np.r_[ps, cs[neg]]
            one = np.ones(len(sc))
            rows.append(dict(line=line, kind="loop", w=np.nan, scorer=name, n_items=len(sc), n_pos=len(pk),
                             auroc=E.weighted_auroc(sc, lab, one), ap=E.weighted_ap(sc, lab, one),
                             base_rate=lab.mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
