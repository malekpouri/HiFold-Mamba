"""H3 limited-data regimes (manuscript §2.11; protocol §5, G4): nested contiguous-block subsets.

The official training contigs of the held-out lines (contigs.bed, label "train") are ordered by a fixed
random permutation (numpy default_rng(0)); whole contigs are added in that order until 10 %, 25 % and 50 %
of the training windows are covered. The subsets are nested and identical across arms, seeds and lines
(all four lines share the same windows). Writes configs/h3_subsets.json (record indices into the h3 cache).

python scripts/h3_subsets.py CONTIGS_BED H3_CACHE_ROOT [OUT_JSON]
"""

import json
import sys
from pathlib import Path

import numpy as np

REGIMES = (10, 25, 50)


def build(contigs_bed, cache_root):
    contigs = [l.split("\t") for l in Path(contigs_bed).read_text().splitlines() if l.strip()]
    train = [(c[0], int(c[1]), int(c[2])) for c in contigs if c[3].strip() == "train"]
    windows = [l.split("\t") for l in (Path(cache_root) / "train" / "windows.tsv").read_text().splitlines()]
    windows = [(w[0], int(w[1]), int(w[2])) for w in windows]
    owner = np.full(len(windows), -1)
    for i, (ch, s, e) in enumerate(windows):
        hit = [k for k, (cc, cs, ce) in enumerate(train) if cc == ch and cs <= s and e <= ce]
        if len(hit) != 1:
            raise ValueError(f"training window {i} {ch}:{s}-{e} lies in {len(hit)} training contigs")
        owner[i] = hit[0]
    order = np.random.default_rng(0).permutation(len(train))
    out, chosen, covered = {}, [], 0
    for r in REGIMES:
        for k in order[len(chosen):]:
            if covered >= r / 100 * len(windows):
                break
            chosen.append(int(k))
            covered += int((owner == k).sum())
        idx = np.flatnonzero(np.isin(owner, chosen))
        out[str(r)] = {"windows": idx.tolist(), "n_windows": int(len(idx)), "n_contigs": len(chosen),
                       "fraction": round(len(idx) / len(windows), 4)}
    out["full"] = {"windows": list(range(len(windows))), "n_windows": len(windows), "n_contigs": len(train),
                   "fraction": 1.0}
    out["_info"] = {"contig_order_seed": 0, "n_train_contigs": len(train), "n_train_windows": len(windows),
                    "contig_order": [f"{train[k][0]}:{train[k][1]}-{train[k][2]}" for k in order]}
    return out


if __name__ == "__main__":
    out = build(sys.argv[1], sys.argv[2])
    dst = Path(sys.argv[3] if len(sys.argv) > 3 else "configs/h3_subsets.json")
    dst.write_text(json.dumps(out, indent=1))
    for r in (*map(str, REGIMES), "full"):
        print(f"{r:>4}: {out[r]['n_windows']:5d} windows ({out[r]['fraction']:.1%}), {out[r]['n_contigs']} contigs")
