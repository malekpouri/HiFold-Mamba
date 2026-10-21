"""Check whether H3 evaluation windows overlap loci used to train the 5-line model.

Rule (CLAUDE.md): H3 evaluation regions are excluded from pretraining. The 4 extra
DNALongBench lines ship with their own split, so their valid/test windows may cover
loci that are in Akita's training split.

Usage: python -I scripts/check_h3_overlap.py AKITA_SEQUENCES_BED EXTRA_SEQUENCES_BED
"""

import bisect
import sys
from collections import defaultdict


def read_bed(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            chrom, start, end, split = line.split()[:4]
            rows.append((chrom, int(start), int(end), split))
    return rows


def index(rows, split):
    """Per chromosome: sorted starts and the matching ends, for one split."""
    by_chrom = defaultdict(list)
    for chrom, start, end, s in rows:
        if s == split:
            by_chrom[chrom].append((start, end))
    out = {}
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        out[chrom] = ([s for s, _ in ivs], [e for _, e in ivs], max(e - s for s, e in ivs))
    return out


def overlap_bp(idx, chrom, start, end):
    """Largest single-window overlap (bp) of [start, end) with any indexed window."""
    if chrom not in idx:
        return 0
    starts, ends, max_len = idx[chrom]
    lo = bisect.bisect_left(starts, start - max_len)
    hi = bisect.bisect_left(starts, end)
    best = 0
    for k in range(lo, hi):
        best = max(best, min(end, ends[k]) - max(start, starts[k]))
    return best


def main(akita_bed, extra_bed):
    akita = read_bed(akita_bed)
    extra = read_bed(extra_bed)
    akita_train = index(akita, "train")
    akita_eval = {s: index(akita, s) for s in ("valid", "test")}

    print(f"Akita windows: {len(akita)}; extra-line windows: {len(extra)}")
    for split in ("train", "valid", "test"):
        wins = [r for r in extra if r[3] == split]
        ov = [overlap_bp(akita_train, c, s, e) for c, s, e, _ in wins]
        n_any = sum(o > 0 for o in ov)
        n_half = sum(o >= (e - s) / 2 for o, (_, s, e, _) in zip(ov, wins))
        print(f"extra {split:5s}: {len(wins):5d} windows | overlap Akita TRAIN: "
              f"any {n_any} ({n_any / len(wins):.1%}), >=50% {n_half} ({n_half / len(wins):.1%}), "
              f"clean {len(wins) - n_any}")
    # Reverse direction: do extra-line TRAIN windows cover Akita valid/test loci?
    extra_train = index(extra, "train")
    for split, _ in akita_eval.items():
        wins = [r for r in akita if r[3] == split]
        n_any = sum(overlap_bp(extra_train, c, s, e) > 0 for c, s, e, _ in wins)
        print(f"Akita {split:5s}: {len(wins)} windows, overlapping extra-line TRAIN: {n_any}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
