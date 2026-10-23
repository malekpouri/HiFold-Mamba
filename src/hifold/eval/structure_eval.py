"""H2 evaluation (manuscript §2.8-2.9): independent boundaries and loops, paired block bootstrap.

Items are genomic objects, not window pixels, so overlapping test windows count each object once:
  boundary item = one 2,048 bp genomic bin; score = s_i of the predicted map, averaged over every
                  test window in which the bin is scorable, then over seeds;
  loop item     = one pair of genomic bins; score = predicted map value, averaged likewise.
Per line, AUROC / AP is computed over that line's items; the statistic is the mean over lines.
Bootstrap: resample 1 Mb blocks (an item's block = block of its bin, or of the pair midpoint) with
replacement, identically for both arms and all lines; an item enters with weight = number of times
its block was drawn; Delta = statistic(Struct) - statistic(Base) on the same resample.
"""

from dataclasses import dataclass

import numpy as np

BIN_BP = 2048
CROP = 32
BLOCK_BP = 1_000_000


def output_bin_starts(window_start: int, n_out: int = 448, crop: int = CROP, bin_bp: int = BIN_BP) -> np.ndarray:
    """Genomic start (bp) of each output bin of a window."""
    return window_start + (crop + np.arange(n_out)) * bin_bp


def aggregate(keys, values):
    """Mean of `values` per unique key. keys: list of hashables. Returns (unique_keys, means)."""
    acc = {}
    for k, v in zip(keys, values):
        if np.isnan(v):
            continue
        s, n = acc.get(k, (0.0, 0))
        acc[k] = (s + v, n + 1)
    uk = list(acc)
    return uk, np.array([acc[k][0] / acc[k][1] for k in uk])


def label_bins(bins, intervals, pad_bp: int = BIN_BP) -> np.ndarray:
    """bins: list of (chrom, start). Positive if [start, start + BIN_BP) overlaps an interval padded by pad_bp."""
    by_chrom = {}
    for c, s, e in intervals:
        by_chrom.setdefault(c, []).append((s - pad_bp, e + pad_bp))
    for c in by_chrom:
        by_chrom[c] = np.array(sorted(by_chrom[c]))
    out = np.zeros(len(bins), bool)
    for i, (c, s) in enumerate(bins):
        iv = by_chrom.get(c)
        if iv is None:
            continue
        j = np.searchsorted(iv[:, 0], s + BIN_BP, side="left")   # intervals starting before bin end
        if j and (iv[:j, 1] > s).any():
            out[i] = True
    return out


def weighted_auroc(scores: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    """AUROC with integer-like item weights (equals AUROC of the data with items repeated w times).
    Ties between positive and negative scores count 1/2."""
    order = np.argsort(scores, kind="stable")
    s, y, w = scores[order], labels[order], w[order]
    wp, wn = w * y, w * ~y
    W_pos, W_neg = wp.sum(), wn.sum()
    if W_pos == 0 or W_neg == 0:
        return np.nan
    # group equal scores so ties get half credit
    uniq, start = np.unique(s, return_index=True)
    gp = np.add.reduceat(wp, start)
    gn = np.add.reduceat(wn, start)
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (W_pos * W_neg))


def weighted_ap(scores: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    """Average precision with integer item weights, exactly equal to AP of the data with items repeated
    w times (stable tie order). A positive repeated k times after T true and S total earlier items adds
    sum_{j=1..k} (T + j) / (S + j) = k - (S - T) * (psi(S + k + 1) - psi(S + 1))."""
    from scipy.special import digamma
    order = np.argsort(-scores, kind="stable")
    y, w = labels[order].astype(bool), w[order]
    W_pos = (w * y).sum()
    if W_pos == 0:
        return np.nan
    tp_before = np.cumsum(w * y) - w * y
    seen_before = np.cumsum(w) - w
    contrib = w - (seen_before - tp_before) * (digamma(seen_before + w + 1) - digamma(seen_before + 1))
    return float((contrib * y).sum() / W_pos)


@dataclass
class LineItems:
    """Per cell line: item labels, block ids and the two arms' scores (already averaged over seeds)."""
    labels: np.ndarray
    blocks: np.ndarray
    base: np.ndarray
    struct: np.ndarray


@dataclass
class H2Result:
    metric: str
    base: float
    struct: float
    delta: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    n_blocks: int


def paired_block_bootstrap_metric(lines: list, metric: str = "auroc", n_boot: int = 10_000, seed: int = 0,
                                  alpha: float = 0.025) -> H2Result:
    """Delta of the line-averaged metric (Struct - Base), paired block bootstrap shared across lines and arms."""
    fn = weighted_auroc if metric == "auroc" else weighted_ap
    all_blocks = np.unique(np.concatenate([li.blocks for li in lines]))
    index = {b: i for i, b in enumerate(all_blocks)}
    bidx = [np.array([index[b] for b in li.blocks]) for li in lines]

    def stat(counts):
        vb, vs = [], []
        for li, bi in zip(lines, bidx):
            w = counts[bi].astype(float)
            vb.append(fn(li.base, li.labels, w))
            vs.append(fn(li.struct, li.labels, w))
        return np.nanmean(vb), np.nanmean(vs)

    ones = np.ones(len(all_blocks))
    b0, s0 = stat(ones)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    nb = len(all_blocks)
    for k in range(n_boot):
        counts = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        b, s = stat(counts)
        deltas[k] = s - b
    lo, hi = np.percentile(deltas, [100 * alpha, 100 * (1 - alpha)])
    p = (1 + np.sum(deltas <= 0)) / (n_boot + 1)
    return H2Result(metric, float(b0), float(s0), float(s0 - b0), float(lo), float(hi), float(p), nb)


def block_of(chrom: str, pos: int, block_bp: int = BLOCK_BP):
    return (chrom, int(pos) // block_bp)


def distance_matched_negatives(pos_pairs, candidate_pairs, n_per_pos: int = 5, n_bins: int = 10, seed: int = 0):
    """Sample negatives from candidate (chrom, a, b) bin pairs, matched to positives on log-separation.

    pos_pairs / candidate_pairs: arrays of shape (n, 2) of bin starts (bp) on the same chromosome; returns
    indices into candidate_pairs. Separation bins: n_bins log-spaced between the min and max positive separation.
    """
    rng = np.random.default_rng(seed)
    sp = np.abs(pos_pairs[:, 1] - pos_pairs[:, 0]).astype(float)
    sc = np.abs(candidate_pairs[:, 1] - candidate_pairs[:, 0]).astype(float)
    edges = np.logspace(np.log10(sp.min()), np.log10(sp.max() + 1), n_bins + 1)
    bp, bc = np.digitize(sp, edges), np.digitize(sc, edges)
    out = []
    for b in np.unique(bp):
        need = n_per_pos * int((bp == b).sum())
        pool = np.flatnonzero(bc == b)
        if len(pool):
            out.append(rng.choice(pool, size=min(need, len(pool)), replace=False))
    return np.concatenate(out) if out else np.array([], int)
