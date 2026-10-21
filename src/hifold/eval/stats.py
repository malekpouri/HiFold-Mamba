"""Primary statistical test (protocol §3): per-block best baseline, paired block bootstrap.

Pipeline: score[model, seed, window] -> mean over seeds -> mean per non-overlapping 1 Mb block
-> Delta_b = ours_b - max_{m in B} baseline_b -> resample blocks with replacement (10,000x)
-> one-sided test H0: Delta <= 0 at alpha = 0.025; report the 95 % percentile CI.
"""

from dataclasses import dataclass

import numpy as np

BLOCK_BP = 1_000_000
ALPHA = 0.025


def block_ids(windows, block_bp: int = BLOCK_BP) -> np.ndarray:
    """Assign each (chrom, start, end) window to the 1 Mb block containing its midpoint."""
    keys = [(c, (int(s) + int(e)) // 2 // block_bp) for c, s, e in windows]
    lookup = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([lookup[k] for k in keys])


def per_block(window_scores: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Mean of window scores within each block; NaN windows ignored. Shape (..., n_windows) -> (..., n_blocks)."""
    n_blocks = blocks.max() + 1
    out = np.full(window_scores.shape[:-1] + (n_blocks,), np.nan)
    for b in range(n_blocks):
        sel = window_scores[..., blocks == b]
        with np.errstate(all="ignore"):
            out[..., b] = np.nanmean(sel, axis=-1) if sel.shape[-1] else np.nan
    return out


@dataclass
class TestResult:
    mean_delta: float
    ci_low: float        # 2.5 % percentile = 97.5 % one-sided lower bound
    ci_high: float
    p_one_sided: float
    n_blocks: int
    success: bool        # mean_delta >= margin and ci_low > 0

    def as_dict(self):
        return self.__dict__.copy()


def block_bootstrap(ours: np.ndarray, baselines: np.ndarray, blocks: np.ndarray,
                    margin: float = 0.01, n_boot: int = 10_000, seed: int = 0) -> TestResult:
    """Paired block bootstrap of Delta_b = ours_b - max_m baseline_{m,b}.

    ours:      (n_seeds, n_windows) scores of our model
    baselines: (n_models, n_seeds, n_windows) scores of the models in the comparison set B
    blocks:    (n_windows,) block id per window
    """
    ours_b = per_block(np.nanmean(ours, axis=0), blocks)                 # seeds first, then blocks
    base_b = per_block(np.nanmean(baselines, axis=1), blocks)            # (n_models, n_blocks)
    with np.errstate(all="ignore"):
        delta = ours_b - np.nanmax(base_b, axis=0)
    delta = delta[~np.isnan(delta)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boots = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * ALPHA, 100 * (1 - ALPHA)])
    p = (1 + np.sum(boots <= 0)) / (n_boot + 1)
    mean = float(delta.mean())
    return TestResult(mean, float(lo), float(hi), float(p), len(delta), bool(mean >= margin and lo > 0))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, blocks: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Block bootstrap of mean(a - b) for a single comparator (H2, H3, ablations, global-mean check).

    a, b: (n_seeds, n_windows) or (n_windows,)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return block_bootstrap(a, b[None], blocks, margin=0.0, n_boot=n_boot, seed=seed)


def holm(pvalues, alpha: float = ALPHA):
    """Holm step-down: returns a boolean 'reject' per p-value."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            reject[i] = True
        else:
            break
    return reject
