"""Contact-map metrics: matrix reconstruction, SCC (two variants), PCC.

Conventions (CLAUDE.md, protocol §2):
- Targets/predictions are stored as the upper triangle of a 448x448 map with diagonal
  offset 2, row-major (np.triu_indices(448, k=2)): 99,681 values. Same order as Akita
  and the DNALongBench CNN baseline (torch.triu_indices(..., offset=2)).
- NaN never counts. Diagonals |i-j| < offset are NaN in reconstructed matrices.

SCC variants
- scc_benchmark: verbatim port of DNALongBench scripts/metrics.py::scc. Used for every
  number that is compared with published benchmark values. Its stratum weight is
  N_k * var(0..N_k-1) / N_k**4 ~ 1/(12 N_k), i.e. it upweights long-distance strata,
  unlike HiCRep, and it applies no 2D smoothing.
- scc_hicrep: stratum weights as in Yang et al. 2017, N_k * sqrt(var(r1/N_k) var(r2/N_k))
  with within-stratum ranks; optional 2D mean-filter smoothing h (default 0 = none).
Which of the two is primary is a protocol decision (see notes/); both are always reported.
"""

import numpy as np
from scipy.stats import pearsonr, rankdata

N_BINS = 448
DIAG_OFFSET = 2


def triu_to_matrix(vec, n: int = N_BINS, offset: int = DIAG_OFFSET) -> np.ndarray:
    """(n_triu,) or (n_triu, k) upper-triangle vector -> symmetric (n, n[, k]) matrix, NaN near diagonal."""
    vec = np.asarray(vec, dtype=np.float64)
    iu = np.triu_indices(n, k=offset)
    if vec.shape[0] != len(iu[0]):
        raise ValueError(f"expected {len(iu[0])} values for n={n}, offset={offset}; got {vec.shape[0]}")
    mat = np.full((n, n) + vec.shape[1:], np.nan)
    mat[iu] = vec
    mat[(iu[1], iu[0])] = vec
    return mat


def matrix_to_triu(mat, offset: int = DIAG_OFFSET) -> np.ndarray:
    n = mat.shape[0]
    return np.asarray(mat)[np.triu_indices(n, k=offset)]


# --- DNALongBench SCC (verbatim logic) ----------------------------------------------------

def _vstrans_benchmark(nk):
    return np.var(np.arange(nk)) / nk**4


def scc_benchmark(mat1: np.ndarray, mat2: np.ndarray, max_bins: int = None) -> float:
    """Port of DNALongBench scripts/metrics.py::scc (commit on main, read 2026-10-21)."""
    assert mat1.shape == mat2.shape
    if max_bins is None:
        max_bins = int(mat1.shape[0] - 5)
    else:
        max_bins = min(max_bins, mat1.shape[0] - 5)
    corr_diag = np.zeros(max_bins)
    weight_diag = np.zeros_like(corr_diag)
    for d in range(max_bins):
        d1 = mat1.diagonal(d)
        d2 = mat2.diagonal(d)
        mask = ~np.isnan(d1) & ~np.isnan(d2)
        d1 = d1[mask]
        d2 = d2[mask]
        corr_diag[d] = pearsonr(d1, d2)[0] if np.var(d1) > 0 and np.var(d2) > 0 else np.nan
        r2k = _vstrans_benchmark(len(d1))
        weight_diag[d] = len(d1) * r2k
    corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
    mask = ~np.isnan(corr_diag)
    corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]
    weight_diag /= sum(weight_diag)
    return float(np.nansum(corr_diag * weight_diag))


# --- HiCRep SCC ---------------------------------------------------------------------------

def _mean_filter(mat: np.ndarray, h: int) -> np.ndarray:
    """2D (2h+1)x(2h+1) mean over non-NaN neighbours (HiCRep smoothing); NaN stays NaN."""
    if h == 0:
        return mat
    valid = ~np.isnan(mat)
    x = np.where(valid, mat, 0.0)
    pad = lambda a: np.pad(a, h)  # noqa: E731
    xs, vs = pad(x), pad(valid.astype(np.float64))
    k = 2 * h + 1
    n = mat.shape[0]
    s = np.zeros_like(x)
    c = np.zeros_like(x)
    for di in range(k):
        for dj in range(k):
            s += xs[di:di + n, dj:dj + n]
            c += vs[di:di + n, dj:dj + n]
    out = s / np.maximum(c, 1)
    out[~valid] = np.nan
    return out


def scc_hicrep(mat1: np.ndarray, mat2: np.ndarray, max_bins: int = None,
               min_dist: int = DIAG_OFFSET, h: int = 0) -> float:
    """Stratum-adjusted correlation coefficient (Yang et al. 2017) on strata d in [min_dist, max_bins)."""
    assert mat1.shape == mat2.shape
    n = mat1.shape[0]
    max_bins = n if max_bins is None else min(max_bins, n)
    m1, m2 = _mean_filter(mat1, h), _mean_filter(mat2, h)
    corrs, weights = [], []
    for d in range(min_dist, max_bins):
        d1, d2 = m1.diagonal(d), m2.diagonal(d)
        mask = ~np.isnan(d1) & ~np.isnan(d2)
        d1, d2 = d1[mask], d2[mask]
        nk = len(d1)
        if nk < 2 or np.ptp(d1) == 0 or np.ptp(d2) == 0:  # exact: constant stratum
            continue
        r2k = np.sqrt(np.var(rankdata(d1) / nk) * np.var(rankdata(d2) / nk))
        corrs.append(pearsonr(d1, d2)[0])
        weights.append(nk * r2k)
    if not weights:
        return float("nan")
    w = np.asarray(weights) / np.sum(weights)
    return float(np.sum(np.asarray(corrs) * w))


# --- PCC ----------------------------------------------------------------------------------

def pcc_window(pred_vec, target_vec) -> float:
    """Pearson r over one window's valid upper-triangle values."""
    p, t = np.asarray(pred_vec, np.float64), np.asarray(target_vec, np.float64)
    m = ~np.isnan(p) & ~np.isnan(t)
    return float(pearsonr(p[m], t[m])[0])


def pcc_pooled(pred_vecs, target_vecs) -> float:
    """Pearson r over all values of all windows, as in the DNALongBench CNN train.py."""
    p = np.asarray(pred_vecs, np.float64).ravel()
    t = np.asarray(target_vecs, np.float64).ravel()
    m = ~np.isnan(p) & ~np.isnan(t)
    return float(pearsonr(p[m], t[m])[0])


def stratum_scc(mat1, mat2, lo_bins: int, hi_bins: int, variant: str = "hicrep") -> float:
    """SCC restricted to separations [lo_bins, hi_bins) bins (distance-stratified metric)."""
    m1, m2 = mat1.copy(), mat2.copy()
    n = m1.shape[0]
    i, j = np.indices((n, n))
    out = (np.abs(i - j) < lo_bins) | (np.abs(i - j) >= hi_bins)
    m1[out] = np.nan
    m2[out] = np.nan
    if variant == "benchmark":
        return scc_benchmark(m1, m2)
    return scc_hicrep(m1, m2, min_dist=max(lo_bins, DIAG_OFFSET), max_bins=hi_bins)
