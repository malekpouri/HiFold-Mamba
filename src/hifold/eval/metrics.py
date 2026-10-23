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


def _diagonals(m: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Stack diagonals d0..d1-1 of a square matrix into (D, n), NaN-padded on the right."""
    n = m.shape[0]
    i = np.arange(n)[None, :]
    d = np.arange(d0, d1)[:, None]
    j = i + d
    ok = j < n
    out = np.full((d1 - d0, n), np.nan)
    out[ok] = m[np.broadcast_to(i, ok.shape)[ok], j[ok]]
    return out


def _rowwise_pearson(x: np.ndarray, y: np.ndarray):
    """Pearson r per row over positions where both are finite. Returns (r, count, var_x, var_y)."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    cnt = mask.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(mask, x, 0.0).sum(1) / cnt
        ym = np.where(mask, y, 0.0).sum(1) / cnt
        dx = np.where(mask, x - xm[:, None], 0.0)
        dy = np.where(mask, y - ym[:, None], 0.0)
        vx, vy = (dx * dx).sum(1), (dy * dy).sum(1)
        r = (dx * dy).sum(1) / np.sqrt(vx * vy)
    return r, cnt, vx / np.maximum(cnt, 1), vy / np.maximum(cnt, 1), mask


def scc_benchmark(mat1: np.ndarray, mat2: np.ndarray, max_bins: int = None) -> float:
    """DNALongBench scripts/metrics.py::scc (read 2026-10-21), vectorised over diagonals.

    Same semantics as the original loop (tests/test_metrics.py compares against a verbatim copy):
    diagonals 0..max_bins-1 with max_bins = n - 5, diagonal 0 dropped, NaN correlation when a
    stratum has zero variance, stratum weight N_k * var(0..N_k-1) / N_k**4.
    """
    assert mat1.shape == mat2.shape
    n = mat1.shape[0]
    max_bins = n - 5 if max_bins is None else min(max_bins, n - 5)
    r, cnt, vx, vy, _ = _rowwise_pearson(_diagonals(mat1, 0, max_bins), _diagonals(mat2, 0, max_bins))
    r = np.where((vx > 0) & (vy > 0), r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = cnt * ((cnt.astype(float) ** 2 - 1) / 12) / cnt.astype(float) ** 4   # var(arange(N)) = (N^2-1)/12
    r, w = r[1:], w[1:]
    keep = ~np.isnan(r)
    r, w = r[keep], w[keep]
    return float(np.nansum(r * w / w.sum()))


def _scc_benchmark_loop(mat1: np.ndarray, mat2: np.ndarray, max_bins: int = None) -> float:
    """Original loop form of scc_benchmark (kept for cross-checking)."""
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
    """Stratum-adjusted correlation coefficient (Yang et al. 2017) on strata d in [min_dist, max_bins).

    Weight per stratum: N_k * sqrt(var(r1/N_k) var(r2/N_k)), r = within-stratum ranks (ties averaged).
    Strata with fewer than 2 pairs or a constant side are skipped. Vectorised over diagonals.
    """
    assert mat1.shape == mat2.shape
    n = mat1.shape[0]
    max_bins = n if max_bins is None else min(max_bins, n)
    if max_bins <= min_dist:
        return float("nan")
    x = _diagonals(_mean_filter(mat1, h), min_dist, max_bins)
    y = _diagonals(_mean_filter(mat2, h), min_dist, max_bins)
    r, cnt, _, _, mask = _rowwise_pearson(x, y)
    xm, ym = np.where(mask, x, np.nan), np.where(mask, y, np.nan)
    with np.errstate(invalid="ignore"):
        const = (np.nanmax(xm, 1) == np.nanmin(xm, 1)) | (np.nanmax(ym, 1) == np.nanmin(ym, 1))
    keep = (cnt >= 2) & ~const
    if not keep.any():
        return float("nan")
    rx = rankdata(xm[keep], axis=1, nan_policy="omit")
    ry = rankdata(ym[keep], axis=1, nan_policy="omit")
    nk = cnt[keep].astype(float)[:, None]
    r2k = np.sqrt(np.nanvar(rx / nk, axis=1) * np.nanvar(ry / nk, axis=1))
    w = cnt[keep] * r2k
    return float(np.sum(r[keep] * w / w.sum()))


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
