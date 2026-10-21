import numpy as np
import pytest

from hifold.eval import metrics as M

N = M.N_BINS
N_TRIU = len(np.triu_indices(N, k=M.DIAG_OFFSET)[0])


def _random_map(rng, decay=True):
    """Symmetric log-O/E-like map with some distance structure."""
    i, j = np.indices((N, N))
    base = -0.002 * np.abs(i - j) if decay else 0.0
    noise = rng.normal(0, 0.5, (N, N))
    m = base + (noise + noise.T) / 2
    return np.clip(m, -2, 2)


def test_triu_length_matches_benchmark():
    assert N_TRIU == 99_681


def test_triu_roundtrip_and_symmetry():
    rng = np.random.default_rng(0)
    vec = rng.normal(size=N_TRIU)
    mat = M.triu_to_matrix(vec)
    assert np.allclose(M.matrix_to_triu(mat), vec)
    assert np.allclose(np.nan_to_num(mat), np.nan_to_num(mat.T))
    assert np.isnan(np.diagonal(mat, 0)).all() and np.isnan(np.diagonal(mat, 1)).all()
    assert not np.isnan(np.diagonal(mat, 2)).any()


def test_triu_rejects_wrong_length():
    with pytest.raises(ValueError):
        M.triu_to_matrix(np.zeros(N_TRIU - 1))


def _scc_reference(mat1, mat2, max_bins=None):
    """Verbatim copy of DNALongBench scripts/metrics.py (indentation normalised)."""
    from scipy.stats import pearsonr

    def vstrans(nk):
        return np.var(np.arange(nk)) / nk**4

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
        r2k = vstrans(len(d1))
        weight_diag[d] = len(d1) * r2k
    corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
    mask = ~np.isnan(corr_diag)
    corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]
    weight_diag /= sum(weight_diag)
    return np.nansum(corr_diag * weight_diag)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scc_benchmark_matches_reference():
    rng = np.random.default_rng(1)
    t = M.triu_to_matrix(M.matrix_to_triu(_random_map(rng)))
    p = M.triu_to_matrix(M.matrix_to_triu(_random_map(rng)))
    p = 0.6 * t + 0.4 * p
    assert M.scc_benchmark(p, t) == pytest.approx(_scc_reference(p, t), abs=1e-12)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("fn", [M.scc_benchmark, M.scc_hicrep])
def test_scc_perfect_and_independent(fn):
    rng = np.random.default_rng(2)
    t = M.triu_to_matrix(M.matrix_to_triu(_random_map(rng)))
    assert fn(t, t) == pytest.approx(1.0, abs=1e-9)
    u = M.triu_to_matrix(M.matrix_to_triu(_random_map(rng)))
    assert abs(fn(u, t)) < 0.05


def test_scc_hicrep_ignores_pure_distance_decay():
    """A map that is only P(s) must not score well against a map with structure."""
    rng = np.random.default_rng(3)
    i, j = np.indices((N, N))
    decay_only = -0.004 * np.abs(i - j).astype(float)
    t = M.triu_to_matrix(M.matrix_to_triu(decay_only + _random_map(rng, decay=False)))
    p = M.triu_to_matrix(M.matrix_to_triu(decay_only))
    # constant within each stratum -> zero variance -> strata skipped -> nan
    assert np.isnan(M.scc_hicrep(p, t))
    # whereas naive PCC rewards the decay
    assert M.pcc_window(M.matrix_to_triu(p), M.matrix_to_triu(t)) > 0.3


def test_scc_hicrep_weights_are_n_over_12_for_untied_ranks():
    rng = np.random.default_rng(4)
    t = M.triu_to_matrix(M.matrix_to_triu(_random_map(rng)))
    p = t + rng.normal(0, 0.3, t.shape)
    p = (p + p.T) / 2
    p[np.isnan(t)] = np.nan
    # manual: weights N_k * sqrt(var(r1/N) var(r2/N)) ~ N_k/12 * (1 - 1/N_k^2)
    from scipy.stats import pearsonr
    cs, ws = [], []
    for d in range(2, N):
        a, b = p.diagonal(d), t.diagonal(d)
        nk = len(a)
        if nk < 2:
            continue
        cs.append(pearsonr(a, b)[0])
        ws.append(nk * (nk**2 - 1) / (12 * nk**2))
    expected = np.sum(np.array(cs) * np.array(ws) / np.sum(ws))
    assert M.scc_hicrep(p, t) == pytest.approx(expected, abs=1e-9)


def test_mean_filter_constant_map_unchanged():
    m = np.full((10, 10), 3.0)
    m[0, 0] = np.nan
    out = M._mean_filter(m, 1)
    assert np.isnan(out[0, 0]) and np.allclose(out[~np.isnan(out)], 3.0)


def test_pcc_pooled_vs_window():
    rng = np.random.default_rng(5)
    t = rng.normal(size=(3, 100))
    p = t + rng.normal(0, 0.5, t.shape)
    assert -1 <= M.pcc_pooled(p, t) <= 1
    assert M.pcc_window(p[0], t[0]) == pytest.approx(np.corrcoef(p[0], t[0])[0, 1])
