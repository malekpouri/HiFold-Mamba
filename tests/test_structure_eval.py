"""H2 evaluation: weighted metrics equal metrics on repeated data; bootstrap detects a real gain."""

import numpy as np

from hifold.eval import structure_eval as E
from hifold.structure import auroc, average_precision


def test_weighted_auroc_equals_repeated_data():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200); y = rng.random(200) < 0.3; w = rng.integers(0, 4, 200)
    rep = np.repeat(np.arange(200), w)
    assert np.isclose(E.weighted_auroc(s, y, w.astype(float)), auroc(s[rep], y[rep]))


def test_weighted_ap_unit_weights_equals_ap():
    rng = np.random.default_rng(1)
    s = rng.normal(size=300); y = rng.random(300) < 0.2
    assert np.isclose(E.weighted_ap(s, y, np.ones(300)), average_precision(s, y))


def test_weighted_ap_equals_repeated_data():
    rng = np.random.default_rng(4)
    s = rng.normal(size=150); y = rng.random(150) < 0.3; w = rng.integers(0, 4, 150)
    rep = np.repeat(np.arange(150), w)
    assert np.isclose(E.weighted_ap(s, y, w.astype(float)), average_precision(s[rep], y[rep]))


def test_label_bins_padding():
    bins = [("chr1", 0), ("chr1", 2048), ("chr1", 4096), ("chr1", 8192), ("chr2", 0)]
    lab = E.label_bins(bins, [("chr1", 6000, 6100)])          # padded to [3952, 8148)
    assert lab.tolist() == [False, True, True, False, False]


def test_aggregate_deduplicates_overlapping_windows():
    k, v = E.aggregate([("chr1", 0), ("chr1", 0), ("chr1", 2048)], [1.0, 3.0, np.nan])
    assert k == [("chr1", 0)] and v.tolist() == [2.0]


def test_bootstrap_detects_gain_and_not_null():
    rng = np.random.default_rng(2)
    lines = []
    for _ in range(3):
        n = 4000
        y = rng.random(n) < 0.1
        blocks = rng.integers(0, 150, n)
        base = rng.normal(size=n) + 0.5 * y
        lines.append(E.LineItems(y, blocks, base, base + 0.6 * y + 0.1 * rng.normal(size=n)))
    r = E.paired_block_bootstrap_metric(lines, "auroc", n_boot=300)
    assert r.delta > 0.05 and r.ci_low > 0 and r.p_one_sided < 0.025
    null = [E.LineItems(li.labels, li.blocks, li.base, li.base.copy()) for li in lines]
    r0 = E.paired_block_bootstrap_metric(null, "ap", n_boot=200)
    assert r0.delta == 0 and r0.p_one_sided > 0.5


def test_distance_matched_negatives_match_separation():
    rng = np.random.default_rng(3)
    pos = np.stack([np.zeros(50), rng.integers(10, 100, 50) * 2048], 1)
    cand = np.stack([np.zeros(5000), rng.integers(10, 400, 5000) * 2048], 1)
    idx = E.distance_matched_negatives(pos, cand, n_per_pos=2)
    sep = cand[idx, 1]
    assert sep.max() <= pos[:, 1].max() + 2048 and len(idx) > 50
