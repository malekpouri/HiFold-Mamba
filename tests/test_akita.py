import numpy as np
import torch

from hifold.data.dataset import rot180_triu_perm
from hifold.eval.metrics import matrix_to_triu, triu_to_matrix
from hifold.model.akita import Akita, AkitaConfig, _rates


def test_dilation_rates_match_basenji():
    # basenji dilated_residual: rate starts at 1, x1.75 per repeat, int(np.round(rate))
    assert _rates(8, 1.75) == [1, 2, 3, 5, 9, 16, 29, 50]
    assert _rates(6, 1.75) == [1, 2, 3, 5, 9, 16]


def test_akita_shapes_symmetry_and_cell_selection():
    cfg = AkitaConfig(seq_len=2**15, tower_repeat=5, filters=16, res_filters=8, res_repeat=3,
                      trunk_out=8, head_filters=8, head_res_filters=4, head_res_repeat=2)
    torch.manual_seed(0)
    m = Akita(cfg).eval()
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, cfg.seq_len)), 4).float()
    y_all = m(x, torch.arange(5).repeat(2, 1))
    assert y_all.shape == (2, 5, 448, 448)
    torch.testing.assert_close(y_all, y_all.transpose(-1, -2))
    y_sel = m(x, torch.tensor([[3, 1], [0, 4]]))
    torch.testing.assert_close(y_sel[0, 0], y_all[0, 3])
    torch.testing.assert_close(y_sel[1, 1], y_all[1, 4])


def test_akita_paper_size():
    n = sum(p.numel() for p in Akita(AkitaConfig()).parameters())
    assert 600_000 < n < 900_000, n          # ~0.75 M; logged in notes


def test_rot180_triu_permutation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(448, 448))
    a = (a + a.T) / 2
    vec = matrix_to_triu(a)
    perm = rot180_triu_perm()
    assert sorted(perm.tolist()) == list(range(len(vec)))     # a permutation
    rot = triu_to_matrix(vec[perm])
    np.testing.assert_allclose(rot, triu_to_matrix(vec)[::-1, ::-1])
