"""Dataset over the memory-mapped cache written by scripts/build_cache.py."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def rot180_triu_perm(n: int = 448, offset: int = 2) -> np.ndarray:
    """Index permutation p with vec_rc = vec[p]: upper-triangle vector of rot180(M) from that of M.

    rot180(M)[i, j] = M[n-1-i, n-1-j]; for symmetric M this equals M[n-1-j, n-1-i], which lies in the
    upper triangle with the same offset.
    """
    iu = np.triu_indices(n, k=offset)
    pos = -np.ones((n, n), dtype=np.int64)
    pos[iu] = np.arange(len(iu[0]))
    return pos[n - 1 - iu[1], n - 1 - iu[0]]


class ContactCache(Dataset):
    """Yields (onehot (L, 4) float32, target (99681, K) float32, cells (K,) long, index).

    lines:     which cached lines to return (names from meta.json), in that order
    cell_ids:  model cell indices for those lines (same length)
    max_shift: random shift in bp (Akita: 11), filled with N (all-zero one-hot); 0 at evaluation
    subset:    optional array of record indices (e.g. H3 limited-data regimes)
    """

    def __init__(self, root, split, lines=None, cell_ids=None, max_shift=0, subset=None, seed=0, augment_rc=False):
        d = Path(root) / split
        self.meta = json.loads((d / "meta.json").read_text())
        self.seq = np.load(d / "seq.u8.npy", mmap_mode="r")
        self.tgt = np.load(d / "target.f16.npy", mmap_mode="r")
        names = self.meta["lines"]
        lines = lines or names
        self.cols = [names.index(x) for x in lines]
        self.cells = torch.as_tensor(cell_ids if cell_ids is not None else range(len(lines)), dtype=torch.long)
        self.windows = [tuple(l.split("\t")) for l in (d / "windows.tsv").read_text().splitlines()]
        self.index = np.arange(len(self.seq)) if subset is None else np.asarray(subset)
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)
        # RC augmentation for non-equivariant baselines (Akita): sequence and map transformed together
        self.rc_perm = rot180_triu_perm() if augment_rc else None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = int(self.index[k])
        codes = np.asarray(self.seq[i])
        if self.max_shift:
            s = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
            if s:
                shifted = np.full_like(codes, 4)
                if s > 0:
                    shifted[s:] = codes[:-s]
                else:
                    shifted[:s] = codes[-s:]
                codes = shifted
        onehot = np.zeros((len(codes), 4), np.float32)
        ok = codes < 4
        onehot[np.nonzero(ok)[0], codes[ok]] = 1.0
        target = np.asarray(self.tgt[i][:, self.cols], dtype=np.float32)
        if self.rc_perm is not None and self.rng.random() < 0.5:
            onehot = np.ascontiguousarray(onehot[::-1, ::-1])           # reverse-complement (ACGT order)
            target = np.ascontiguousarray(target[self.rc_perm])        # rot180 of the map
        return torch.from_numpy(onehot), torch.from_numpy(target), self.cells.clone(), i

    def window(self, i):
        c, s, e = self.windows[i]
        return c, int(s), int(e)
