"""H3 subsets, expanding checkpoint load (5 -> 6 cell lines) and the chromosome-held-out split."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hifold import train as T
from hifold.model.hifold import HiFold, HiFoldConfig

ROOT = Path(__file__).parents[1]


def _subsets_module():
    spec = importlib.util.spec_from_file_location("h3s", ROOT / "scripts" / "h3_subsets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_h3_subsets_nested_whole_contigs(tmp_path):
    rng = np.random.default_rng(1)
    contigs, windows = [], []
    for k in range(40):
        start = k * 10_000_000
        n = int(rng.integers(1, 12))
        contigs.append(f"chr2\t{start}\t{start + 2_000_000 + n * 262_144}\t{'train' if k % 5 else 'valid'}")
        if k % 5:
            windows += [f"chr2\t{start + j * 262_144}\t{start + j * 262_144 + 1_048_576}" for j in range(n)]
    (tmp_path / "contigs.bed").write_text("\n".join(contigs) + "\n")
    (tmp_path / "h3" / "train").mkdir(parents=True)
    (tmp_path / "h3" / "train" / "windows.tsv").write_text("\n".join(windows) + "\n")
    out = _subsets_module().build(tmp_path / "contigs.bed", tmp_path / "h3")
    sets = [set(out[r]["windows"]) for r in ("10", "25", "50", "full")]
    assert sets[0] < sets[1] < sets[2] < sets[3] and len(sets[3]) == len(windows)
    for r, s in zip((10, 25, 50), sets):
        assert len(s) >= r / 100 * len(windows)
    # whole contigs only: a window is in a subset iff every window of its contig is
    owner = {i: int(w.split("\t")[1]) // 10_000_000 for i, w in enumerate(windows)}
    for s in sets[:3]:
        for c in {owner[i] for i in s}:
            assert all(i in s for i, o in owner.items() if o == c)
    assert out == _subsets_module().build(tmp_path / "contigs.bed", tmp_path / "h3")   # deterministic


def test_committed_h3_subsets_are_nested():
    out = json.loads((ROOT / "configs" / "h3_subsets.json").read_text())
    sets = [set(out[r]["windows"]) for r in ("10", "25", "50", "full")]
    assert sets[0] < sets[1] < sets[2] < sets[3] and len(sets[3]) == 7009


def _small(**kw):
    return HiFoldConfig(seq_len=2**15, tower_channels=[8, 8, 8, 8, 16, 16], d_model=16, n_blocks=3, d_state=4,
                        d_pair=8, cell_dim=4, bin_norm=True, **kw)


def test_load_expanding_keeps_trained_rows():
    torch.manual_seed(0)
    src = HiFold(_small(n_cells=5))
    with torch.no_grad():
        src.head.dist_bias.normal_()
    dst = HiFold(_small(n_cells=6))
    grown = T.load_expanding(dst, src.state_dict())
    assert set(grown) == {"cell_emb.weight", "head.dist_bias"}
    assert torch.equal(dst.cell_emb.weight[:5], src.cell_emb.weight)
    assert torch.equal(dst.head.dist_bias[:5], src.head.dist_bias)
    assert torch.count_nonzero(dst.head.dist_bias[5]) == 0                    # new line: fresh zero bias
    sd_src, sd_dst = src.state_dict(), dst.state_dict()
    assert all(torch.equal(sd_dst[k], v) for k, v in sd_src.items() if k not in grown)   # rest copied exactly
    with pytest.raises(ValueError):
        T.load_expanding(HiFold(_small(n_cells=4)), src.state_dict())        # cannot shrink


def test_chrom_holdout_split(tmp_path):
    L, n_triu = 2**15, 99_681
    chroms = ["chr1", "chr2", "chr3", "chr6", "chr8", "chr12", "chrX"]
    for split in ("train", "valid", "test"):
        d = tmp_path / "cache" / split
        d.mkdir(parents=True)
        n = len(chroms)
        np.save(d / "seq.u8.npy", np.zeros((n, L), np.uint8))
        np.save(d / "target.f16.npy", np.ones((n, n_triu, 1), np.float16))
        (d / "windows.tsv").write_text("".join(f"{c}\t0\t{L}\n" for c in chroms))
        (d / "meta.json").write_text(json.dumps({"lines": ["A"], "n": n}))
    tr, va = T.make_datasets({"root": str(tmp_path / "cache"), "split_scheme": "chrom_holdout"}, 0)
    seen_tr = {ds.windows[i][0] for ds in tr.datasets for i in ds.index}
    seen_va = {ds.windows[i][0] for ds in va.datasets for i in ds.index}
    assert seen_tr == {"chr2", "chrX"} and seen_va == {"chr3", "chr6"}
    assert len(tr) == 2 * 3 and len(va) == 2 * 3                          # windows from all three files
