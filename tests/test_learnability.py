"""The model must use the sequence: memorise a few REAL training windows beyond the distance-only floor.

Added 2026-11-17 after 15 pilot runs learned only the distance profile (notes/2026-11-16_5090-queue-batch1.md).
The earlier smoke test used a synthetic target that a distance bias alone can fit, so it could not catch this.
Needs a GPU and the DNALongBench cache; about 5 min on an RTX 5050.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hifold.data.dataset import ContactCache
from hifold.model.hifold import HiFold, HiFoldConfig
from hifold.train import contact_loss, lr_at, to_triu, triu_index

CFG = Path(__file__).parents[1] / "configs" / "hifold_paper.yaml"
N_WIN, STEPS = 4, 200


def _root():
    return yaml.safe_load(CFG.read_text())["data"]["root"]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(not os.path.isdir(_root()), reason="needs the DNALongBench akita5 cache")
def test_paper_config_memorises_real_windows_beyond_distance_floor():
    cfg = yaml.safe_load(CFG.read_text())
    d, dev = cfg["data"], torch.device("cuda")
    torch.manual_seed(0)
    ds = ContactCache(d["root"], "train", d["lines"], d["cell_ids"], 0, subset=np.arange(N_WIN))
    batch = [ds[i] for i in range(N_WIN)]
    mcfg = HiFoldConfig(**cfg["model"])
    iu = triu_index(mcfg.out_bins, device=dev)
    Y = torch.stack([b[1] for b in batch]).permute(0, 2, 1).to(dev)            # (W, K, T)

    # distance-only floor: best per-cell, per-diagonal constant on these same windows
    diag = iu[1] - iu[0]
    fp = torch.zeros_like(Y[0])
    for k in range(Y.shape[1]):
        for dv in diag.unique():
            sel = diag == dv
            fp[k, sel] = torch.nanmean(Y[:, k, sel])
    floor = float(np.mean([contact_loss(fp, Y[w]).item() for w in range(N_WIN)]))

    model = HiFold(mcfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for s in range(STEPS):
        x, y, c, _ = batch[s % N_WIN]
        for g in opt.param_groups:
            g["lr"] = lr_at(s, STEPS, 20, cfg["lr"], "constant")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x[None].to(dev), c[None].to(dev))
        contact_loss(to_triu(p.float(), iu), y[None].permute(0, 2, 1).to(dev)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = float(np.mean([contact_loss(to_triu(model(x[None].to(dev), c[None].to(dev)).float(), iu),
                                           Y[w:w + 1]).item() for w, (x, _, c, _) in enumerate(batch)]))
    assert loss < 0.75 * floor, f"memorisation loss {loss:.4f} vs distance-only floor {floor:.4f}"
