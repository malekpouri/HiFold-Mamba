"""Training and evaluation loop (docs/architecture.md §3).

python -m hifold.train CONFIG.yaml [key=value ...]

Every run writes runs/<name>-<config hash>-s<seed>/ with the resolved config, git commit,
environment, metrics.jsonl, and the best (EMA) checkpoint. Checkpoint selection: mean
validation SCC (variant set by `select_metric`).
"""

import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hifold.data.dataset import ContactCache
from hifold.eval import metrics as M
from hifold.model.akita import Akita, AkitaConfig
from hifold.model.hifold import HiFold, HiFoldConfig
from hifold.structure import boundary_strength_t, insulation

N_TRIU = 99_681


# --- losses --------------------------------------------------------------------------------

def triu_index(n, offset=2, device="cpu"):
    return torch.triu_indices(n, n, offset=offset, device=device)


def to_triu(mat, iu):
    return mat[..., iu[0], iu[1]]


def from_triu(vec, n, iu):
    """(..., n_triu) -> (..., n, n) symmetric with NaN near the diagonal."""
    out = vec.new_full(vec.shape[:-1] + (n, n), float("nan"))
    out[..., iu[0], iu[1]] = vec
    out[..., iu[1], iu[0]] = vec
    return out


def contact_loss(pred_vec, tgt_vec):
    m = ~torch.isnan(tgt_vec)
    return ((pred_vec - torch.nan_to_num(tgt_vec)) ** 2 * m).sum() / m.sum().clamp_min(1)


def insulation_loss(pred_mat, tgt_mat, w):
    ip, it = insulation(pred_mat.float(), w), insulation(tgt_mat.float(), w)
    m = ~torch.isnan(it) & ~torch.isnan(ip)
    return ((torch.nan_to_num(ip) - torch.nan_to_num(it)) ** 2 * m).sum() / m.sum().clamp_min(1)


def boundary_loss(pred_mat, tgt_mat, w, window=5):
    """L_bnd (architecture §2): masked MSE between the boundary-strength profiles (depth of the log2
    insulation dip relative to its +-window neighbourhood) of prediction and target, same operator on both."""
    bp = boundary_strength_t(insulation(pred_mat.float(), w), window)
    bt = boundary_strength_t(insulation(tgt_mat.float(), w), window)
    m = ~torch.isnan(bt) & ~torch.isnan(bp)
    return ((torch.nan_to_num(bp) - torch.nan_to_num(bt)) ** 2 * m).sum() / m.sum().clamp_min(1)


# --- utilities -----------------------------------------------------------------------------

def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"], text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("seed", "run_root", "name")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def lr_at(step, total, warmup, base, schedule="cosine"):
    if step < warmup:
        return base * (step + 1) / warmup
    if schedule == "constant":
        return base
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


@torch.no_grad()
def evaluate(model, loader, device, amp=True, max_batches=None):
    """Per-(window, line) SCC (both variants) and PCC; returns dict of arrays + window ids."""
    model.eval()
    n = model.cfg.out_bins
    rows = []
    for b, (x, y, cells, idx) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            pred = model(x.to(device), cells.to(device)).float().cpu().numpy()   # (B, K, n, n)
        y = y.numpy()
        for i in range(pred.shape[0]):
            for k in range(pred.shape[1]):
                t = M.triu_to_matrix(y[i, :, k], n)
                p = pred[i, k].astype(np.float64)
                p[np.isnan(t)] = np.nan
                rows.append((int(idx[i]), int(cells[i, k]), M.scc_benchmark(p, t), M.scc_hicrep(p, t),
                             M.pcc_window(M.matrix_to_triu(p), y[i, :, k])))
    a = np.array(rows, dtype=float)
    return {"window": a[:, 0].astype(int), "cell": a[:, 1].astype(int), "scc_benchmark": a[:, 2],
            "scc_hicrep": a[:, 3], "pcc": a[:, 4]}


# --- main loop -----------------------------------------------------------------------------

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_type = cfg.get("model_type", "hifold")
    mcfg = (AkitaConfig if model_type == "akita" else HiFoldConfig)(**cfg["model"])
    run = Path(cfg.get("run_root", "runs")) / f"{cfg['name']}-{config_hash(cfg)}-s{cfg['seed']}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "provenance.json").write_text(json.dumps({
        "commit": git_commit(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "python": sys.version.split()[0], "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    d = cfg["data"]
    train_ds = ContactCache(d["root"], "train", d.get("lines"), d.get("cell_ids"), d.get("max_shift", 11),
                            subset=d.get("train_subset"), seed=cfg["seed"], augment_rc=d.get("augment_rc", False))
    valid_ds = ContactCache(d["root"], "valid", d.get("lines"), d.get("cell_ids"), 0)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=d.get("workers", 2),
                    drop_last=True, generator=torch.Generator().manual_seed(cfg["seed"]))
    vl = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=d.get("workers", 2))

    model = (Akita if model_type == "akita" else HiFold)(mcfg).to(device)
    if cfg.get("init_from"):
        model.load_state_dict(torch.load(cfg["init_from"], map_location=device)["model"], strict=False)
    if cfg.get("optimizer", "adamw") == "sgd":     # Akita's published optimiser
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    ema = EMA(model, cfg.get("ema_decay", 0.999))
    accum = cfg.get("grad_accum", 1)
    total = cfg["steps"]
    iu = triu_index(mcfg.out_bins, device=device)
    lam = cfg.get("struct", {})
    struct_from = lam.get("start_step", None)
    select = cfg.get("select_metric", "scc_hicrep")
    best, step, log = -np.inf, 0, open(run / "metrics.jsonl", "a")
    t0 = time.time()

    while step < total:
        for x, y, cells, _ in tl:
            model.train()
            x, y, cells = x.to(device, non_blocking=True), y.to(device), cells.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x, cells)                                   # (B, K, n, n)
            pred = pred.float()
            pv = to_triu(pred, iu)                                       # (B, K, n_triu)
            tv = y.permute(0, 2, 1)
            loss = contact_loss(pv, tv)
            parts = {"contact": loss.item()}
            if struct_from is not None and step >= struct_from:
                tm = None
                if lam.get("ins", 0) > 0:
                    tm = from_triu(tv, mcfg.out_bins, iu)
                    li = insulation_loss(pred, tm, lam.get("w", 40))
                    loss = loss + lam["ins"] * li
                    parts["ins"] = li.item()
                if lam.get("bnd", 0) > 0:
                    tm = from_triu(tv, mcfg.out_bins, iu) if tm is None else tm
                    lb = boundary_loss(pred, tm, lam.get("w", 40), lam.get("bnd_window", 5))
                    loss = loss + lam["bnd"] * lb
                    parts["bnd"] = lb.item()
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
                for gp in opt.param_groups:
                    gp["lr"] = lr_at(step // accum, total // accum, cfg.get("warmup", 1000) // accum, cfg["lr"],
                                     cfg.get("schedule", "cosine"))
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            step += 1
            if step % cfg.get("log_every", 50) == 0:
                rec = {"step": step, "loss": loss.item(), **parts, "lr": opt.param_groups[0]["lr"],
                       "sec": round(time.time() - t0, 1)}
                if device.type == "cuda":
                    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % cfg.get("eval_every", 1000) == 0 or step == total:
                res = evaluate(ema.shadow, vl, device, max_batches=cfg.get("eval_batches"))
                score = float(np.nanmean(res[select]))
                rec = {"step": step, "eval": True, **{k: float(np.nanmean(v)) for k, v in res.items()
                                                      if k in ("scc_benchmark", "scc_hicrep", "pcc")}}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if score > best:
                    best = score
                    torch.save({"model": ema.shadow.state_dict(), "step": step, "config": cfg}, run / "best.pt")
            if step >= total:
                break
    log.close()
    return run, best


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for o in overrides:
        key, val = o.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


if __name__ == "__main__":
    run, best = train(load_config(sys.argv[1], sys.argv[2:]))
    print(f"{run}: best validation score {best:.4f}")
