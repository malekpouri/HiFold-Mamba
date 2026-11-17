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
import os
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
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"], text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        # inside the Docker image there is no .git; the image records the commit it was built from
        return os.environ.get("HIFOLD_COMMIT", "unknown")


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("seed", "run_root", "name")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def run_dir(cfg):
    return Path(cfg.get("run_root", "runs")) / f"{cfg['name']}-{config_hash(cfg)}-s{cfg['seed']}"


def write_status(run, state, **extra):
    """status.json for monitoring (hifold.monitor); written atomically, never read by training."""
    rec = {"state": state, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
           "host": os.uname().nodename, **extra}
    tmp = Path(run) / "status.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(Path(run) / "status.json")


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


# --- data and initialisation ---------------------------------------------------------------

# Robustness split (protocol §1, manuscript §2.13): chromosome held-out, windows from all official splits.
CHROM_HOLDOUT = {"test": ("chr1", "chr8", "chr12"), "valid": ("chr3", "chr6")}


def make_datasets(d, seed):
    """Training and validation datasets for a config's `data` section.

    split_scheme "official" (default): the DNALongBench train/valid files. Optional training subset from
      subset_file + subset_regime (H3 limited-data regimes, configs/h3_subsets.json).
    split_scheme "chrom_holdout": the windows of all three official files, re-assigned by chromosome
      (CHROM_HOLDOUT). Its test chromosomes are never loaded here.
    """
    lines, cells, shift, rc = d.get("lines"), d.get("cell_ids"), d.get("max_shift", 11), d.get("augment_rc", False)
    scheme = d.get("split_scheme", "official")
    if scheme == "official":
        subset = d.get("train_subset")
        if d.get("subset_file"):
            subset = json.loads(Path(d["subset_file"]).read_text())[str(d["subset_regime"])]["windows"]
        return (ContactCache(d["root"], "train", lines, cells, shift, subset=subset, seed=seed, augment_rc=rc),
                ContactCache(d["root"], "valid", lines, cells, 0))
    if scheme == "chrom_holdout":
        held = set(CHROM_HOLDOUT["test"]) | set(CHROM_HOLDOUT["valid"])
        tr, va = [], []
        for split in ("train", "valid", "test"):
            wins = ContactCache(d["root"], split, lines, cells, 0).windows
            t_idx = [i for i, w in enumerate(wins) if w[0] not in held]
            v_idx = [i for i, w in enumerate(wins) if w[0] in CHROM_HOLDOUT["valid"]]
            if t_idx:
                tr.append(ContactCache(d["root"], split, lines, cells, shift, subset=t_idx, seed=seed, augment_rc=rc))
            if v_idx:
                va.append(ContactCache(d["root"], split, lines, cells, 0, subset=v_idx))
        return torch.utils.data.ConcatDataset(tr), torch.utils.data.ConcatDataset(va)
    raise ValueError(f"unknown split_scheme {scheme}")


def load_expanding(model, state):
    """Load a checkpoint into a model that may have more cell lines (H3: 5 trained + 1 new line).

    Per-cell tables (cell embedding, distance bias) whose first dimension grew keep the trained rows; the
    new rows keep the model's fresh initialisation (random embedding, zero distance bias). Every other
    tensor must match exactly.
    """
    own = model.state_dict()
    grown = []
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            if own[k].shape[1:] != v.shape[1:] or own[k].shape[0] < v.shape[0]:
                raise ValueError(f"{k}: checkpoint {tuple(v.shape)} does not fit model {tuple(own[k].shape)}")
            new = own[k].clone()
            new[: v.shape[0]] = v
            state = {**state, k: new}
            grown.append(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [m for m in missing if not m.endswith("dist")]:
        raise ValueError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return grown


# --- main loop -----------------------------------------------------------------------------

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_type = cfg.get("model_type", "hifold")
    mcfg = (AkitaConfig if model_type == "akita" else HiFoldConfig)(**cfg["model"])
    run = run_dir(cfg)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "provenance.json").write_text(json.dumps({
        "commit": git_commit(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "python": sys.version.split()[0], "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    train_ds, valid_ds = make_datasets(cfg["data"], cfg["seed"])
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["data"].get("workers", 2),
                    drop_last=True, generator=torch.Generator().manual_seed(cfg["seed"]))
    vl = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["data"].get("workers", 2))

    model = (Akita if model_type == "akita" else HiFold)(mcfg).to(device)
    if cfg.get("init_from"):
        load_expanding(model, torch.load(cfg["init_from"], map_location=device)["model"])
    if cfg.get("optimizer", "adamw") == "sgd":     # Akita's published optimiser
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    ema = EMA(model, cfg.get("ema_decay", 0.999))
    accum = cfg.get("grad_accum", 1)
    total = cfg["steps"]
    save_at = set(cfg.get("save_at") or [])          # full training state for forks (λ pilot, manuscript §2.6)
    assert all(s % accum == 0 for s in save_at), "save_at steps must fall on optimiser steps"
    start = 0
    if cfg.get("resume_from"):
        # Fork: weights, EMA, optimiser state and step continue exactly; the LR schedule continues from the
        # step. The data order after the fork is a fresh shuffle (the loader position is not restored).
        st = torch.load(cfg["resume_from"], map_location=device)
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        assert start % accum == 0 and start < total, f"cannot resume at step {start} of {total}"
    iu = triu_index(mcfg.out_bins, device=device)
    lam = cfg.get("struct", {})
    struct_from = lam.get("start_step", None)
    select = cfg.get("select_metric", "scc_hicrep")
    best, step, log = -np.inf, start, open(run / "metrics.jsonl", "a")
    t0 = time.time()
    write_status(run, "running", steps=total)

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
            if step in save_at:
                torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                            "step": step, "config": cfg}, run / f"fork_step{step}.pt")
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
    write_status(run, "finished", steps=total, best=float(best), select_metric=select)
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
    import signal
    # docker stop sends SIGTERM (PID 1 would otherwise ignore it): turn it into a clean, logged exit
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("stopped by SIGTERM"))
    cfg = load_config(sys.argv[1], sys.argv[2:])
    try:
        run, best = train(cfg)
    except BaseException as e:          # also Ctrl-C / docker stop: record why the run ended
        import traceback
        if run_dir(cfg).is_dir():
            write_status(run_dir(cfg), "failed", error=f"{type(e).__name__}: {e}",
                         traceback=traceback.format_exc()[-4000:])
        raise
    print(f"{run}: best validation score {best:.4f}")
