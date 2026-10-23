"""Structural operators on contact maps, shared by the loss and the evaluation.

The same functions are applied to predicted and true maps (CLAUDE.md, protocol G2):
  g(x) = exp(2 * tanh(x / 2))                          (soft-clipped exp, bounded to [e^-2, e^2])
  I_i  = mean of g(C)[a, b] over a in [i-w, i), b in (i, i+w], b - a >= d_min, both valid
Insulation is defined for i in [w, n - w); other positions are NaN.
"""

import numpy as np
import torch

D_MIN = 2


def g_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(2.0 * torch.tanh(x / 2.0))


def insulation(cmap: torch.Tensor, w: int = 20, valid: torch.Tensor = None,
               transform: bool = True) -> torch.Tensor:
    """Crossing-square insulation.

    cmap:  (..., n, n) symmetric map in target space (log O/E); NaN allowed, treated as invalid.
    valid: optional (..., n, n) bool mask of usable pixels (AND-ed with ~isnan).
    Returns (..., n) insulation, NaN outside [w, n - w) or where no valid pixel exists.
    """
    n = cmap.shape[-1]
    ok = ~torch.isnan(cmap)
    if valid is not None:
        ok = ok & valid
    # exclude the near-diagonal band (|a - b| < D_MIN); a < i < b already gives b - a >= 2
    # nan_to_num first: torch.where would otherwise backpropagate NaN through the masked branch
    safe = torch.nan_to_num(cmap, nan=0.0)
    x = torch.where(ok, g_transform(safe) if transform else safe, torch.zeros((), dtype=cmap.dtype, device=cmap.device))
    okf = ok.to(x.dtype)

    def integral(t):  # zero-padded 2D cumulative sum: S[r, c] = sum t[:r, :c]
        s = t.cumsum(-1).cumsum(-2)
        return torch.nn.functional.pad(s, (1, 0, 1, 0))

    S, C = integral(x), integral(okf)
    i = torch.arange(w, n - w, device=cmap.device)
    r0, r1 = i - w, i          # rows a in [i-w, i)
    c0, c1 = i + 1, i + w + 1  # cols b in (i, i+w]

    def box(T):
        return T[..., r1, c1] - T[..., r0, c1] - T[..., r1, c0] + T[..., r0, c0]

    total, count = box(S), box(C)
    ins = total / count.clamp_min(1)
    ins = torch.where(count > 0, ins, torch.full_like(ins, float("nan")))
    out = torch.full(cmap.shape[:-2] + (n,), float("nan"), dtype=x.dtype, device=cmap.device)
    out[..., w:n - w] = ins
    return out


def insulation_np(cmap: np.ndarray, w: int = 20) -> np.ndarray:
    return insulation(torch.as_tensor(cmap, dtype=torch.float64), w=w).numpy()


def boundary_strength_t(ins: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Differentiable `boundary_strength` over the last dim: nan-mean of log2(I) in [k-window, k+window]
    (clipped at the edges) minus log2(I_k). NaN where I is NaN. Insulation is >= e^-2, so log2 is safe."""
    li = torch.log2(ins)
    ok = ~torch.isnan(li)
    v = torch.where(ok, li, torch.zeros_like(li))

    def run(t):  # sum over the clipped window via a zero-padded cumulative sum
        c = torch.nn.functional.pad(t.cumsum(-1), (1, 0))
        n = t.shape[-1]
        k = torch.arange(n, device=t.device)
        hi, lo = (k + window + 1).clamp(max=n), (k - window).clamp(min=0)
        return c[..., hi] - c[..., lo]

    out = run(v) / run(ok.to(v.dtype)).clamp_min(1) - v
    return torch.where(ok, out, torch.full_like(out, float("nan")))


def boundary_strength(ins: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-bin boundary score: depth of the insulation dip relative to its +-window neighbourhood.

    Uses log2 of insulation; higher = stronger boundary. NaN where insulation is NaN.
    """
    li = np.log2(ins)
    n = len(li)
    out = np.full(n, np.nan)
    for k in range(n):
        if np.isnan(li[k]):
            continue
        lo, hi = max(0, k - window), min(n, k + window + 1)
        neigh = li[lo:hi]
        out[k] = np.nanmean(neigh) - li[k]
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney); NaN scores are dropped."""
    from scipy.stats import rankdata
    m = ~np.isnan(scores)
    s, y = scores[m], labels[m].astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    m = ~np.isnan(scores)
    s, y = scores[m], labels[m].astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y].sum() / y.sum())
