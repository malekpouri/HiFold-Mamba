"""Mamba-1 selective state-space block in pure PyTorch (no custom kernels).

    h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t x_t,     y_t = C_t . h_t + D x_t

`selective_scan_ref` is a sequential reference. `selective_scan` evaluates the same recurrence
in chunks: inside a chunk with exact pairwise decays exp(cumsum_t - cumsum_s) (all exponents
<= 0, so no overflow), across chunks a short sequential carry. Both are exact; a unit test
checks they agree.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def selective_scan_ref(x, dt, A, B, C, D):
    """x, dt: (b, L, E); A: (E, N); B, C: (b, L, N); D: (E,). Returns (b, L, E)."""
    b, L, E = x.shape
    h = x.new_zeros(b, E, A.shape[1])
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t, :, None] * A)                         # (b, E, N)
        h = dA * h + (dt[:, t, :, None] * B[:, t, None, :]) * x[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + x * D


def selective_scan(x, dt, A, B, C, D, chunk: int = 16):
    b, L, E = x.shape
    N = A.shape[1]
    assert L % chunk == 0, "sequence length must be a multiple of the chunk size"
    nc = L // chunk
    xs = x.view(b, nc, chunk, E)
    dts = dt.view(b, nc, chunk, E)
    Bs = B.view(b, nc, chunk, N)
    Cs = C.view(b, nc, chunk, N)
    causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=x.device))
    # All chunks at once (few large kernels instead of many small ones); only the carry is sequential.
    logd = dts[..., None] * A                                           # (b, c, T, E, N), <= 0
    cs = logd.cumsum(2)                                                 # cumulative log-decay per chunk
    u = dts[..., None] * Bs[:, :, :, None, :] * xs[..., None]           # (b, c, s, E, N)
    # intra-chunk: y_t += sum_{s<=t} C_t . exp(cs_t - cs_s) u_s
    seg = cs[:, :, :, None] - cs[:, :, None, :]                         # (b, c, t, s, E, N)
    seg = seg.masked_fill(~causal[None, None, :, :, None, None], float("-inf"))
    intra = torch.einsum("bctsen,bcsen,bctn->bcte", torch.exp(seg), u, Cs)
    # chunk summaries: state added within chunk c, and total decay across it
    last = cs[:, :, -1]                                                 # (b, c, E, N)
    added = (torch.exp(last[:, :, None] - cs) * u).sum(2)               # (b, c, E, N)
    decay = torch.exp(last)
    carries = [x.new_zeros(b, E, N)]                                    # state entering chunk c
    for c in range(nc - 1):
        carries.append(decay[:, c] * carries[-1] + added[:, c])
    h = torch.stack(carries, 1)                                         # (b, c, E, N)
    inter = torch.einsum("bcten,bcen,bctn->bcte", torch.exp(cs), h, Cs)
    return (intra + inter).reshape(b, L, E) + x * D


class Mamba(nn.Module):
    """Unidirectional Mamba-1 block: in_proj -> causal depthwise conv -> SiLU -> selective SSM -> gate -> out_proj."""

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_rank=None, chunk=16,
                 dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        e = expand * d_model
        self.e, self.n, self.chunk = e, d_state, chunk
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * e, bias=False)
        self.conv = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e, bias=True)
        # Mamba init: dt in [dt_min, dt_max] log-uniform via inverse softplus of the bias
        dt = torch.exp(torch.rand(e) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d_model, bias=False)

    def forward(self, u):
        b, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)        # causal
        x = F.silu(x)
        dbc = self.x_proj(x)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.n, self.n], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        # the scan runs in fp32 regardless of autocast (stability of exp/cumsum)
        with torch.autocast(device_type=u.device.type, enabled=False):
            y = selective_scan(x.float(), dt.float(), A, Bm.float(), Cm.float(), self.D.float(), self.chunk)
        y = y.to(u.dtype) * F.silu(z)
        return self.out_proj(y)


class BiMamba(nn.Module):
    """Bidirectional Mamba with tied weights: Mamba(x) + flipL(Mamba(flipL(x)))."""

    def __init__(self, d_model, **kw):
        super().__init__()
        self.mamba = Mamba(d_model, **kw)

    def forward(self, x):
        return self.mamba(x) + self.mamba(x.flip(1)).flip(1)
