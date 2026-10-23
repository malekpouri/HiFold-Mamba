"""HiFold-Mamba (docs/architecture.md).

one-hot (B, 2^20, 4)
 -> RC conv tower, 11 x (conv, norm, GELU, max-pool 2)      -> (B, 512, 2D)   [cell-independent]
 -> n_blocks x [RCPS(BiMamba) + FiLM(cell) + RC FFN], residual -> (B*, 512, 2D)
 -> multi-scale mixer over shallow / intermediate / deep states
 -> symmetric RC-paired biaffine head + learned distance bias, crop 32 -> (B*, 448, 448)

Every component is RC-equivariant, so f(RC(x), c) = rot180(f(x, c)) holds for the whole network
(tests/test_equivariance.py). The conv tower runs once per window and is shared by all cell lines.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hifold.model.rc import RCLayerNorm, RCMaxPool, RCAvgPool, RCPS, rc_conv, rc_linear, tie
from hifold.model.ssm import BiMamba


@dataclass
class HiFoldConfig:
    seq_len: int = 1_048_576
    n_bins: int = 512
    crop: int = 32
    tower_channels: list = field(default_factory=lambda: [64, 64, 96, 96, 128, 128, 160, 160, 192, 192, 256])
    stem_kernel: int = 15
    tower_kernel: int = 5
    d_model: int = 256            # full (paired) width 2D
    n_blocks: int = 8
    d_state: int = 16
    expand: int = 2
    ffn_mult: int = 2
    pyramid: tuple = (1, 2, 4, 8)
    d_pair: int = 64
    n_cells: int = 5
    cell_dim: int = 32
    dropout: float = 0.0
    # Per-cell distance bias b_{c,|i-j|}. Off = the spec in docs/architecture.md (shared b_|i-j|).
    # Without it the model cannot express cell-specific distance profiles (notes/2026-10-21_phase3-5.md).
    cell_dist_bias: bool = False
    checkpoint: bool = True
    scan_chunk: int = 16          # with the compiled scan: 1.34 s/step, 3.8 GiB (notes/2026-10-23_g1-profiling-reference.md)

    @property
    def out_bins(self):
        return self.n_bins - 2 * self.crop


class TowerBlock(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.conv = rc_conv(c_in, c_out, k)
        self.norm = RCLayerNorm(c_out)
        self.pool = RCMaxPool(2)

    def forward(self, x):
        return self.pool(nn.functional.gelu(self.norm(self.conv(x))))


class FFN(nn.Module):
    def __init__(self, d, mult, dropout):
        super().__init__()
        self.up = rc_linear(d, mult * d)
        self.down = rc_linear(mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(nn.functional.gelu(self.up(x))))


class MambaBlock(nn.Module):
    """Pre-norm residual block: x += RCPS(BiMamba)(norm x) * FiLM ; x += FFN(norm x)."""

    def __init__(self, cfg: HiFoldConfig):
        super().__init__()
        half = cfg.d_model // 2
        self.norm1 = RCLayerNorm(cfg.d_model)
        # RCPS inner maps 2D -> D; BiMamba works at width D, so project in first
        self.mixer = RCPS(nn.Sequential(nn.Linear(cfg.d_model, half, bias=False),
                                        BiMamba(half, d_state=cfg.d_state, expand=cfg.expand, chunk=cfg.scan_chunk)))
        self.film = nn.Linear(cfg.cell_dim, cfg.d_model)   # -> (gamma_half, beta_half)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm2 = RCLayerNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.ffn_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cell_emb):
        gb = self.film(cell_emb)                            # (B, D) -> gamma, beta halves
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + tie(gamma)[:, None, :]                # RC-symmetric per-channel scale
        beta = tie(beta)[:, None, :]
        x = x + self.drop(gamma * self.mixer(self.norm1(x)) + beta)
        return x + self.drop(self.ffn(self.norm2(x)))


class MultiScaleMixer(nn.Module):
    """Pools shallow / intermediate / deep states at several scales and sums RC-equivariant projections."""

    def __init__(self, cfg: HiFoldConfig, n_sources=3):
        super().__init__()
        self.pyramid = cfg.pyramid
        self.proj = nn.ModuleList(rc_linear(cfg.d_model, cfg.d_model) for _ in range(n_sources * len(cfg.pyramid)))
        self.norm = RCLayerNorm(cfg.d_model)

    def forward(self, states):
        out = 0
        k = 0
        for h in states:
            for s in self.pyramid:
                p = h if s == 1 else RCAvgPool(s)(h).repeat_interleave(s, dim=1)
                out = out + self.proj[k](p)
                k += 1
        return self.norm(out)


class PairHead(nn.Module):
    """C_ij = sym( (U_i . V_j) / sqrt(2 d) + b_|i-j| ), with U = [Z P, (J Z) P], V = [Z Q, (J Z) Q].

    The [Z, JZ] pairing makes the bilinear form satisfy J M J = M, which together with the
    |i-j| bias and symmetrisation gives f(RC x) = rot180(f(x)).
    """

    def __init__(self, cfg: HiFoldConfig):
        super().__init__()
        self.P = nn.Linear(cfg.d_model, cfg.d_pair, bias=False)
        self.Q = nn.Linear(cfg.d_model, cfg.d_pair, bias=False)
        self.per_cell = cfg.cell_dist_bias
        self.dist_bias = nn.Parameter(torch.zeros(cfg.n_cells if self.per_cell else 1, cfg.n_bins))
        self.scale = (2 * cfg.d_pair) ** -0.5
        self.crop = cfg.crop
        idx = torch.arange(cfg.n_bins)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs(), persistent=False)

    def forward(self, z, cells=None):
        zj = z.flip(-1)
        u = torch.cat([self.P(z), self.P(zj)], -1)
        v = torch.cat([self.Q(z), self.Q(zj)], -1)
        rows = cells if self.per_cell else torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        bias = self.dist_bias[rows][:, self.dist]                          # (B, n, n)
        m = torch.einsum("bid,bjd->bij", u, v) * self.scale + bias
        m = 0.5 * (m + m.transpose(1, 2))
        c = self.crop
        return m[:, c:-c, c:-c] if c else m


class HiFold(nn.Module):
    def __init__(self, cfg: HiFoldConfig = HiFoldConfig()):
        super().__init__()
        self.cfg = cfg
        n_pool = len(cfg.tower_channels)
        assert cfg.seq_len // 2**n_pool == cfg.n_bins, "tower must downsample seq_len to n_bins"
        chans = cfg.tower_channels
        self.stem = rc_conv(4, chans[0], cfg.stem_kernel)
        self.tower = nn.ModuleList(TowerBlock(c_in, c_out, cfg.tower_kernel)
                                   for c_in, c_out in zip([chans[0]] + chans[:-1], chans))
        self.to_model = rc_linear(chans[-1], cfg.d_model)
        self.cell_emb = nn.Embedding(cfg.n_cells, cfg.cell_dim)
        self.blocks = nn.ModuleList(MambaBlock(cfg) for _ in range(cfg.n_blocks))
        self.mixer = MultiScaleMixer(cfg)
        self.head = PairHead(cfg)

    def _ckpt(self, fn, *args):
        if self.cfg.checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode(self, x):
        """One-hot (B, L, 4) -> (B, n_bins, d_model). Cell-independent."""
        h = self.stem(x)
        for blk in self.tower:
            h = self._ckpt(blk, h)
        return self.to_model(h)

    def decode(self, h, cells):
        """h: (B, n_bins, d_model); cells: (B,) long. Returns (B, out_bins, out_bins)."""
        e = self.cell_emb(cells)
        states = []
        n = len(self.blocks)
        marks = {n // 3 - 1, 2 * n // 3 - 1, n - 1}
        for k, blk in enumerate(self.blocks):
            h = self._ckpt(blk, h, e)
            if k in marks:
                states.append(h)
        return self.head(self.mixer(states), cells)

    def forward(self, x, cells):
        """x: (B, L, 4) one-hot; cells: (B, K) long -> (B, K, out, out), tower shared over the K cells."""
        b, k = cells.shape
        h = self.encode(x)
        h = h[:, None].expand(b, k, *h.shape[1:]).reshape(b * k, *h.shape[1:])
        out = self.decode(h, cells.reshape(-1))
        return out.view(b, k, *out.shape[1:])
