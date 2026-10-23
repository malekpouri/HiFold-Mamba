"""Akita (Fudenberg, Kelley & Pollard 2020) ported to PyTorch.

Faithful to calico/basenji manuscripts/akita/params.json and basenji/blocks.py + layers.py
(read 2026-10-23). Keras semantics kept: pre-activation blocks (ReLU -> conv -> BN -> dropout ->
pool), bias-free convs before BN, BN momentum 0.9265 (PyTorch 0.0735), eps 1e-3, second BN of every
residual block initialised with gamma = 0, dilation x1.75 rounded per repeat, he_normal init.

Interface matches HiFold: forward(x (B, L, 4), cells (B, K)) -> (B, K, 448, 448). Akita predicts all
lines jointly through its final dense layer; `cells` selects which output channels are returned.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BN_MOMENTUM = 1 - 0.9265
BN_EPS = 1e-3


def _bn1(c, gamma):
    bn = nn.BatchNorm1d(c, momentum=BN_MOMENTUM, eps=BN_EPS)
    nn.init.constant_(bn.weight, gamma)
    return bn


def _bn2(c, gamma):
    bn = nn.BatchNorm2d(c, momentum=BN_MOMENTUM, eps=BN_EPS)
    nn.init.constant_(bn.weight, gamma)
    return bn


def _he(conv):
    # Keras he_normal = truncated normal, std sqrt(2 / fan_in)
    fan_in = conv.weight[0].numel()
    nn.init.trunc_normal_(conv.weight, std=(2 / fan_in) ** 0.5, a=-2 * (2 / fan_in) ** 0.5, b=2 * (2 / fan_in) ** 0.5)
    return conv


class ConvBlock1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation=1, pool=1, dropout=0.0, gamma=1.0):
        super().__init__()
        self.conv = _he(nn.Conv1d(c_in, c_out, k, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False))
        self.bn = _bn1(c_out, gamma)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):  # (B, C, L)
        return self.pool(self.drop(self.bn(self.conv(F.relu(x)))))


class ConvBlock2d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation=1, dropout=0.0, gamma=1.0):
        super().__init__()
        self.conv = _he(nn.Conv2d(c_in, c_out, k, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False))
        self.bn = _bn2(c_out, gamma)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):  # (B, C, H, W)
        return self.drop(self.bn(self.conv(F.relu(x))))


def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


def _rates(repeat, mult):
    r, out = 1.0, []
    for _ in range(repeat):
        out.append(int(np.round(r)))
        r *= mult
    return out


@dataclass
class AkitaConfig:
    seq_len: int = 1_048_576
    n_bins: int = 512
    crop: int = 32
    n_cells: int = 5
    filters: int = 96
    tower_repeat: int = 10
    res_filters: int = 48
    res_repeat: int = 8
    res_dropout: float = 0.4
    trunk_out: int = 64
    head_filters: int = 48
    head_res_filters: int = 24
    head_res_repeat: int = 6
    head_dropout: float = 0.1
    rate_mult: float = 1.75
    # Off by default: recomputation would update BatchNorm running statistics twice per step
    checkpoint: bool = False

    @property
    def out_bins(self):
        return self.n_bins - 2 * self.crop


class Akita(nn.Module):
    def __init__(self, cfg: AkitaConfig = AkitaConfig()):
        super().__init__()
        self.cfg = cfg
        f = cfg.filters
        assert cfg.seq_len // 2 ** (cfg.tower_repeat + 1) == cfg.n_bins
        self.stem = ConvBlock1d(4, f, 11, pool=2)
        self.tower = nn.ModuleList(ConvBlock1d(f, f, 5, pool=2) for _ in range(cfg.tower_repeat))
        self.res = nn.ModuleList(
            nn.ModuleList([ConvBlock1d(f, cfg.res_filters, 3, dilation=r, gamma=1.0),
                           ConvBlock1d(cfg.res_filters, f, 1, dropout=cfg.res_dropout, gamma=0.0)])
            for r in _rates(cfg.res_repeat, cfg.rate_mult))
        self.trunk_out = ConvBlock1d(f, cfg.trunk_out, 5)
        self.head_in = ConvBlock2d(cfg.trunk_out + 1, cfg.head_filters, 3)
        self.head_res = nn.ModuleList(
            nn.ModuleList([ConvBlock2d(cfg.head_filters, cfg.head_res_filters, 3, dilation=r, gamma=1.0),
                           ConvBlock2d(cfg.head_res_filters, cfg.head_filters, 1, dropout=cfg.head_dropout, gamma=0.0)])
            for r in _rates(cfg.head_res_repeat, cfg.rate_mult))
        self.final = nn.Conv2d(cfg.head_filters, cfg.n_cells, 1)      # Dense(units=5) applied per pixel
        _he(self.final)
        nn.init.zeros_(self.final.bias)
        idx = torch.arange(cfg.n_bins, dtype=torch.float32)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs(), persistent=False)

    def _ckpt(self, fn, *a):
        if self.cfg.checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
        return fn(*a)

    def forward(self, x, cells):
        h = x.transpose(1, 2)                                         # (B, 4, L)
        h = self._ckpt(self.stem, h)
        for blk in self.tower:
            h = self._ckpt(blk, h)
        for a, b in self.res:
            h = h + self._ckpt(lambda t: b(a(t)), h)
        h = self.trunk_out(h)                                          # (B, 64, 512)
        n = h.shape[-1]
        two = 0.5 * (h[:, :, :, None] + h[:, :, None, :])              # one_to_two, mean
        two = torch.cat([two, self.dist.to(two.dtype).expand(h.shape[0], 1, n, n)], 1)   # concat_dist_2d
        two = _sym(self.head_in(two))
        for a, b in self.head_res:
            two = _sym(two + self._ckpt(lambda t: b(a(t)), two))
        c = self.cfg.crop
        two = two[..., c:-c, c:-c]
        out = self.final(two)                                          # (B, n_cells, 448, 448), symmetric
        idx = cells[:, :, None, None].expand(-1, -1, *out.shape[-2:])
        return torch.gather(out, 1, idx)
