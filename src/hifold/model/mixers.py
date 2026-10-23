"""Alternative sequence mixers for the matched baselines (protocol §4, architecture §5).

Each maps (B, L, D) -> (B, L, D) and is used inside the same RCPS wrapper as BiMamba, so every
variant stays RC-equivariant and only the mixer changes. Sizes are chosen to match BiMamba's
parameter count (tests/test_baselines_models.py checks within 5 %).
"""

import torch
import torch.nn.functional as F
from torch import nn


class AttentionMixer(nn.Module):
    """Multi-head self-attention over bins with a learned per-head bias on |i - j| (no absolute positions).

    The |i-j| bias is symmetric, so the mixer has no preferred direction (like BiMamba).
    """

    def __init__(self, d_model, n_bins=512, heads=8, d_attn=None, dropout=0.0):
        super().__init__()
        d_attn = d_attn or d_model
        assert d_attn % heads == 0
        self.h, self.dh = heads, d_attn // heads
        self.qkv = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.out = nn.Linear(d_attn, d_model, bias=False)
        self.rel = nn.Parameter(torch.zeros(heads, n_bins))
        idx = torch.arange(n_bins)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs(), persistent=False)
        self.dropout = dropout

    def forward(self, x):
        b, L, _ = x.shape
        q, k, v = self.qkv(x).view(b, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = self.rel[:, self.dist[:L, :L]].to(q.dtype)                    # (h, L, L)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, L, -1))


class DilatedConvMixer(nn.Module):
    """Stack of dilated residual 1D convolutions (rates x2), symmetric padding, GELU."""

    def __init__(self, d_model, width=None, kernel=3, layers=4):
        super().__init__()
        width = width or d_model
        self.inp = nn.Linear(d_model, width)
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel, padding=(kernel // 2) * 2 ** i, dilation=2 ** i) for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.out = nn.Linear(width, d_model)

    def forward(self, x):
        h = self.inp(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(F.gelu(norm(h)).transpose(1, 2)).transpose(1, 2)
        return self.out(h)
