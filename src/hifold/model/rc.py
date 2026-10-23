"""Reverse-complement-equivariant building blocks (channels-last: (B, L, C)).

RC action on features: R(h) = h flipped along length AND channels. With one-hot input in
ACGT order, flipping channels is complementing, so R on the input is exactly reverse-complement.
A layer f is RC-equivariant if f(R h) = R f(h).

RCPS (Caduceus-PS style parameter sharing): for any F: (B, L, C_in) -> (B, L, D),
    RCPS(F)(x) = concat[F(x), R(F(R x))]      : (B, L, 2D)
is exactly equivariant, whatever F is.
"""

import contextlib
import copy

import torch
from torch import nn

# Ablation "- RC equivariance" (architecture §5): layers built inside `untied_rc()` keep their shapes and
# FLOPs but give the second (reverse) half its own parameters, so the network is no longer equivariant.
_UNTIED = False


@contextlib.contextmanager
def untied_rc(enabled: bool = True):
    global _UNTIED
    prev, _UNTIED = _UNTIED, enabled
    try:
        yield
    finally:
        _UNTIED = prev


def is_untied() -> bool:
    return _UNTIED


def rc(h: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a channels-last tensor (B, L, C)."""
    return h.flip(-2).flip(-1)


class RCPS(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.inner2 = None
        if _UNTIED:   # independent copy for the second half; re-initialised so the halves differ
            self.inner2 = copy.deepcopy(inner)
            for m in self.inner2.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self, x, *args, **kwargs):
        fwd = self.inner(x, *args, **kwargs)
        if self.inner2 is not None:
            return torch.cat([fwd, self.inner2(x, *args, **kwargs)], dim=-1)
        rev = rc(self.inner(rc(x), *args, **kwargs))
        return torch.cat([fwd, rev], dim=-1)


class ChannelsLastConv1d(nn.Module):
    """Conv1d on (B, L, C) tensors with 'same' padding."""

    def __init__(self, c_in, c_out, k, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=bias)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


def rc_conv(c_in: int, c_out: int, k: int) -> RCPS:
    """RC-equivariant conv: (B, L, c_in) -> (B, L, c_out); c_out must be even."""
    assert c_out % 2 == 0
    return RCPS(ChannelsLastConv1d(c_in, c_out // 2, k))


def rc_linear(c_in: int, c_out: int, bias: bool = True) -> RCPS:
    assert c_out % 2 == 0
    return RCPS(nn.Linear(c_in, c_out // 2, bias=bias))


class RCLayerNorm(nn.Module):
    """LayerNorm over channels with tied affine parameters gamma = [g, flip(g)]."""

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        assert c % 2 == 0
        self.eps = eps
        self.untied = _UNTIED
        self.g = nn.Parameter(torch.ones(c if self.untied else c // 2))
        self.b = nn.Parameter(torch.zeros(c if self.untied else c // 2))

    def forward(self, x):
        x = nn.functional.layer_norm(x, x.shape[-1:], eps=self.eps)
        if self.untied:
            return x * self.g + self.b
        g = torch.cat([self.g, self.g.flip(0)])
        b = torch.cat([self.b, self.b.flip(0)])
        return x * g + b


def tie(v: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., 2D) as [v, flip(v)] so the per-channel vector is RC-symmetric."""
    return torch.cat([v, v.flip(-1)], dim=-1)


class RCMaxPool(nn.Module):
    """Max-pool along length; equivariant when L is divisible by the pool size."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x):
        b, length, c = x.shape
        assert length % self.k == 0, "length must be divisible by the pool size"
        return x.view(b, length // self.k, self.k, c).amax(2)


class RCAvgPool(nn.Module):
    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x):
        b, length, c = x.shape
        assert length % self.k == 0
        return x.view(b, length // self.k, self.k, c).mean(2)
