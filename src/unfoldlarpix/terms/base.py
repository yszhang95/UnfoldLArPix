"""Objective-term protocol and the per-iteration shared context.

The objective is  F(q) = sum_i smooth_i(q)  +  prox-able part, where the
prox part must be COORDINATEWISE (positivity + weighted L1 + support
compose into one closed-form prox; non-separable prox terms require a
different engine, e.g. ADMM — that is the sanctioned escape hatch).

``IterCtx`` is a lazily-cached view of the current iterate: expensive
intermediates (the block-space convolution) are computed by whichever
term asks first and reused by the rest — terms never call ``op.conv``
directly.
"""
from __future__ import annotations

from typing import Protocol

import torch


class IterCtx:
    """Lazy per-iteration cache bound to one iterate ``q``."""

    def __init__(self, q: torch.Tensor, op):
        self.q = q
        self.op = op
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def block_pred(self) -> torch.Tensor:
        """conv(q): per-bin collected charge on the block grid."""
        if "bp" not in self._cache:
            self._cache["bp"] = self.op.conv(self.q)
        return self._cache["bp"]

    @property
    def q_fft_t(self) -> torch.Tensor:
        """rfft of q along time (spectral terms)."""
        if "qft" not in self._cache:
            self._cache["qft"] = torch.fft.rfft(self.q, dim=2)
        return self._cache["qft"]


class SmoothTerm(Protocol):
    """A differentiable objective term."""

    def value(self, ctx: IterCtx) -> torch.Tensor: ...

    def grad_into(self, ctx: IterCtx, out: torch.Tensor) -> None:
        """Accumulate dF/dq into ``out`` (same shape as q)."""
        ...

    def curvature(self) -> float:
        """Upper bound on the Hessian norm (contribution to the FISTA
        step bound).  Zero for terms handled as subgradients."""
        ...


class CoordProx:
    """The single coordinatewise prox: weighted L1 + positivity + support.

    prox_step(v) = max(v - step * alpha, 0) * support
    """

    def __init__(self, alpha: torch.Tensor | float,
                 support: torch.Tensor | None = None):
        self.alpha = alpha
        self.support = support

    def __call__(self, v: torch.Tensor, step: float) -> torch.Tensor:
        out = torch.clamp(v - step * self.alpha, min=0.0)
        if self.support is not None:
            out = out * self.support
        return out
