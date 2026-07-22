"""FISTA engine: minimize sum(smooth terms) + coordinatewise prox.

The engine owns only the optimization mechanics; terms own the math.
Step size = 1 / (1.05 * sum of term curvature bounds).  Per iteration a
fresh :class:`IterCtx` is bound to the extrapolated point so expensive
intermediates are computed once and shared across terms.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from ..terms.base import CoordProx, IterCtx, SmoothTerm


class Fista:
    def __init__(self, n_iter: int = 150, safety: float = 1.05):
        self.n_iter = int(n_iter)
        self.safety = float(safety)

    def minimize(
        self,
        op,
        terms: Sequence[SmoothTerm],
        prox: CoordProx,
        q0: torch.Tensor | None = None,
        callback: Callable[[int, IterCtx], None] | None = None,
    ) -> torch.Tensor:
        L = sum(t.curvature() for t in terms)
        step = 1.0 / (self.safety * max(L, 1e-12))
        x = (torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
             if q0 is None else prox(q0.to(op.device, op.dtype), 0.0))
        y = x.clone()
        t = 1.0
        for k in range(self.n_iter):
            ctx = IterCtx(y, op)
            grad = torch.zeros_like(y)
            for term in terms:
                term.grad_into(ctx, grad)
            x_new = prox(y - step * grad, step)
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new
            if callback is not None:
                callback(k, ctx)
        return x
