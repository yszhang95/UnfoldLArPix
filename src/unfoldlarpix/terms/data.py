"""Data-fidelity term: 1/2 ||A q - d||^2."""
from __future__ import annotations

import torch

from .base import IterCtx


class DataFidelity:
    """1/2 ||sample(conv(q)) - target||^2.

    ``target`` defaults to the operator's recorded ``d`` and is the ONLY
    sanctioned place to express background subtraction (e.g. the
    final-refit frozen-faint charge: target = d - A q_faint).  The
    operator itself is immutable — never mutate ``op.d``.
    """

    def __init__(self, op, target: torch.Tensor | None = None):
        self.op = op
        self.target = op.d if target is None else target

    def _resid(self, ctx: IterCtx) -> torch.Tensor:
        return self.op.sample(ctx.block_pred) - self.target

    def value(self, ctx: IterCtx) -> torch.Tensor:
        r = self._resid(ctx)
        return 0.5 * (r * r).sum()

    def grad_into(self, ctx: IterCtx, out: torch.Tensor) -> None:
        out += self.op.conv_adjoint(self.op.sample_adjoint(self._resid(ctx)))

    def curvature(self) -> float:
        return self.op.lipschitz
