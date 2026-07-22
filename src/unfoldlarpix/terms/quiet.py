"""Quiet-bin inequality: beta/2 ||relu(conv(q)[quiet] - thr)||^2.

Silence on never-fired pixels, per bin.  Measured status (FINDINGS
items 17/18): truth-feasible (penalty exactly 0 at the truth) but
INACTIVE under the deconv support — kept for wide/absent-support
configurations.
"""
from __future__ import annotations

import torch

from .base import IterCtx


class QuietHinge:
    def __init__(self, op, quiet_mask, threshold: float, beta: float = 1.0):
        self.op = op
        self.mask = op.to_tensor(quiet_mask, torch.bool)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self._zero = torch.zeros((), dtype=op.dtype, device=op.device)

    def _viol(self, ctx: IterCtx) -> torch.Tensor:
        return torch.where(
            self.mask,
            torch.clamp(ctx.block_pred - self.threshold, min=0.0),
            self._zero)

    def value(self, ctx: IterCtx) -> torch.Tensor:
        v = self._viol(ctx)
        return 0.5 * self.beta * (v * v).sum()

    def grad_into(self, ctx: IterCtx, out: torch.Tensor) -> None:
        v = self._viol(ctx)
        if bool(v.any()):
            out += self.beta * self.op.conv_adjoint(v)

    def curvature(self) -> float:
        return 2.0 * self.beta
