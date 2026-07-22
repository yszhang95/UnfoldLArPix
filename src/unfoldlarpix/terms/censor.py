"""Exact ZS censoring: silence of ARMED discriminators (FINDINGS item 19).

Statistic per pixel: running MAX (bipolar response — the peak, not the
total) of the cumulative REFERENCED at the CSA restart (last latch +
csa_reset), taken ONLY over the armed window [re-arm (col4), end of the
readout-covered range).  Penalty: hinge on (peak - threshold);
``norm="l2"`` squared hinge (soft; adds curvature -> needs ~4x
iterations), ``norm="l1"`` linear hinge (exact penalty, zero curvature,
no step cost).

Measured (nb4/nb1): truth-feasible after the col3 fix; INERT in the
dense regime; in the sparse regime (nb1) it improves r by ~+0.02 and
halves the integral bias.  Region boundaries are built from the typed
hits accessors — bare column indexing caused the original bug.
"""
from __future__ import annotations

import numpy as np
import torch

from ..io.hits import HitsView
from .base import IterCtx


class CensorRunningMax:
    def __init__(self, op, censor_reset: np.ndarray, censor_arm: np.ndarray,
                 censor_end: int, threshold: float, beta: float = 1.0,
                 norm: str = "l2"):
        if norm not in ("l1", "l2"):
            raise ValueError(f"censor norm must be l1|l2, got {norm}")
        self.op = op
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.norm = norm
        nt = op.block_shape[2]
        r_idx = op.to_tensor(np.asarray(censor_reset, np.int64), torch.long)
        a_idx = op.to_tensor(np.asarray(censor_arm, np.int64), torch.long)
        t_axis = torch.arange(nt, device=op.device)[None, None, :]
        self.ref = t_axis >= r_idx[:, :, None]
        self.armed = (t_axis >= a_idx[:, :, None]) & (t_axis < int(censor_end))
        self._zero = torch.zeros((), dtype=op.dtype, device=op.device)
        self._neg_inf = torch.tensor(float("-inf"), dtype=op.dtype,
                                     device=op.device)
        self._curv = (2.0 * self.beta * max(self._power_iter(), 1.0)
                      if norm == "l2" else 0.0)

    @classmethod
    def from_hits(cls, op, hits: HitsView, block_offset, *,
                  csa_reset_time: float, threshold: float,
                  npad_bins: int = 50, beta: float = 1.0,
                  margin: float = 3.0, norm: str = "l2"):
        """Build region boundaries from the DATA via typed accessors."""
        B = hits.adc_hold_delay
        nx, ny, nt = op.block_shape
        reset = np.full((nx, ny), npad_bins, np.int64)   # never-fired
        arm = np.full((nx, ny), npad_bins, np.int64)
        px = (hits.pixel_x - int(block_offset[0])).astype(int)
        py = (hits.pixel_y - int(block_offset[1])).astype(int)
        ll = hits.last_latch - float(block_offset[2])
        ra = hits.rearm - float(block_offset[2])
        trig = hits.trigger
        latest: dict[tuple[int, int], tuple[float, float, float]] = {}
        for i in range(len(px)):
            if not (0 <= px[i] < nx and 0 <= py[i] < ny):
                continue
            k = (px[i], py[i])
            if k not in latest or trig[i] > latest[k][0]:
                latest[k] = (trig[i], ll[i], ra[i])
        for (x, y), (_t, lli, rai) in latest.items():
            reset[x, y] = min(max(int(np.ceil((lli + csa_reset_time) / B)), 0), nt)
            arm[x, y] = min(max(int(np.ceil(rai / B)), 0), nt)
        return cls(op, reset, arm, censor_end=nt - npad_bins,
                   threshold=threshold + margin, beta=beta, norm=norm)

    def _power_iter(self, n: int = 6) -> float:
        """Worst-case (full-span cumulative row) linearization bound."""
        g = torch.Generator(device="cpu").manual_seed(1)
        xc = torch.randn(self.op.q_shape, generator=g, dtype=self.op.dtype)
        xc = (xc / torch.linalg.vector_norm(xc)).to(self.op.device)
        lam = 0.0
        for _ in range(n):
            b = torch.where(self.ref, self.op.conv(xc), self._zero)
            row = b.sum(dim=2)
            yc = self.op.conv_adjoint(
                torch.where(self.ref, row[:, :, None].expand_as(b),
                            self._zero))
            lam = float(torch.linalg.vector_norm(yc))
            if lam <= 0:
                break
            xc = yc / lam
        return lam

    def _peaks(self, ctx: IterCtx):
        C = torch.cumsum(torch.where(self.ref, ctx.block_pred, self._zero),
                         dim=2)
        Cm = torch.where(self.armed, C, self._neg_inf)
        peak, arg = Cm.max(dim=2)
        viol = torch.where(torch.isfinite(peak),
                           torch.clamp(peak - self.threshold, min=0.0),
                           self._zero)
        return viol, arg

    def value(self, ctx: IterCtx) -> torch.Tensor:
        v, _ = self._peaks(ctx)
        return (0.5 * self.beta * (v * v).sum() if self.norm == "l2"
                else self.beta * v.sum())

    def grad_into(self, ctx: IterCtx, out: torch.Tensor) -> None:
        viol, arg = self._peaks(ctx)
        if not bool(viol.any()):
            return
        coeff = viol if self.norm == "l2" else (viol > 0).to(self.op.dtype)
        nt = ctx.block_pred.shape[2]
        t_axis = torch.arange(nt, device=self.op.device)[None, None, :]
        upto = t_axis <= arg[:, :, None]
        g_c = torch.where(self.ref & upto,
                          coeff[:, :, None].expand(*coeff.shape, nt),
                          self._zero)
        out += self.beta * self.op.conv_adjoint(g_c)

    def curvature(self) -> float:
        return self._curv
