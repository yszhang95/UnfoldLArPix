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
        """``censor_reset`` / ``censor_arm`` are per-pixel boundaries in BIN
        units and may be FRACTIONAL.  The boundary bin of the running sum
        contributes with its overlap fraction (uniform-current-within-bin,
        the same first-order convention as the operator's window sampling);
        integer boundaries reproduce the old hard mask on the reset side.
        The armed test asks whether the END of bin b, at (b+1)*B, is past
        the re-arm instant -- C(b) is the accumulator at that instant.
        """
        if norm not in ("l1", "l2"):
            raise ValueError(f"censor norm must be l1|l2, got {norm}")
        self.op = op
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.norm = norm
        nt = op.block_shape[2]
        r_t = op.to_tensor(np.asarray(censor_reset, np.float64))
        a_t = op.to_tensor(np.asarray(censor_arm, np.float64))
        t_axis = torch.arange(nt, device=op.device, dtype=op.dtype)[None, None, :]
        # fraction of bin b after the CSA restart: clip(b+1 - reset, 0, 1)
        self.w = torch.clamp(t_axis + 1.0 - r_t[:, :, None], 0.0, 1.0)
        self.armed = ((t_axis + 1.0 >= a_t[:, :, None])
                      & (t_axis < float(censor_end)))
        self._zero = torch.zeros((), dtype=op.dtype, device=op.device)
        self._neg_inf = torch.tensor(float("-inf"), dtype=op.dtype,
                                     device=op.device)
        self._curv = (2.0 * self.beta * max(self._power_iter(), 1.0)
                      if norm == "l2" else 0.0)

    @classmethod
    def from_hits(cls, op, hits: HitsView, block_offset, *,
                  csa_reset_time: float, threshold: float,
                  npad_bins: int = 50, beta: float = 1.0,
                  margin: float = 3.0, norm: str = "l2", bin_ticks=None):
        """Build region boundaries from the DATA via typed accessors.

        ``bin_ticks`` is the operator's time-bin width in fine ticks; it
        defaults to the physical ``hits.adc_hold_delay`` but MUST be set to
        the operator bin (e.g. adc_hold_delay/time_subbin) when the operator
        runs on a sub-divided time grid, or the reset/arm boundaries land at
        the wrong bin index.
        """
        B = hits.adc_hold_delay if bin_ticks is None else float(bin_ticks)
        nx, ny, nt = op.block_shape
        reset = np.full((nx, ny), float(npad_bins))      # never-fired
        arm = np.full((nx, ny), float(npad_bins))
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
            reset[x, y] = min(max((lli + csa_reset_time) / B, 0.0), float(nt))
            arm[x, y] = min(max(rai / B, 0.0), float(nt))
        return cls(op, reset, arm, censor_end=nt - npad_bins,
                   threshold=threshold + margin, beta=beta, norm=norm)

    def _power_iter(self, n: int = 6) -> float:
        """Worst-case (full-span cumulative row) linearization bound."""
        g = torch.Generator(device="cpu").manual_seed(1)
        xc = torch.randn(self.op.q_shape, generator=g, dtype=self.op.dtype)
        xc = (xc / torch.linalg.vector_norm(xc)).to(self.op.device)
        lam = 0.0
        for _ in range(n):
            b = self.w * self.op.conv(xc)
            row = b.sum(dim=2)
            yc = self.op.conv_adjoint(self.w * row[:, :, None])
            lam = float(torch.linalg.vector_norm(yc))
            if lam <= 0:
                break
            xc = yc / lam
        return lam

    def _peaks(self, ctx: IterCtx):
        C = torch.cumsum(self.w * ctx.block_pred, dim=2)
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
        upto = (t_axis <= arg[:, :, None]).to(self.op.dtype)
        g_c = self.w * upto * coeff[:, :, None]
        out += self.beta * self.op.conv_adjoint(g_c)

    def curvature(self) -> float:
        return self._curv
