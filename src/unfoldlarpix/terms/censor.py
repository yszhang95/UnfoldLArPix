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

:func:`pre_trigger_censors` applies the same statistic to the silent
intervals BEFORE each trigger, which ``from_hits`` does not cover and
which ``split_trigger`` constrains only at the endpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from ..io.hits import HitsView
from .base import IterCtx


class CensorRunningMax:
    def __init__(self, op, censor_reset: np.ndarray, censor_arm: np.ndarray,
                 censor_end, threshold: float, beta: float = 1.0,
                 norm: str = "l2"):
        """``censor_reset`` / ``censor_arm`` are per-pixel boundaries in BIN
        units and may be FRACTIONAL.  The boundary bin of the running sum
        contributes with its overlap fraction (uniform-current-within-bin,
        the same first-order convention as the operator's window sampling);
        integer boundaries reproduce the old hard mask on the reset side.
        The armed test asks whether the END of bin b, at (b+1)*B, is past
        the re-arm instant -- C(b) is the accumulator at that instant.

        ``censor_end`` closes the armed window and is either a scalar (the
        end of the readout-covered range, the post-latch use) or a per-pixel
        (nx, ny) array in BIN units (the pre-trigger use, where each pixel's
        window closes at its own next trigger).  Bin b is checked when its
        END is at or before that instant, ``b + 1 <= end``; for the integer
        scalar of the post-latch use that is the same set of bins as the
        original ``b < end`` test.
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
        e = np.asarray(censor_end, np.float64)
        if e.ndim == 0:
            hi = t_axis + 1.0 <= float(e)
        elif e.shape == tuple(op.block_shape[:2]):
            hi = t_axis + 1.0 <= op.to_tensor(e)[:, :, None]
        else:
            raise ValueError(
                f"censor_end must be scalar or {tuple(op.block_shape[:2])}, "
                f"got shape {e.shape}")
        self.armed = (t_axis + 1.0 >= a_t[:, :, None]) & hi
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


def pre_trigger_censors(op, hits: HitsView, block_offset, *,
                        csa_reset_time: float, threshold: float,
                        acq_start=None, npad_bins: int = 50,
                        beta: float = 1.0, margin: float = 3.0,
                        norm: str = "l1", bin_ticks=None,
                        one_tick: float = 1.0, close_back: float = 20.0,
                        include_post_reset: bool = False,
                        ) -> list[CensorRunningMax]:
    """Silence BEFORE each trigger, as a list of :class:`CensorRunningMax`.

    ``CensorRunningMax.from_hits`` covers only the interval AFTER a pixel's
    last burst.  The silent intervals before each trigger are not covered by
    it, and ``split_trigger`` states only the ENDPOINT there: its pseudo row
    asserts the accumulator equalled the threshold AT the trigger, which
    leaves free the excursions on the way -- with a bipolar response the
    accumulator may cross the threshold early and be pulled back down by the
    negative lobe before the recorded trigger, a path the data term has no
    reason to reject.  The peak statement forbids it, for the same reason the
    post-latch term uses the peak rather than the endpoint.

    Two kinds of interval, named as the row metadata names the sequences they
    belong to (``RowMeta.post_reset``):

    - PRE-TRIGGER: before a pixel's first trigger.  The accumulator has no
      prior reset, so the cumulative is referenced to ``acq_start``.
    - POST-RESET: before any later trigger, referenced to that sequence's CSA
      restart.  Off by default (see ``include_post_reset``).

    One term is emitted per interval ORDINAL -- the pre-trigger interval, then
    the first post-reset one, and so on -- so a pixel contributes at most one
    row per term and the terms share one cumulative sum.  Pixels with no
    interval in a given term get ``reset = nt`` (zero weight everywhere, hence
    no contribution to the curvature bound) and an empty armed mask.

    Boundaries, per interval, mirroring the post-latch conventions:

    - the cumulative is referenced to the CSA restart, ``previous last latch +
      csa_reset_time`` (pre-trigger: ``acq_start``, no prior reset), because
      that is where the amplifier resumes accumulating;
    - the armed window OPENS at the previous burst's discriminator re-arm
      (hits col4).  Between the restart and the re-arm the readout gates
      triggering but not accumulation, so a non-detection statement there
      carries no information;
    - it CLOSES ``one_tick + close_back`` before the trigger.  At the trigger
      the discriminator did fire, and the discrete-crossing overshoot can
      exceed ``margin``, so the last instant at which the pixel is known NOT
      to have fired is the tick before.  ``close_back`` [ticks] backs that
      boundary off further: the check instants nearest the crossing are the
      ones where the accumulator is climbing through the threshold, so their
      slack is ~0 and the operator's within-bin model error shows up there as
      a violation rather than as a solution error.  Backing off trades check
      instants for honest ones.

    The suppression the burst gate exists for needs no separate gate here: a
    pixel that re-fired the instant it re-armed has ``re-arm >= trigger``, the
    interval is empty and no row is emitted.  ``norm`` defaults to ``"l1"``
    (exact penalty, zero curvature) so the term costs no step size -- the
    post-latch ``l2`` term already carries 3-5x the data term's curvature.

    ``close_back`` defaults to 20 ticks (\\SI{1}{\\micro s}, 2/3 of a fit bin at
    the standard readout) because that is what the truth-feasibility gate
    measures (\\path{censor_pre_probe.py}, nb1 scan, bound = threshold +
    margin = 8 ke; truth violations / intervals):

    ============  ==================  ====================  ===============
    close_back    pre-trigger viol    post-reset viol       post-reset
    [ticks]       (mu00/mu50/p50/p75) (mu00/mu50/p50/p75)   checks kept
    ============  ==================  ====================  ===============
    0             0 / 0 / 1 / 5       12 / 4 / 7 / 10       100%
    10            0 / 0 / 1 / 2       0 / 0 / 2 / 2         71 / 60 / 81 / 92%
    20            0 / 0 / 0 / 0       0 / 0 / 0 / 0         49 / 32 / 68 / 87%
    ============  ==================  ====================  ===============

    At ``close_back = 0`` the intervals that open right after a latch are
    violated BY THE TRUTH on 3-10% of pixels: they sit where the bipolar
    response is in its negative lobe and the coarse-bin prediction is biased
    HIGH -- the same mechanism that made the old ceil reset boundary falsely
    tight -- so they would penalise operator error, not solution error.  The
    offending check instants are the bin ends within ~20 ticks of the
    trigger, where the accumulator is climbing through the threshold and the
    slack is ~0; backing off that far removes them and costs the pre-trigger
    intervals only 1-2% of their check instants while also clearing their own
    residual violations (positron).

    ``include_post_reset`` adds the post-reset intervals.  It defaults to
    False: at ``close_back = 0`` they are not truth-feasible, and the
    solve-level effect of adding them at ``close_back = 20`` (where they are)
    is still being measured.
    """
    B = hits.adc_hold_delay if bin_ticks is None else float(bin_ticks)
    nx, ny, nt = op.block_shape
    boff = np.asarray(block_offset, dtype=float)
    px = (hits.pixel_x - int(boff[0])).astype(int)
    py = (hits.pixel_y - int(boff[1])).astype(int)
    trig = np.asarray(hits.trigger, float) - boff[2]
    ll = np.asarray(hits.last_latch, float) - boff[2]
    ra = np.asarray(hits.rearm, float) - boff[2]
    if acq_start is None:                    # legacy -inf edge: fall back to
        acq = np.full(len(px), float(npad_bins) * B)   # the covered range
    elif callable(acq_start):
        acq = np.array([float(acq_start(int(gx), int(gy))) - boff[2]
                        for gx, gy in zip(hits.pixel_x, hits.pixel_y)])
    else:
        acq = np.full(len(px), float(acq_start) - boff[2])

    by_ordinal: list[dict[tuple[int, int], tuple[float, float, float]]] = []
    prev_key = None
    prev_ll = prev_ra = 0.0
    k = 0
    for i in np.lexsort((trig, py, px)):
        if not (0 <= px[i] < nx and 0 <= py[i] < ny):
            continue
        key = (int(px[i]), int(py[i]))
        if key != prev_key:
            k, ref, arm = 0, acq[i], acq[i]
        else:
            k += 1
            ref, arm = prev_ll + float(csa_reset_time or 0.0), prev_ra
        end = trig[i] - float(one_tick) - float(close_back)
        if arm < end:
            while len(by_ordinal) <= k:
                by_ordinal.append({})
            by_ordinal[k][key] = (ref / B, arm / B, end / B)
        prev_key, prev_ll, prev_ra = key, ll[i], ra[i]

    terms: list[CensorRunningMax] = []
    if not include_post_reset:
        by_ordinal = by_ordinal[:1]                 # the pre-trigger interval only
    for s in by_ordinal:
        if not s:
            continue
        reset = np.full((nx, ny), float(nt))          # w == 0 everywhere
        arm = np.full((nx, ny), float(nt) + 2.0)      # armed mask empty
        end = np.zeros((nx, ny))
        for (x, y), (r, a, e) in s.items():
            reset[x, y] = min(max(r, 0.0), float(nt))
            arm[x, y] = min(max(a, 0.0), float(nt))
            end[x, y] = min(max(e, 0.0), float(nt))
        terms.append(CensorRunningMax(op, reset, arm, censor_end=end,
                                     threshold=threshold + margin,
                                     beta=beta, norm=norm))
    return terms
