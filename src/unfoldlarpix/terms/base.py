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


class SimplexProx:
    """Positivity, support, and a PINNED total charge.

    Projects onto

        C = { x : x_v >= 0 on the support, x_v = 0 off it,
                  sum_v x_v = target }

    i.e. the scaled simplex.  Used to pin the reconstruction's global charge
    to the self-triggered record sum, so the fit can no longer choose its own
    normalisation.

    NOTE ON alpha.  On the non-negative orthant ``sum_v x_v == ||x||_1``, so
    once the sum is pinned the l1 penalty is a CONSTANT (= alpha * target)
    and cannot influence the minimiser.  This prox therefore takes no alpha:
    with the sum pinned, l1 regularisation is not merely weak, it is
    inoperative.  A soft penalty ``(beta/2)(sum x - target)^2`` was rejected
    instead: its Hessian is ``beta * 1 1^T`` with spectral norm ``beta * N``
    over N ~ 10^6 cells, which would dominate the Lipschitz constant and
    collapse the FISTA step for any beta large enough to bind.

    Projection: the Euclidean projection onto the simplex is
    ``x_v = max(v_v - tau, 0)`` for the unique ``tau`` with
    ``sum_v max(v_v - tau, 0) = target`` over the support.  The left side is
    continuous and non-increasing in ``tau``, so a bisection converges;
    ``tau`` may be negative (when the input sums to less than the target,
    the projection ADDS charge).  Brackets: ``tau_hi = max(v)`` gives 0,
    and ``tau_lo = min(v) - target/n_support`` gives at least ``target``.
    """

    def __init__(self, support, target: float, n_bisect: int = 80):
        self.support = support
        self.target = float(target)
        self.n_bisect = int(n_bisect)
        if self.target < 0:
            raise ValueError(f"target must be >= 0, got {target}")

    def __call__(self, v: torch.Tensor, step: float) -> torch.Tensor:
        s = (self.support if self.support is not None
             else torch.ones_like(v))
        if self.target <= 0:
            return torch.zeros_like(v)
        big = float(torch.finfo(v.dtype).max) / 4.0
        v_in = torch.where(s > 0, v, torch.full_like(v, -big))
        n_s = float((s > 0).sum())
        if n_s <= 0:
            return torch.zeros_like(v)

        def g(tau):
            return float(torch.clamp(v_in - tau, min=0.0).sum())

        hi = float(v_in.max())
        lo = float(torch.where(s > 0, v,
                               torch.full_like(v, big)).min()) \
            - self.target / n_s
        for _ in range(self.n_bisect):
            mid = 0.5 * (lo + hi)
            if g(mid) > self.target:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        return torch.clamp(v_in - tau, min=0.0) * s


class GroupFloorProx:
    """Small negatives per cell, non-negative per group of cells.

    The constraint set is

        C = { x : x_v >= -floor  for every cell v,
                  sum_{v in G} x_v >= 0  for every group G }

    with the groups a fixed partition of the time axis (here one readout
    window, ``group`` cells of the unknown basis).  It is convex, and since
    the groups do not overlap it is BLOCK separable: the projection splits
    into independent problems of size ``group``.  ``Fista.minimize`` only
    calls ``prox(v, step)``, so a block prox drops in where
    :class:`CoordProx` would go; what it gives up is the coordinatewise
    property, not convergence.

    The penalty carried alongside is the LINEAR charge price ``alpha *
    sum_v x_v``, not ``alpha ||x||_1``.  On the non-negative orthant the two
    are identical, which is the regime every campaign so far has run in, and
    the linear form is the one that stays convex and prox-friendly once
    cells may be slightly negative.  It also keeps alpha's meaning: within a
    group the penalty depends only on the group sum, so it prices charge and
    never prefers one shape inside a window over another.

    Projection of ``u`` onto ``C`` for one group:

    1. ``w = max(u, -floor)``.  If ``sum w >= 0`` this is the projection.
    2. Otherwise the sum constraint is active, and the projection is
       ``x_i = max(u_i + lam, -floor)`` with the unique ``lam > 0`` solving
       ``sum_i max(u_i + lam, -floor) = 0``.  The left side is continuous and
       non-decreasing in ``lam``, so a bisection converges; ``n_bisect``
       steps are taken on all groups at once.

    Cells outside the support are held at zero and excluded from the group
    sum, so the support behaves exactly as it does under ``CoordProx``.

    ``floor = 0`` reproduces ``CoordProx`` exactly (asserted in the tests).

    ``group_phase`` anchors the grouping to an EXTERNAL grid instead of the
    unknown's own cell index 0.  Without it, group ``g`` is cells
    ``[g*group, (g+1)*group)`` of the unknown, i.e. windows start wherever
    the operator's own ``block_offset`` happens to start -- a per-event
    accident.  With ``group_phase = p`` (``0 <= p < group``), ``p`` virtual
    all-zero cells are prepended before grouping, so group ``g`` becomes
    cells ``[g*group - p, (g+1)*group - p)`` of the unknown: the SAME
    absolute grid at every event, provided ``p`` is computed from the
    block's own absolute offset (see :func:`universal_group_phase`).  The
    virtual cells carry support 0, so they cost nothing and add no charge.
    """

    def __init__(self, alpha, support, floor: float, group: int,
                 n_bisect: int = 60, group_phase: int = 0):
        self.alpha = alpha
        self.support = support
        self.floor = float(floor)
        self.group = int(group)
        self.n_bisect = int(n_bisect)
        self.group_phase = int(group_phase) % max(int(group), 1)
        if self.group < 1:
            raise ValueError(f"group must be >= 1, got {group}")
        if self.floor < 0:
            raise ValueError(f"floor is a magnitude >= 0, got {floor}")

    def __call__(self, v: torch.Tensor, step: float) -> torch.Tensor:
        u = v - step * self.alpha
        s = (self.support if self.support is not None
             else torch.ones_like(u))
        nx, ny, nt = u.shape
        g = self.group
        ph = self.group_phase
        if ph:
            u = torch.nn.functional.pad(u, (ph, 0))
            s = torch.nn.functional.pad(s, (ph, 0))
        pad = (-(nt + ph)) % g                # last partial group, if any
        if pad:
            u = torch.nn.functional.pad(u, (0, pad))
            s = torch.nn.functional.pad(s, (0, pad))
        ug = u.reshape(nx, ny, -1, g)
        sg = s.reshape(nx, ny, -1, g)

        def clipped(lam):
            """max(u + lam, -floor) on the support, 0 off it."""
            return torch.where(sg > 0,
                               torch.clamp(ug + lam, min=-self.floor),
                               torch.zeros_like(ug))

        x = clipped(0.0)
        tot = x.sum(dim=3, keepdim=True)
        need = tot < 0
        if bool(need.any()):
            # lam = 0 gives sum < 0; the sum is non-decreasing in lam and
            # reaches >= 0 at lam_hi, so bracket then bisect.
            lo = torch.zeros_like(tot)
            hi = torch.full_like(tot, 1.0)
            for _ in range(40):
                still = clipped(hi).sum(dim=3, keepdim=True) < 0
                if not bool(still.any()):
                    break
                hi = torch.where(still, hi * 2.0, hi)
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                neg = clipped(mid).sum(dim=3, keepdim=True) < 0
                lo = torch.where(neg, mid, lo)
                hi = torch.where(neg, hi, mid)
            lam = 0.5 * (lo + hi)
            x = torch.where(need, clipped(lam), x)
        out = x.reshape(nx, ny, -1)
        return out[:, :, ph:ph + nt].contiguous()


def universal_group_phase(block_offset_tick: float, cell_ticks: int,
                          group_ticks: int = 30) -> tuple[int, float]:
    """The ``group_phase`` that anchors ``GroupFloorProx`` to absolute tick 0.

    ``block_offset_tick`` is the unknown grid's own absolute origin
    (``block_offset[2]``, fine ticks); ``cell_ticks`` is the unknown's cell
    width ``c``; ``group_ticks`` is the window width ``B`` (30 by default,
    one readout window).  Returns ``(group_phase, misalignment_ticks)``.

    Group boundaries live at integer multiples of ``B`` measured from
    absolute tick 0 -- the ``eval/universal.py`` convention
    (``edge_anchor="universal"``, ``phi=0``).  Cell boundaries live at
    ``block_offset_tick + k*c``.  A boundary coincides with a group boundary
    only for ``k`` with ``k*c == -block_offset_tick (mod B)``, which has an
    exact integer solution iff ``block_offset_tick mod c == 0`` (``c``
    divides ``B`` by construction of the cell basis, so ``gcd(c, B) = c``).
    When that holds, ``misalignment_ticks`` is exactly 0.  When it does not
    (the block origin is not itself a multiple of the cell width), the
    phase is rounded to the NEAREST cell -- the finest anchoring the basis
    can represent -- and the residual is reported so a caller can assert it
    is small (below one cell width) rather than silently absorbing it.
    """
    B, c = int(group_ticks), int(cell_ticks)
    if B % c:
        raise ValueError(f"group_ticks {B} is not a multiple of cell_ticks "
                         f"{c}; universal-grid anchoring needs c | B")
    g = B // c
    r_ticks = float(block_offset_tick) % B          # in [0, B)
    phase = int(round(r_ticks / c)) % g
    misalign = r_ticks - phase * c
    # round() can push the residual to the OTHER side of a group edge
    # (e.g. r_ticks = B - 0.4*c): fold back into (-c/2, c/2]
    if misalign > B / 2:
        misalign -= B
    return phase, float(misalign)
