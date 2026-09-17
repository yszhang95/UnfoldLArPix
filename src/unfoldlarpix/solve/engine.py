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
    """Plain FISTA by default.  Opt-in accelerations, all leaving the fixed
    point unchanged:

    ``restart``   gradient-based adaptive restart (O'Donoghue & Candes 2015):
                  when ``<y_k - x_{k+1}, x_{k+1} - x_k> > 0`` the momentum
                  is pointing uphill; reset ``t = 1`` and ``y = x_{k+1}``.
                  Removes the oscillation plain FISTA shows on badly
                  conditioned problems (here L is 15x inflated by the
                  censor terms, so the fixed step is 15x smaller than the
                  data term alone would need).
    ``rel_tol``   stop when ``||x_{k+1} - x_k|| / max(||x_{k+1}||, 1e-12)``
                  is below ``rel_tol`` for ``patience`` consecutive
                  iterations.  ``stopped_at`` records the iteration used.
    ``backtrack``  Beck-Teboulle line search on the step.  The fixed step
                  ``1/(safety*L)`` uses the sum of the terms' curvature
                  BOUNDS; for the censor terms that bound assumes every
                  armed interval is active at once, which is never the case,
                  so it is far too pessimistic (measured on iso50 d=16.5,
                  c=5: L_data 623, L_post 2482, L_pre 6165).  With
                  backtracking the iteration starts from ``L_k = L /
                  bt_L0_div``, accepts the step if the sufficient-decrease
                  condition holds,
                      f(x+) <= f(y) + <grad f(y), x+ - y> + L_k/2 ||x+ - y||^2,
                  else multiplies ``L_k`` by ``bt_up`` and retries; after an
                  accepted step ``L_k`` is divided by ``bt_down`` so the step
                  can grow back.  ``L_k`` is capped at the proven bound ``L``.
                  Each attempt costs one objective evaluation (one forward
                  pass, shared by all terms through ``IterCtx``).  The fixed
                  point is unchanged.
    ``trace_every``  record ``(iteration, sum(x))`` every that many
                  iterations into ``self.trace`` -- the convergence curve of
                  the total charge, which is what this campaign scores.
    """

    def __init__(self, n_iter: int = 150, safety: float = 1.05,
                 restart: bool = False, rel_tol: float | None = None,
                 patience: int = 5, backtrack: bool = False,
                 bt_L0_div: float = 16.0, bt_up: float = 2.0,
                 bt_down: float = 1.5, trace_every: int = 0):
        self.n_iter = int(n_iter)
        self.safety = float(safety)
        self.restart = bool(restart)
        self.rel_tol = None if rel_tol is None else float(rel_tol)
        self.patience = int(patience)
        self.backtrack = bool(backtrack)
        self.bt_L0_div = float(bt_L0_div)
        self.bt_up = float(bt_up)
        self.bt_down = float(bt_down)
        self.trace_every = int(trace_every)
        self.n_restarts = 0
        self.n_backtracks = 0
        self.trace: list = []
        self.steps: list = []

    def minimize(
        self,
        op,
        terms: Sequence[SmoothTerm],
        prox: CoordProx,
        q0: torch.Tensor | None = None,
        callback: Callable[[int, IterCtx], None] | None = None,
        stop_when: Callable[[IterCtx], bool] | None = None,
    ) -> torch.Tensor:
        stop_when = stop_when or getattr(self, "stop_when", None)
        L = sum(t.curvature() for t in terms)
        step = 1.0 / (self.safety * max(L, 1e-12))
        x = (torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
             if q0 is None else prox(q0.to(op.device, op.dtype), 0.0))
        y = x.clone()
        t = 1.0
        self.n_restarts = 0
        self.n_backtracks = 0
        self.trace = []
        self.steps = []
        quiet = 0
        L_full = float(max(L, 1e-12))
        L_k = L_full / self.bt_L0_div if self.backtrack else L_full
        for k in range(self.n_iter):
            ctx = IterCtx(y, op)
            grad = torch.zeros_like(y)
            for term in terms:
                term.grad_into(ctx, grad)
            if self.backtrack:
                f_y = float(sum(float(term.value(ctx)) for term in terms))
                while True:
                    step = 1.0 / (self.safety * L_k)
                    x_new = prox(y - step * grad, step)
                    d = x_new - y
                    ctx_x = IterCtx(x_new, op)
                    f_x = float(sum(float(term.value(ctx_x))
                                    for term in terms))
                    quad = (f_y + float((grad * d).sum())
                            + 0.5 * L_k * float((d * d).sum()))
                    if f_x <= quad * (1.0 + 1e-9) + 1e-9 or L_k >= L_full:
                        break
                    L_k = min(L_k * self.bt_up, L_full)
                    self.n_backtracks += 1
                self.steps.append(step)
                L_k = max(L_k / self.bt_down, L_full / 1e6)
            else:
                x_new = prox(y - step * grad, step)
            dx = x_new - x
            if self.restart and k > 0 and float(((y - x_new) * dx).sum()) > 0:
                # momentum points uphill: drop it
                t_new = 1.0
                y = x_new.clone()
                self.n_restarts += 1
            else:
                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
                y = x_new + ((t - 1.0) / t_new) * dx
            if self.rel_tol is not None:
                rel = float(torch.linalg.vector_norm(dx)) / max(
                    float(torch.linalg.vector_norm(x_new)), 1e-12)
                quiet = quiet + 1 if rel < self.rel_tol else 0
            x, t = x_new, t_new
            if self.trace_every and (k + 1) % self.trace_every == 0:
                self.trace.append((k + 1, float(x_new.sum())))
            if callback is not None:
                callback(k, ctx)
            if self.rel_tol is not None and quiet >= self.patience:
                self.stopped_at = k + 1
                return x
            # Discrepancy principle: stop as soon as the data term reaches the
            # level the noise model predicts.  Minimising past it means fitting
            # structure the data cannot carry, which on this operator is
            # absorbed as displaced charge rather than as a smaller error.
            if stop_when is not None and stop_when(ctx):
                self.stopped_at = k + 1
                return x_new
        self.stopped_at = self.n_iter
        return x
