"""Solve strategies: schedules that re-invoke the engine.

Strategies operate on an explicit :class:`SolveState` that flows between
stages — no hidden state in the engine, so stage transitions are unit
testable in isolation and strategies compose as pipelines:

    state = Ladder(...).run(...)
    state = FinalRefit(...).run(...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..constrained_solver import exponential_alpha_field
from ..terms.base import CoordProx
from ..terms.data import DataFidelity
from .engine import Fista


@dataclass
class StageRecord:
    label: str
    alpha: Any
    q_sum: float
    nnz: int


@dataclass
class SolveState:
    q: torch.Tensor
    skeleton: torch.Tensor | None = None
    history: list[StageRecord] = field(default_factory=list)
    # optimization trace: one row per sampled iteration, filled only when
    # a stage is run with trace_every > 0 (see make_tracer).
    trace: list[dict] = field(default_factory=list)


def make_tracer(terms, prox, stage: str, every: int, out: list):
    """Callback for Fista.minimize recording the objective components.

    Costs one extra value() per term per sampled iteration -- opt-in and
    sampled, so a solve is unaffected unless tracing is requested.
    """
    def cb(k: int, ctx) -> None:
        if every <= 0 or (k % every and k != 0):
            return
        row = {"stage": stage, "iter": int(k)}
        tot = 0.0
        for t in terms:
            v = float(t.value(ctx))
            row[type(t).__name__] = v
            tot += v
        a = prox.alpha
        l1 = float((a * ctx.q).sum()) if torch.is_tensor(a) else \
            float(a) * float(ctx.q.sum())
        row["l1"] = l1
        row["objective"] = tot + l1
        row["q_sum"] = float(ctx.q.sum())
        row["nnz"] = int((ctx.q > 0.01).sum())
        out.append(row)
    return cb


class Ladder:
    """Strong-charge-first homotopy (soft-seeded alpha ladder).

    Stage 0 solves at alphas[0] on the base support; each later stage
    lowers alpha with the weighted-L1 soft seed prior
    alpha_i = a * exp((d_i / soft_len)^p), d = Manhattan distance from
    the previous stage's skeleton (q > seed_cut).
    """

    def __init__(self, alphas, seed_cut: float = 0.5, soft_len: float = 2.0,
                 soft_exponent: float = 1.0, n_iter: int = 150,
                 alpha_scale=None, trace_every: int = 0,
                 soft_axis_cost=None):
        if not list(alphas):
            raise ValueError("ladder alphas cannot be empty")
        self.alphas = [float(a) for a in alphas]
        self.seed_cut = float(seed_cut)
        self.soft_len = float(soft_len)
        self.soft_exponent = float(soft_exponent)
        # Per-axis step cost for the seed distance.  None keeps the
        # grid-index Manhattan metric, under which a time step (2.395 mm
        # for the standard readout) costs the same as a pixel step
        # (4.434 mm), so the prior is 1.85x tighter per mm along time --
        # an accident of the grid, not a statement about charge.
        self.soft_axis_cost = (None if soft_axis_cost is None
                               else [float(c) for c in soft_axis_cost])
        self.n_iter = int(n_iter)
        # optional STATIC per-voxel multiplier on every stage's weights
        # (e.g. the measurement gain c_v: a coordinate activates when
        # |A^T r|_v > alpha_v, and A^T r itself scales with c_v, so a
        # uniform alpha systematically suppresses weak-coverage voxels;
        # alpha_v ~ c_v equalises the activation condition).
        self.alpha_scale = alpha_scale
        self.trace_every = int(trace_every)

    def alpha_field(self, op, a: float, skeleton) -> torch.Tensor | float:
        """Per-voxel L1 weights for one stage (exposed for unit tests)."""
        if skeleton is None or not bool(skeleton.any()):
            field = a
        else:
            field_np = exponential_alpha_field(
                skeleton.cpu().numpy().astype(bool), a, self.soft_len,
                exponent=self.soft_exponent,
                axis_cost=self.soft_axis_cost)
            field = op.to_tensor(field_np)
        if self.alpha_scale is not None:
            field = field * self.alpha_scale
        return field

    def run(self, engine: Fista, op, smooth_terms, support,
            state: SolveState) -> SolveState:
        stage_engine = Fista(n_iter=self.n_iter, safety=engine.safety)
        for k, a in enumerate(self.alphas):
            alpha = self.alpha_field(op, a, state.skeleton)
            prox = CoordProx(alpha, support)
            cb = (make_tracer(smooth_terms, prox, f"ladder[{k}]",
                              self.trace_every, state.trace)
                  if self.trace_every else None)
            state.q = stage_engine.minimize(op, smooth_terms, prox,
                                            q0=state.q, callback=cb)
            state.skeleton = state.q > self.seed_cut
            state.history.append(StageRecord(
                label=f"ladder[{k}]", alpha=a,
                q_sum=float(state.q.sum()),
                nnz=int((state.q > 0.01).sum())))
        return state


class FinalRefit:
    """Near-unbiased amplitude refit on the frozen strong support.

    Faint charges are FROZEN as background — expressed at the TERM level
    (DataFidelity with target d - A q_faint); the operator is never
    mutated (immutability contract).
    """

    def __init__(self, eps: float = 0.5, alpha: float = 0.0,
                 n_iter: int = 150, trace_every: int = 0):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.trace_every = int(trace_every)

    def run(self, engine: Fista, op, smooth_terms, support,
            state: SolveState) -> SolveState:
        strong = state.q > self.eps
        q_faint = torch.where(strong, torch.zeros_like(state.q), state.q)
        target = op.d - op.forward(q_faint)
        terms = [DataFidelity(op, target=target) if isinstance(t, DataFidelity)
                 else t for t in smooth_terms]
        prox = CoordProx(self.alpha, strong.to(op.dtype))
        cb = (make_tracer(terms, prox, "refit", self.trace_every,
                          state.trace) if self.trace_every else None)
        q_strong = Fista(n_iter=self.n_iter, safety=engine.safety).minimize(
            op, terms, prox, q0=torch.where(strong, state.q,
                                            torch.zeros_like(state.q)),
            callback=cb)
        state.q = q_strong + q_faint
        state.history.append(StageRecord(
            label="refit", alpha=self.alpha,
            q_sum=float(state.q.sum()), nnz=int((state.q > 0.01).sum())))
        return state
