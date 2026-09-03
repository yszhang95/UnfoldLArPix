"""RESS: the Wire-Cell regularised sparse solver, on this operator's Gram.

Wire-Cell solves ``m = R s`` by CYCLIC COORDINATE DESCENT with an active
set (``WireCellUtil/{Ress,LassoModel,ElasticNetModel}``, WCP
``ress/src/LassoModel.cxx``).  The engine this package ships is FISTA --
PROXIMAL (projected, at alpha = 0) GRADIENT DESCENT on the same objective.
Both minimise

    F(q) = 1/2 ||A q - d||^2 + sum_v gamma_v |q_v|,   q >= 0 on the support

so on a convex problem they must agree up to the minimiser set; whether
they DO is a statement about convergence, not about the model, and it is
what :mod:`unfoldlarpix.algs.ress_algs` measures.

Why the Gram form
-----------------
``LassoModel::Fit`` itself precomputes ``ydX = X^T y`` and ``XdX = X^T X``
and never touches ``X`` again -- the port below is that same loop.  It is
also the only form this operator can afford: ``A`` is never materialised
(FFT convolution + row selection over 1.26 M rows), whereas ``G = A^T A``
restricted to an ROI of n voxels is n columns of ``A^T A e_v``, one
forward+adjoint each.  The constant ``1/2||d||^2`` keeps the objective
numerically equal to the operator's own ``L``, which is asserted.

The two penalty conventions (a trap)
------------------------------------
WCP's live code thresholds ``delta_j / ||X_j||^2`` at ``lambda * w_j``,
i.e. the effective L1 weight is ``gamma_j = lambda * w_j * ||X_j||^2`` --
per-column, not uniform.  (The commented-out line above it, and the
docstring, say ``N * lambda``; the code does neither.)  This package's
:class:`~unfoldlarpix.terms.base.CoordProx` uses a UNIFORM ``alpha``.  The
same number means different priors, so ``penalty`` is explicit here:
``"ress"`` reproduces Wire-Cell, ``"uniform"`` reproduces ``CoordProx``.
At lambda = alpha = 0 (pure NNLS) they coincide and the question is moot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# the Gram system
# ---------------------------------------------------------------------------
@dataclass
class GramSystem:
    """``F(beta) = 1/2 beta^T G beta - b^T beta + const`` on ROI voxels.

    ``idx`` are the ROI voxels' (x, y, t) indices on the operator's charge
    grid, so :meth:`embed` puts a solution back where the evaluation stack
    expects it.  ``const = 1/2 ||d||^2`` makes ``F`` equal to the
    operator's ``L = 1/2||A q - d||^2`` for any q supported on the ROI --
    :func:`build_gram_system` asserts exactly that.
    """

    idx: np.ndarray            # (n, 3) int, voxel indices on the charge grid
    q_shape: tuple[int, int, int]
    G: np.ndarray              # (n, n) float64
    b: np.ndarray              # (n,) float64
    const: float
    meta: dict

    @classmethod
    def from_matrix(cls, X: np.ndarray, y: np.ndarray) -> "GramSystem":
        """Explicit ``X`` (rows x columns) -- unit tests and small ROIs.

        The same system Wire-Cell's ``LassoModel`` builds internally, so a
        port can be checked against the compiled original on a matrix both
        can hold.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[1]
        idx = np.zeros((n, 3), dtype=int)
        idx[:, 0] = np.arange(n)
        return cls(idx=idx, q_shape=(n, 1, 1), G=X.T @ X, b=X.T @ y,
                   const=float(0.5 * y @ y), meta={"source": "from_matrix"})

    @property
    def n(self) -> int:
        return int(self.b.size)

    @property
    def col_norm(self) -> np.ndarray:
        """``||X_j||^2`` -- the Gram diagonal, RESS's ``norm(j)``."""
        return np.diag(self.G).copy()

    def embed(self, beta: np.ndarray) -> np.ndarray:
        q = np.zeros(self.q_shape, dtype=np.float64)
        q[self.idx[:, 0], self.idx[:, 1], self.idx[:, 2]] = beta
        return q

    def data_term(self, beta: np.ndarray) -> float:
        """``1/2 ||A q - d||^2``."""
        return float(0.5 * beta @ (self.G @ beta) - self.b @ beta + self.const)

    def gamma(self, lam: float = 0.0, alpha: float = 1.0,
              weight: np.ndarray | None = None,
              penalty: str = "ress") -> np.ndarray:
        """Per-coordinate L1 weight for the two conventions (see module doc)."""
        w = np.ones(self.n) if weight is None else np.asarray(weight, float)
        if penalty == "ress":
            return lam * alpha * w * self.col_norm
        if penalty == "uniform":
            return lam * alpha * w
        raise ValueError(f"penalty {penalty!r} (want 'ress' or 'uniform')")

    def objective(self, beta: np.ndarray, gamma: np.ndarray | float = 0.0,
                  l2: float = 0.0) -> dict:
        g = np.broadcast_to(np.asarray(gamma, float), (self.n,))
        data = self.data_term(beta)
        l1 = float((g * np.abs(beta)).sum())
        ridge = float(0.5 * l2 * (beta * beta).sum())
        return {"data": data, "l1": l1, "l2": ridge,
                "objective": data + l1 + ridge}

    def kkt(self, beta: np.ndarray, gamma: np.ndarray | float = 0.0,
            l2: float = 0.0) -> dict:
        """First-order optimality of ``min F, beta >= 0``.

        ``g = G beta - b + gamma + l2 beta``.  At a minimiser
        ``g_j = 0`` where ``beta_j > 0`` and ``g_j >= 0`` where
        ``beta_j = 0``.  Scaled by ``max|b|`` so the two solvers'
        violations are comparable independently of the charge scale.
        """
        g = self.G @ beta - self.b + np.broadcast_to(
            np.asarray(gamma, float), (self.n,)) + l2 * beta
        on = beta > 0
        scale = float(np.abs(self.b).max()) or 1.0
        act = float(np.abs(g[on]).max()) if on.any() else 0.0
        ina = float(np.maximum(0.0, -g[~on]).max()) if (~on).any() else 0.0
        return {"kkt_active_max": act, "kkt_inactive_max": ina,
                "kkt_max": max(act, ina), "kkt_scale": scale,
                "kkt_active_rel": act / scale, "kkt_inactive_rel": ina / scale,
                "kkt_rel": max(act, ina) / scale}


def build_gram_system(op, roi_mask: np.ndarray, verify: int = 3,
                      progress: Callable[[int, int], None] | None = None,
                      symmetrize: bool = True) -> GramSystem:
    """``G = A^T A`` and ``b = A^T d`` restricted to the ROI voxels.

    One ``op.adjoint(op.forward(e_v))`` per ROI voxel -- the operator's own
    forward and adjoint, so the reduced system is the production system
    with ``support = roi_mask`` and nothing is re-derived.  ``verify``
    random non-negative vectors are pushed through BOTH the Gram objective
    and ``1/2||A q - d||^2`` on the full grid; they must agree.
    """
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if roi_mask.shape != tuple(op.q_shape):
        raise ValueError(f"roi_mask {roi_mask.shape} != q_shape {op.q_shape}")
    idx = np.argwhere(roi_mask)
    n = len(idx)
    if n == 0:
        raise ValueError("empty ROI")
    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), op.q_shape)
    flat_t = torch.as_tensor(flat, device=op.device)

    G = np.empty((n, n), dtype=np.float64)
    e = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
    for k in range(n):
        i, j, t = idx[k]
        e[i, j, t] = 1.0
        col = op.adjoint(op.forward(e)).reshape(-1)[flat_t]
        G[:, k] = col.double().cpu().numpy()
        e[i, j, t] = 0.0
        if progress is not None and (k % 500 == 0 or k == n - 1):
            progress(k + 1, n)
    asym = float(np.abs(G - G.T).max())
    if symmetrize:
        G = 0.5 * (G + G.T)
    b = op.adjoint(op.d).reshape(-1)[flat_t].double().cpu().numpy()
    const = float(0.5 * (op.d.double() ** 2).sum())

    sysm = GramSystem(idx=idx, q_shape=tuple(int(s) for s in op.q_shape),
                      G=G, b=b, const=const,
                      meta={"gram_asymmetry_max": asym,
                            "gram_diag_min": float(np.diag(G).min()),
                            "gram_diag_max": float(np.diag(G).max())})

    checks = []
    rng = np.random.default_rng(0)
    for _ in range(int(verify)):
        beta = np.abs(rng.normal(size=n))
        qt = op.to_tensor(sysm.embed(beta))
        L_op = float(0.5 * ((op.forward(qt).detach() - op.d) ** 2).sum())
        L_gr = sysm.data_term(beta)
        checks.append({"L_operator": L_op, "L_gram": L_gr,
                       "rel_diff": abs(L_op - L_gr) / max(abs(L_op), 1e-12)})
    sysm.meta["closure_checks"] = checks
    return sysm


# ---------------------------------------------------------------------------
# RESS: Wire-Cell's coordinate descent
# ---------------------------------------------------------------------------
def _soft_threshold(delta: float, lam: float, non_negative: bool) -> float:
    """``ElasticNetModel::_soft_thresholding`` verbatim.

    Note the asymmetry: with ``non_negative`` a delta below ``+lam``
    returns 0 -- there is no negative branch at all, which is what makes
    ``lambda = 0`` a pure NNLS update ``max(delta, 0)``.
    """
    if delta > lam:
        return delta - lam
    if non_negative:
        return 0.0
    if delta < -lam:
        return delta + lam
    return 0.0


def ress_fit(sysm: GramSystem, lam: float = 0.0, alpha: float = 1.0,
             weight: np.ndarray | None = None, penalty: str = "ress",
             max_iter: int = 100000, tol: float = 1e-3,
             non_negative: bool = True, beta0: np.ndarray | None = None,
             trace_every: int = 0) -> tuple[np.ndarray, dict]:
    """Cyclic coordinate descent, ported from WCP ``LassoModel::Fit``.

    Kept verbatim, including the three parts that are easy to lose:

    * the **active set** -- a coordinate that lands under ``1e-6`` is
      frozen and skipped by later sweeps;
    * the **double check** -- a sweep whose step is under
      ``tol^2 * n`` reactivates EVERY coordinate and only a second
      consecutive small sweep terminates, so a frozen coordinate always
      gets one more chance before the fit is declared converged;
    * the update itself, ``beta_j = soft(delta_j / ||X_j||^2,
      lambda*alpha*w_j) / (1 + lambda*(1-alpha))``, whose thresholding
      AFTER the division is what makes the effective L1 weight scale with
      the column norm (see the module docstring).

    ``delta_j`` is recomputed from the current iterate every visit
    (Gauss-Seidel), as in the original.
    """
    G, b = sysm.G, sysm.b
    n = sysm.n
    gamma_w = (np.ones(n) if weight is None
               else np.asarray(weight, dtype=float).copy())
    norm = np.diag(G).copy()
    n_unused = int((norm < 1e-6).sum())
    norm[norm < 1e-6] = 1.0                      # WCP's guard, same constant
    if penalty == "ress":
        thr = lam * alpha * gamma_w              # thresholded after /norm
    elif penalty == "uniform":
        thr = lam * alpha * gamma_w / norm       # same gamma as CoordProx
    else:
        raise ValueError(f"penalty {penalty!r} (want 'ress' or 'uniform')")
    shrink = 1.0 + lam * (1.0 - alpha)

    beta = (np.zeros(n) if beta0 is None
            else np.asarray(beta0, dtype=float).copy())
    active = np.ones(n, dtype=bool)
    tol2 = tol * tol * n
    trace: list[dict] = []

    double_check = 0
    sweeps = 0
    converged = False
    for it in range(int(max_iter)):
        betalast = beta.copy()
        for j in range(n):
            if not active[j]:
                continue
            # ydX(j) - sum_{i != j} XdX(i,j) beta(i)
            delta = b[j] - (G[j] @ beta - G[j, j] * beta[j])
            beta[j] = _soft_threshold(delta / norm[j], thr[j],
                                      non_negative) / shrink
            if abs(beta[j]) < 1e-6:
                active[j] = False
        sweeps += 1
        double_check += 1
        step2 = float(((beta - betalast) ** 2).sum())
        if trace_every and (it % trace_every == 0):
            trace.append({"sweep": it, "step2": step2,
                          "n_active": int(active.sum()),
                          "nnz": int((beta > 0).sum()),
                          "sum": float(beta.sum()),
                          **sysm.objective(beta, sysm.gamma(
                              lam, alpha, weight, penalty))})
        if step2 < tol2:
            if double_check != 1:
                double_check = 0
                active[:] = True
            else:
                converged = True
                break
    info = {"solver": "ress_cd", "sweeps": sweeps, "converged": converged,
            "final_step2": step2, "tol2": tol2, "n_unused_columns": n_unused,
            "penalty": penalty, "lambda": lam, "alpha": alpha,
            "non_negative": non_negative, "trace": trace}
    return beta, info


# ---------------------------------------------------------------------------
# the shipped engine, on the same system
# ---------------------------------------------------------------------------
def pgd_fit(sysm: GramSystem, gamma: np.ndarray | float = 0.0,
            n_iter: int = 3000, safety: float = 1.05, accel: bool = True,
            positivity: bool = True, beta0: np.ndarray | None = None,
            trace_every: int = 0, device: str = "cpu"
            ) -> tuple[np.ndarray, dict]:
    """FISTA / plain projected gradient on the SAME Gram system.

    Mirrors :class:`~unfoldlarpix.solve.engine.Fista` and
    :class:`~unfoldlarpix.terms.base.CoordProx` line for line -- step
    ``1/(safety * L)`` with ``L = lambda_max(G) = ||A^T A||``, prox
    ``max(v - step*gamma, 0)``, Nesterov momentum with the same ``t``
    recursion -- but with the gradient evaluated exactly on ``G`` instead
    of through the FFT operator.  Running this beside the operator arm
    separates "the algorithm" from "the FFT round-off".

    ``accel=False`` is unaccelerated projected gradient, which is what the
    phrase "projected gradient descent" means literally; the shipped
    solver is the accelerated one.
    """
    n = sysm.n
    g_np = np.broadcast_to(np.asarray(gamma, float), (n,)).copy()
    L = (float(np.linalg.eigvalsh(sysm.G)[-1]) if n <= 6000
         else _power_iter(sysm.G))
    step = 1.0 / (safety * max(L, 1e-12))
    # the iteration is one n x n matvec per step; above a few thousand
    # unknowns that is memory bound, so it moves to the GPU in float64 --
    # same arithmetic, ~25x the bandwidth.
    xp, dev = (np, None) if device == "cpu" else (torch, torch.device(device))
    if dev is None:
        G, b, gam = sysm.G, sysm.b, g_np
        zeros = lambda: np.zeros(n)                          # noqa: E731
        clamp = lambda v: np.maximum(v, 0.0)                 # noqa: E731
    else:
        G = torch.as_tensor(sysm.G, dtype=torch.float64, device=dev)
        b = torch.as_tensor(sysm.b, dtype=torch.float64, device=dev)
        gam = torch.as_tensor(g_np, dtype=torch.float64, device=dev)
        zeros = lambda: torch.zeros(n, dtype=torch.float64,  # noqa: E731
                                    device=dev)
        clamp = lambda v: torch.clamp(v, min=0.0)            # noqa: E731

    def prox(v):
        out = v - step * gam
        return clamp(out) if positivity else out

    def host(v):
        return v if dev is None else v.cpu().numpy()

    x = zeros() if beta0 is None else prox(
        xp.asarray(beta0) if dev is None
        else torch.as_tensor(np.asarray(beta0, float), dtype=torch.float64,
                             device=dev))
    y = x.clone() if dev is not None else x.copy()
    t = 1.0
    trace: list[dict] = []
    for k in range(int(n_iter)):
        grad = G @ y - b
        x_new = prox(y - step * grad)
        if accel:
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        else:
            y = x_new
        x = x_new
        if trace_every and (k % trace_every == 0):
            xh = host(x)
            trace.append({"iter": k, "nnz": int((xh > 0).sum()),
                          "sum": float(xh.sum()), **sysm.objective(xh, g_np)})
    return host(x), {"solver": "pgd_fista" if accel else "pgd_plain",
                     "n_iter": int(n_iter), "step": step, "lipschitz": L,
                     "safety": safety, "positivity": positivity,
                     "device": str(device), "trace": trace}


# ---------------------------------------------------------------------------
# the SHIPPED engine, unmodified, on the same system
# ---------------------------------------------------------------------------
class GramOperator:
    """An operator whose normal equations ARE ``(G, b)``.

    The point of this class is that nothing about the solver has to be
    re-implemented to compare it: :class:`~unfoldlarpix.solve.engine.Fista`,
    :class:`~unfoldlarpix.terms.data.DataFidelity`,
    :class:`~unfoldlarpix.terms.base.IterCtx` and
    :class:`~unfoldlarpix.terms.base.CoordProx` run against this object
    exactly as they run against :class:`~unfoldlarpix.model.operator.ZSOperator`
    in production.  The arm that uses it is therefore the repository's own
    projected-gradient solver, not a second reading of it.

    Construction is the Cholesky factor: with ``G = R^T R`` and
    ``d := R^-T b``,

        A q := R q      =>   A^T A = G,   A^T d = b

    so ``1/2||A q - d||^2`` and the Gram objective differ by the constant
    ``1/2||d||^2 - const`` -- which cannot move an argmin, and the arm's
    reported objective is evaluated with :meth:`GramSystem.objective`
    anyway, like every other arm's.  ``G`` positive definite is required
    and checked; on this campaign's ROIs it is (residual 4e-16).

    ``block`` and ``q`` are the same space here: ``conv`` is the identity
    and the whole operator lives in ``sample``.  That keeps ``IterCtx``'s
    cache meaningful (one ``R q`` per iteration, shared by the terms).
    """

    def __init__(self, sysm: GramSystem, device: str = "cuda",
                 dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = sysm.n
        self.q_shape = (sysm.n, 1, 1)
        self.block_shape = self.q_shape
        R = np.linalg.cholesky(np.ascontiguousarray(sysm.G,
                                                    dtype=np.float64)).T
        self.gram_factor_resid = float(np.abs(R.T @ R - sysm.G).max())
        d = np.linalg.solve(R.T, np.ascontiguousarray(sysm.b,
                                                      dtype=np.float64))
        self._R = torch.as_tensor(np.ascontiguousarray(R), dtype=dtype,
                                  device=self.device)
        self.d = torch.as_tensor(d, dtype=dtype, device=self.device)
        self.n_data = int(self.n)
        self._lipschitz = float(_power_iter(sysm.G))

    # -- the ZSOperator interface -----------------------------------------
    def conv(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def conv_adjoint(self, r_block: torch.Tensor) -> torch.Tensor:
        return r_block

    def sample(self, block: torch.Tensor) -> torch.Tensor:
        return self._R @ block.reshape(-1)

    def sample_adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return (self._R.T @ r).reshape(self.q_shape)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.sample(self.conv(q))

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return self.conv_adjoint(self.sample_adjoint(r))

    @property
    def lipschitz(self) -> float:
        return self._lipschitz

    def to_tensor(self, arr, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr),
                               dtype=dtype or self.dtype, device=self.device)


class _SupportOnlyProx:
    """Support mask, no positivity -- mirrors ``fixedgrid_algs._SupportProx``."""

    def __init__(self, support=None):
        self.support = support
        self.alpha = 0.0

    def __call__(self, v, step):
        return v if self.support is None else v * self.support


def pgd_engine_fit(sysm: GramSystem, gamma: np.ndarray | float = 0.0,
                   n_iter: int = 3000, positivity: bool = True,
                   device: str = "cuda", dtype: str = "float64",
                   safety: float = 1.05, trace_every: int = 0
                   ) -> tuple[np.ndarray, dict]:
    """Run THE REPOSITORY'S solver on this system.

    ``Fista.minimize(op, [DataFidelity(op)], CoordProx(alpha, support))``
    -- the production call, with ``op`` a :class:`GramOperator`.  The only
    freedom here is ``dtype``: production runs float32 through the FFT, and
    ``float64`` isolates the algorithm from that.
    """
    from ..solve.engine import Fista
    from ..terms.base import CoordProx
    from ..terms.data import DataFidelity

    td = {"float32": torch.float32, "float64": torch.float64}[dtype]
    op = GramOperator(sysm, device=device, dtype=td)
    g_np = np.broadcast_to(np.asarray(gamma, float), (sysm.n,)).copy()
    alpha = (float(g_np[0]) if np.ptp(g_np) == 0
             else op.to_tensor(g_np.reshape(op.q_shape)))
    support = torch.ones(op.q_shape, dtype=td, device=op.device)
    prox = CoordProx(alpha, support) if positivity else _SupportOnlyProx(support)
    terms = [DataFidelity(op)]

    trace: list[dict] = []

    def cb(k, ctx):
        if trace_every and (k % trace_every == 0):
            x = ctx.q.reshape(-1).double().cpu().numpy()
            trace.append({"iter": k, "nnz": int((x > 0).sum()),
                          "sum": float(x.sum()), **sysm.objective(x, g_np)})

    q = Fista(n_iter=int(n_iter), safety=safety).minimize(
        op, terms, prox, q0=None, callback=cb if trace_every else None)
    beta = q.reshape(-1).double().cpu().numpy()
    info = {"solver": "pgd_repo_engine", "n_iter": int(n_iter),
            "dtype": dtype, "device": str(device), "safety": safety,
            "lipschitz": op.lipschitz, "step": 1.0 / (safety * op.lipschitz),
            "positivity": positivity, "alpha_is_tensor": torch.is_tensor(alpha),
            "gram_factor_resid": op.gram_factor_resid,
            "engine": "unfoldlarpix.solve.engine.Fista + terms.data."
                      "DataFidelity + terms.base.CoordProx",
            "trace": trace}
    del op, prox, terms, q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return beta, info


def _power_iter(G: np.ndarray, n_iter: int = 200) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(size=G.shape[0])
    x /= np.linalg.norm(x)
    lam = 1.0
    for _ in range(n_iter):
        y = G @ x
        lam = float(np.linalg.norm(y))
        if lam <= 0:
            return 1.0
        x = y / lam
    return lam


# ---------------------------------------------------------------------------
def compare(sysm: GramSystem, a: np.ndarray, b: np.ndarray,
            active_tol: float = 1e-6) -> dict:
    """How far apart two solutions of the same system are.

    ``active_tol`` is RESS's own deactivation constant: a coordinate the
    coordinate descent sets to exactly 0 and the gradient solve leaves at
    1e-12 is the SAME answer, and a support overlap measured at ``> 0``
    would report it as a disagreement.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    sa, sb = float(a.sum()), float(b.sum())
    diff = a - b
    on_a, on_b = a > active_tol, b > active_tol
    inter = int((on_a & on_b).sum())
    union = int((on_a | on_b).sum())
    return {
        "sum_a": sa, "sum_b": sb,
        "d_sum": sa - sb, "d_sum_rel": (sa - sb) / max(abs(sa), 1e-12),
        "l1_diff": float(np.abs(diff).sum()),
        "l1_diff_rel": float(np.abs(diff).sum() / max(abs(sa), 1e-12)),
        "linf_diff": float(np.abs(diff).max()),
        "l2_diff_rel": float(np.linalg.norm(diff)
                             / max(np.linalg.norm(a), 1e-12)),
        "nnz_a": int(on_a.sum()), "nnz_b": int(on_b.sum()),
        "active_tol": float(active_tol),
        "support_jaccard": (inter / union) if union else 1.0,
        "max_a": float(a.max()), "max_b": float(b.max()),
    }
