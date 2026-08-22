"""Spectral diagnostics of the measurement operator.

Two algorithms over the same object at two cost classes.

``OperatorConditioning`` -- matrix-free.  Lanczos on the restricted Gram, so
the cost is ``steps * probes`` operator applications *regardless of the system
size*.  Reports the extreme eigenvalues, and optionally a stochastic-Lanczos-
quadrature (SLQ) estimate of the trace and of the effective ranks.  Cheap
enough to leave on.

``OperatorSpectrum`` -- dense.  Builds the restricted Gram exactly, one column
per unknown (``n`` operator applications), then diagonalises it.  Gives the full
spectrum plus the per-mode geometry: participation ratio, sign balance, spatial
and time extent, and the charge level each mode occupies.  Opt-in.

Both read ``op`` from the store, so what they characterise is by construction
the same immutable operator the solver used -- the store is write-once
(``fwk/store.py``), which a standalone replay script can only assert.

Definitions
-----------
Two Gram matrices of the same restricted operator ``A P``:

    measurement space   G = A P A^T      dim = op.n_data  (one per latch row)
    charge space        H = P A^T A P    dim = #restricted voxels

They share their non-zero spectrum, so ``rank`` and the non-zero eigenvalues
agree; which one to use is a matter of dimension and of what the eigenvectors
are wanted for (measurement-space vectors are combinations of recorded window
integrals, charge-space vectors are charge patterns).

NOTE on the name: the standalone scripts called the first one "channel space".
That is misleading -- its index is a latch *row*, and one pixel contributes as
many rows as it has latches (856 rows over ~395 pixels for ``mu_a00_nb1``),
while "channel" in LArPix means a readout channel, i.e. a pixel.  ``space:
channel`` is still accepted as an alias.  Pixel language is kept where it
belongs: the coupling profile really is indexed by pixel separation.

The restriction ``P`` is a diagonal 0/1 mask on the fit grid:

    free       P = I, the whole fit grid -- the geometry of the measurement
               itself, no reconstruction in it
    support    the hard ROI mask from BuildSupport
    active     the voxels the solution uses, ``solve.q > active_cut``.  The
               weighted l1 is linear in q on the positive orthant, so it adds
               no curvature: its whole effect on this geometry is which voxels
               survive.

Caveats that are properties of the problem, not of the implementation
--------------------------------------------------------------------
* Whenever the restricted system has fewer rows than unknowns it has an exact
  null space, so ``lambda_min = 0`` and the condition number is infinite.  That
  is the normal case for charge space on the active set (e.g. 219 rows against
  275 active voxels at 75 deg).  ``cond_sqrt`` is therefore reported as null
  rather than as a large finite number, and ``rank_deficit`` is reported
  instead.  A finite condition number is meaningful mainly for measurement space
  with ``restrict: free``.
* Where the spectrum is exactly degenerate -- and the null space of a singular
  system always is -- the individual eigenvectors are arbitrary up to a rotation
  inside the degenerate block, so ``weak_modes[i]`` is basis-dependent and two
  runs of the same job can report different participation or extent for "the
  weakest mode".  The aggregates are not: ``spectrum_deciles`` and any statistic
  over a group of modes are invariant, and they are what should be quoted.
* Rows whose restriction sees no charge have ``G_ii = 0``: the active set is
  blind to them.  They are counted (``n_blind``) and excluded from the spectrum
  and from the correlation statistics, because they cannot be normalised.
* Lanczos returns the extreme *Ritz* values.  The largest converges quickly and
  from below; the smallest converges from above, so ``lambda_min`` from the
  cheap algorithm is an UPPER bound and the condition number derived from it is
  a LOWER bound -- reported as ``lambda_min_upper_bound`` and
  ``cond_sqrt_lower_bound`` so that neither can be read as exact.  On a
  genuinely singular system the cheap algorithm therefore returns a finite
  number where the truth is infinite; ``OperatorSpectrum`` is the reference.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..fwk.component import Algorithm, algorithm

SPACES = ("measurement", "charge")
# "channel" was the name in the standalone scripts.  It is wrong: the index of
# G is a ROW -- one latch window -- and a pixel that latches four times
# contributes four rows, while "channel" in LArPix already means a readout
# channel, i.e. a pixel.  Kept as an alias so archived configs still run.
SPACE_ALIASES = {"channel": "measurement"}
RESTRICTS = ("free", "support", "active")


# --------------------------------------------------------------------------
# restriction and matrix-free products
# --------------------------------------------------------------------------
def _restriction_mask(store, op, restrict: str, active_cut: float):
    """Diagonal 0/1 mask on the fit grid, or None for ``free``."""
    if restrict == "free":
        return None
    if restrict == "support":
        m = np.asarray(store.get("support")).astype(bool)
    elif restrict == "active":
        m = np.asarray(store.get("solve.q")) > float(active_cut)
    else:
        raise ValueError(f"restrict must be one of {RESTRICTS}, got {restrict!r}")
    if tuple(m.shape) != tuple(op.q_shape):
        raise ValueError(f"restriction mask shape {m.shape} != q_shape "
                         f"{tuple(op.q_shape)}")
    return torch.as_tensor(m, dtype=op.dtype, device=op.device)


def _charge_indices(op, mask):
    """Flat fit-grid indices the charge-space system is built on."""
    if mask is None:
        return torch.arange(int(np.prod(op.q_shape)), device=op.device)
    return torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).reshape(-1)


class _MatVec:
    """``y = M x`` for M = G (measurement) or M = H (charge), matrix-free.

    Counts its own applications so the cost is reported, not guessed.
    """

    def __init__(self, op, space: str, mask, idx=None):
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        self.op, self.space, self.mask, self.idx = op, space, mask, idx
        self.n = int(op.n_data) if space == "measurement" else int(idx.numel())
        self.calls = 0
        if space == "charge":
            self._qbuf = torch.zeros(op.q_shape, dtype=op.dtype,
                                     device=op.device)
            self._qflat = self._qbuf.reshape(-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        op = self.op
        if self.space == "measurement":
            v = op.adjoint(x.to(op.dtype))
            if self.mask is not None:
                v = v * self.mask
            return op.forward(v).to(torch.float64)
        self._qflat.zero_()
        self._qflat[self.idx] = x.to(op.dtype)
        y = op.adjoint(op.forward(self._qbuf)).reshape(-1)[self.idx]
        return y.to(torch.float64)

    def unit(self, j: int) -> torch.Tensor:
        e = torch.zeros(self.n, dtype=torch.float64, device=self.op.device)
        e[j] = 1.0
        return e


# --------------------------------------------------------------------------
# Lanczos with full reorthogonalisation
# --------------------------------------------------------------------------
def _lanczos(mv: _MatVec, steps: int, seed: int):
    """m-step Lanczos from a random unit start.

    Returns ``(theta, tau, m_used, (depths, trail))``: Ritz values, their SLQ
    weights (squared first component of each Ritz vector), the number of steps
    actually taken (the recursion can break down early on an exactly singular
    system, which is information, not an error), and the smallest Ritz value
    read off the nested tridiagonal at a quarter, half, three quarters and full
    depth -- the convergence trail.
    """
    n, dev = mv.n, mv.op.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, generator=g, dtype=torch.float64).to(dev)
    v = v / torch.linalg.vector_norm(v)
    m = int(min(steps, n))
    V = torch.zeros((m, n), dtype=torch.float64, device=dev)
    alpha = np.zeros(m)
    beta = np.zeros(max(m - 1, 1))
    V[0] = v
    w = mv(v)
    a = float(torch.dot(w, v))
    alpha[0] = a
    w = w - a * v
    used = 1
    for j in range(1, m):
        b = float(torch.linalg.vector_norm(w))
        if b < 1e-13:                     # invariant subspace reached
            break
        beta[j - 1] = b
        v = w / b
        # full reorthogonalisation: O(j n) and it is what makes the small
        # Ritz values usable at all
        v = v - V[:j].T @ (V[:j] @ v)
        nv = float(torch.linalg.vector_norm(v))
        if nv < 1e-13:
            break
        v = v / nv
        V[j] = v
        w = mv(v)
        a = float(torch.dot(w, v))
        alpha[j] = a
        w = w - a * v - b * V[j - 1]
        w = w - V[:j + 1].T @ (V[:j + 1] @ w)
        used = j + 1
    def tridiag(k):
        T = np.diag(alpha[:k])
        if k > 1:
            i = np.arange(k - 1)
            T[i, i + 1] = beta[:k - 1]
            T[i + 1, i] = beta[:k - 1]
        return T

    theta, S = np.linalg.eigh(tridiag(used))
    # The Krylov space is nested, so the leading k x k block of the same
    # tridiagonal is exactly what k steps would have produced.  Reading the
    # smallest Ritz value at a few depths therefore costs nothing and shows
    # whether it has stopped descending.
    depths = sorted({max(2, used * f // 4) for f in (1, 2, 3, 4)})
    trail = [float(np.linalg.eigvalsh(tridiag(k)).min()) for k in depths]
    return theta, S[0] ** 2, used, (depths, trail)


def _slq_ranks(theta_all, tau_all, n: int, fracs=(0.9, 0.99, 0.999)):
    """Effective ranks from the SLQ spectral-density estimate.

    The density estimate is ``n * mean_probes sum_i tau_i delta(x - theta_i)``,
    so sorting the nodes downward gives an estimated eigenvalue count and an
    estimated eigenvalue mass above any threshold; ``rank_p`` is the count at
    which the mass first reaches ``p`` of the trace.
    """
    theta = np.concatenate(theta_all)
    tau = np.concatenate(tau_all) / len(theta_all)
    o = np.argsort(theta)[::-1]
    theta, tau = np.clip(theta[o], 0.0, None), tau[o]
    count = n * np.cumsum(tau)
    mass = n * np.cumsum(tau * theta)
    trace = float(mass[-1]) if mass.size else 0.0
    out = {"trace_est": trace}
    for p in fracs:
        key = f"rank_{p * 100:g}pct_est".replace(".0pct", "pct")
        if trace <= 0:
            out[key] = None
            continue
        k = int(np.searchsorted(mass, p * trace))
        k = min(k, count.size - 1)
        out[key] = int(round(float(count[k])))
    return out


# --------------------------------------------------------------------------
# geometry of one mode (charge space)
# --------------------------------------------------------------------------
def _mode_stats(v, ix, iy, it, q, lam):
    w = v ** 2                                       # sum(w) = 1
    cx, cy, ct = (w * ix).sum(), (w * iy).sum(), (w * it).sum()
    return {
        "eig": float(lam),
        "participation": float(1.0 / (w ** 2).sum()),
        "pixel_rms": float(np.sqrt((w * ((ix - cx) ** 2
                                         + (iy - cy) ** 2)).sum())),
        "time_rms_bins": float(np.sqrt((w * (it - ct) ** 2).sum())),
        "q_weighted_mean_ke": float((w * q).sum()),
        "sign_balance": float(abs(v.sum()) / np.abs(v).sum()),
    }


def _effective_ranks(w, fracs=(0.9, 0.99, 0.999)):
    tot = max(float(w.sum()), 1e-30)
    cum = np.cumsum(w) / tot
    return {f"rank_{p * 100:g}pct".replace(".0pct", "pct"):
            int(np.searchsorted(cum, p) + 1) for p in fracs}


def _cond_sqrt(lmax, lmin, tol):
    """None when the system is singular -- a huge finite number there is a
    division by round-off, not a condition number."""
    if lmin <= tol * max(lmax, 1e-30):
        return None
    return float(np.sqrt(lmax / lmin))


class _SpectrumAlgorithm(Algorithm):
    """Shared props, store contract and job-summary bookkeeping."""

    def __init__(self, **props):
        super().__init__(**props)
        space = str(props.get("space", "measurement"))
        self.space = SPACE_ALIASES.get(space, space)
        if space in SPACE_ALIASES:
            print(f"[{self.name}] space={space!r} is a deprecated alias for "
                  f"{self.space!r} (the index is a latch row, not a pixel)")
        self.restrict = str(props.get("restrict", "free"))
        if self.space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}")
        if self.restrict not in RESTRICTS:
            raise ValueError(f"restrict must be one of {RESTRICTS}")
        self.active_cut = float(props.get("active_cut", 0.01))
        self.sing_tol = float(props.get("singular_tol", 1e-12))
        self.out_path = props.get("out_path")
        # reads depend on the restriction, so they are set per instance;
        # validate_sequence reads the instance attribute, which is what makes
        # 'restrict: active' provably ordered after Solve.
        reads = ["op"]
        if self.restrict == "support":
            reads.append("support")
        elif self.restrict == "active":
            reads.append("solve.q")
        self.reads = tuple(reads)
        self.writes = (f"{self._prefix}.{self.space}.{self.restrict}",)
        self._records: list[dict] = []

    def _setup(self, store):
        op = store.get("op")
        mask = _restriction_mask(store, op, self.restrict, self.active_cut)
        idx = _charge_indices(op, mask) if self.space == "charge" else None
        mv = _MatVec(op, self.space, mask, idx)
        rec = {"space": self.space, "restrict": self.restrict,
               "n": mv.n, "n_rows": int(op.n_data),
               "q_shape": [int(s) for s in op.q_shape],
               "q_voxels": int(np.prod(op.q_shape)),
               "device": str(op.device), "dtype": str(op.dtype)}
        if mask is not None:
            rec["n_restricted_voxels"] = int(mask.sum().item())
        if self.restrict == "active":
            rec["active_cut"] = self.active_cut
        rec["rank_deficit"] = max(0, mv.n - int(op.n_data)) \
            if self.space == "charge" else 0
        return op, mask, idx, mv, rec

    def _emit(self, store, rec):
        self.put(store, self.writes[0], rec)
        self._records.append(rec)

    def finalize(self):
        if not self._records:
            return {}
        if self.out_path:
            with open(self.out_path, "w") as fh:
                json.dump(self._records if len(self._records) > 1
                          else self._records[0], fh, indent=1)
            print(f"[{self.name}] wrote {self.out_path}")
        return (self._records[0] if len(self._records) == 1
                else {"events": self._records})


# --------------------------------------------------------------------------
# the cheap one
# --------------------------------------------------------------------------
@algorithm("OperatorConditioning")
class OperatorConditioning(_SpectrumAlgorithm):
    """Matrix-free conditioning of the restricted operator.

    Props: ``space`` (measurement|charge), ``restrict`` (free|support|active),
    ``active_cut``, ``steps`` (Lanczos steps per probe; default
    ``max(120, 4*sqrt(n))`` capped at ``n``), ``probes`` (independent random
    starts -- for the SLQ density only, they do NOT help the extremes),
    ``density`` (bool, add the trace and effective-rank estimates), ``seed``,
    ``out_path``.

    Cost is ``steps * probes`` operator applications.  ``converged`` reports
    whether the smallest Ritz value stopped moving between half depth and full
    depth; if it is False, raise ``steps``, do not raise ``probes``.
    """

    _prefix = "conditioning"

    def execute(self, store):
        op, mask, idx, mv, rec = self._setup(store)
        # lambda_min converges with Krylov DEPTH, not with averaging: measured
        # on measurement/free, 480 steps x 1 probe reaches the exact value on
        # an n=11769 system while 240 x 4 -- twice the cost -- is still 15% high.
        # So the default scales the depth with n and leaves probes at 1; probes
        # exist for the SLQ density, not for the extremes.  8*sqrt(n) is where
        # `converged` came out True on both calibration systems (n=856 and
        # n=11769), at 12x fewer applications than the dense algorithm needs.
        steps = int(self.props.get(
            "steps", min(mv.n, max(120, int(8 * np.sqrt(mv.n))))))
        probes = int(self.props.get("probes", 1))
        density = bool(self.props.get("density", probes > 1))
        seed = int(self.props.get("seed", 0))
        t0 = time.time()
        thetas, taus, used, trails = [], [], [], []
        for p in range(probes):
            th, ta, m, tr = _lanczos(mv, steps, seed + 1000 * p)
            thetas.append(th)
            taus.append(ta)
            used.append(m)
            trails.append(tr)
        lmax = float(max(t.max() for t in thetas))
        lmin = float(min(t.min() for t in thetas))
        lmin = max(lmin, 0.0)
        best_trail = min(trails, key=lambda tr: tr[1][-1])
        tr = best_trail[1]
        last_step = (float(tr[-2] / tr[-1] - 1.0)
                     if len(tr) > 1 and tr[-1] > 0 else None)
        rec.update({
            "method": "lanczos",
            "steps_requested": steps, "steps_used": used, "probes": probes,
            "lambda_max": lmax,
            "lambda_min_upper_bound": lmin,
            # convergence trail of the best probe: lambda_min at a quarter,
            # half, three quarters and full depth.  Judged on the LAST step --
            # the half-to-full change is a much looser proxy and reads as "not
            # converged" on systems whose lambda_min is already exact.
            "lanczos_depths": best_trail[0],
            "lambda_min_trail": best_trail[1],
            "lambda_min_last_step": last_step,
            "converged": (None if last_step is None
                          else bool(last_step < 0.02)),
            # lambda_min is an upper bound, so the ratio is a LOWER bound on
            # the condition number.  Naming it cond_sqrt would invite reading
            # a finite number off a system whose true lambda_min is 0.
            "cond_sqrt_lower_bound": _cond_sqrt(lmax, lmin, self.sing_tol),
            "n_matvecs": mv.calls,
            "seconds": round(time.time() - t0, 2),
        })
        if density:
            rec.update(_slq_ranks(thetas, taus, mv.n))
            r99 = rec.get("rank_99pct_est")
            if r99 and self.space == "charge":
                rec["rank_99_over_n"] = r99 / mv.n
        print(f"[{self.name}] {self.space}/{self.restrict} n={mv.n} "
              f"lmax={lmax:.4g} lmin<={lmin:.4g} "
              f"sqrt_kappa>={rec['cond_sqrt_lower_bound']} "
              f"conv={rec['converged']} last_step={last_step} "
              f"({mv.calls} matvecs, {rec['seconds']}s)")
        self._emit(store, rec)


# --------------------------------------------------------------------------
# the expensive one
# --------------------------------------------------------------------------
@algorithm("OperatorSpectrum")
class OperatorSpectrum(_SpectrumAlgorithm):
    """Exact spectrum of the restricted operator, and the mode geometry.

    Props: ``space``, ``restrict``, ``active_cut``, ``n_modes`` (how many
    extreme modes to characterise), ``deciles`` (bool), ``max_dim`` (refuse
    above this, pointing at OperatorConditioning), ``out_path``.

    Cost is ``n`` operator applications and ``n^2`` memory.
    """

    _prefix = "spectrum"

    def execute(self, store):
        op, mask, idx, mv, rec = self._setup(store)
        n_modes = int(self.props.get("n_modes", 20))
        max_dim = int(self.props.get("max_dim", 6000))
        if mv.n > max_dim:
            raise ValueError(
                f"{self.name}: {self.space}/{self.restrict} has n={mv.n} > "
                f"max_dim={max_dim}; the dense Gram would need "
                f"{mv.n ** 2 * 8 / 1e9:.1f} GB and {mv.n} operator "
                f"applications.  Use OperatorConditioning (matrix-free), or "
                f"raise max_dim deliberately.")
        t0 = time.time()
        M = np.zeros((mv.n, mv.n))
        for c in range(mv.n):
            M[:, c] = mv(mv.unit(c)).cpu().numpy()
        M = 0.5 * (M + M.T)                        # symmetrise round-off
        t_gram = time.time() - t0

        # rows/voxels the restriction is blind to: M_ii = 0, cannot be
        # normalised, and they contribute exact zeros to the spectrum
        dg = np.clip(np.diag(M), 0.0, None)
        pos = dg[dg > 0]
        live = (dg > 1e-8 * np.median(pos)) if pos.size else (dg > 1)
        n_blind = int((~live).sum())
        Ml = M[np.ix_(live, live)]
        w, V = np.linalg.eigh(Ml)
        w = np.clip(w[::-1], 0.0, None)
        V = V[:, ::-1]
        lmax, lmin = float(w[0]), float(w[-1])
        rec.update({
            "method": "dense_eigh",
            "n_live": int(live.sum()), "n_blind": n_blind,
            "lambda_max": lmax, "lambda_min": lmin,
            "cond_sqrt": _cond_sqrt(lmax, lmin, self.sing_tol),
            "trace": float(w.sum()),
            "eig_top20": [float(x) for x in w[:20]],
            "eig_tail20": [float(x) for x in w[-20:]],
            "n_matvecs": mv.calls,
            "seconds_gram": round(t_gram, 2),
            "seconds": round(time.time() - t0, 2),
        })
        rec.update(_effective_ranks(w))

        if self.space == "charge":
            self._charge_geometry(store, op, idx, live, w, V, n_modes, rec)
        else:
            self._measurement_geometry(op, live, w, V, n_modes, rec)
        print(f"[{self.name}] {self.space}/{self.restrict} n={mv.n} "
              f"live={rec['n_live']} lmax={lmax:.4g} lmin={lmin:.4g} "
              f"rank99={rec['rank_99pct']} ({mv.calls} matvecs, "
              f"{rec['seconds']}s)")
        self._emit(store, rec)

    # -- charge space: what the modes are, and where the charge is ---------
    def _charge_geometry(self, store, op, idx, live, w, V, n_modes, rec):
        keep = idx.cpu().numpy()[live]
        ix, iy, it = np.unravel_index(keep, tuple(op.q_shape))
        ix, iy, it = ix.astype(float), iy.astype(float), it.astype(float)
        q = (np.asarray(store.get("solve.q"), dtype=np.float64).reshape(-1)[keep]
             if "solve.q" in store else np.zeros(keep.size))
        rec["q_ke"] = {"median": float(np.median(q)), "mean": float(q.mean()),
                       "min": float(q.min()), "max": float(q.max())}
        k = min(n_modes, V.shape[1])
        rec["weak_modes"] = [_mode_stats(V[:, -i], ix, iy, it, q, w[-i])
                             for i in range(1, k + 1)]
        rec["strong_modes"] = [_mode_stats(V[:, i], ix, iy, it, q, w[i])
                               for i in range(k)]
        if bool(self.props.get("deciles", True)):
            n = V.shape[1]
            dec = []
            for j in range(10):
                sl = slice(j * n // 10, (j + 1) * n // 10)
                Vk = V[:, sl]
                if Vk.shape[1] == 0:
                    continue
                Wk = Vk ** 2
                part = 1.0 / (Wk ** 2).sum(0)
                sb = np.abs(Vk.sum(0)) / np.abs(Vk).sum(0)
                ct = (Wk * it[:, None]).sum(0)
                trms = np.sqrt((Wk * (it[:, None] - ct) ** 2).sum(0))
                cx = (Wk * ix[:, None]).sum(0)
                cy = (Wk * iy[:, None]).sum(0)
                prms = np.sqrt((Wk * ((ix[:, None] - cx) ** 2
                                      + (iy[:, None] - cy) ** 2)).sum(0))
                dec.append({
                    "decile": j + 1,
                    "eig_median": float(np.median(w[sl])),
                    "q_weighted_mean_ke": float((Wk.sum(1) / Wk.sum()) @ q),
                    "participation_median": float(np.median(part)),
                    "sign_balance_median": float(np.median(sb)),
                    "pixel_rms_median": float(np.median(prms)),
                    "time_rms_bins_median": float(np.median(trms)),
                })
            rec["spectrum_deciles"] = dec

    # -- measurement space: which windows see the same charge -------------
    def _measurement_geometry(self, op, live, w, V, n_modes, rec):
        # each latch window sits on one pixel, so the row's pixel is the
        # pixel of any block cell it samples
        nx, ny, nt = op.block_shape
        rows = op._rows.cpu().numpy()
        cols = op._cols.cpu().numpy()
        rpx = np.full(int(op.n_data), -1.0)
        rpy = np.full(int(op.n_data), -1.0)
        first = np.full(int(op.n_data), -1, dtype=np.int64)
        for r, c in zip(rows, cols):
            if first[r] < 0:
                first[r] = c
        has = first >= 0
        pix = first[has] // nt
        rpx[has] = pix // ny
        rpy[has] = pix % ny
        rpx, rpy = rpx[live], rpy[live]
        dpx = np.abs(rpx[:, None] - rpx[None, :])
        dpy = np.abs(rpy[:, None] - rpy[None, :])
        dpix = np.maximum(dpx, dpy)
        # normalised coupling rho_ij = G_ij / sqrt(G_ii G_jj)
        Gl = V @ np.diag(w) @ V.T
        d = np.sqrt(np.clip(np.diag(Gl), 1e-30, None))
        RHO = Gl / d[:, None] / d[None, :]
        np.fill_diagonal(RHO, np.nan)
        prof = []
        for k in range(0, int(self.props.get("max_sep", 30)) + 1):
            m = (dpix == k) & np.isfinite(RHO)
            if m.sum() < 3:
                continue
            a = np.abs(RHO[m])
            prof.append({"d": k, "n": int(m.sum()), "mean": float(a.mean()),
                         "p90": float(np.percentile(a, 90)),
                         "max": float(a.max()),
                         "frac_gt_0.1": float((a > 0.1).mean())})
        rec["rho_profile"] = prof
        same = (dpix == 0) & np.isfinite(RHO)
        rec["mean_abs_rho_same_pixel"] = (float(np.abs(RHO[same]).mean())
                                          if same.sum() else None)
        # pixel-space spread of the least-constrained window combinations
        k = min(n_modes, V.shape[1])
        loc = []
        for i in range(1, k + 1):
            v2 = V[:, -i] ** 2
            cx, cy = (v2 * rpx).sum(), (v2 * rpy).sum()
            loc.append({"eig": float(w[-i]),
                        "pixel_rms": float(np.sqrt(
                            (v2 * ((rpx - cx) ** 2 + (rpy - cy) ** 2)).sum())),
                        "sign_balance": float(abs(V[:, -i].sum())
                                              / np.abs(V[:, -i]).sum())})
        rec["weak_dirs"] = loc
        rec["weak_dirs_mean_pixel_rms"] = float(
            np.mean([o["pixel_rms"] for o in loc])) if loc else None
