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

# An eigenvector is treated as well separated when its eigenvalue gap exceeds
# this fraction of lambda_max.  float32 eps is 1.2e-7; 1e-4 leaves three orders
# of headroom, which is what the measured spread between a float32 and a
# float64 run of the same job needs.
EIGVEC_GAP_FACTOR = 1e-4
LMAX = [1.0]                       # set per record, read by _mode_stats

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
def _mode_stats(v, ix, iy, it, q, lam, gap=None, q_med=None):
    w = v ** 2                                       # sum(w) = 1
    cx, cy, ct = (w * ix).sum(), (w * iy).sum(), (w * it).sum()
    med = np.median(q) if q_med is None else q_med
    out = {
        "eig": float(lam),
        "participation": float(1.0 / (w ** 2).sum()),
        "pixel_rms": float(np.sqrt((w * ((ix - cx) ** 2
                                         + (iy - cy) ** 2)).sum())),
        "time_rms_bins": float(np.sqrt((w * (it - ct) ** 2).sum())),
        "q_weighted_mean_ke": float((w * q).sum()),
        "q_frac_below_median": float(w[q < med].sum()),
        "sign_balance": float(abs(v.sum()) / np.abs(v).sum()),
    }
    if gap is not None:
        # An individual eigenvector is only defined to within a rotation inside
        # the cluster of eigenvalues it is degenerate with.  Under a matrix
        # perturbation of size delta the eigenvector moves by ~delta/gap, and
        # delta here is the operator's own precision, ~eps(float32)*lambda_max.
        # So this flag says whether the per-mode geometry above means anything
        # for THIS mode; the aggregates over a group of modes are invariant
        # either way.
        out["eig_gap_to_neighbour"] = float(gap)
        out["eigvec_well_separated"] = bool(gap > EIGVEC_GAP_FACTOR * LMAX[0])
    return out


def _eig_gaps(w):
    """Distance from each eigenvalue to its nearest neighbour in the spectrum."""
    if w.size < 2:
        return np.full(w.size, np.inf)
    d = np.abs(np.diff(w))
    return np.minimum(np.r_[np.inf, d], np.r_[d, np.inf])


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
        # An eigenvalue below sing_tol * lambda_max is indistinguishable from
        # zero at the precision the operator applies in.  The default is tied
        # to float32 (eps ~ 1.2e-7), not to float64: on mu_a75_nb1's active
        # system the smallest eigenvalues come out at 5e-7 .. 5e-9 against
        # lambda_max = 121, i.e. 4e-9 .. 4e-11 relative -- pure round-off, and
        # a condition number built from them is a division by noise.
        self.sing_tol = float(props.get("singular_tol", 1e-6))
        self.out_path = props.get("out_path")
        # reads depend on the restriction, so they are set per instance;
        # validate_sequence reads the instance attribute, which is what makes
        # 'restrict: active' provably ordered after Solve.
        reads = ["op"]
        if self.space == "measurement":
            # row identity, for the per-kind coupling statistics; published by
            # BuildMeasurement so nothing has to re-run build_latch_rows
            reads.append("row_meta")
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

    @staticmethod
    def _coarse_bin(store):
        """adc_hold_delay, in the operator's own bin units."""
        if "readout_config" not in store:
            return None
        B = int(store.get("readout_config").adc_hold_delay)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        return B // S

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
        n_noise = int((w <= self.sing_tol * max(lmax, 1e-30)).sum())
        rec.update({
            "method": "dense_eigh",
            "n_live": int(live.sum()), "n_blind": n_blind,
            "lambda_max": lmax, "lambda_min": lmin,
            "singular_tol": self.sing_tol,
            # eigenvalues at or below sing_tol * lambda_max: numerically zero
            # at the operator's precision, so neither they nor anything derived
            # from them (cond_sqrt, the individual weakest eigenvectors) carries
            # information
            "n_eig_at_roundoff": n_noise,
            "numerically_singular": bool(n_noise > 0),
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
            self._measurement_geometry(
                op, live, w, V, n_modes, rec,
                store.get("row_meta") if "row_meta" in store else None,
                Ml, coarse_bin=self._coarse_bin(store))
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
        LMAX[0] = float(w[0]) if w.size else 1.0
        q_med = float(np.median(q))
        gaps = _eig_gaps(w)
        rec["weak_modes"] = [
            _mode_stats(V[:, -i], ix, iy, it, q, w[-i],
                        gaps[len(w) - i], q_med) for i in range(1, k + 1)]
        rec["strong_modes"] = [
            _mode_stats(V[:, i], ix, iy, it, q, w[i], gaps[i], q_med)
            for i in range(k)]
        rec["n_modes_well_separated"] = int(
            (gaps > EIGVEC_GAP_FACTOR * LMAX[0]).sum())
        # charge level occupied by the strong and the weak half of the
        # spectrum: the aggregate answer to "is weak the same as low charge?".
        # A half is a subspace, so unlike an individual mode it is invariant
        # under rotations inside a degenerate cluster -- except where the
        # halving boundary itself falls inside one, which is why the gap at the
        # boundary is reported alongside.
        nh = V.shape[1] // 2
        for lab, sl in (("strong_half", slice(0, nh)),
                        ("weak_half", slice(nh, V.shape[1]))):
            W = V[:, sl] ** 2
            tot = W.sum()
            rec[f"q_weighted_mean_{lab}_ke"] = (
                float((W.sum(1) / tot) @ q) if tot > 0 else None)
        rec["eig_gap_at_half_boundary"] = (float(gaps[nh])
                                           if nh < gaps.size else None)
        rec["half_boundary_well_separated"] = (
            bool(gaps[nh] > EIGVEC_GAP_FACTOR * LMAX[0])
            if nh < gaps.size else None)
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
    def _measurement_geometry(self, op, live, w, V, n_modes, rec, meta=None,
                              Gl=None, coarse_bin=None):
        rt = None
        if meta is not None:
            rpx = np.asarray(meta["px"], dtype=float)
            rpy = np.asarray(meta["py"], dtype=float)
            kind = kind_all = np.asarray(meta["kind"], dtype=object)
            # the row's latch instant is its window's upper edge
            rt = np.asarray(meta["t_hi"], dtype=float)
        else:
            # fallback: each latch window sits on one pixel, so the row's pixel
            # is the pixel of any block cell it samples
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
            kind = kind_all = None
        rpx, rpy = rpx[live], rpy[live]
        if rt is not None:
            rt = rt[live]
        if kind is not None:
            kind = kind[live]
        dpx = np.abs(rpx[:, None] - rpx[None, :])
        dpy = np.abs(rpy[:, None] - rpy[None, :])
        dpix = np.maximum(dpx, dpy)
        # normalised coupling rho_ij = G_ij / sqrt(G_ii G_jj), from the Gram
        # ITSELF.  An earlier version reconstructed it as V diag(w) V^T, which
        # is a round trip through an eigendecomposition whose negative
        # eigenvalues have been clipped to zero -- on a near-singular system
        # the clipped mass is exactly the size of the smallest rho entries, so
        # the reconstruction corrupted precisely the long-range coupling the
        # profile is there to measure.
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
                         "frac_gt_0.1": float((a > 0.1).mean()),
                         "frac_gt_0.5": float((a > 0.5).mean())})
        rec["rho_profile"] = prof
        same = (dpix == 0) & np.isfinite(RHO)
        rec["mean_abs_rho_same_pixel"] = (float(np.abs(RHO[same]).mean())
                                          if same.sum() else None)
        # beyond the response half-width the two kernels do not overlap, so
        # anything left here is round-off (or a normalisation artefact on rows
        # that see almost no charge)
        khalf = int(self.props.get("kernel_half_width", 12))
        far = (dpix > khalf) & np.isfinite(RHO)
        rec["mean_abs_rho_beyond_kernel"] = (float(np.abs(RHO[far]).mean())
                                             if far.sum() else None)
        # the coupling resolved on the two pixel axes separately, not just on
        # the Chebyshev distance: an anisotropic response shows up here
        M = np.full((khalf + 3, khalf + 3), np.nan)
        for a1 in range(M.shape[0]):
            for a2 in range(M.shape[1]):
                m = (dpx == a1) & (dpy == a2) & np.isfinite(RHO)
                if m.sum() >= 3:
                    M[a1, a2] = float(np.abs(RHO[m]).mean())
        rec["map_dpx_dpy"] = [[None if np.isnan(x) else float(x) for x in row]
                              for row in M]
        # -- time-resolved coupling ---------------------------------------
        # The pixel-distance profile marginalises Delta t away, so the
        # same-pixel column is one number covering every time separation at
        # once.  Re-bin the SAME rho by the separation between the two rows'
        # latch instants.  A re-binning, not a new measurement of the
        # operator: ports channel_coupling_dt/dt_coupling.py, whose bin k
        # covers |Delta t| / B in [k-1, k) and is labelled by its upper edge.
        if rt is not None and coarse_bin:
            B = float(coarse_bin)
            one_tick_us = float(self.props.get("tick_us", 0.05))
            nbin = int(self.props.get("dt_bins", 24))
            # four defensible definitions of "how far apart" two windows are;
            # dt_coupling.py used the first and dt_definition.py asked whether
            # that choice matters, so all four are reported rather than one
            # being picked silently
            lo_e = np.asarray(meta["t_lo"], dtype=float)[live]
            lo_e = np.maximum(lo_e, 0.0)
            mid = 0.5 * (lo_e + rt)
            DEFS = {
                "latch": np.abs(rt[:, None] - rt[None, :]) / B,
                "mid": np.abs(mid[:, None] - mid[None, :]) / B,
                "gap": np.maximum(
                    np.maximum(lo_e[:, None], lo_e[None, :])
                    - np.minimum(rt[:, None], rt[None, :]), 0.0) / B,
            }
            which = str(self.props.get("dt_definition_used", "latch"))
            DT = DEFS[which]
            # a pair is UNORDERED and is not a row with itself: the Gram is
            # symmetric, so counting the full matrix double-counts every pair
            # and adds n diagonal entries of rho = 1.
            iu = np.triu(np.ones_like(RHO, dtype=bool), k=1)
            # bin by the NEAREST whole bin: the latch grid is quantised, so
            # a pseudo/remainder pair sits at exactly 1 bin and belongs in
            # bin 1, not in a [0,1) bucket.
            dtb = np.rint(DT).astype(int)
            exact = np.abs(DT - dtb) < 1e-6
            fin = np.isfinite(RHO) & iu
            # "near" is a whole number of bins, applied to the SAME nearest-bin
            # index the steps use, so a pair cannot be near by one definition
            # and far by the other
            near_us = float(self.props.get("near_us", 3.0))
            near_bins = int(round(near_us / (B * one_tick_us)))
            near = DT <= near_bins        # raw ratio, inclusive

            def st(mask, extra=None):
                m = mask & fin
                if m.sum() < 1:
                    return None
                a = np.abs(RHO[m])
                out = {"n": int(m.sum()), "mean": float(a.mean()),
                       "median": float(np.median(a)),
                       "p90": float(np.percentile(a, 90)),
                       "max": float(a.max()),
                       "frac_gt_0.1": float((a > 0.1).mean()),
                       "frac_gt_0.5": float((a > 0.5).mean())}
                if extra:
                    out.update(extra)
                return out

            sp = dpix == 0
            steps = []
            for k in range(1, nbin + 2):
                m = sp & (dtb == k)
                r = st(m, {"n_exact": int((m & exact & fin).sum())})
                if r:
                    r["dt_bins"] = k
                    r["dt_us"] = k * B * one_tick_us
                    steps.append(r)
            rec["same_pixel_dt_steps"] = steps
            if kind is not None:
                nops = kind != "pseudo"
                np_mask = sp & nops[:, None] & nops[None, :]
                steps2 = []
                for k in range(1, nbin + 2):
                    r = st(np_mask & (dtb == k))
                    if r:
                        r["dt_bins"] = k
                        r["dt_us"] = k * B * one_tick_us
                        steps2.append(r)
                rec["same_pixel_dt_steps_no_pseudo"] = steps2
            khalf = int(self.props.get("kernel_half_width", 12))
            rec["same_pixel_within"] = st(sp & near)
            rec["same_pixel_beyond"] = st(sp & ~near)
            rec["d1_within"] = st((dpix == 1) & near)
            rec["d1_beyond"] = st((dpix == 1) & ~near)
            rec["beyond_kernel_within"] = st((dpix > khalf) & near)
            rec["beyond_kernel_beyond"] = st((dpix > khalf) & ~near)
            sp0 = (dpix == 0) & fin
            rec["dt_definition_sensitivity"] = {
                k: {"mean_abs_rho_within": (
                        float(np.abs(RHO[sp0 & (v <= near_bins)]).mean())
                        if (sp0 & (v <= near_bins)).any() else None),
                    "n_within": int((sp0 & (v <= near_bins)).sum())}
                for k, v in DEFS.items()}
            rec["dt_definition"] = {
                "used": which,
                "delta_t": "|t_hi_i - t_hi_j|, the two rows' latch instants",
                "bin_k_covers": "|dt|/B in [k-1, k), labelled by the upper edge",
                "near_split_us": near_us,
                "near_split_bins": near_bins,
                "coarse_bin_ticks": coarse_bin,
                "tick_us": one_tick_us}
        # mean |rho| between rows of each pair of kinds: are two windows of the
        # same kind more alike than two of different kinds?
        if kind is not None:
            kinds = sorted(set(kind.tolist()))
            kp = {}
            for k1 in kinds:
                i1 = np.flatnonzero(kind == k1)
                for k2 in kinds:
                    i2 = np.flatnonzero(kind == k2)
                    if not (i1.size and i2.size):
                        continue
                    sub = RHO[np.ix_(i1, i2)]
                    sub = sub[np.isfinite(sub)]
                    kp[f"{k1}|{k2}"] = (float(np.abs(sub).mean())
                                        if sub.size else None)
            rec["kind_pairs"] = kp
            # over ALL rows, matching the archived top-level field: the row
            # kinds are a property of the operator, not of the restriction.
            # An earlier version counted only the live rows, which made this
            # move between the free and the restricted systems of one event.
            rec["row_kinds"] = {k: int(v) for k, v in
                                zip(*np.unique(kind_all, return_counts=True))}
            rec["row_kinds_live"] = {k: int((kind == k).sum()) for k in kinds}
        # pixel-space spread of the least-constrained window combinations
        k = min(n_modes, V.shape[1])
        loc = []
        for i in range(1, k + 1):
            v2 = V[:, -i] ** 2
            cx, cy = (v2 * rpx).sum(), (v2 * rpy).sum()
            loc.append({"eig": float(w[-i]),
                        "pixel_rms": float(np.sqrt(
                            (v2 * ((rpx - cx) ** 2 + (rpy - cy) ** 2)).sum())),
                        "participation": float(1.0 / (v2 ** 2).sum()),
                        "sign_balance": float(abs(V[:, -i].sum())
                                              / np.abs(V[:, -i]).sum())})
        rec["weak_dirs"] = loc
        rec["weak_dirs_mean_pixel_rms"] = float(
            np.mean([o["pixel_rms"] for o in loc])) if loc else None
        gaps = _eig_gaps(w)
        for i, o in enumerate(loc, start=1):
            g = float(gaps[len(w) - i])
            o["eig_gap_to_neighbour"] = g
            o["eigvec_well_separated"] = bool(g > EIGVEC_GAP_FACTOR * w[0])
        n_strong = int(self.props.get("n_strong", 5))
        strong = []
        for i in range(min(n_strong, V.shape[1])):
            v2 = V[:, i] ** 2
            cx, cy = (v2 * rpx).sum(), (v2 * rpy).sum()
            strong.append({"eig": float(w[i]),
                           "pixel_rms": float(np.sqrt(
                               (v2 * ((rpx - cx) ** 2
                                      + (rpy - cy) ** 2)).sum())),
                           "participation": float(1.0 / (v2 ** 2).sum()),
                           "eig_gap_to_neighbour": float(gaps[i]),
                           "eigvec_well_separated":
                               bool(gaps[i] > EIGVEC_GAP_FACTOR * w[0])})
        rec["strong_dirs"] = strong
