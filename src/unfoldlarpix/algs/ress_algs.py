"""RESS bench: Wire-Cell's coordinate descent against the shipped FISTA.

The question these two algorithms exist to answer is narrow and it is
about the SOLVER, not the model: on one and the same convex problem --
same operator, same data, same support, same prior -- does proximal
(projected) gradient descent land where Wire-Cell's coordinate descent
lands?  Anything that differs between the two arms other than the
optimisation algorithm would make the answer meaningless, so the system
is built ONCE, published to the store, and every arm reads that one
product.

``BuildRESSProblem``
    Materialises ``G = A^T A`` and ``b = A^T d`` on an ROI of the charge
    grid, using the operator's own forward and adjoint (one pair per ROI
    voxel).  The reduced system is EXACTLY the production problem with
    ``support = ROI``: no row is dropped, no response is re-derived, and
    ``const = 1/2||d||^2`` makes the Gram objective numerically equal to
    the operator's own ``L`` -- which is checked, not assumed.

``RESSSolverScan``
    Runs the arms on that system: ``ress`` (coordinate descent, ported
    from WCP ``LassoModel::Fit``), ``pgd`` (FISTA/projected gradient on
    the same Gram) and ``pgd_operator`` (the production
    :func:`~unfoldlarpix.algs.fixedgrid_algs.solve_arm`, i.e. FISTA
    through the FFT operator, so the FFT round-off is separable from the
    algorithm).  Each arm reports its objective, its KKT violation, the
    production metrics, and every pair reports how far apart the two
    solutions actually are.

Why an ROI and not the whole grid: coordinate descent needs a column of
``A`` per unknown.  Wire-Cell's ROIs are ~10^2-10^3 unknowns with an
explicit response matrix; this operator has 3.66 M unknowns and an
81250-entry kernel, so ``A`` cannot be materialised and neither can
``A^T A``.  The ROI is the largest sub-problem that can be, and it is
chosen to CONTAIN THE TRUTH so the reduced problem is a real
reconstruction and not a toy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..fwk.component import Algorithm, algorithm
from ..solve.ress import (GramSystem, build_gram_system, compare,
                          pgd_engine_fit, pgd_fit, ress_fit)
from .fixedgrid_algs import (_JsonRecorder, grid_truth, loss, resolve_support,
                             score_universal, solve_arm, voxel_stats)


def roi_mask(op, spec: dict, base: np.ndarray | None = None) -> np.ndarray:
    """ROI on the charge grid from an inclusive index box.

    ``{x: [lo, hi], y: [lo, hi], t: [lo, hi]}`` in CHARGE-GRID indices
    (the frame ``grid_truth`` and the solver both use); a missing axis is
    the full extent.  ``base`` is an optional mask to intersect with --
    the production support, so the ROI can never be larger than what the
    production arm was allowed.
    """
    nx, ny, nt = op.q_shape
    m = np.zeros((nx, ny, nt), dtype=bool)
    def rng(key, n):
        v = spec.get(key)
        if v is None:
            return 0, n - 1
        lo, hi = int(v[0]), int(v[1])
        if not (0 <= lo <= hi < n):
            raise ValueError(f"roi {key} {v} outside [0, {n - 1}]")
        return lo, hi
    x0, x1 = rng("x", nx); y0, y1 = rng("y", ny); t0, t1 = rng("t", nt)
    m[x0:x1 + 1, y0:y1 + 1, t0:t1 + 1] = True
    if base is not None:
        m &= np.asarray(base, dtype=bool)
    return m


# ---------------------------------------------------------------------------
# the Gram build is the expensive stage; cache it so a scan can be re-run
# ---------------------------------------------------------------------------
def system_stamp(op, roi: dict, support_spec, mask: np.ndarray) -> dict:
    """Everything a cached Gram must agree with to be reusable.

    Not a checksum of convenience: it pins the OPERATOR (row count, the
    data vector, the kernel's DC gain) as well as the ROI, so a cache
    written for another event, another response or another readout can
    never be picked up silently.
    """
    import hashlib
    h = hashlib.sha1(np.ascontiguousarray(
        np.argwhere(mask).astype(np.int32)).tobytes()).hexdigest()[:16]
    return {"q_shape": [int(s) for s in op.q_shape],
            "block_shape": [int(s) for s in op.block_shape],
            "n_rows": int(op.n_data),
            "sum_d": round(float(op.d.double().sum()), 6),
            "sumsq_d": round(float((op.d.double() ** 2).sum()), 4),
            "kernel_dc": round(float(op._K[0, 0, 0].real), 8),
            "roi": roi, "support_spec": str(support_spec),
            "n_columns": int(mask.sum()), "mask_sha1": h}


def load_cache(path, stamp: dict):
    """Return the cached :class:`GramSystem` iff its stamp matches."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("stamp") != stamp:
        print(f"[BuildRESSProblem] cache {p} stamp MISMATCH -- rebuilding\n"
              f"  cached  {meta.get('stamp')}\n  wanted  {stamp}")
        return None
    return GramSystem(idx=z["idx"], q_shape=tuple(int(s) for s in z["q_shape"]),
                      G=z["G"], b=z["b"], const=float(z["const"]), meta=meta)


def save_cache(path, sysm) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with open(tmp, "wb") as fh:      # savez would append a second .npz
        np.savez(fh, idx=sysm.idx, q_shape=np.array(sysm.q_shape), G=sysm.G,
                 b=sysm.b, const=np.array(sysm.const),
                 meta_json=np.array(json.dumps(sysm.meta, default=float)))
    tmp.replace(p)
    print(f"[BuildRESSProblem] cached Gram -> {p} "
          f"({p.stat().st_size / 1e9:.2f} GB)")


@algorithm("BuildRESSProblem")
class BuildRESSProblem(Algorithm):
    """Publish the explicit Gram system of an ROI sub-problem.

    Props
    -----
    roi : dict
        ``{x: [lo, hi], y: [lo, hi], t: [lo, hi]}``, inclusive, charge-grid
        indices.  Omitted axes are full.
    support : str, optional
        Intersect the box with a production support spec (``hits``,
        ``gain:F``, ``none``); default ``none`` -- the box alone.
    max_cols : int
        Refuse to build above this (the build is one forward+adjoint per
        column and the Gram is ``n^2`` doubles).  Default 30000.
    verify : int
        Random non-negative vectors pushed through both the Gram objective
        and the operator's ``L``; they must agree.  Default 3.
    """

    reads = ("op", "support")
    writes = ("ress.problem",)

    def execute(self, store):
        op = store.get("op")
        spec = dict(self.props.get("roi") or {})
        supp_spec = self.props.get("support", "none")
        base = resolve_support(store, op, supp_spec)
        m = roi_mask(op, spec, base)
        n = int(m.sum())
        cap = int(self.props.get("max_cols", 30000))
        if n > cap:
            raise ValueError(
                f"ROI has {n} columns > max_cols {cap}: the Gram build is "
                f"{n} forward+adjoint pairs and {n * n * 8 / 1e9:.1f} GB")
        print(f"[{self.name}] ROI {spec} & support {supp_spec!r}: {n} of "
              f"{int(np.prod(op.q_shape))} voxels, Gram "
              f"{n * n * 8 / 1e9:.2f} GB")
        stamp = system_stamp(op, spec, supp_spec, m)
        cached = load_cache(self.props.get("cache"), stamp)
        if cached is not None:
            print(f"[{self.name}] reusing cached Gram "
                  f"{self.props['cache']} ({n} columns)")
            self.put(store, "ress.problem", cached)
            return
        t0 = time.time()

        def progress(k, tot):
            el = time.time() - t0
            print(f"[{self.name}] Gram column {k}/{tot} "
                  f"({el:.0f} s, {el / max(k, 1) * tot:.0f} s total)",
                  flush=True)

        sysm = build_gram_system(op, m, verify=int(self.props.get("verify", 3)),
                                 progress=progress)
        sysm.meta.update({"roi": spec, "support_spec": supp_spec,
                          "n_columns": n, "n_rows": int(op.n_data),
                          "q_shape": list(op.q_shape),
                          "build_seconds": time.time() - t0})
        sysm.meta["stamp"] = stamp
        for c in sysm.meta["closure_checks"]:
            print(f"[{self.name}] closure: L_operator {c['L_operator']:.6g} "
                  f"L_gram {c['L_gram']:.6g} rel {c['rel_diff']:.3e}")
        save_cache(self.props.get("cache"), sysm)
        self.put(store, "ress.problem", sysm)


@algorithm("RESSSolverScan")
class RESSSolverScan(_JsonRecorder):
    """Solve one published system several ways and compare the answers.

    Props
    -----
    arms : list of dict
        ``solver``: ``ress`` (the port of Wire-Cell's coordinate descent),
        ``ress_cxx`` (Wire-Cell's OWN compiled code, driven through the
        Cholesky bridge in :mod:`unfoldlarpix.solve.ress_wirecell`; needs
        ``lib:`` and does not scale past ~10^3 unknowns), ``pgd`` (FISTA
        on the Gram), ``pgd_plain`` (unaccelerated projected gradient) or
        ``pgd_engine`` (THE REPOSITORY'S OWN ``Fista`` + ``DataFidelity`` +
        ``CoordProx``, run unmodified on this Gram through
        :class:`~unfoldlarpix.solve.ress.GramOperator`) or
        ``pgd_operator`` (that same solver through the real FFT operator).
        ``lambda`` / ``alpha`` / ``penalty`` set the prior -- ``penalty``
        is ``ress`` (Wire-Cell: the L1 weight scales with the column norm)
        or ``uniform`` (this package's ``CoordProx``).  ``tol`` /
        ``max_iter`` for the coordinate descent, ``iters`` for the
        gradient arms, ``positivity`` for both.
    active_spectrum_max : int
        Diagonalise ``G`` on the union of the arms' active sets when it is
        no larger than this (default 4000).  A positive definite active
        block means the minimiser is UNIQUE, which is what decides whether
        two solvers may legitimately disagree.
    out : str, optional
    """

    reads = ("ress.problem", "op", "event", "readout_config", "block_offset")
    writes = ("ress.compare",)

    def execute(self, store):
        sysm = store.get("ress.problem")
        op = store.get("op")
        m = np.zeros(op.q_shape, dtype=bool)
        m[sysm.idx[:, 0], sysm.idx[:, 1], sysm.idx[:, 2]] = True
        qg = grid_truth(store, op)
        truth_roi = float(qg[m].sum())
        truth_all = float(qg.sum())
        beta_truth = qg[sysm.idx[:, 0], sysm.idx[:, 1], sysm.idx[:, 2]]
        print(f"[{self.name}] truth {truth_all:.1f} ke on the grid, "
              f"{truth_roi:.1f} ke ({100 * truth_roi / truth_all:.2f}%) in the "
              f"ROI; L(q_truth|ROI) {sysm.data_term(beta_truth):.6g}")

        arms, sols = [], {}
        for spec in self.props.get("arms", []):
            kind = str(spec.get("solver", "ress"))
            lam = float(spec.get("lambda", 0.0))
            al = float(spec.get("alpha", 1.0))
            pen = str(spec.get("penalty", "ress"))
            pos = bool(spec.get("positivity", True))
            label = spec.get("label", kind)
            gamma = sysm.gamma(lam, al, None, pen)
            t0 = time.time()
            if kind == "ress":
                beta, info = ress_fit(
                    sysm, lam=lam, alpha=al, penalty=pen,
                    max_iter=int(spec.get("max_iter", 100000)),
                    tol=float(spec.get("tol", 1e-3)), non_negative=pos,
                    trace_every=int(spec.get("trace_every", 0)))
            elif kind == "ress_cxx":
                from ..solve import ress_wirecell
                if pen != "ress":
                    raise ValueError("ress_cxx runs Wire-Cell's own code; its "
                                     "penalty convention is 'ress' by "
                                     "construction")
                beta, info = ress_wirecell.solve(
                    sysm, lam=lam, alpha=al,
                    model=str(spec.get("model", "lasso")),
                    max_iter=int(spec.get("max_iter", 100000)),
                    tol=float(spec.get("tol", 1e-3)), non_negative=pos,
                    lib=spec.get("lib"))
            elif kind == "pgd_engine":
                beta, info = pgd_engine_fit(
                    sysm, gamma=gamma, n_iter=int(spec.get("iters", 3000)),
                    positivity=pos, device=str(spec.get("device", "cuda")),
                    dtype=str(spec.get("dtype", "float64")),
                    trace_every=int(spec.get("trace_every", 0)))
            elif kind in ("pgd", "pgd_plain"):
                beta, info = pgd_fit(
                    sysm, gamma=gamma, n_iter=int(spec.get("iters", 3000)),
                    accel=(kind == "pgd"), positivity=pos,
                    device=str(spec.get("device", "cpu")),
                    trace_every=int(spec.get("trace_every", 0)))
            elif kind == "pgd_operator":
                if pen != "uniform" and lam != 0.0:
                    raise ValueError("pgd_operator carries a uniform alpha "
                                     "prox; use penalty: uniform")
                q = solve_arm(op, m, lam * al, pos,
                              int(spec.get("iters", 3000)))
                beta = q[sysm.idx[:, 0], sysm.idx[:, 1], sysm.idx[:, 2]]
                info = {"solver": "pgd_operator",
                        "n_iter": int(spec.get("iters", 3000)),
                        "leak_outside_roi": float(q.sum() - beta.sum())}
            else:
                raise ValueError(f"unknown solver {kind!r}")
            wall = time.time() - t0

            q = sysm.embed(beta)
            obj = sysm.objective(beta, gamma)
            kkt = sysm.kkt(beta, gamma)
            sc = score_universal(store, op, q); tp = sc.pop("transport")
            rec = {
                "label": label, "solver": kind, "lambda": lam, "alpha": al,
                "penalty": pen, "positivity": pos, "wall_seconds": wall,
                "info": {k: v for k, v in info.items() if k != "trace"},
                "trace": info.get("trace", []),
                **obj, **kkt,
                "L_operator": loss(op, q),
                "sum_q": float(beta.sum()),
                "ratio_truth_roi": float(beta.sum() / truth_roi),
                "nnz": int((beta > 1e-6).sum()),
                "max_q": float(beta.max()), "min_q": float(beta.min()),
                "voxels": voxel_stats(q),
                "universal": sc, "transport": tp,
            }
            arms.append(rec); sols[label] = beta
            del q
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{self.name}] {label:22s} {wall:7.1f} s  F {obj['objective']:.8g}"
                  f"  data {obj['data']:.8g}  L_op {rec['L_operator']:.8g}"
                  f"  KKT {kkt['kkt_rel']:.3e}  sum {beta.sum():9.2f} ke"
                  f" ({rec['ratio_truth_roi']:.4f}x truth)  nnz {rec['nnz']}")

        labels = list(sols)
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                c = compare(sysm, sols[a], sols[b])
                c["pair"] = f"{a} | {b}"
                # Each arm's F is evaluated under ITS OWN gamma, so dF is
                # only a comparison when the two arms share a prior; the
                # data term always is one, and it is the honest column for
                # a cross-prior pair.
                c["d_objective"] = (arms[i]["objective"] - arms[j]["objective"])
                c["d_data"] = arms[i]["data"] - arms[j]["data"]
                c["same_prior"] = bool(
                    arms[i]["lambda"] == arms[j]["lambda"]
                    and arms[i]["alpha"] == arms[j]["alpha"]
                    and arms[i]["penalty"] == arms[j]["penalty"])
                pairs.append(c)
                print(f"[{self.name}] {c['pair']:44s} "
                      f"{'dF' if c['same_prior'] else 'dL'} "
                      f"{(c['d_objective'] if c['same_prior'] else c['d_data']):+.6g}"
                      f"  |dq|_1/sum {c['l1_diff_rel']:.3e}"
                      f"  max|dq| {c['linf_diff']:.4g} ke"
                      f"  jaccard {c['support_jaccard']:.4f}")

        spec_block = None
        cap = int(self.props.get("active_spectrum_max", 4000))
        if sols:
            on = np.zeros(sysm.n, dtype=bool)
            for v in sols.values():
                on |= v > 1e-6
            k = int(on.sum())
            if 0 < k <= cap:
                ev = np.linalg.eigvalsh(sysm.G[np.ix_(on, on)])
                spec_block = {
                    "n_active_union": k, "lambda_min": float(ev[0]),
                    "lambda_max": float(ev[-1]),
                    "condition": float(ev[-1] / ev[0]) if ev[0] > 0 else None,
                    "rank_tol": float(ev[-1] * k * 2.22e-16),
                    "numerical_rank": int(
                        (ev > ev[-1] * k * 2.22e-16).sum()),
                }
                print(f"[{self.name}] active block {k}x{k}: lambda_min "
                      f"{ev[0]:.6g}, lambda_max {ev[-1]:.6g}, rank "
                      f"{spec_block['numerical_rank']}/{k}")
            else:
                spec_block = {"n_active_union": k, "skipped_above": cap}

        self._emit(store, {
            "problem": {k: v for k, v in sysm.meta.items()},
            "truth_on_grid": truth_all, "truth_in_roi": truth_roi,
            "L_truth_roi": sysm.data_term(beta_truth),
            "arms": arms, "pairs": pairs, "active_spectrum": spec_block})
