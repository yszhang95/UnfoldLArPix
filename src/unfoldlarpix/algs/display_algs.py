"""Event-display algorithms: what a reconstruction looks like, voxel by voxel.

The scan algorithms in :mod:`~unfoldlarpix.algs.fixedgrid_algs` reduce a
reconstruction to scalars.  This one keeps the picture: it runs the SAME
estimators on the SAME operator, rebins them with the SAME adopted eval
(:func:`~unfoldlarpix.algs.fixedgrid_algs.universal_blocks`), and writes the
projections of the truth, of each reconstruction, and of the per-voxel error

    e_v = |truth_v - reco_v| / sum(truth)

so the error map is a FRACTION OF THE EVENT'S TOTAL CHARGE and every panel of
every arm shares one normalisation.  Summing a whole map gives the relative
L1 error, which is the honest headline the scalar table does not carry:
``integral_pct`` is a signed total and cancels, ``r``/``slope`` are computed
on a reco-selected mask, and neither sees charge that merely MOVED.

Two cut states are written for every arm:

``nocut``   the reconstruction as the solver produced it;
``cut``     the reconstruction with every voxel at or below ``cut_ke``
            zeroed -- what a downstream analysis actually keeps.  Charge
            removed by the threshold therefore appears in the error map at
            full weight, which is the point: it quantifies the loss the
            threshold causes.

Projections are SUMS along the missing axis, so the total of a map is
conserved and the three views of one arm agree on their total.
"""
from __future__ import annotations

import numpy as np
import torch

from ..eval.universal import metrics_from_blocks
from ..fwk.component import algorithm
from ..model.warm_start import deconv_fft_torch, gaussian_filter_3d_torch
from .fixedgrid_algs import (_JsonRecorder, block_from_rows, fit_bin_ticks,
                             score_universal,
                             grid_truth, loss, resolve_support, solve_arm,
                             universal_blocks, voxel_stats)


def _projections(vol: np.ndarray) -> dict:
    """Sum-projections of a block onto the three coordinate planes."""
    return {"yt": vol.sum(axis=0), "xt": vol.sum(axis=1),
            "xy": vol.sum(axis=2)}


@algorithm("ResidualDisplay")
class ResidualDisplay(_JsonRecorder):
    """Per-voxel error maps for a set of estimators, on one event.

    Props
    -----
    arms : list of dict
        ``kind: fft``     linear inverse; ``sigma_time`` / ``sigma_pixel``
                          are FREQUENCY-domain widths (``sigma_time: null``
                          = unfiltered).  NOTE a non-null filter DOUBLE-
                          SMEARS against the adopted eval, which already
                          deposits at 0.005 / 0.5 -- the error map of a
                          filtered inverse therefore carries a smearing the
                          solver arms do not have, and the two are only
                          comparable with that stated.
        ``kind: solve``   FISTA from ``q0 = 0``; ``alpha`` (scalar or ladder
                          list), ``positivity``, ``support``, ``iters``.
    cut_ke : float
        The downstream threshold, default 0.5 ke (= 500 e).
    npz : str, optional
        Where the projection arrays go.  The JSON at ``out`` carries the
        scalars, the axis origins and the recipe.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("display.residual",)

    def execute(self, store):
        op = store.get("op")
        cut = float(self.props.get("cut_ke", 0.5))
        sig_p = float(self.props.get("sigma_pixel", 0.5))
        sig_t = float(self.props.get("sigma_time", 0.005))
        qg = grid_truth(store, op)
        T_grid = float(qg.sum())

        blobs: dict[str, np.ndarray] = {}
        arms, origin = [], None
        for spec in self.props.get("arms", []):
            label = spec.get("label", spec.get("kind", "?"))
            q = self._reconstruct(store, op, spec)
            tru, reco, origin = universal_blocks(store, op, q, sig_p, sig_t)
            T = float(tru.sum())
            tag = str(spec.get("tag", label)).replace(" ", "_")

            if "truth_yt" not in blobs:
                for k, v in _projections(tru / T).items():
                    blobs[f"truth_{k}"] = v.astype(np.float32)

            rec = {"label": label, "tag": tag, "spec": dict(spec),
                   "sum_q_grid": float(q.sum()),
                   "sum_q_pos": float(q[q > 0].sum()),
                   "sum_q_neg": float(q[q < 0].sum()),
                   "L": loss(op, q), "voxels": voxel_stats(q, cut=cut),
                   "truth_universal": T, "states": {}}
            for state, r in (("nocut", reco),
                             ("cut", np.where(reco > cut, reco, 0.0))):
                err = np.abs(tru - r) / T
                m = dict(metrics_from_blocks(tru, r, corr_threshold=cut))
                m["rel_l1"] = float(err.sum())
                m["rel_l1_pos"] = float(np.clip(r - tru, 0, None).sum() / T)
                m["rel_l1_neg"] = float(np.clip(tru - r, 0, None).sum() / T)
                m["sum_reco_universal"] = float(r.sum())
                rec["states"][state] = m
                for k, v in _projections(err).items():
                    blobs[f"err_{tag}_{state}_{k}"] = v.astype(np.float32)
                for k, v in _projections(r / T).items():
                    blobs[f"reco_{tag}_{state}_{k}"] = v.astype(np.float32)
                print(f"[{self.name}] {label:34s} {state:5s}  "
                      f"rel L1 {m['rel_l1']:.4f}  (+{m['rel_l1_pos']:.4f} / "
                      f"-{m['rel_l1_neg']:.4f})  int% {m['integral_pct']:+7.2f} "
                      f" r {m['pearson_r']:+.4f}  slope {m['slope']:+.4f}  "
                      f"ghostQ {m['ghost_charge']:8.1f}  killed "
                      f"{m['true_killed']:8.1f}")
            arms.append(rec)
            del q, tru, reco
            torch.cuda.empty_cache()

        npz = self.props.get("npz")
        if npz:
            np.savez_compressed(npz, **blobs,
                                origin_u_min=np.array(origin["u_min"]),
                                origin_p_min=np.array(origin["p_min"]),
                                origin_bin_ticks=np.array(
                                    origin["bin_ticks"]),
                                tags=np.array([a["tag"] for a in arms]),
                                labels=np.array([a["label"] for a in arms]))
            print(f"[{self.name}] wrote {npz}")
        self._emit(store, {"truth_on_grid": T_grid, "L_qtruth": loss(op, qg),
                           "cut_ke": cut, "sigma_pixel": sig_p,
                           "sigma_time": sig_t, "npz": npz,
                           "origin": origin, "arms": arms})

    # ------------------------------------------------------------------
    def _reconstruct(self, store, op, spec) -> np.ndarray:
        kind = str(spec.get("kind", "solve"))
        if kind == "solve":
            supp = resolve_support(store, op, spec.get("support"),
                                   spec.get("gain_cut"))
            return solve_arm(op, supp, spec.get("alpha", 0.0),
                             bool(spec.get("positivity", True)),
                             int(spec.get("iters", 3000)),
                             float(spec.get("seed_cut", 0.5)),
                             float(spec.get("soft_len", 2.0)))
        if kind != "fft":
            raise ValueError(f"unknown arm kind {kind!r} (want 'fft'/'solve')")
        B = fit_bin_ticks(store)
        blk = op.to_tensor(block_from_rows(op))
        bs = tuple(blk.shape)
        prep = self.services["detector"].prepared(int(round(B)))
        kern = torch.as_tensor(prep.integrated_response, dtype=op.dtype,
                               device=op.device)
        st_ = spec.get("sigma_time")
        filt = None
        if st_ is not None:
            sp = float(spec.get("sigma_pixel", 0.5))
            filt = gaussian_filter_3d_torch(
                (bs[0] + kern.shape[0] - 1, bs[1] + kern.shape[1] - 1, bs[2]),
                dt=(1, 1, B), sigma=(sp, sp, float(st_)),
                device=op.device, dtype=op.dtype)
        q = deconv_fft_torch(blk, kern, filt).detach().cpu().numpy()
        return q.astype(np.float64)[:, :, :op.q_shape[2]]


@algorithm("TransportVisibility")
class TransportVisibility(_JsonRecorder):
    """Is ``reco - truth`` in the NULL SPACE of ``A``, or does the data see it?

    A difference the data cannot see satisfies ``A d = 0``; one the data can
    see does not.  The discriminating number is the Rayleigh quotient of the
    Gram, ``rq = ||A d||^2 / ||d||^2``, read against two references measured
    on the same operator: random vectors on the same voxels (what a generic,
    fully visible direction scores) and the truth itself.  ``rq`` near zero
    means invisible; ``rq`` of order the random baseline means the operator
    resolves that difference perfectly well and something other than
    degeneracy put it there.

    Also reports, per arm, ``L`` against ``L(q_truth)``.  ``L(q_truth) != 0``
    is the whole point: the truth is NOT a solution of ``A q = d``, so for
    any arm that fits the data, ``A(q - q_truth) = -(A q_truth - d) != 0``
    identically -- so the difference has a nonzero component orthogonal to
    ``N(A)``, which no choice within the solution set can remove.  NOTE this
    Rayleigh quotient proves only that a difference is not ENTIRELY in
    ``N(A)``; for the component itself use :class:`NullSplit`.

    Props: ``arms`` (as :class:`ResidualDisplay`), ``n_random`` (default 8),
    ``out``.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("display.visibility",)

    def execute(self, store):
        op = store.get("op")
        qg = grid_truth(store, op)
        L_truth = loss(op, qg)
        gen = np.random.default_rng(0)

        def rayleigh(v: np.ndarray) -> dict:
            t = op.to_tensor(v)
            n2 = float((t ** 2).sum())
            a2 = float((op.forward(t).detach() ** 2).sum())
            return {"l2": float(np.sqrt(n2)), "l2_A": float(np.sqrt(a2)),
                    "rq": a2 / max(n2, 1e-30)}

        supp = np.asarray(resolve_support(store, op, "gain:0.5"))
        rnd = []
        for _ in range(int(self.props.get("n_random", 8))):
            v = gen.standard_normal(op.q_shape) * supp
            rnd.append(rayleigh(v)["rq"])
        # a deliberately constructed PURE TRANSPORT: take the truth's own
        # charge out of its peak time bin and put it two bins either side.
        # Charge-conserving, spatially identical -- exactly the move the
        # figures show -- so its rq is the scale a transport of this shape
        # would have to beat to be invisible.
        tp = qg.sum(axis=(0, 1))
        k = int(np.argmax(tp))
        mv = np.zeros_like(qg)
        mv[:, :, k] = -qg[:, :, k]
        mv[:, :, k - 2] = 0.5 * qg[:, :, k]
        mv[:, :, k + 2] = 0.5 * qg[:, :, k]

        # POSITIVE CONTROL for the probe itself: the block always extends one
        # kernel length before the first recorded window, so the leading
        # kt-1 time bins have c_v = 0 BY CONSTRUCTION (METHODS.md sec 2).
        # A vector living only there is a genuine null direction; if the
        # probe cannot score it ~0, the probe is measuring nothing.
        cv = op.measurement_gain().cpu().numpy()
        dead = np.abs(cv) < 1e-12
        nul = gen.standard_normal(op.q_shape) * dead

        refs = {"truth": rayleigh(qg), "pure_transport_2bins": rayleigh(mv),
                "null_control": {**rayleigh(nul),
                                 "n_voxels": int(dead.sum())},
                "random_on_support": {
                    "rq_mean": float(np.mean(rnd)),
                    "rq_min": float(np.min(rnd)),
                    "rq_max": float(np.max(rnd)),
                    "n": len(rnd)}}
        print(f"[{self.name}] L(q_truth) {L_truth:.4g}   "
              f"rq truth {refs['truth']['rq']:.4g}   "
              f"rq pure transport {refs['pure_transport_2bins']['rq']:.4g}   "
              f"rq random {refs['random_on_support']['rq_mean']:.4g}   "
              f"rq NULL CONTROL {refs['null_control']['rq']:.4g} "
              f"({refs['null_control']['n_voxels']} voxels)")

        arms = []
        for spec in self.props.get("arms", []):
            q = ResidualDisplay._reconstruct(self, store, op, spec)
            r = rayleigh(q - qg)
            rec = {"label": spec.get("label", "?"), "spec": dict(spec),
                   "L": loss(op, q), "L_over_L_truth": loss(op, q) / L_truth,
                   "sum_q": float(q.sum()), "delta": r,
                   "rq_over_random": r["rq"] / refs["random_on_support"]["rq_mean"]}
            print(f"[{self.name}] {rec['label']:34s} L {rec['L']:11.4g} "
                  f"({rec['L_over_L_truth']:8.2e} x L_truth)  "
                  f"||d|| {r['l2']:9.2f}  ||A d|| {r['l2_A']:9.3f}  "
                  f"rq {r['rq']:.4g}  = {rec['rq_over_random']:.3f} x random")
            arms.append(rec)
            del q
            torch.cuda.empty_cache()
        self._emit(store, {"L_qtruth": L_truth, "references": refs,
                           "arms": arms})


def cgls(op, b, n_iter: int, checkpoints=()):
    """Minimum-norm solution of ``A x = b`` by CGLS from ``x0 = 0``.

    Every iterate lives in ``span{A^T b, (A^T A) A^T b, ...} = Row(A)``, so
    the limit is ``A^+ b`` -- and with ``b = A v`` that limit is exactly the
    ROW-SPACE PROJECTION of ``v``.  ``v - x`` is then the null-space
    component, which is the only honest way to ask whether a difference has
    one: ``||A v||`` alone cannot tell a vector with a large null component
    from one with none.

    The split is threshold-dependent and the trajectory says so: components
    along small singular values are recovered late, so ``||x||`` keeps
    growing after the data residual has stopped moving (CGLS semi-
    convergence).  ``checkpoints`` records the trajectory; read the null
    fraction together with ``resid``, never alone.
    """
    x = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
    r = b.clone()
    s = op.adjoint(r)
    p = s.clone()
    g = float((s ** 2).sum())
    nb = float(torch.linalg.vector_norm(b))
    traj = []
    for k in range(1, n_iter + 1):
        t = op.forward(p)
        tt = float((t ** 2).sum())
        if tt <= 0 or g <= 0:
            break
        a = g / tt
        x += a * p
        r -= a * t
        s = op.adjoint(r)
        g2 = float((s ** 2).sum())
        p = s + (g2 / g) * p
        g = g2
        if k in checkpoints or k == n_iter:
            traj.append({"iter": k,
                         "resid": float(torch.linalg.vector_norm(r)) / nb,
                         "x_norm": float(torch.linalg.vector_norm(x))})
    return x, traj


@algorithm("NullSplit")
class NullSplit(_JsonRecorder):
    """Split each difference into its NULL-SPACE and ROW-SPACE components.

    ``TransportVisibility`` reports ``||A v||^2/||v||^2``, which proves only
    that ``v`` is not ENTIRELY null.  This one does the projection:
    ``v_row = A^+(A v)`` by :func:`cgls`, ``v_null = v - v_row``, and
    reports the null-space fraction ``nu(v) = ||v_null|| / ||v||``.

    The reference for a difference is a SOLUTION, not the truth: the truth
    is not one (``A q_truth != d``).  So every arm is differenced BOTH ways
    -- against ``q_truth`` and against the exact-fit unfiltered inverse --
    and the second is the one where "null-space component" is the right
    question, because two solutions can differ ONLY by a null vector.

    Two controls bracket the method and are run every time: a vector on the
    ``c_v = 0`` voxels (a true null direction -- must score 1.000) and
    ``A^T w`` for random ``w`` (orthogonal to ``N(A)`` -- must score 0.000).

    Props: ``arms`` (as :class:`ResidualDisplay`), ``iters`` (default 300),
    ``out``.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("display.nullsplit",)

    def execute(self, store):
        op = store.get("op")
        nit = int(self.props.get("iters", 300))
        cps = tuple(int(c) for c in self.props.get("checkpoints",
                                                   [10, 30, 100, 300]))
        gen = np.random.default_rng(0)
        qg = grid_truth(store, op)

        def split(name, v_np, baseline=False):
            v = op.to_tensor(np.asarray(v_np, dtype=np.float64))
            nv = float(torch.linalg.vector_norm(v))
            b = op.forward(v)
            x, traj = cgls(op, b, nit, cps)
            nn = float(torch.linalg.vector_norm(v - x))
            # The norm is dominated by the many tiny voxels; the TRANSPORT
            # is a few big ones.  So also split the time profile (summed
            # over both pixel axes) -- that is where "did the null part
            # move the charge, or the row part?" is actually readable.
            def tprof(u):
                return u.sum(dim=(0, 1)).detach().cpu().numpy().tolist()
            rec = {"name": name, "norm": nv,
                   "norm_A": float(torch.linalg.vector_norm(b)),
                   "norm_row": float(torch.linalg.vector_norm(x)),
                   "norm_null": nn, "null_frac": nn / max(nv, 1e-30),
                   "charge": float(v.sum()), "charge_row": float(x.sum()),
                   "charge_null": float((v - x).sum()),
                   "tprof": tprof(v), "tprof_row": tprof(x),
                   "tprof_null": tprof(v - x),
                   # transverse (across the line) and longitudinal (along it)
                   "xprof": v.sum(dim=(1, 2)).detach().cpu().numpy().tolist(),
                   "yprof": v.sum(dim=(0, 2)).detach().cpu().numpy().tolist(),
                   "cgls": traj}
            print(f"[{self.name}] {name:38s} ||v|| {nv:9.2f}  ||v_row|| "
                  f"{rec['norm_row']:9.2f}  ||v_null|| {nn:9.2f}   "
                  f"NULL FRACTION {rec['null_frac']:.4f}   "
                  f"charge {rec['charge']:+9.1f} = row {rec['charge_row']:+9.1f} "
                  f"+ null {rec['charge_null']:+8.3f} ke   "
                  f"(cgls resid {traj[-1]['resid']:.2e})")
            del v, b, x
            torch.cuda.empty_cache()
            if baseline:
                # nu is bounded by WHERE the vector lives: a vector confined
                # to a well-observed region cannot have a large one.  So also
                # split a random vector on the SAME occupancy set -- that is
                # the nu a generic vector supported there would score, and
                # the only honest denominator for reading this one.
                occ = np.abs(np.asarray(v_np)) > 1e-9
                r2 = split(f"  [baseline] random on the occupancy of {name}",
                           gen.standard_normal(op.q_shape) * occ)
                rec["baseline_nu"] = r2["null_frac"]
                rec["baseline_n"] = int(occ.sum())
                rec["nu_over_baseline"] = (r2["null_frac"] and
                                           r2["null_frac"] > 0 and
                                           rec["null_frac"] / r2["null_frac"])
            return rec

        out = []
        # --- controls -----------------------------------------------------
        cv = op.measurement_gain().cpu().numpy()
        out.append(split("CONTROL pure null (c_v = 0 voxels)",
                         gen.standard_normal(op.q_shape) * (np.abs(cv) < 1e-12)))
        w = op.to_tensor(gen.standard_normal(op.n_data))
        out.append(split("CONTROL orthogonal to N(A)  (A^T w)",
                         op.adjoint(w).cpu().numpy()))
        del w
        out.append(split("the truth itself", qg))

        qs = {}
        for spec in self.props.get("arms", []):
            tag = str(spec.get("tag", spec.get("label", "?")))
            qs[tag] = ResidualDisplay._reconstruct(self, store, op, spec)
            out.append(split(f"{tag} - truth", qs[tag] - qg,
                             baseline=bool(self.props.get("occupancy_baseline"))))
        if self.props.get("split_arms"):
            # the ESTIMATE itself, not a difference: how much null-space
            # content did each estimator CHOOSE to put in?  An exact-fit arm
            # with ||P_null q|| > 0 is not the minimum-norm solution, so the
            # solution set is being sampled, not just hit.
            for tag, q in qs.items():
                out.append(split(f"{tag} itself", q))
        ref = self.props.get("solution_ref")
        if ref and ref in qs:
            for tag, q in qs.items():
                if tag != ref:
                    out.append(split(f"{tag} - {ref}  (solution ref)", q - qs[ref],
                                     baseline=bool(self.props.get("occupancy_baseline"))))
            out.append(split(f"truth - {ref}  (solution ref)", qg - qs[ref]))
        self._emit(store, {"iters": nit, "L_qtruth": loss(op, qg),
                           "L_arms": {t: loss(op, op.to_tensor(q))
                                      for t, q in qs.items()},
                           "vectors": out})


def cgls_masked(op, b, mask, n_iter: int, x0=None):
    """Least-squares over vectors SUPPORTED ON ``mask``: min ||A x - b||.

    Identical to :func:`cgls` except the adjoint is masked at every step, so
    every iterate stays in ``span(mask)``.  With ``b = d`` this solves the
    restricted normal equations; with ``b = A v`` it gives the projection of
    ``v`` onto ``span(mask)`` along ``N(A|_mask)^perp`` when that null space
    is trivial.
    """
    x = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device) if x0 is None \
        else x0.clone()
    r = b - op.forward(x)
    s = op.adjoint(r) * mask
    p = s.clone()
    g = float((s ** 2).sum())
    traj = []
    for k in range(1, n_iter + 1):
        t = op.forward(p)
        tt = float((t ** 2).sum())
        if tt <= 0 or g <= 0:
            break
        a = g / tt
        x += a * p
        r -= a * t
        s = op.adjoint(r) * mask
        g2 = float((s ** 2).sum())
        p = s + (g2 / g) * p
        g = g2
        if k % max(n_iter // 4, 1) == 0:
            traj.append({"iter": k, "resid": 0.5 * float((r ** 2).sum())})
    return x, traj


def smallest_singular(op, mask, n_iter: int, lam_max: float = 0.94, seed: int = 0):
    """Upper bound on ``sigma_min`` of ``A`` restricted to ``mask``.

    Shifted power iteration ``w <- (lam_max I - A^T A) w`` with the mask
    applied each step, so the iterate converges to the smallest eigenvector
    of the RESTRICTED Gram.  Monotonically decreasing: report it as a bound
    and quote the trajectory, since it need not have converged.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = op.to_tensor(torch.randn(op.q_shape, generator=g,
                                 dtype=torch.float64).numpy()) * mask
    w = w / torch.linalg.vector_norm(w)
    traj = []
    for k in range(1, n_iter + 1):
        w = lam_max * w - op.adjoint(op.forward(w)) * mask
        w = w / torch.linalg.vector_norm(w)
        if k % max(n_iter // 4, 1) == 0:
            traj.append({"iter": k,
                         "sigma": float(torch.linalg.vector_norm(op.forward(w)))})
    return traj[-1]["sigma"], traj, w


@algorithm("RestrictedColumnAnalysis")
class RestrictedColumnAnalysis(_JsonRecorder):
    """The inverse problem on the PHYSICAL unknowns only.

    The shipped operator shares one ``(nx, ny)`` pixel grid between the
    convolution's input and its output.  A linear convolution needs the
    output larger than the input, and ``FFTWarmStart``'s ``pad_pixels: 12``
    ring exists to hold the induced signal -- but because the grid is shared,
    that ring also becomes UNKNOWNS.  On this event 4536 of the 8036 pixel
    columns carry no data, and they inflate the unknown count by 2.296x for
    no physical reason.  Every null-space statement made on the full grid is
    therefore a statement about the implementation, not about the
    measurement.

    This algorithm restricts the columns to what the readout can actually
    determine and re-does the analysis there:

    ``pixels``  the pixel columns that carry data;
    ``time``    the charge bins whose whole kernel footprint lies inside the
                recorded window, i.e. ``k >= t_first`` and
                ``k + kt - 1 <= t_last`` -- bins outside are only partially
                observed, which is a real property of a finite record and
                not an artefact.

    It then reports (a) ``sigma_min`` of the restricted operator, which
    decides whether any null space survives at all, (b) the restricted
    least-squares solution ``q_C``, which is the linear inverse of the
    PHYSICAL problem, and (c) for every arm, how much of it lies outside the
    restricted set and how visible its departure from ``q_C`` is.

    Props: ``arms`` (as :class:`ResidualDisplay`), ``sigma_iters`` (2000),
    ``cgls_iters`` (400), ``out``.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("display.restricted",)

    def execute(self, store):
        op = store.get("op")
        kt = self.services["detector"].prepared(
            int(round(fit_bin_ticks(store)))).integrated_response.shape[2]
        cols = op._cols.cpu().numpy()
        blk = np.zeros(int(np.prod(op.block_shape)), bool)
        blk[cols] = True
        blk = blk.reshape(op.block_shape)
        pix = blk.any(axis=2)
        rt = np.nonzero(blk.any(axis=(0, 1)))[0]
        # The time cut is MEASURED, never geometric.  A geometric rule
        # ("kernel footprint entirely inside the record", k >= rt.min()) is
        # wrong and is wrong by a depth-dependent amount: this response is
        # BACK-LOADED (peak at tap 126 of 128), so its first ~126 taps carry
        # almost nothing.  The readout truncates before t0 and that costs
        # nearly nothing for a charge bin whose response has not started;
        # at the far end the record simply runs past the response and
        # records the zero tail, which is harmless.  On this event the
        # geometric rule discarded 126 charge bins whose recorded response
        # was at FULL norm.  Tying the cut to a block index would also break
        # at any other depth, where t0 and the block origin move.
        #
        # So: probe ||A e_k|| for every k at one interior recorded pixel and
        # keep the bins whose recorded response reaches col_ratio_cut of the
        # maximum.  Self-calibrating, and the profile is archived.
        cut = float(self.props.get("col_ratio_cut", 0.99))
        py, px_ = np.nonzero(pix)
        ci = int(np.argmin((py - py.mean()) ** 2 + (px_ - px_.mean()) ** 2))
        probe = (int(py[ci]), int(px_[ci]))
        prof = np.zeros(op.q_shape[2])
        for k in range(op.q_shape[2]):
            e = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
            e[probe[0], probe[1], k] = 1.0
            prof[k] = float(torch.linalg.vector_norm(op.forward(e)))
        ratio = prof / prof.max()
        keep = np.nonzero(ratio >= cut)[0]
        k0, k1 = int(keep.min()), int(keep.max())
        rec_prof = {"probe_pixel": list(probe), "col_ratio_cut": cut,
                    "col_norm_max": float(prof.max()),
                    "col_norm_argmax": int(np.argmax(prof)),
                    "kept_contiguous": bool(len(keep) == k1 - k0 + 1),
                    "n_kept_bins": int(len(keep)),
                    "ratio": [float(x) for x in ratio]}
        C = np.zeros(op.q_shape, bool)
        C[pix, k0:k1 + 1] = True
        Ct = op.to_tensor(C.astype(np.float64))
        qg = grid_truth(store, op)
        rec = {"n_columns": int(C.sum()), "n_rows": int(op.n_data),
               "cols_over_rows": float(C.sum() / op.n_data),
               "pixel_columns_with_data": int(pix.sum()),
               "pixel_columns_in_block": int(pix.size),
               "time_bins_kept": [k0, k1],
               "column_norm_profile": rec_prof,
               "kernel_time_bins": int(kt),
               # POST-HOC DIAGNOSTIC ONLY.  The mask is built from op._cols
               # and the kernel length -- no truth enters it.  This number
               # says whether the event happened to lie inside; it is not
               # available to an analysis and must never be used to justify
               # the mask.  The truth-free check is L(q_C) below: if charge
               # sat outside C, the restricted fit could not reproduce d.
               "truth_inside_fraction_POSTHOC": float(qg[C].sum() / qg.sum()),
               "L_qtruth": loss(op, qg)}
        print(f"[{self.name}] restricted columns {rec['n_columns']:,} of "
              f"{qg.size:,}   rows {op.n_data:,}   n/rows "
              f"{rec['cols_over_rows']:.4f}")
        print(f"[{self.name}] pixel columns with data {pix.sum()} of {pix.size}"
              f";  charge bins {k0}..{k1} of {op.q_shape[2]} "
              f"(||A e_k|| >= {cut:g} of max, measured at pixel {probe})"
              f";  [post-hoc only] truth inside "
              f"{100*rec['truth_inside_fraction_POSTHOC']:.4f}%")

        s_min, s_traj, _ = smallest_singular(
            op, Ct, int(self.props.get("sigma_iters", 2000)))
        rec["sigma_min_bound"] = s_min
        rec["sigma_min_trajectory"] = s_traj
        rec["lambda_min_bound"] = s_min ** 2
        print(f"[{self.name}] sigma_min(A|C) <= {s_min:.5g}  (trajectory "
              + " -> ".join(f"{t['sigma']:.4g}" for t in s_traj) + ")")
        print(f"[{self.name}] N(A|C) is trivial: "
              f"{'YES' if s_min > 1e-4 else 'NOT ESTABLISHED'}")

        nit = int(self.props.get("cgls_iters", 400))
        qC, tr = cgls_masked(op, op.d, Ct, nit)
        qC_np = qC.detach().cpu().numpy()
        rec["L_zero"] = 0.5 * float((op.d ** 2).sum())   # scale for L(q_C)
        rec["qC"] = {"L": loss(op, qC), "sum_q": float(qC_np.sum()),
                     "sum_q_pos": float(qC_np[qC_np > 0].sum()),
                     "sum_q_neg": float(qC_np[qC_np < 0].sum()),
                     "cgls": tr}
        rec["qC"]["universal"] = score_universal(store, op, qC_np)
        rec["qC"]["universal"].pop("transport", None)
        u = rec["qC"]["universal"]
        print(f"[{self.name}] restricted LS solution q_C: L {rec['qC']['L']:.6g}"
              f"  of L(q=0) {rec['L_zero']:.6g}"
              f"  [L(q_truth) {rec['L_qtruth']:.6g}]   sum_q "
              f"{rec['qC']['sum_q']:.1f}  q- {rec['qC']['sum_q_neg']:.1f}"
              f"  int% {u['integral_pct']:+.2f}  r {u['pearson_r']:.4f}"
              f"  slope {u['slope']:.4f}")

        arms = []
        for spec in self.props.get("arms", []):
            tag = str(spec.get("tag", spec.get("label", "?")))
            q = ResidualDisplay._reconstruct(self, store, op, spec)
            out_frac = float(np.linalg.norm(q[~C]) / max(np.linalg.norm(q), 1e-30))
            dq = q * C - qC_np
            nd = float(np.linalg.norm(dq))
            sv = (float(torch.linalg.vector_norm(op.forward(op.to_tensor(dq))))
                  / max(nd, 1e-30))
            a = {"tag": tag, "label": spec.get("label", tag), "L": loss(op, q),
                 "norm_outside_C_frac": out_frac,
                 "charge_outside_C": float(q[~C].sum()),
                 "sum_q": float(q.sum()),
                 "dist_to_qC_restricted": nd,
                 "effective_sigma": sv,
                 "sigma_over_sigma_min": sv / max(s_min, 1e-30)}
            print(f"[{self.name}] {a['label']:22s} L {a['L']:10.4g}  "
                  f"||q outside C||/||q|| {out_frac:.4f}  charge outside C "
                  f"{a['charge_outside_C']:9.1f} ke  ||q|C - q_C|| {nd:8.2f}"
                  f"  its effective sigma {sv:.4f} = {a['sigma_over_sigma_min']:.1f}"
                  f" x sigma_min")
            arms.append(a)
            del q
            torch.cuda.empty_cache()
        rec["arms"] = arms
        self._emit(store, rec)


@algorithm("ResidualSpectrum")
class ResidualSpectrum(_JsonRecorder):
    """Where the reconstruction error sits in the operator's own spectrum.

    The measurement the mode studies never did.  For each arm:

    ``r = q_arm - q_truth`` on the charge grid.  ``act = {q_arm > cut}`` is
    the same mask ``qside_modes.py`` uses.  ``H = P^T A^T A P`` on that mask
    is built EXACTLY, one column per active voxel, and eigendecomposed;
    ``c = V^T r|_act`` is the residual expanded in the eigenbasis, so

        ||r|_act||^2 = sum_i c_i^2          how the error is distributed
        ||A r|_act||^2 = sum_i lam_i c_i^2  how much of it the data sees

    If ``c_i^2`` concentrates at small ``lam_i`` the error lives where the
    measurement is blind and a different prior could have avoided it; if it
    is spread or sits high the error is visible to the data and is not a
    degeneracy at all.

    COMPLETENESS IS REPORTED FIRST AND GATES EVERYTHING.  ``r`` is NOT
    supported on ``act``: the solver zeroes voxels the truth occupies, and
    that part of the error is invisible to this projection.  Measured on the
    isoline, completeness is 1.0000 for the sparse arms and 0.3966 for the
    unfiltered inverse -- for that arm the projection describes 40% of the
    error and the aggregate Rayleigh quotient (also reported, computed on
    the full grid and on the support, where completeness is ~1) is the only
    honest summary.

    Props: ``arms``, ``cut_active`` (0.01), ``max_active`` (6000, above which
    the exact H is skipped and only the aggregates are reported), ``out``.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("display.residual_spectrum",)

    def execute(self, store):
        op = store.get("op")
        cut = float(self.props.get("cut_active", 0.01))
        nmax = int(self.props.get("max_active", 6000))
        qg = grid_truth(store, op)
        hits = np.asarray(store.get("support"))
        gain = np.asarray(resolve_support(store, op, "gain:0.5"))
        out = []
        for spec in self.props.get("arms", []):
            tag = str(spec.get("tag", spec.get("label", "?")))
            q = ResidualDisplay._reconstruct(self, store, op, spec)
            act = q > cut
            r = q - qg
            n2 = float((r ** 2).sum())
            rt = op.to_tensor(r)
            rec = {"tag": tag, "label": spec.get("label", tag),
                   "n_active": int(act.sum()), "cut_active": cut,
                   "r_norm": float(np.sqrt(n2)),
                   "completeness_active": float((r[act] ** 2).sum() / n2),
                   "completeness_hits": float((r[hits] ** 2).sum() / n2),
                   "completeness_gain05": float((r[gain] ** 2).sum() / n2),
                   # aggregate: the lambda-weighted mean of r's spectral
                   # distribution, on the FULL grid -- no mask, no eigenbasis
                   "rayleigh_full": float(
                       (torch.linalg.vector_norm(op.forward(rt)) ** 2 / n2)),
                   }
            print(f"[{self.name}] {rec['label']:24s} ||r|| {rec['r_norm']:8.2f}"
                  f"  n_act {rec['n_active']:6d}  completeness act "
                  f"{rec['completeness_active']:.4f} / hits "
                  f"{rec['completeness_hits']:.4f}  Rayleigh(full) "
                  f"{rec['rayleigh_full']:.5f}")
            if rec["n_active"] <= nmax:
                idx = np.flatnonzero(act.reshape(-1))
                n = idx.size
                H = np.zeros((n, n))
                e = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
                flat = e.reshape(-1)
                ti = torch.as_tensor(idx, device=op.device)
                for cix in range(n):
                    flat.zero_()
                    flat[int(idx[cix])] = 1.0
                    H[:, cix] = op.adjoint(op.forward(e)).reshape(-1)[ti].cpu().numpy()
                H = 0.5 * (H + H.T)
                lam, V = np.linalg.eigh(H)
                lam = np.clip(lam[::-1], 0, None); V = V[:, ::-1]
                cc = (V.T @ r.reshape(-1)[idx]) ** 2
                tot = cc.sum()
                dec = []
                for k in range(10):
                    sl = slice(k * n // 10, (k + 1) * n // 10)
                    dec.append({"decile": k + 1,
                                "lam_median": float(np.median(lam[sl])),
                                "energy_frac": float(cc[sl].sum() / tot),
                                "lam_weighted": float((lam[sl] * cc[sl]).sum())})
                rec["spectrum"] = {
                    "lam_max": float(lam[0]), "lam_min": float(lam[-1]),
                    "cond_sqrt": float(np.sqrt(lam[0] / max(lam[-1], 1e-30))),
                    "energy_in_weakest_decile": dec[-1]["energy_frac"],
                    "energy_in_strongest_decile": dec[0]["energy_frac"],
                    "deciles": dec,
                    "lam": [float(x) for x in lam],
                    "c2": [float(x) for x in cc]}
                print(f"{'':>22s}    lam {lam[-1]:.3g}..{lam[0]:.3g}   energy in"
                      f" weakest decile {dec[-1]['energy_frac']:.4f}"
                      f"   strongest {dec[0]['energy_frac']:.4f}")
            else:
                rec["spectrum"] = None
                print(f"{'':>22s}    exact H skipped: {rec['n_active']} > {nmax}")
            out.append(rec)
            del q, rt
            torch.cuda.empty_cache()
        self._emit(store, {"truth_on_grid": float(qg.sum()),
                           "L_qtruth": loss(op, qg), "arms": out})
