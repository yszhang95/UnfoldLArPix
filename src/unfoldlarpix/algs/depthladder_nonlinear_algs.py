"""Nonlinear arms (positivity, positivity + l1) on the isoline DEPTH LADDER.

The linear campaign ``depthladder_isoline`` measured the depth dependence of
the reconstructed charge for ONE estimator, the closed-form filtered
minimum-norm inverse on the ``c = 5`` (250 ns) cell basis with
``cell_model = uniform`` (``P_0``).  This module runs the two NONLINEAR
estimators of :mod:`~unfoldlarpix.algs.finebasis_nonlinear_algs` on the SAME
setup, at the same nine depths and two lifetimes, so that the question
"does positivity, or positivity + l1, change the depth dependence of the
reconstructed charge and hence the lifetime fit?" is answered on the same
records with the same basis, the same support and the same evaluation.

What is identical to the archived linear campaign
-------------------------------------------------
* the events (``isoline_d*_{1ms,20ms}_dense.npz``, tpc 0, fixed interval) and
  their ``acq_start``, so the records and the block are the same;
* the basis: ``c = 5`` fine ticks, ``cell_model = uniform``;
* the operator: :class:`~unfoldlarpix.algs.finebasis_algs.FineOperator` with
  ``kernel_cut_tick = None`` -- the impact-averaged ``Kbar``, NO kernel
  truncation.  The ``t_0``-truncated variants of
  :mod:`~unfoldlarpix.algs.depthladder_algs` are not run here;
* the evaluation: :class:`~unfoldlarpix.algs.evalharness_algs.EvalHarness`,
  ``sigma_H`` = 1.5 and 2.0 us, ``P_0`` prolongation, ``margin_windows = 40``,
  ``line_pixel_y_range = [5, 131]``, 7-``pixel_y`` segments with a 3-pad end
  trim, and the Chebyshev pad rings ``0 / 1 / 2 / 3-5 / >= 6`` of
  :func:`~unfoldlarpix.algs.depthladder_algs.ring_masks`.

What is new
-----------
The estimators.  Both are the SHIPPED FISTA of
:func:`~unfoldlarpix.algs.finebasis_nonlinear_algs.solve_fine_arm` on
:class:`~unfoldlarpix.algs.finebasis_nonlinear_algs.FineZSOperator`, from
``q0 = 0``, with the ``gain:0.5`` support of
:func:`~unfoldlarpix.algs.finebasis_nonlinear_algs.upsample_support` --
nothing about the solver, the data term or the prox is re-implemented here:

    ``pos_a0``          ``x >= 0`` on the support, ``alpha = 0``
    ``pos_l1_<a>``      ``x >= 0`` on the support plus ``alpha sum_v x_v``

``alpha`` is in ke per cell and does NOT scale with the cell width: the l1
term prices reconstructed CHARGE, and one unit of charge is credited the same
total by the records on any cell basis (measured, ``STUDIES`` Sec. 4.3).

The archived LINEAR result is REUSED, never re-run
--------------------------------------------------
``archived_json`` is the per-event JSON of the linear campaign
(``outputs/depthladder_isoline/event_d*_*.json``).  Its ``truth``,
``records``, ``tau_cut``, ``cell_grid`` and ``representation_term`` blocks and
its ``variants["none"]`` entry -- the minimum-norm inverse with the full
kernel, i.e. least squares on the 5-tick basis -- are copied verbatim into
this record, so that the fit and the figures see the linear and the nonlinear
arms in one file and the linear numbers are bit-for-bit the archived ones.
The copied entry keeps the name ``none``, so
:class:`~unfoldlarpix.algs.depthladder_algs.DepthLadderFit` reads it as
``xhat_none`` exactly as before.  The nonlinear entries carry the SAME
``kernel`` block (the kernel is the same; only the estimator differs).

Quantities reported per event and arm, all literal
--------------------------------------------------
``sum_xhat_ke``                 ``sum_v xhat_v`` over the real pads.
``sum_xhat_over_sum_effq``      that divided by the event's ``sum effq``.
``sum_xhat_line_pads_ke`` / ``sum_xhat_off_line_ke``
    the split over PADS: pads that carry any truth charge (ring 0) against
    all others.  No time selection; this is the raw estimate.
``by_ring``                     per Chebyshev ring: ``sum``, ``sum`` of the
    positive part, ``sum`` of the negative part, and each per pad -- the ring
    ledger of the RAW estimate.  The ``H``-space ledger at each ``sigma_H`` is
    the ``zero_preservation`` block inside ``scores``.
``scores``                      ``E_rel``, conservation, ``zero_preservation``
    and the segment sums at each ``sigma_H``, from
    :func:`~unfoldlarpix.algs.finebasis_algs.score_rows`.
``residual_rel``                ``||A xhat - y|| / ||y||``.
``nnz`` / ``nnz_1e-3``          cells with ``xhat > 0`` / ``> 1e-3`` ke.
``wall_s``                      FISTA wall time for the arm.

:class:`DepthLadderNonlinearFit`
    :class:`~unfoldlarpix.algs.depthladder_algs.DepthLadderFit` with the two
    nonlinear estimates added to its estimate list and the per-event off-line
    charge, ``E_rel`` and segment error carried through for the figures.  The
    fit itself -- ``ln E(d) = a - lambda t_drift(d)``, unweighted, with the
    residual-scatter error -- is the parent's, unchanged.

:class:`DepthLadderNonlinearFigures`
    E1-E4.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..fwk.component import algorithm
from .depthladder_algs import (DRIFT_VELOCITY_CM_PER_US, RING_LABELS,
                               DepthLadderFit, ring_masks)
from .evalharness_algs import EvalHarness, TICK_US
from .exactrows_algs import _Recorder
from .finebasis_algs import (CellGrid, FineOperator, cell_xhat, embed_pads,
                             ieee_style, save, score_rows)
from .finebasis_nonlinear_algs import (FineZSOperator, record_row_mask,
                                       solve_fine_arm, upsample_support)
from .fixedgrid_algs import block_from_rows, fit_bin_ticks, resolve_support

# Okabe-Ito, the colours of this campaign
C_TRUTH = "#000000"          # truth
C_GREY = "#666666"           # controls (the records)
C_LIN = "#D55E00"            # vermillion: LINEAR minimum norm (LS), full kernel
C_POS = "#009E73"            # green:      positivity, alpha = 0
C_POSL1 = "#CC79A7"          # magenta:    positivity + l1

# the blocks of the archived linear per-event JSON that this record reuses
COPIED_BLOCKS = ("event", "basis", "truth", "records", "tau_cut", "cell_grid",
                 "representation_term")


def ring_ledger(xf: np.ndarray, masks: dict, sum_effq: float) -> dict:
    """Per-ring sum, positive part, negative part -- total and per pad.

    ``xf`` is ``(n_pads, n_cells)``; ``masks`` the Chebyshev classes of
    :func:`~unfoldlarpix.algs.depthladder_algs.ring_masks`.
    """
    out = {}
    for k, m in masks.items():
        blk = xf[m]
        n = max(int(m.sum()), 1)
        s = float(blk.sum())
        p = float(blk[blk > 0].sum())
        q = float(blk[blk < 0].sum())
        out[k] = {"n_pads": int(m.sum()), "sum_ke": s, "per_pad_ke": s / n,
                  "sum_pos_ke": p, "sum_neg_ke": q,
                  "pos_per_pad_ke": p / n, "neg_per_pad_ke": q / n,
                  "over_sum_effq": s / sum_effq}
    return out


# ---------------------------------------------------------------------------
@algorithm("DepthLadderNonlinearEvent")
class DepthLadderNonlinearEvent(_Recorder):
    """One depth, one lifetime: the nonlinear arms beside the archived linear one.

    Props
    -----
    depth_cm, tau_ms : float          metadata of this event.
    velocity_cm_per_us : float, default 0.159645.
    cell_ticks : int, default 5.      cell_model : str, default ``uniform``.
    arms : list of ``{label, alpha, iters}``; positivity is always on.
    support : str, default ``"gain:0.5"``.
    sigma_H_us : list, default ``[1.5, 2.0]``.
    prolongation : str, default ``uniform`` (``P_0``).
    archived_json : str
        The linear campaign's per-event JSON, reused (never re-run).
    margin_windows, line_pixel_y_range, segment_pixels, segment_edge_exclude
        as :class:`~unfoldlarpix.algs.evalharness_algs.ResolutionScore`.
    dtype : ``float32`` (default) or ``float64``.
    out_json : str
    """

    reads = ("op", "support", "event", "readout_config", "block_offset",
             "charge_model")
    writes = ("depthladder.nonlinear_event",)

    DEFAULT_ARMS = [{"label": "pos_a0", "alpha": 0.0, "iters": 1000},
                    {"label": "pos_l1_0.01", "alpha": 0.01, "iters": 1000}]

    def execute(self, store):
        op = store.get("op")
        ev = store.get("event")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        B = int(round(fit_bin_ticks(store)))
        dtype = (torch.float64 if str(self.props.get("dtype", "float32"))
                 == "float64" else torch.float32)
        c = int(self.props.get("cell_ticks", 5))
        cell_model = str(self.props.get("cell_model", "uniform"))
        pname = str(self.props.get("prolongation", "uniform"))
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [1.5, 2.0])]
        specs = self.props.get("arms") or self.DEFAULT_ARMS
        supp_spec = str(self.props.get("support", "gain:0.5"))
        margin = int(self.props.get("margin_windows", 40))
        depth = float(self.props.get("depth_cm", float("nan")))
        tau_ms = float(self.props.get("tau_ms", float("nan")))
        v_cm_us = float(self.props.get("velocity_cm_per_us",
                                       DRIFT_VELOCITY_CM_PER_US))
        dev = op.device

        prep = self.services["detector"].prepared(B)
        K_full = np.asarray(prep.full_response, dtype=np.float64)

        # ---- the archived LINEAR result, reused ---------------------------
        arch_path = self.props.get("archived_json")
        with open(arch_path) as fh:
            arch = json.load(fh)
        ar = arch.get("result", arch)
        rec: dict = {k: ar[k] for k in COPIED_BLOCKS if k in ar}
        eq_all = float(ar["truth"]["sum_effq_ke"])
        lin = dict(ar["variants"]["none"])
        lin["estimator"] = "linear, closed-form filtered minimum norm (LS)"
        lin["reused_from"] = str(arch_path)
        rec["archived_linear"] = {
            "json": str(arch_path),
            "algorithm": arch.get("algorithm"),
            "git": (arch.get("job_config") or {}).get("_meta", {}).get("git"),
            "note": ("the linear arm 'none' of this record is copied verbatim "
                     "from the archived depthladder_isoline per-event JSON; it "
                     "is NOT re-run here")}
        rec["event"] = {**rec.get("event", {}), "depth_cm": depth,
                        "tau_ms": tau_ms, "velocity_cm_per_us": v_cm_us,
                        "t_drift_us": depth / v_cm_us,
                        "t_drift_ms": depth / v_cm_us * 1e-3}

        # ---- the operator, the harness and the cell grid --------------------
        t0 = time.time()
        F = FineOperator(K_full, op.block_shape, B, device=dev, dtype=dtype,
                         cell_ticks=c, cell_model=cell_model,
                         kernel_cut_tick=None)
        t_build = time.time() - t0
        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))
        pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
        win_lo = int(H.fine[0]) - pad_ext
        win_hi = int(H.fine[-1]) + 1 + pad_ext
        grid = CellGrid(b, c, F.N)
        m_lo, m_hi = grid.window(win_lo, win_hi)
        masks = ring_masks(H.chebyshev)

        rec["basis"] = {**rec.get("basis", {}),
                        "cell_ticks": c, "cell_model": cell_model,
                        "prolongation": pname, "dtype": str(dtype),
                        "kernel_cut": "none",
                        "B_fine_ticks": B, "block_offset_b": b,
                        "n_cells_per_pad": F.N,
                        "n_unknowns_real_pads": int(H.nx * H.ny * F.N),
                        "stored_cell_window": [m_lo, m_hi],
                        "build_wall_s": t_build}
        print(f"[{self.name}] d = {depth} cm, tau = {tau_ms} ms: operator "
              f"c = {c} ({cell_model}) {F.nxp}x{F.nyp}x{F.N} in {t_build:.1f} s;"
              f" {H.nx * H.ny * F.N / 1e6:.1f} M unknowns; sum effq "
              f"{eq_all:.2f} ke; archived LS sum xhat "
              f"{lin['sum_xhat_ke']:.2f} ke "
              f"({lin['sum_xhat_over_sum_effq']:.5f} of truth)")

        # ---- the support ----------------------------------------------------
        rowm = record_row_mask(op, F, H.nx, H.ny, dev, dtype)
        zop = FineZSOperator(F, torch.zeros((F.nxp, F.nyp, F.M), dtype=dtype,
                                            device=dev), H.nx, H.ny)
        cv = zop.measurement_gain(rowm)
        cv_max = float(cv.max())
        gain_cut = (float(supp_spec.split(":", 1)[1])
                    if supp_spec.startswith("gain:") else 0.0)
        gain_mask = cv > gain_cut * cv_max
        del cv
        torch.cuda.empty_cache()
        base_c = np.asarray(resolve_support(store, op, "hits"))
        base_f = upsample_support(base_c, H.c, H.B, b, F.N, c)
        supp_t = torch.as_tensor(base_f, device=dev) & gain_mask
        del gain_mask
        torch.cuda.empty_cache()
        n_keep = int(supp_t.sum().item())

        jj = grid.index(H.truth_tick)
        inside = (jj >= 0) & (jj < F.N)
        keep_t = np.zeros(len(jj), dtype=bool)
        idx = (torch.as_tensor(H.truth_ix[inside], device=dev),
               torch.as_tensor(H.truth_iy[inside], device=dev),
               torch.as_tensor(jj[inside], device=dev))
        keep_t[inside] = supp_t[idx].cpu().numpy()
        q_excl = float(H.truth_q[~keep_t].sum())
        rec["support"] = {
            "spec": supp_spec, "gain_cut": gain_cut,
            "c_cell_max_ke_per_unit": cv_max,
            "n_cells_total": int(H.nx * H.ny * F.N),
            "n_cells_kept": n_keep,
            "fraction_kept": n_keep / float(H.nx * H.ny * F.N),
            "n_cells_kept_by_hits_only": int(base_f.sum()),
            "truth_charge_excluded_ke": q_excl,
            "truth_charge_excluded_fraction":
                q_excl / max(H.truth_total, 1e-30)}
        print(f"[{self.name}]   support {supp_spec}: {n_keep} of "
              f"{H.nx * H.ny * F.N} cells "
              f"({100 * n_keep / (H.nx * H.ny * F.N):.3f} %); truth charge "
              f"excluded {q_excl:.4g} ke")
        del base_f
        torch.cuda.empty_cache()

        # ---- the data --------------------------------------------------------
        blk = block_from_rows(op)
        y_t = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
        y_norm = float(torch.linalg.vector_norm(y_t))
        zop.d = y_t
        rec["data"] = {"sum_records_ke": float(blk.sum()),
                       "sum_records_over_sum_effq": float(blk.sum() / eq_all),
                       "y_norm": y_norm}

        # ---- the arms --------------------------------------------------------
        rec["variants"] = {"none": lin}
        for spec in specs:
            lab = str(spec["label"])
            alpha = float(spec.get("alpha", 0.0))
            iters = int(spec.get("iters", 1000))
            print(f"[{self.name}]   solving {lab}: positivity, "
                  f"alpha = {alpha:g} ke per cell, {iters} iterations",
                  flush=True)
            xh, hist, wall = solve_fine_arm(zop, supp_t, alpha, iters,
                                            log_every=250, tag=lab)
            resid = float(torch.linalg.vector_norm(zop.forward(xh) - y_t)
                          / y_norm)
            nnz = int((xh > 0).sum())
            nnz3 = int((xh > 1e-3).sum())
            xhn = xh[:, :, m_lo:m_hi].cpu().numpy()
            tot = float(xh.sum())
            pos = float(xh[xh > 0].sum())
            neg = float(xh[xh < 0].sum())
            del xh
            torch.cuda.empty_cache()

            xf = xhn.reshape(H.n_pads, -1)
            by_ring = ring_ledger(xf, masks, eq_all)
            on_line = float(xf[masks["0"]].sum())
            sc = {}
            for s in sigmas:
                m = score_rows(H, cell_xhat(H, grid, xhn, pname, m_lo, m_hi, s,
                                            x_lo=m_lo), s)
                m.pop("_profiles")
                sc[f"s{s:g}"] = {
                    "E_rel": m["E_rel"],
                    "conservation_rel": m["conservation_rel"],
                    "sum_xhat_ke": m["sum_xhat_ke"],
                    "sum_Hx_ke": m["sum_Hx_ke"],
                    "zero_preservation": m["zero_preservation"],
                    "segments": {k: m["segments"][k] for k in
                                 ("n", "rel_error_mean", "rel_error_rms",
                                  "rel_error_max_abs")}}
            ent = {
                "estimator": ("nonlinear, FISTA, positivity"
                              + (f" + l1 alpha = {alpha:g} ke per cell"
                                 if alpha > 0 else " (alpha = 0)")),
                "kernel_cut": "none",
                "kernel": lin["kernel"],
                "alpha_ke_per_cell": alpha, "iters": iters,
                "support": supp_spec,
                "wall_s": wall, "build_wall_s": t_build,
                "iteration_history": hist,
                "sum_xhat_ke": float(xf.sum()),
                "sum_xhat_all_pads_ke": tot,
                "sum_xhat_pos_ke": pos, "sum_xhat_neg_ke": neg,
                "sum_xhat_line_pads_ke": on_line,
                "sum_xhat_off_line_ke": float(xf.sum()) - on_line,
                "off_line_over_sum_effq": (float(xf.sum()) - on_line) / eq_all,
                "sum_xhat_over_sum_effq": float(xf.sum()) / eq_all,
                "by_ring": by_ring,
                "nnz": nnz, "nnz_1e-3": nnz3,
                "residual_rel": resid,
                "scores": sc}
            rec["variants"][lab] = ent
            s15 = sc.get("s1.5", {})
            print(f"[{self.name}]   {lab:14s} sum xhat {xf.sum():9.2f} ke "
                  f"({ent['sum_xhat_over_sum_effq']:.5f} of truth)  off-line "
                  f"{ent['sum_xhat_off_line_ke']:+9.2f} ke  E_rel "
                  f"{s15.get('E_rel', float('nan')):.4f}  |Ax-y|/|y| "
                  f"{resid:.4e}  nnz {nnz}  {wall:.1f} s")

        del supp_t, y_t, rowm
        zop.d = None
        del F, zop
        torch.cuda.empty_cache()
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("DepthLadderNonlinearFit")
class DepthLadderNonlinearFit(DepthLadderFit):
    """The ladder fit with the nonlinear estimates added.

    The fit itself is the parent's, unchanged: unweighted least squares of
    ``ln E(d) = a - lambda t_drift(d)`` over the depths, all depths and
    ``d >= deep_only_min_depth_cm``, per lifetime, with the residual-scatter
    error.  This subclass only

    * extends the estimate list (prop ``estimates_extra``, default the two
      nonlinear arms) so that ``xhat_pos_a0`` and ``xhat_pos_l1_0.01`` are
      fitted beside ``sum_effq``, ``sum_records`` and ``xhat_none``, and
    * carries the per-event off-line charge fraction, ``E_rel`` and segment
      error of every arm into ``per_event`` for the figures.

    Props: those of the parent, plus ``estimates_extra``.
    """

    writes = ("depthladder.nonlinear_fit",)

    DEFAULT_EXTRA = (("xhat_pos_a0", "positivity, alpha = 0"),
                     ("xhat_pos_l1_0.01",
                      "positivity + l1, alpha = 0.01 ke per cell"))

    def execute(self, store):
        extra = self.props.get("estimates_extra")
        extra = (tuple((str(a), str(b)) for a, b in extra) if extra
                 else self.DEFAULT_EXTRA)
        keep = tuple(e for e in DepthLadderFit.ESTIMATES
                     if e[0] in ("sum_effq", "sum_records", "xhat_none"))
        self.ESTIMATES = keep + extra
        super().execute(store)
        rec = self._records[-1]
        # per-event quantities the parent does not carry
        for item in self.props.get("inputs", []):
            with open(item["json"]) as fh:
                r = json.load(fh)
            r = r.get("result", r)
            e = rec["per_event"][str(float(item["tau_ms"]))][
                str(float(item["depth_cm"]))]
            eq = float(r["truth"]["sum_effq_ke"])
            e["off_line_over_sum_effq"] = {
                n: float(v["sum_xhat_off_line_ke"]) / eq
                for n, v in r["variants"].items()}
            e["line_over_sum_effq"] = {
                n: float(v["sum_xhat_line_pads_ke"]) / eq
                for n, v in r["variants"].items()}
            e["by_ring_over_sum_effq"] = {
                n: {k: float(o["sum_ke"]) / eq
                    for k, o in v["by_ring"].items()}
                for n, v in r["variants"].items()}
            e["nnz"] = {n: v.get("nnz") for n, v in r["variants"].items()}
            e["residual_rel"] = {n: v["residual_rel"]
                                 for n, v in r["variants"].items()}
            e["wall_s"] = {n: v.get("wall_s", v.get("solve_wall_s"))
                           for n, v in r["variants"].items()}
        rec["estimates"] = [list(x) for x in self.ESTIMATES]


# ---------------------------------------------------------------------------
@algorithm("DepthLadderNonlinearFigures")
class DepthLadderNonlinearFigures(_Recorder):
    """E1-E4 from the nonlinear ladder JSON.

    Props
    -----
    ladder_json : str        the :class:`DepthLadderNonlinearFit` output.
    figdir : str
    out_json : str
    """

    reads = ()
    writes = ("depthladder.nonlinear_figures",)

    # (fit key, per-event variant key, colour, marker, label)
    ARMS = (("xhat_none", "none", C_LIN, "s", "min-norm, full kernel (LS)"),
            ("xhat_pos_a0", "pos_a0", C_POS, "^", r"positivity, $\alpha=0$"),
            ("xhat_pos_l1_0.01", "pos_l1_0.01", C_POSL1, "o",
             r"positivity + $\ell_1$, $\alpha=0.01$"))

    def execute(self, store):
        with open(self.props["ladder_json"]) as fh:
            self._doc = json.load(fh)["result"]
        self.put(store, self.writes[0], {"pending": True})
        self._recipe = {"job_config": store.get("job.config"),
                        "provenance": store.provenance()}

    def finalize(self):
        plt = ieee_style()
        R = self._doc
        outdir = Path(self.props.get("figdir", "figs"))
        made: list = []
        keys = sorted(R["ratios"], key=lambda s: float(s[:-2]))
        depths = np.array(R["ratios"][keys[0]]["depths_cm"])
        pe = R["per_event"]

        def ev(key):
            k = key[:-2]
            return pe[str(float(k))] if str(float(k)) in pe else pe[k]

        # ---- E1 charge ratio vs depth --------------------------------------
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for ki, key in enumerate(keys):
            fill = ki == 0
            for name, _vn, col, mk, lab in self.ARMS:
                if name not in R["ratios"][key]:
                    continue
                ax.plot(depths, R["ratios"][key][name], mk, ls="-", color=col,
                        ms=3.6, lw=0.9, mfc=(col if fill else "none"), mew=0.9,
                        label=(lab if fill else None))
        ax.axhline(1.0, color=C_TRUTH, lw=0.8, ls="--", label="truth")
        ax.set_xlabel("drift depth [cm]")
        ax.set_ylabel(r"$\Sigma\hat{x}\,/\,\Sigma\,\mathrm{effq}$")
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.set_title("filled: $\\tau$ = %s   open: $\\tau$ = %s"
                     % (keys[0], keys[1] if len(keys) > 1 else "-"),
                     fontsize=7)
        save(fig, outdir, "E1_charge_ratio_vs_depth", made)

        # ---- E2 lambda per estimate ----------------------------------------
        names = [("sum_effq", C_TRUTH, "truth"),
                 ("sum_records", C_GREY, "records")] + \
                [(n, c, l) for n, _v, c, _m, l in self.ARMS]
        names = [t for t in names if t[0] in R["fits"][keys[0]]]
        fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 2.9),
                                squeeze=False)
        for ax, key in zip(axs[0], keys):
            F = R["fits"][key]
            for i, (n, col, _lab) in enumerate(names):
                f = F[n]
                ax.bar(i, f["lambda_per_ms"], 0.6, color=col, alpha=0.75)
                ax.errorbar(i, f["lambda_per_ms"], yerr=f["lambda_err"],
                            fmt="none", ecolor=C_TRUTH, elinewidth=0.8,
                            capsize=2)
                if f.get("deep_only"):
                    d = f["deep_only"]
                    ax.errorbar(i + 0.24, d["lambda_per_ms"],
                                yerr=d["lambda_err"], fmt="o", ms=4.0,
                                mfc="none", mec=col, ecolor=col,
                                elinewidth=0.8, capsize=2)
            lt = 1.0 / float(key[:-2])
            ax.axhline(lt, color=C_TRUTH, lw=0.8, ls="--")
            ax.set_ylim(lt - 1.3, lt + 0.6)
            for i, (n, _c, _l) in enumerate(names):
                f = F[n]
                if f["lambda_per_ms"] > lt + 0.6:
                    ax.annotate(f"{f['lambda_per_ms']:.2f}"
                                f"$\\pm${f['lambda_err']:.2f}",
                                (i, lt + 0.56), ha="center", va="top",
                                fontsize=5.5, rotation=90)
                if f["lambda_per_ms"] < lt - 1.3:
                    ax.annotate(f"{f['lambda_per_ms']:.2f}"
                                f"$\\pm${f['lambda_err']:.2f}",
                                (i, lt - 1.26), ha="center", va="bottom",
                                fontsize=5.5, rotation=90)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels([t[2] for t in names], rotation=35, ha="right",
                               fontsize=5.5)
            ax.set_ylabel(r"$\lambda$ [1/ms]")
            ax.set_title(f"$\\tau$ = {key}  (open: $d \\geq$ "
                         f"{R['deep_only_min_depth_cm']:g} cm)", fontsize=7)
        save(fig, outdir, "E2_lambda_per_estimate", made)

        # ---- E3 off-line charge fraction and segment error vs depth ---------
        e0 = ev(keys[0])
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for _n, vn, col, mk, lab in self.ARMS:
            off = [e0[str(d)]["off_line_over_sum_effq"][vn] for d in depths]
            axs[0].plot(depths, off, mk, ls="-", color=col, ms=3.4, lw=0.9,
                        label=lab)
            mu = np.array([e0[str(d)]["segments"][vn]["s1.5"]["mean"]
                           for d in depths]) * 100
            sd = np.array([e0[str(d)]["segments"][vn]["s1.5"]["rms"]
                           for d in depths]) * 100
            axs[1].errorbar(depths, mu, yerr=sd, fmt=mk + "-", color=col,
                            ms=3.4, lw=0.9, elinewidth=0.8, capsize=2,
                            label=lab)
        for a in axs:
            a.axhline(0.0, color=C_TRUTH, lw=0.6, ls="--")
            a.set_xlabel("drift depth [cm]")
            a.legend(frameon=False, fontsize=6)
        axs[0].set_ylabel(r"$\Sigma\hat{x}_{\rm off-line}\,/\,"
                          r"\Sigma\,\mathrm{effq}$")
        axs[1].set_ylabel("segment-sum relative error [%]")
        axs[0].set_title(f"$\\tau$ = {keys[0]}", fontsize=7)
        axs[1].set_title(r"$\sigma_H = 1.5\,\mu$s, mean $\pm$ rms",
                         fontsize=7)
        save(fig, outdir, "E3_offline_charge_and_segments", made)

        # ---- E4 E_rel vs depth ----------------------------------------------
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for _n, vn, col, mk, lab in self.ARMS:
            e = [e0[str(d)]["E_rel"][vn]["s1.5"] for d in depths]
            ax.plot(depths, e, mk, ls="-", color=col, ms=3.4, lw=0.9,
                    label=lab)
        ax.set_xlabel("drift depth [cm]")
        ax.set_ylabel(r"$E_{\rm rel}$ at $\sigma_H$ = 1.5 $\mu$s")
        ax.legend(frameon=False, fontsize=6)
        ax.set_title(f"$\\tau$ = {keys[0]}", fontsize=7)
        save(fig, outdir, "E4_Erel_vs_depth", made)

        print(f"[{self.name}] {len(made)} figures in {outdir}")
        body = {"figures": made, "ladder_json": self.props["ladder_json"],
                "colours": {"truth": C_TRUTH, "records": C_GREY,
                            "linear_minnorm": C_LIN, "positivity": C_POS,
                            "positivity_l1": C_POSL1},
                "ring_labels": list(RING_LABELS)}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": body,
                           **self._recipe}, fh, indent=1, default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return body
