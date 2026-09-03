"""Depth dependence of the reconstructed charge on the isoline depth ladder.

One estimator setup throughout: the ``c = 5`` (250 ns) cell basis with
``cell_model = uniform`` (``P_0``), the closed-form filtered minimum-norm
inverse ``xhat = A^T (A A^T + lambda I)^{-1} y`` of
:mod:`~unfoldlarpix.algs.finebasis_algs`, ``lambda_rel = 1e-6``.  What varies
is the KERNEL: the full impact-averaged response, or the same response with
its leading part deleted.

The truncation
--------------
tred zeroes every current sample before the event time reference ``t_0``
(``tred/plots/graph_effq.py:148-159``, ``concatenate_waveforms(...,
event_t = global_tref[1] // tspace)``; ``t_0`` = fine tick 0 in every campaign
here).  The fine truth tick ``j`` of a charge is the time it crosses the
RESPONSE PLANE (30.431 cm from the anode, ``tau = 0`` of the kernel), so the
absolute time of response tick ``tau`` is ``j + tau`` and the deletion removes
exactly the kernel ticks ``tau < t_0 - j``.

For a charge CREATED at depth ``d`` at ``t_0``, ``j`` is negative: the charge
would have had to cross the response plane ``|j|`` ticks BEFORE the event
existed.  It did not; that part of the induced current never happened.  So

    tau_cut(d) = t_0 - t_plane = - j = ARRIVAL_TICK - t_drift(d) / Delta_t ,

with ``ARRIVAL_TICK = 3812`` the kernel's own plane-to-anode transit and
``t_drift(d) = d / v``.  This is physics, not an artefact: by Ramo's theorem
a pad ``k`` records the net induced charge of the motion that actually took
place, so a neighbour pad ends at ``-q W_k(d)`` (``W_k`` its weighting
potential at the CREATION point) and the whole ``25 x 25`` array ends at
``q [1 - sum_k W_k(d)]``.  In kernel terms

    sum_{d, tau >= tau_cut} Kbar_d(tau)  =  1 - sum_k W_k(depth d)

up to ``sum Kbar = 1.000266``, so the deficit is exactly the leading kernel
integral that the truncation deletes.  It grows toward the anode (shallow
``d`` = large ``tau_cut``) and is INDEPENDENT of the lifetime: attenuation
scales ``q``, the acceptance is a ratio.

Because one depth gives one ``tau_cut``, the truncation is a fixed kernel
modification and the operator stays shift invariant:

    h_trunc_d(tau) = sum_{m = tau-B+1}^{tau} Kbar_d(m) [m >= tau_cut]
                   = box_B * (Kbar . [tau >= tau_cut]) ,

built by :func:`~unfoldlarpix.algs.finebasis_algs.truncated_response` inside
:class:`~unfoldlarpix.algs.finebasis_algs.FineOperator` through its
``kernel_cut_tick`` argument, and the closed-form inverse applies unchanged.
Its conservation identity becomes

    sum xhat = sum y / sum_{d, tau >= tau_cut} Kbar_d(tau)   (lambda -> 0)

instead of ``sum y / sum Kbar``: the recorded total is divided by what the
array could actually record, not by the full-drift gain.

``kernel_cut`` variants (prop ``variants`` of :class:`DepthLadderEvent`)
    ``none``
        ``tau_cut = None``: the full-drift kernel, the operator of every
        earlier campaign.  This is the DEFAULT everywhere else, so nothing
        archived changes.
    ``truth``
        ``tau_cut = -median(effq tick)``, the charge-weighted median over the
        event's own fine truth.  Uses truth; the main result.
    ``observed_first_above``
        ``tau_cut = ARRIVAL_TICK - t_arrival``, with ``t_arrival`` the MIDPOINT
        of the first record window whose pad-summed charge reaches
        ``arrival_fraction`` (default 0.5) of the largest pad-summed window
        charge: ``t_arrival = b + B w* + (B+1)/2``.  Records only.
    ``observed_shape_fit``
        ``tau_cut = -t_plane``, ``t_plane`` the integer minimising the squared
        difference between the normalised cumulative pad-summed record
        ``C[w]/C[M-1]`` and the kernel's own normalised cumulative
        ``[F(l_w - t_p) - F(-t_p - 1)] / [F(inf) - F(-t_p - 1)]``,
        ``F(tau) = sum_{d, m <= tau} Kbar_d(m)``, ``l_w = b + B(w+1)``.
        Scanned over ``+-scan_ticks`` around the ``first_above`` estimate.
        Records and the response file only; no truth.

Ring definitions (NEW here only as a grouping, the classes are the ones
``exactrows_algs`` and ``evalharness_algs`` already use)
    PADS are grouped by Chebyshev distance to the nearest pad carrying truth
    charge (``EvalHarness.chebyshev``): ``0`` (the line pads), ``1``, ``2``,
    ``3-5``, ``>= 6``.  KERNEL OFFSETS are grouped by Chebyshev norm of the
    pixel offset ``d`` in the ``25 x 25`` array: ``0`` is the own pad.

Quantities reported per event, all literal
    ``sum_effq_ke``            the truth, ``sum`` of the event's ``effq``.
    ``sum_records_ke``         ``sum_p sum_w y_p[w]``, the recorded total.
    ``sum_xhat_ke``            the reconstructed total per kernel variant.
    ``kernel_truncation_fraction``  ``1 - sum h_trunc / sum h_full``, summed
        over the whole ``25 x 25`` array and over ``tau``; what the array
        cannot record for a charge created at that depth.
    ``E_rel``                  as ``evalharness_algs``: ``sum |xhat - Hx| /
        sum Hx`` at ``sigma_H``, with the ``P_0`` representation term
        ``H (P_0 R_c - I) x`` scored the same way as its own arm.
    segment sums               7 ``pixel_y`` (3.1 cm) blocks, 3-pad end trim,
        relative error mean and rms.

:class:`DepthLadderFit`
    Source algorithm over the per-event JSONs: unweighted least squares of
    ``ln E(d) = a - lambda t_drift(d)`` over the nine depths, per lifetime and
    per charge estimate, with the residual-scatter error

        se(lambda) = sqrt( ( sum_i r_i^2 / (n - 2) ) / sum_i (t_i - tbar)^2 ) .

:class:`DepthLadderFigures`
    D1-D5 from the archived JSONs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..fwk.component import algorithm
from .evalharness_algs import EvalHarness, TICK_US
from .exactrows_algs import ARRIVAL_TICK, _Recorder
from .finebasis_algs import (CellGrid, FineOperator, cell_charge_model_taps,
                             direct_sum_records, embed_pads, fine_xhat,
                             ieee_style, prolong_truth_to_fine, save,
                             score_rows)
from .fixedgrid_algs import block_from_rows, fit_bin_ticks

DRIFT_VELOCITY_CM_PER_US = 0.159645     # the ladder generator's own velocity

# Okabe-Ito, consistent with the earlier campaigns
C_TRUTH = "#000000"          # truth
C_GREY = "#666666"           # controls (the records, the METHODS Sec. 10 points)
C_FULL = "#D55E00"           # vermillion: minimum-norm, FULL kernel
C_TRUNC = "#0072B2"          # blue:      minimum-norm, TRUNCATED kernel (truth)
C_TRUNC_OBS = "#56B4E9"      # sky blue:  minimum-norm, TRUNCATED (observed)

RING_LABELS = ("0", "1", "2", "3-5", ">=6")


def ring_masks(cheb: np.ndarray) -> dict:
    """The five Chebyshev-distance classes as boolean masks."""
    c = np.asarray(cheb)
    return {"0": c == 0, "1": c == 1, "2": c == 2,
            "3-5": (c >= 3) & (c <= 5), ">=6": c >= 6}


def kernel_offset_rings(kx: int, ky: int) -> dict:
    """The same five classes on the ``25 x 25`` kernel offset grid."""
    ax = np.abs(np.arange(kx) - (kx - 1) // 2)
    ay = np.abs(np.arange(ky) - (ky - 1) // 2)
    return ring_masks(np.maximum(ax[:, None], ay[None, :]))


def charge_weighted_median(tick: np.ndarray, q: np.ndarray) -> float:
    """The 50 % point of the charge distribution over fine ticks."""
    t = np.asarray(tick, dtype=np.int64)
    w = np.asarray(q, dtype=float)
    o = np.argsort(t)
    t, w = t[o], w[o]
    cw = np.cumsum(w)
    return float(t[int(np.searchsorted(cw, 0.5 * cw[-1]))])


def arrival_first_above(Y: np.ndarray, b: int, B: int,
                        fraction: float) -> dict:
    """First record window at or above ``fraction`` of the largest one.

    ``Y[w]`` is the pad-summed window charge.  Window ``w`` covers the fine
    ticks ``(b + Bw, b + B(w+1)]``, so its midpoint is ``b + Bw + (B+1)/2``,
    which is the arrival-time estimate; the estimator's own resolution is the
    window, ``+- B/2`` fine ticks.
    """
    Y = np.asarray(Y, dtype=float)
    thr = float(fraction) * float(Y.max())
    w = int(np.argmax(Y >= thr))
    t_arr = float(b) + B * w + (B + 1) / 2.0
    return {"window": w, "threshold_ke": thr, "max_window_ke": float(Y.max()),
            "latch_tick": int(b + B * (w + 1)),
            "t_arrival_tick": t_arr,
            "t_plane_tick": t_arr - ARRIVAL_TICK,
            "tau_cut": int(round(ARRIVAL_TICK - t_arr)),
            "resolution_ticks": B / 2.0}


def arrival_shape_fit(Y: np.ndarray, K: np.ndarray, b: int, B: int,
                      t_plane_start: float, scan_ticks: int) -> dict:
    """Fit the plane-crossing tick to the shape of the cumulative record.

    ``F(tau) = sum_{d} sum_{m <= tau} Kbar_d(m)`` is the kernel's own
    pad-summed cumulative.  Under the truncation the pad-summed cumulative
    record of an isochronous event at plane tick ``t_p`` is

        C(l) = q [ F(l - t_p) - F(-t_p - 1) ] ,   l = b + B(w+1) ,

    so the NORMALISED cumulative ``C(l)/C(l_last)`` depends on ``t_p`` alone.
    The estimate is the integer ``t_p`` minimising the sum of squared
    differences to the measured normalised cumulative.  Truth is not used:
    only the records and the response file.
    """
    Y = np.asarray(Y, dtype=float)
    C = np.cumsum(Y)
    if C[-1] <= 0:
        raise ValueError("cumulative record is not positive")
    Cn = C / C[-1]
    F = np.cumsum(np.asarray(K, dtype=np.float64).sum(axis=(0, 1)))
    nt = len(F)

    def Fv(tau):
        t = np.clip(np.asarray(tau), -1, nt - 1)
        return np.where(np.asarray(tau) < 0, 0.0, F[np.maximum(t, 0)])

    latch = b + B * (np.arange(len(Y)) + 1)
    best = None
    cands = np.arange(int(round(t_plane_start)) - int(scan_ticks),
                      int(round(t_plane_start)) + int(scan_ticks) + 1)
    for tp in cands:
        base = Fv(-tp - 1)
        top = F[-1] - base
        if top <= 0:
            continue
        m = (Fv(latch - tp) - base) / top
        ss = float(((m - Cn) ** 2).sum())
        if best is None or ss < best[1]:
            best = (int(tp), ss)
    if best is None:
        raise ValueError("no admissible plane tick in the scan range")
    return {"t_plane_tick": best[0], "sum_squares": best[1],
            "t_arrival_tick": best[0] + ARRIVAL_TICK,
            "tau_cut": int(-best[0]),
            "scan_ticks": int(scan_ticks),
            "scan_center": int(round(t_plane_start))}


def ls_line(t: np.ndarray, y: np.ndarray) -> dict:
    """``y = a - lambda t`` by unweighted least squares, residual-scatter error.

    ``se(lambda) = sqrt( (sum r^2 / (n-2)) / sum (t - tbar)^2 )`` -- the
    textbook standard error of a straight-line slope with the variance
    estimated from the fit's own residuals, which is the only estimate
    available on a fluctuation-free sample (see ``METHODS.md`` Sec. 10).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    tb = t.mean()
    Stt = float(((t - tb) ** 2).sum())
    slope = float(((t - tb) * (y - y.mean())).sum() / Stt)
    inter = float(y.mean() - slope * tb)
    r = y - (inter + slope * t)
    s2 = float((r ** 2).sum() / max(n - 2, 1))
    return {"lambda_per_ms": -slope, "lambda_err": float(np.sqrt(s2 / Stt)),
            "intercept": inter, "q0_ke": float(np.exp(inter)),
            "n_depths": int(n), "rms_resid_lnE": float(np.sqrt(s2)),
            "residual_lnE": [float(v) for v in r],
            "t_drift_ms": [float(v) for v in t],
            "lnE": [float(v) for v in y]}


# ---------------------------------------------------------------------------
@algorithm("DepthLadderEvent")
class DepthLadderEvent(_Recorder):
    """One depth, one lifetime: charge, deficit by ring, and both kernels.

    Props
    -----
    depth_cm, tau_ms : float          metadata of this event (not used to cut).
    cell_ticks : int, default 5.      cell_model : str, default ``uniform``.
    lambda_rel : float, default 1e-6.
    sigma_H_us : list, default ``[1.5, 2.0]``.
    variants : list of str, default
        ``[none, truth, observed_first_above, observed_shape_fit]``.
    arrival_fraction : float, default 0.5   (``observed_first_above``).
    shape_scan_ticks : int, default 900     (``observed_shape_fit``).
    velocity_cm_per_us : float, default 0.159645.
    margin_windows, line_pixel_y_range, segment_pixels, segment_edge_exclude
        as :class:`~unfoldlarpix.algs.evalharness_algs.ResolutionScore`.
    dtype : ``float64`` (default) or ``float32``.
    out_json : str
    """

    reads = ("op", "event", "readout_config", "block_offset", "charge_model")
    writes = ("depthladder.event",)

    def execute(self, store):
        op = store.get("op")
        ev = store.get("event")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        B = int(round(fit_bin_ticks(store)))
        dtype = (torch.float64 if str(self.props.get("dtype", "float64"))
                 == "float64" else torch.float32)
        tol_fwd = 1e-6 if dtype == torch.float64 else 1e-4
        tol_lin = 1e-5 if dtype == torch.float64 else 1e-3
        c = int(self.props.get("cell_ticks", 5))
        cell_model = str(self.props.get("cell_model", "uniform"))
        lam_rel = float(self.props.get("lambda_rel", 1e-6))
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [1.5, 2.0])]
        variants = [str(v) for v in self.props.get(
            "variants", ["none", "truth", "observed_first_above",
                         "observed_shape_fit"])]
        frac = float(self.props.get("arrival_fraction", 0.5))
        scan = int(self.props.get("shape_scan_ticks", 900))
        v_cm_us = float(self.props.get("velocity_cm_per_us",
                                       DRIFT_VELOCITY_CM_PER_US))
        depth = float(self.props.get("depth_cm", float("nan")))
        tau_ms = float(self.props.get("tau_ms", float("nan")))
        margin = int(self.props.get("margin_windows", 40))
        dev = op.device

        prep = self.services["detector"].prepared(B)
        K_full = np.asarray(prep.full_response, dtype=np.float64)
        nt_resp = K_full.shape[-1]

        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))
        pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
        win_lo = int(H.fine[0]) - pad_ext
        win_hi = int(H.fine[-1]) + 1 + pad_ext

        # ---- truth ---------------------------------------------------------
        eq_all = float(np.asarray(ev.effq.data, dtype=float)[:, -1].sum())
        t_med = charge_weighted_median(H.truth_tick, H.truth_q)
        t_mean = float((H.truth_tick * H.truth_q).sum() / H.truth_q.sum())
        t_drift_us = depth / v_cm_us
        t_drift_ticks = t_drift_us / TICK_US

        rec: dict = {
            "event": {"depth_cm": depth, "tau_ms": tau_ms,
                      "velocity_cm_per_us": v_cm_us,
                      "t_drift_us": t_drift_us,
                      "t_drift_ticks": t_drift_ticks,
                      "t_drift_ms": t_drift_us * 1e-3},
            "basis": {"cell_ticks": c, "cell_model": cell_model,
                      "lambda_rel": lam_rel, "B_fine_ticks": B,
                      "dtype": str(dtype), "block_offset_b": b,
                      "block_shape": [int(v) for v in op.block_shape]},
            "truth": {
                "sum_effq_ke": eq_all,
                "sum_effq_in_block_ke": float(H.truth_total),
                "effq_outside_block_ke": float(H.truth_dropped),
                "n_cells": int(len(H.truth_tick)),
                "n_pads_with_truth": int(len(H.truth_pad_rows)),
                "tick_min": int(H.truth_tick.min()),
                "tick_max": int(H.truth_tick.max()),
                "tick_charge_weighted_median": t_med,
                "tick_charge_weighted_mean": t_mean},
        }

        # ---- records --------------------------------------------------------
        blk = block_from_rows(op)                       # (nx, ny, M)
        yflat = blk.reshape(H.n_pads, -1)
        Y = yflat.sum(axis=0)                           # pad-summed per window
        masks = ring_masks(H.chebyshev)
        y_ring = {k: {"n_pads": int(m.sum()),
                      "sum_ke": float(yflat[m].sum()),
                      "per_pad_ke": float(yflat[m].sum() / max(int(m.sum()), 1)),
                      "over_sum_effq": float(yflat[m].sum() / eq_all)}
                  for k, m in masks.items()}
        rec["records"] = {
            "sum_records_ke": float(blk.sum()),
            "sum_abs_records_ke": float(np.abs(blk).sum()),
            "sum_records_over_sum_effq": float(blk.sum() / eq_all),
            "deficit_frac": float(1.0 - blk.sum() / eq_all),
            "M_windows": int(blk.shape[2]),
            "by_ring": y_ring,
            "deficit_by_ring": {
                k: float((eq_all - yflat[m].sum()) / eq_all) if k == "0"
                else float(-yflat[m].sum() / eq_all)
                for k, m in masks.items()},
            "note": ("deficit_by_ring: ring 0 contributes 1 - sum y_0/sum x, "
                     "every other ring -sum y_r/sum x, so the five entries "
                     "sum to 1 - sum y / sum x")}

        # ---- tau_cut, truth and observed ------------------------------------
        cuts = {"none": None, "truth": int(round(-t_med))}
        obs: dict = {}
        fa = arrival_first_above(Y, b, B, frac)
        obs["first_above"] = fa
        cuts["observed_first_above"] = int(fa["tau_cut"])
        sf = arrival_shape_fit(Y, K_full, b, B, fa["t_plane_tick"], scan)
        obs["shape_fit"] = sf
        cuts["observed_shape_fit"] = int(sf["tau_cut"])
        cen = float((Y * (b + B * (np.arange(len(Y)) + 1))).sum() / Y.sum())
        obs["padsummed_window_charge_centroid_tick"] = cen
        tc_truth = cuts["truth"]
        rec["tau_cut"] = {
            "truth": tc_truth,
            "truth_definition": "-(charge-weighted median effq tick)",
            "predicted_from_depth": float(ARRIVAL_TICK - t_drift_ticks),
            "truth_minus_predicted_ticks": float(tc_truth
                                                 - (ARRIVAL_TICK
                                                    - t_drift_ticks)),
            "observed": obs,
            "observed_first_above_minus_truth_ticks":
                int(cuts["observed_first_above"] - tc_truth),
            "observed_shape_fit_minus_truth_ticks":
                int(cuts["observed_shape_fit"] - tc_truth),
            "effq_tick_span_ticks": int(H.truth_tick.max()
                                        - H.truth_tick.min()),
            "arrival_tick_constant": ARRIVAL_TICK}
        print(f"[{self.name}] d = {depth} cm, tau = {tau_ms} ms: effq "
              f"{eq_all:.2f} ke, records {blk.sum():.2f} ke "
              f"({blk.sum()/eq_all:.5f} of truth); tau_cut truth {tc_truth}, "
              f"first_above {cuts['observed_first_above']}, shape_fit "
              f"{cuts['observed_shape_fit']}")

        # ---- kernel integrals per variant ------------------------------------
        krings = kernel_offset_rings(K_full.shape[0], K_full.shape[1])
        sumK_full = float(K_full.sum())

        def kernel_report(cut):
            Kt = K_full if cut is None else np.where(
                np.arange(nt_resp)[None, None, :] >= int(cut), K_full, 0.0)
            per = Kt.sum(axis=2)
            perf = K_full.sum(axis=2)
            out = {"kernel_cut_tick": (None if cut is None else int(cut)),
                   "sum_K_trunc": float(Kt.sum()),
                   "sum_K_full": sumK_full,
                   "sum_h_trunc_over_B": float(Kt.sum()),
                   "truncation_fraction": float(1.0 - Kt.sum() / sumK_full),
                   "by_offset_ring": {}}
            for k, m in krings.items():
                out["by_offset_ring"][k] = {
                    "n_offsets": int(m.sum()),
                    "sum_K_trunc": float(per[m].sum()),
                    "sum_K_full": float(perf[m].sum()),
                    "deleted": float(perf[m].sum() - per[m].sum()),
                    "deleted_over_sum_K_full": float(
                        (perf[m].sum() - per[m].sum()) / sumK_full)}
            return out

        # ---- what the truncation predicts for the recorded total --------------
        # Cumulative of the pad-summed kernel, F(tau) = sum_{d, m <= tau} Kbar.
        Fcum = np.cumsum(K_full.sum(axis=(0, 1)))
        def sumK_from(cut):
            cut = int(cut)
            if cut <= 0:
                return sumK_full
            if cut >= nt_resp:
                return 0.0
            return float(sumK_full - Fcum[cut - 1])
        # (a) one fixed tau_cut for the whole event (what the operator uses)
        pred_fixed = eq_all * sumK_from(tc_truth)
        # (b) the exact per-tick rule: every truth tick j has its own
        #     tau_cut = -j, which is what tred actually applied.  The gap
        #     between (a) and (b) is the cost of making the operator shift
        #     invariant at one depth.
        tj = np.asarray(H.truth_tick)
        uj, inv = np.unique(tj, return_inverse=True)
        qj = np.bincount(inv, weights=np.asarray(H.truth_q))
        pred_pertick = float(sum(q * sumK_from(-int(j))
                                 for j, q in zip(uj, qj)))
        rec["records"].update({
            "predicted_total_fixed_tau_cut_ke": float(pred_fixed),
            "predicted_total_per_tick_tau_cut_ke": pred_pertick,
            "measured_minus_fixed_ke": float(blk.sum() - pred_fixed),
            "measured_minus_fixed_rel": float(blk.sum() / pred_fixed - 1.0),
            "measured_minus_per_tick_ke": float(blk.sum() - pred_pertick),
            "measured_minus_per_tick_rel": float(blk.sum() / pred_pertick - 1.0),
            "predicted_note": (
                "fixed: sum effq * sum_{d, tau >= tau_cut} Kbar with one "
                "tau_cut for the event; per_tick: sum_j x(j) * "
                "sum_{d, tau >= -j} Kbar, tred's own rule.  Both use the "
                "in-block truth ticks and the full event charge")})
        print(f"[{self.name}]   records {blk.sum():.3f} ke; truncation "
              f"predicts {pred_fixed:.3f} (fixed cut, "
              f"{blk.sum()/pred_fixed-1:+.3e}) / {pred_pertick:.3f} "
              f"(per tick, {blk.sum()/pred_pertick-1:+.3e})")

        # ---- the cell grid (shared by every variant) --------------------------
        grid = CellGrid(b, c, (B // c) * int(op.block_shape[2]))
        m_lo, m_hi = grid.window(win_lo, win_hi)
        Rx_c = grid.restrict(H.truth_ix, H.truth_iy, H.truth_tick, H.truth_q,
                             H.nx, H.ny)
        rec["cell_grid"] = grid.report(n_probe=1)
        rec["cell_grid"].update({
            "Rc_truth_total_ke": float(Rx_c.sum()),
            "Rc_truth_minus_truth_ke": float(Rx_c.sum() - H.truth_total),
            "stored_cell_window": [m_lo, m_hi]})

        def score(xcells, tag):
            out = {}
            pf = grid.to_fine(xcells, "uniform", m_lo, m_hi)
            for s in sigmas:
                m = score_rows(H, fine_xhat(H, pf, grid.fine_origin(m_lo), s), s)
                m.pop("_profiles")
                out[f"s{s:g}"] = m
            del pf
            return out

        rep = score(Rx_c, "representation")
        rec["representation_term"] = {
            k: {"E_rel": v["E_rel"], "conservation_rel": v["conservation_rel"],
                "segments_rel_error_rms": v["segments"]["rel_error_rms"]}
            for k, v in rep.items()}

        y_t = embed_pads(blk, H.nx + 24, H.ny + 24, blk.shape[2], dev, dtype)
        y_norm = float(torch.linalg.vector_norm(y_t))
        taps, wts = cell_charge_model_taps(grid, cell_model)
        IX, IY, TT, QQ = prolong_truth_to_fine(grid, Rx_c, taps, wts)

        # ---- the variants ------------------------------------------------------
        rec["variants"] = {}
        for vname in variants:
            cut = cuts[vname]
            t0 = time.time()
            F = FineOperator(K_full, op.block_shape, B, device=dev,
                             dtype=dtype, cell_ticks=c, cell_model=cell_model,
                             kernel_cut_tick=cut)
            t_build = time.time() - t0
            kr = kernel_report(cut)
            # validation: the cell forward against the exact-functional direct
            # sum of the SAME (possibly truncated) fine window function
            xc = embed_pads(Rx_c, F.nxp, F.nyp, F.N, dev, dtype)
            y_fft = F.forward(xc).cpu().numpy()[:H.nx, :H.ny]
            del xc
            torch.cuda.empty_cache()
            y_dir = direct_sum_records(F.h_np, IX, IY, TT, QQ, b, B, H.nx,
                                       H.ny, F.M, dev, dtype)
            den = float(np.abs(y_dir).sum())
            v1 = float(np.abs(y_fft - y_dir).sum() / den)
            if not v1 < tol_fwd:
                raise AssertionError(f"{vname}: forward vs direct sum {v1:.3e}")
            # validation: adjoint and the A A^T symbol
            gg = torch.Generator(device="cpu").manual_seed(11)
            xr = torch.randn((F.nxp, F.nyp, F.N), generator=gg,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            yr = torch.randn((F.nxp, F.nyp, F.M), generator=gg,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            a1 = float((F.forward(xr) * yr).sum())
            a2 = float((xr * F.adjoint(yr)).sum())
            u, w = F.AAt_fft(yr), F.forward(F.adjoint(yr))
            v_aat = float(torch.abs(u - w).sum() / torch.abs(w).sum())
            v_adj = abs(a1 - a2) / max(abs(a1), 1e-30)
            del xr, yr, u, w
            torch.cuda.empty_cache()
            if not (v_adj < tol_lin and v_aat < tol_lin):
                raise AssertionError(f"{vname}: adjoint {v_adj:.3e} "
                                     f"AAt {v_aat:.3e}")
            # the truth's own data residual under this forward model
            r_truth = float(torch.linalg.vector_norm(
                torch.as_tensor(y_fft, dtype=dtype, device=dev)
                - y_t[:H.nx, :H.ny]) / y_norm)

            t0 = time.time()
            xh = F.solve(y_t, lam_rel * F.G_max)
            t_solve = time.time() - t0
            resid = float(torch.linalg.vector_norm(F.forward(xh) - y_t)
                          / y_norm)
            xhn = xh[:H.nx, :H.ny].cpu().numpy()
            sum_all = float(xh.sum())
            del xh
            torch.cuda.empty_cache()

            xf = xhn.reshape(H.n_pads, -1)
            xr_ring = {k: {"n_pads": int(m.sum()),
                           "sum_ke": float(xf[m].sum()),
                           "per_pad_ke": float(xf[m].sum()
                                               / max(int(m.sum()), 1)),
                           "over_sum_effq": float(xf[m].sum() / eq_all)}
                       for k, m in masks.items()}
            sc = score(xhn, vname)
            pred = float(blk.sum() / kr["sum_K_trunc"])
            ent = {
                "kernel_cut": vname,
                "kernel": kr,
                "build_wall_s": t_build, "solve_wall_s": t_solve,
                "G_max": F.G_max, "G_dc": F.G_dc,
                "sum_xhat_ke": float(xf.sum()),
                "sum_xhat_all_pads_ke": sum_all,
                "sum_xhat_padding_pads_ke": sum_all - float(xf.sum()),
                "sum_xhat_pos_ke": float(xf[xf > 0].sum()),
                "sum_xhat_neg_ke": float(xf[xf < 0].sum()),
                "sum_xhat_line_pads_ke": float(xf[masks["0"]].sum()),
                "sum_xhat_off_line_ke": float(xf[~masks["0"]].sum()),
                "sum_xhat_over_sum_effq": float(xf.sum() / eq_all),
                "by_ring": xr_ring,
                "conservation_identity_sum_y_over_sum_K": pred,
                "conservation_identity_ratio": float(sum_all / pred),
                "lambda_shrinkage_predicted": float(
                    F.G_dc / (F.G_dc + lam_rel * F.G_max)),
                "residual_rel": resid,
                "truth_data_residual_rel": r_truth,
                "validation": {
                    "forward_vs_direct_sum_rel": v1,
                    "adjoint_rel": v_adj, "AAt_rel": v_aat,
                    "sum_g_over_D_minus_sum_K_trunc": float(
                        F.g_np.sum() / F.D - kr["sum_K_trunc"]),
                    "G_dc_over_ghat0_sq_over_D": float(
                        F.G_dc / (F.g_np.sum() ** 2 / F.D)),
                    "tolerance_forward": tol_fwd,
                    "tolerance_linear": tol_lin},
                "scores": {k: {"E_rel": v["E_rel"],
                               "conservation_rel": v["conservation_rel"],
                               "sum_xhat_ke": v["sum_xhat_ke"],
                               "sum_Hx_ke": v["sum_Hx_ke"],
                               "zero_preservation": v["zero_preservation"],
                               "segments": {
                                   "n": v["segments"]["n"],
                                   "rel_error_mean":
                                       v["segments"]["rel_error_mean"],
                                   "rel_error_rms":
                                       v["segments"]["rel_error_rms"],
                                   "rel_error_max_abs":
                                       v["segments"]["rel_error_max_abs"]}}
                           for k, v in sc.items()},
            }
            rec["variants"][vname] = ent
            del F
            torch.cuda.empty_cache()
            s15 = ent["scores"].get("s1.5", {})
            print(f"[{self.name}]   {vname:22s} cut "
                  f"{str(kr['kernel_cut_tick']):>5s}  sum K "
                  f"{kr['sum_K_trunc']:.6f}  trunc frac "
                  f"{kr['truncation_fraction']:.5f}  sum xhat "
                  f"{ent['sum_xhat_ke']:9.2f} ke  "
                  f"({ent['sum_xhat_over_sum_effq']:.5f} of truth)  E_rel "
                  f"{s15.get('E_rel', float('nan')):.4f}  |Ax-y|/|y| "
                  f"{resid:.3e}")
        del y_t
        torch.cuda.empty_cache()
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("DepthLadderFit")
class DepthLadderFit(_Recorder):
    """The lifetime fit over the ladder, from the archived per-event JSONs.

    A SOURCE algorithm: ``reads = ()``.  It re-reads nothing but JSON, so a
    different depth cut never re-runs a solve.

    Props
    -----
    inputs : list of ``{depth_cm, tau_ms, json}``.
    deep_only_min_depth_cm : float, default 16.5.
    velocity_cm_per_us : float, default 0.159645.
    out_json : str
    """

    reads = ()
    writes = ("depthladder.fit",)

    ESTIMATES = (
        ("sum_effq", "TRUTH (control), sum effq"),
        ("sum_records", "records, sum y"),
        ("xhat_none", "min-norm, full kernel"),
        ("xhat_truth", "min-norm, truncated kernel (truth tau_cut)"),
        ("xhat_observed_first_above",
         "min-norm, truncated kernel (observed tau_cut, first above 0.5 max)"),
        ("xhat_observed_shape_fit",
         "min-norm, truncated kernel (observed tau_cut, cumulative shape fit)"),
    )

    @staticmethod
    def _values(doc: dict) -> dict:
        r = doc.get("result", doc)
        out = {"sum_effq": float(r["truth"]["sum_effq_ke"]),
               "sum_records": float(r["records"]["sum_records_ke"])}
        for name, ent in r.get("variants", {}).items():
            out[f"xhat_{name}"] = float(ent["sum_xhat_ke"])
        return out

    def execute(self, store):
        v = float(self.props.get("velocity_cm_per_us",
                                 DRIFT_VELOCITY_CM_PER_US))
        deep = float(self.props.get("deep_only_min_depth_cm", 16.5))
        per: dict = {}
        events: dict = {}
        for item in self.props.get("inputs", []):
            d = float(item["depth_cm"])
            tau = float(item["tau_ms"])
            with open(item["json"]) as fh:
                doc = json.load(fh)
            r = doc.get("result", doc)
            per.setdefault(tau, {})[d] = self._values(doc)
            events.setdefault(str(tau), {})[str(d)] = {
                "tau_cut_truth": r["tau_cut"]["truth"],
                "tau_cut_predicted_from_depth":
                    r["tau_cut"]["predicted_from_depth"],
                "tau_cut_observed_first_above":
                    r["tau_cut"]["observed"]["first_above"]["tau_cut"],
                "tau_cut_observed_shape_fit":
                    r["tau_cut"]["observed"]["shape_fit"]["tau_cut"],
                "observed_first_above_minus_truth_ticks":
                    r["tau_cut"]["observed_first_above_minus_truth_ticks"],
                "observed_shape_fit_minus_truth_ticks":
                    r["tau_cut"]["observed_shape_fit_minus_truth_ticks"],
                "deficit_by_ring": r["records"]["deficit_by_ring"],
                "records_by_ring": {k: e["over_sum_effq"] for k, e
                                    in r["records"]["by_ring"].items()},
                "kernel_truncation_fraction": {
                    n: e["kernel"]["truncation_fraction"]
                    for n, e in r["variants"].items()},
                "kernel_truncation_by_offset_ring": {
                    n: {k: o["deleted_over_sum_K_full"] for k, o
                        in e["kernel"]["by_offset_ring"].items()}
                    for n, e in r["variants"].items()},
                "E_rel": {n: {k: s["E_rel"] for k, s in e["scores"].items()}
                          for n, e in r["variants"].items()},
                "segments": {n: {k: {"mean": s["segments"]["rel_error_mean"],
                                     "rms": s["segments"]["rel_error_rms"]}
                                 for k, s in e["scores"].items()}
                             for n, e in r["variants"].items()},
                "representation_term": r.get("representation_term", {}),
            }

        rec: dict = {"velocity_cm_per_us": v,
                     "deep_only_min_depth_cm": deep,
                     "lambda_error_definition":
                         "sqrt( (sum r^2/(n-2)) / sum (t - tbar)^2 ), "
                         "unweighted straight line ln E = a - lambda t_drift",
                     "per_event": events, "fits": {}, "ratios": {}}
        for tau, byd in sorted(per.items()):
            depths = np.array(sorted(byd))
            t_ms = depths / v * 1e-3
            key = f"{tau:g}ms"
            rec["fits"][key] = {}
            rec["ratios"][key] = {"depths_cm": [float(x) for x in depths]}
            hdr = (f"{'estimate':52s} {'lambda':>9s} {'+/-':>8s} "
                   f"{'deep':>9s} {'+/-':>8s}  rms lnE")
            print(f"[{self.name}] tau = {tau} ms   " + hdr)
            for name, label in self.ESTIMATES:
                if not all(name in byd[d] for d in depths):
                    continue
                E = np.array([byd[d][name] for d in depths])
                if not np.all(E > 0):
                    continue
                f_all = ls_line(t_ms, np.log(E))
                sel = depths >= deep
                f_dp = (ls_line(t_ms[sel], np.log(E[sel]))
                        if sel.sum() >= 3 else None)
                f_all["label"] = label
                f_all["depths_cm"] = [float(x) for x in depths]
                f_all["E_ke"] = [float(x) for x in E]
                f_all["deep_only"] = f_dp
                f_all["lambda_true_per_ms"] = 1.0 / tau
                f_all["pull"] = ((f_all["lambda_per_ms"] - 1.0 / tau)
                                 / max(f_all["lambda_err"], 1e-12))
                rec["fits"][key][name] = f_all
                rec["ratios"][key][name] = [
                    float(byd[d][name] / byd[d]["sum_effq"]) for d in depths]
                print(f"[{self.name}] {label:52s} "
                      f"{f_all['lambda_per_ms']:9.4f} "
                      f"{f_all['lambda_err']:8.4f} "
                      + (f"{f_dp['lambda_per_ms']:9.4f} "
                         f"{f_dp['lambda_err']:8.4f}" if f_dp else " " * 18)
                      + f"  {f_all['rms_resid_lnE']:.5f}")

        # the ratio table's lifetime independence: |R_1ms(d) - R_20ms(d)|
        keys = sorted(rec["ratios"])
        if len(keys) == 2:
            a, bkey = rec["ratios"][keys[0]], rec["ratios"][keys[1]]
            da = {float(x): i for i, x in enumerate(a["depths_cm"])}
            db = {float(x): i for i, x in enumerate(bkey["depths_cm"])}
            common = sorted(set(da) & set(db))
            diff = {"depths_cm": common}
            for name, _ in self.ESTIMATES:
                if name in a and name in bkey:
                    d = np.abs(np.array([a[name][da[x]] for x in common])
                               - np.array([bkey[name][db[x]] for x in common]))
                    diff[name] = {"max_abs_difference": float(d.max()),
                                  "per_depth": [float(x) for x in d]}
            rec["ratio_lifetime_difference"] = {
                "keys": keys, "note":
                    "|E(d)/sum effq(d) at one lifetime - the other|; a pure "
                    "acceptance is lifetime independent", **{"by_estimate": diff}}
            for name, e in diff.items():
                if not isinstance(e, dict):
                    continue
                print(f"[{self.name}] ratio 1ms-20ms max |diff| {name:32s} "
                      f"{e['max_abs_difference']:.2e}")
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("DepthLadderFigures")
class DepthLadderFigures(_Recorder):
    """Figures D1-D5 from the ladder JSON (and its embedded per-event table).

    Props
    -----
    ladder_json : str        the :class:`DepthLadderFit` output.
    figdir : str
    methods_ls_over_truth : list of float, optional
        The ``METHODS.md`` Sec. 10 LS/truth acceptance per depth, overlaid on
        D1 as grey crosses.
    out_json : str
    """

    reads = ()
    writes = ("depthladder.figures",)

    # (key, colour, marker, linestyle, label)
    ARMS = (("sum_records", C_GREY, "o", "-", "records, $\\Sigma y$"),
            ("xhat_none", C_FULL, "s", "-", "min-norm, full kernel"),
            ("xhat_truth", C_TRUNC, "^", "-",
             "min-norm, truncated (truth $\\tau_{cut}$)"),
            ("xhat_observed_shape_fit", C_TRUNC_OBS, "v", "-",
             "min-norm, truncated (observed $\\tau_{cut}$, shape fit)"),
            ("xhat_observed_first_above", C_TRUNC_OBS, "D", "--",
             "min-norm, truncated (observed $\\tau_{cut}$, first above)"))

    def execute(self, store):
        with open(self.props["ladder_json"]) as fh:
            self._doc = json.load(fh)["result"]
        self.put(store, "depthladder.figures", {"pending": True})
        self._recipe = {"job_config": store.get("job.config"),
                        "provenance": store.provenance()}

    def finalize(self):
        plt = ieee_style()
        R = self._doc
        outdir = Path(self.props.get("figdir", "figs"))
        made: list = []
        keys = sorted(R["ratios"], key=lambda s: float(s[:-2]))
        depths = np.array(R["ratios"][keys[0]]["depths_cm"])

        # ---- D1 charge ratio vs depth ------------------------------------
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for ki, key in enumerate(keys):
            fill = ki == 0
            for name, col, mk, ls, lab in self.ARMS:
                if name not in R["ratios"][key]:
                    continue
                wide = name == "sum_records"
                ax.plot(depths, R["ratios"][key][name], mk, ls=ls, color=col,
                        ms=(6.0 if wide else 3.4), lw=(3.0 if wide else 0.9),
                        alpha=(0.45 if wide else 1.0), zorder=(1 if wide else 3),
                        mfc=(col if fill else "none"), mew=0.9,
                        label=(f"{lab}" if fill else None))
        m = self.props.get("methods_ls_over_truth")
        if m:
            ax.plot(depths[:len(m)], m, "x", color=C_TRUTH, ms=5, mew=1.0,
                    ls="none", zorder=6,
                    label="METHODS Sec. 10, LS/truth")
        ax.axhline(1.0, color=C_TRUTH, lw=0.6, ls="--")
        ax.set_xlabel("drift depth [cm]")
        ax.set_ylabel(r"$E(d)\,/\,\Sigma\,\mathrm{effq}(d)$")
        ax.set_ylim(0.84, 1.01)
        ax.legend(loc="lower right", frameon=False)
        ax.set_title("filled: $\\tau$ = %s   open: $\\tau$ = %s"
                     % (keys[0], keys[1] if len(keys) > 1 else "-"),
                     fontsize=7)
        save(fig, outdir, "D1_charge_ratio_vs_depth", made)

        # ---- D2 deficit by ring ------------------------------------------
        pe = R["per_event"]
        key0 = keys[0][:-2]
        ev = pe[str(float(key0))] if str(float(key0)) in pe else pe[key0]
        cols = {"0": C_TRUTH, "1": C_FULL, "2": C_TRUNC, "3-5": C_TRUNC_OBS,
                ">=6": C_GREY}
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for r in RING_LABELS:
            v = [ev[str(d)]["deficit_by_ring"][r] for d in depths]
            axs[0].plot(depths, v, "o-", color=cols[r], ms=3, lw=0.9,
                        label=f"ring {r}")
            k = [ev[str(d)]["kernel_truncation_by_offset_ring"]["truth"][r]
                 for d in depths]
            axs[1].plot(depths, k, "s-", color=cols[r], ms=3, lw=0.9,
                        label=f"ring {r}")
        tot = [1.0 - sum(ev[str(d)]["records_by_ring"][r]
                         for r in RING_LABELS) for d in depths]
        axs[0].plot(depths, tot, "k--", lw=0.9, label="total")
        axs[1].plot(depths, [ev[str(d)]["kernel_truncation_fraction"]["truth"]
                             for d in depths], "k--", lw=0.9, label="total")
        for a, t in zip(axs, ("(a) measured: deficit of the records",
                              "(b) predicted: deleted kernel integral")):
            a.set_xlabel("drift depth [cm]")
            a.set_title(t, fontsize=7)
            a.axhline(0.0, color=C_GREY, lw=0.5)
        axs[0].set_ylabel(r"contribution to $1-\Sigma y/\Sigma x$")
        axs[1].set_ylabel(r"$1-\Sigma h_{trunc}/\Sigma h_{full}$, by ring")
        axs[0].legend(frameon=False, ncol=2, fontsize=6)
        save(fig, outdir, "D2_deficit_by_ring", made)

        # ---- D3 ln E vs t_drift ------------------------------------------
        fig, axs = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 2.6),
                                squeeze=False)
        for ax, key in zip(axs[0], keys):
            F = R["fits"][key]
            for name, col, mk, ls, lab in (
                    ("sum_effq", C_TRUTH, "o", "-",
                     "truth, $\\Sigma$ effq"),) + self.ARMS:
                if name not in F:
                    continue
                f = F[name]
                t = np.array(f["t_drift_ms"])
                ax.plot(t, f["lnE"], mk, color=col, ms=3, mfc="none", mew=0.9)
                ax.plot(t, f["intercept"] - f["lambda_per_ms"] * t, ls=ls,
                        color=col, lw=0.9,
                        label=f"{lab}: $\\lambda$ = "
                              f"{f['lambda_per_ms']:.3f}")
            # the y range is set by every arm EXCEPT the first-above one,
            # whose shallow points are far off scale (they are in the table)
            vals = [v for n in F if n != "xhat_observed_first_above"
                    for v in F[n]["lnE"]]
            lo, hi = min(vals), max(vals)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.75 * (hi - lo))
            ax.set_xlabel(r"$t_{drift}$ [ms]")
            ax.set_ylabel(r"$\ln E$  [$E$ in ke]")
            ax.set_title(f"$\\tau$ = {key}", fontsize=7)
            ax.legend(frameon=False, fontsize=5.5, loc="upper right")
        save(fig, outdir, "D3_lnE_vs_tdrift", made)

        # ---- D4 lambda per estimate --------------------------------------
        names = [n for n, _, _, _, _ in
                 (("sum_effq", C_TRUTH, "o", "-", ""),) + self.ARMS
                 if n in R["fits"][keys[0]]]
        colmap = {"sum_effq": C_TRUTH, **{n: c for n, c, _, _, _ in self.ARMS}}
        fig, axs = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 2.8),
                                squeeze=False)
        for ax, key in zip(axs[0], keys):
            F = R["fits"][key]
            xs = np.arange(len(names))
            for i, n in enumerate(names):
                f = F[n]
                ax.bar(i, f["lambda_per_ms"], 0.6, color=colmap[n], alpha=0.75)
                ax.errorbar(i, f["lambda_per_ms"], yerr=f["lambda_err"],
                            fmt="none", ecolor=C_TRUTH, elinewidth=0.8,
                            capsize=2)
                if f.get("deep_only"):
                    d = f["deep_only"]
                    ax.errorbar(i + 0.24, d["lambda_per_ms"],
                                yerr=d["lambda_err"], fmt="o", ms=4.0,
                                mfc="none", mec=colmap[n], ecolor=colmap[n],
                                elinewidth=0.8, capsize=2)
            lt = 1.0 / float(key[:-2])
            ax.axhline(lt, color=C_TRUTH, lw=0.8, ls="--")
            ax.set_ylim(lt - 1.3, lt + 0.6)
            for i, n in enumerate(names):
                f = F[n]
                if f["lambda_per_ms"] > lt + 0.6:
                    ax.annotate(f"{f['lambda_per_ms']:.2f}"
                                f"$\\pm${f['lambda_err']:.2f}",
                                (i, lt + 0.56), ha="center", va="top",
                                fontsize=5.5, rotation=90)
            ax.set_xticks(xs)
            ax.set_xticklabels([n.replace("xhat_", "") for n in names],
                               rotation=35, ha="right", fontsize=5.5)
            ax.set_ylabel(r"$\lambda$ [1/ms]")
            ax.set_title(f"$\\tau$ = {key}  (open: $d \\geq$ "
                         f"{R['deep_only_min_depth_cm']:g} cm)", fontsize=7)
        save(fig, outdir, "D4_lambda_per_estimate", made)

        # ---- D5 E_rel and segment error vs depth --------------------------
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for vn, col, lab in (("none", C_FULL, "full kernel"),
                             ("truth", C_TRUNC, "truncated (truth)"),
                             ("observed_shape_fit", C_TRUNC_OBS,
                              "truncated (observed, shape fit)")):
            e = [ev[str(d)]["E_rel"][vn]["s1.5"] for d in depths]
            axs[0].plot(depths, e, "o-", color=col, ms=3, lw=0.9, label=lab)
            mu = np.array([ev[str(d)]["segments"][vn]["s1.5"]["mean"]
                           for d in depths]) * 100
            sd = np.array([ev[str(d)]["segments"][vn]["s1.5"]["rms"]
                           for d in depths]) * 100
            axs[1].errorbar(depths, mu, yerr=sd, fmt="o-", color=col, ms=3,
                            lw=0.9, elinewidth=0.8, capsize=2, label=lab)
        axs[0].set_ylabel(r"$E_{rel}$ at $\sigma_H$ = 1.5 $\mu$s")
        # the first-above variant is left out here: its shallow points
        # (E_rel 2.50, segment error +100 %) compress everything else; they
        # are in D1, D3, D4 and in the tables
        axs[1].set_ylabel("segment-sum relative error [%]")
        axs[1].axhline(0.0, color=C_GREY, lw=0.5)
        for a in axs:
            a.set_xlabel("drift depth [cm]")
            a.legend(frameon=False, fontsize=6)
        save(fig, outdir, "D5_Erel_and_segments_vs_depth", made)

        print(f"[{self.name}] {len(made)} figures in {outdir}")
        body = {"figures": made, "ladder_json": self.props["ladder_json"],
                "colours": {"truth": C_TRUTH, "controls": C_GREY,
                            "minnorm_full_kernel": C_FULL,
                            "minnorm_truncated_truth": C_TRUNC,
                            "minnorm_truncated_observed": C_TRUNC_OBS}}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": body,
                           **self._recipe}, fh, indent=1, default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return body
