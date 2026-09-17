"""Where the objective moves the created ionisation charge.

Every campaign so far reported the ESTIMATE: start the solver at ``q0 = 0``,
run a fixed number of iterations, and compare what comes out with the created
ionisation charge.  This module asks the complementary question, which needs
no solver run to state and one to follow:

    put the created ionisation charge itself into the objective and measure
    the force on it.

Definitions
-----------
``f(x)``
    the smooth part of the objective actually minimised:
    ``1/2 ||A x - y||^2`` plus, in the censored variants, the two censor
    terms of :mod:`unfoldlarpix.terms.censor`, exactly as
    :class:`~unfoldlarpix.algs.zsbasis_algs.ZSCensorArms` builds them.

``g(x)``
    the prox-able part: ``alpha ||x||_1`` plus the indicator of ``x >= 0`` on
    the support (``CoordProx``), or the indicator of the support alone
    (``_SupportProx``, the least-squares arm).

``x_truth``
    ``R_c x``: the created ionisation charge of the event summed onto the
    ``c``-tick cell grid at the registration shift ``delta``.  This is the
    same array the campaigns score against, not a smoothed version of it.

``grad_at_truth``
    ``nabla f(x_truth) = A^T (A x_truth - y)`` (+ censor gradients), in ke.
    It is zero if and only if the created charge is a stationary point of the
    data misfit.  It is NOT zero here, and its sign per cell is the direction
    in which one solver step moves charge away from the truth.

``first_step_ke``
    ``prox(x_truth - step * grad_at_truth) - x_truth``, in ke: the charge one
    iteration adds to, or removes from, each cell when the iterate IS the
    created charge.  ``step = 1 / (1.05 * L)`` with ``L`` the sum of the
    terms' curvature bounds — the shipped FISTA step, so this is the actual
    first move of the production solver if it were started at the truth.

``gradient_mapping``
    ``G(x) = (x - prox(x - step * nabla f(x))) / step``.  ``G(x) = 0`` if and
    only if ``x`` minimises ``f + g``; ``||G(x_truth)||`` is therefore the
    literal statement "how far the created charge is from being the answer",
    measured in the objective's own units rather than in charge.

``projected gradient flow``
    the trajectory ``x_{k+1} = prox(x_k - step * nabla f(x_k))``: proximal
    gradient descent with no momentum, i.e. the forward-Euler discretisation
    of ``dx/dt = -nabla f(x)`` under the same constraint set the production
    solver uses.  Started at ``x_truth`` it shows where the objective takes
    the truth; started at ``0`` it is the production start without FISTA's
    extrapolation.  FISTA itself is run only for its endpoint, with the
    shipped engine and one call, so those numbers reproduce the archived
    campaign; a per-iteration FISTA trajectory would have to record the
    extrapolated point, which is not an iterate of the constraint set.

``distance to the truth``
    reported five ways, because no one of them is sufficient (the campaign's
    own finding 24): the charge ratio ``sum(x)/sum(x_truth)``; the cell L1
    distance ``sum|x - x_truth| / sum(x_truth)``; the relative L2 distance;
    ``E_rel`` at the goal smoothing; and the signed charge per Chebyshev
    distance class.  ``closest_approach`` is the minimum of the L1 distance
    over the recorded trajectory and the iteration at which it occurs: for a
    flow started at ``0`` it separates "the solver is still approaching the
    truth" from "the solver has passed it and is now fitting the operator's
    error".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..constrained_solver import build_latch_rows
from ..fwk.component import Algorithm, algorithm
from .evalharness_algs import EvalHarness
from .finebasis_algs import fine_xhat, score_rows
from .zsbasis_algs import (CONV_LABEL, CONVENTIONS, VARIANT_LABEL, _BasisJob,
                           _JsonAlg, build_zs_operator, censor_violation,
                           ring_sums, sigma_L_ticks, waveform_stats,
                           zs_windows)

CLASS_LABELS = ("ionised", "plus1", "plus2", "plus3_or_more")


def _class_masks(H, nx: int, ny: int, device) -> dict:
    """Chebyshev distance class masks on the (nx, ny) pad grid."""
    ch = np.asarray(H.chebyshev).reshape(nx, ny)
    sel = {"ionised": ch == 0, "plus1": ch == 1, "plus2": ch == 2,
           "plus3_or_more": ch >= 3}
    return {k: torch.as_tensor(v, device=device) for k, v in sel.items()}


def _class_sums(pad_sum: torch.Tensor, masks: dict) -> dict:
    """Signed charge summed over the pads of each class, in ke."""
    return {k: float(pad_sum[m].sum()) for k, m in masks.items()}


class _Objective:
    """The terms, the prox and the step of one (variant, arm) combination."""

    def __init__(self, op, terms, prox, safety: float = 1.05):
        self.op = op
        self.terms = list(terms)
        self.prox = prox
        self.L = float(sum(t.curvature() for t in self.terms))
        self.step = 1.0 / (safety * max(self.L, 1e-12))

    def grad(self, x: torch.Tensor):
        """(gradient, objective value) at ``x``; one shared context."""
        from ..terms.base import IterCtx
        ctx = IterCtx(x, self.op)
        g = torch.zeros_like(x)
        for t in self.terms:
            t.grad_into(ctx, g)
        val = float(sum(float(t.value(ctx)) for t in self.terms))
        return g, val

    def step_from(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.prox(x - self.step * g, self.step)


@algorithm("ZSGradientFlow")
class ZSGradientFlow(_JsonAlg):
    """The gradient at the created ionisation charge, and the flow it drives.

    For each first-window convention, operator/term variant and arm the
    algorithm reports

    1. ``stationarity``: the residual ``A x_truth - y`` by row kind, the
       gradient at the truth by Chebyshev distance class, the charge the
       first step would move, and ``||G(x_truth)||``.  No solve.
    2. ``runs``: the projected gradient flow from ``x_truth`` and from ``0``,
       with the distance to the truth recorded along the way, and the FISTA
       endpoint of the same objective from both starts.

    The operator, the rows, the censor terms, the support and the scoring are
    built exactly as :class:`ZSCensorArms` builds them, so a variant-A
    ``pos_l1`` endpoint here is comparable with the archived ``arms.json``
    entry of the same configuration.
    """

    reads = ("event", "readout_config", "block", "block_offset", "op",
             "support", "hits_view")
    writes = ("zs.gradflow",)

    def execute(self, store):
        from ..model.conventions import resolve_burst_tau
        from ..solve.engine import Fista
        from ..terms.base import CoordProx
        from ..terms.censor import CensorRunningMax, pre_trigger_censors
        from ..terms.data import DataFidelity
        from .fixedgrid_algs import _SupportProx

        J = _BasisJob(self, store)
        rc, hv, boff = J.rc, store.get("hits_view"), store.get("block_offset")
        p = self.props
        c = int(p.get("cell_ticks", 5))
        kernel_bin = bool(p.get("kernel_bin", False))
        convs = list(p.get("conventions", ["acq_edge"]))
        variants = list(p.get("variants", ["A", "D"]))
        starts = list(p.get("starts", ["truth", "zero"]))
        methods = list(p.get("methods", ["flow"]))
        sigmas = [float(s) for s in p.get("sigma_H_us", [1.5, 2.0])]
        delta = int(p.get("registration_delta", -1))
        iters = int(p.get("iters", 3000))
        rec_every = int(p.get("record_every", 10))
        score_every = int(p.get("score_every", 250))
        margin = float(p.get("censor_margin", 0.0))
        beta = float(p.get("censor_beta", 1.0))
        norm = str(p.get("censor_norm", "l2"))
        npad = int(p.get("censor_npad_bins", 30))
        close_back = float(p.get("censor_close_back", 20.0))
        post_reset = bool(p.get("censor_include_post_reset", True))
        cell_model = str(p.get("cell_model", "uniform"))
        vel = float(p.get("velocity_cm_per_us", 0.159645))
        depth_cm = p.get("depth_cm")
        sig_p = p.get("sigma_p_ticks")
        if cell_model == "gaussian" and sig_p is None:
            if depth_cm is None:
                raise ValueError("cell_model 'gaussian' needs sigma_p_ticks "
                                 "or depth_cm")
            sig_p = 0.5 * sigma_L_ticks(float(depth_cm), vel)
        sig_p = None if sig_p is None else float(sig_p)
        arms_cfg = list(p.get("arms", [{"label": "pos_l1", "alpha": 0.01,
                                        "positivity": True}]))
        wave_pixels = [list(map(int, q)) for q in
                       p.get("waveform_pixels", [[141, 68]])]
        out_npz = p.get("out_npz")
        burst_tau = resolve_burst_tau(rc, None)

        H = EvalHarness(store, store.get("op"),
                        margin_windows=int(p.get("margin_windows", 40)),
                        line_pixel_y_range=p.get("line_pixel_y_range",
                                                 (5, 131)),
                        segment_pixels=int(p.get("segment_pixels", 7)),
                        segment_edge_exclude=int(
                            p.get("segment_edge_exclude", 3)))
        masks = _class_masks(H, J.nx, J.ny, J.comp.device)

        rec = {"truth_total_ke": J.truth_total, "cell_ticks": c,
               "cell_model": cell_model, "sigma_p_ticks": sig_p,
               "registration_delta": delta, "iterations": iters,
               "record_every": rec_every, "score_every": score_every,
               "sigma_H_us": sigmas,
               "burst_tau_ticks": int(burst_tau),
               "flow_definition":
                   "x_{k+1} = prox(x_k - step grad f(x_k)), "
                   "step = 1/(1.05 L), no momentum",
               "stationarity": {}, "runs": [], "start_comparison": []}
        # depth bookkeeping for the calibration fit: the drift time in fine
        # ticks and its remainder modulo the record window, which is the
        # arrival phase the non-linear arms were found to track
        if depth_cm is not None:
            t_drift_us = float(depth_cm) / vel
            t_drift_ticks = t_drift_us / 0.05
            rec["depth_cm"] = float(depth_cm)
            rec["t_drift_us"] = t_drift_us
            rec["t_drift_ticks"] = t_drift_ticks
            rec["arrival_phase_mod_B"] = float(t_drift_ticks % J.B)
        rec["lifetime_us"] = float(getattr(rc, "lifetime", float("nan")))
        npz: dict = {}
        # final iterates, kept so that two starts of the SAME objective can be
        # compared point to point rather than through their summaries
        finals: dict = {}

        # the probe pixels' fine truth, for the waveform moments
        xt_full = np.zeros((J.nx * J.ny, J.nt_fine))
        ok = ((J.truth_ix >= 0) & (J.truth_ix < J.nx)
              & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
        jj = J.truth_tick[ok] + delta - J.b0
        okj = (jj >= 0) & (jj < J.nt_fine)
        np.add.at(xt_full, ((J.truth_ix[ok][okj] * J.ny + J.truth_iy[ok][okj]),
                            jj[okj]), J.truth_q[ok][okj])
        probe = {}
        for pxy in wave_pixels:
            ip, iq = pxy[0] - int(J.boff[0]), pxy[1] - int(J.boff[1])
            if 0 <= ip < J.nx and 0 <= iq < J.ny:
                probe[f"{pxy[0]}_{pxy[1]}"] = (ip, iq, xt_full[ip * J.ny + iq])
                npz[f"truth_fine_{pxy[0]}_{pxy[1]}"] = \
                    xt_full[ip * J.ny + iq].astype(np.float32)
        npz["fine_origin_tick"] = np.array([J.b0])
        npz["nt_fine"] = np.array([J.nt_fine])
        # the pad grid, so the figures can label an absolute pixel_x/pixel_y
        # axis without re-loading the event
        npz["block_offset"] = np.asarray(J.boff, dtype=np.int64)
        npz["pad_shape"] = np.array([J.nx, J.ny], dtype=np.int64)
        npz["cell_ticks"] = np.array([c], dtype=np.int64)
        rec["block_offset"] = [int(v) for v in np.asarray(J.boff)]
        rec["pad_shape"] = [int(J.nx), int(J.ny)]

        sigma_key = f"E_rel_{sigmas[0]}"

        def score(x_np, xt_np):
            """The expensive, resolution-space part of the distance."""
            xf = (J.to_fine(x_np, c) if cell_model == "uniform" else
                  self._expand_fine(J, op_cur, x_np))
            out = {"pixels": ring_sums(H, xf)}
            # The cell L1 distance is measured on the unknown grid and is
            # therefore NOT comparable between bases: a 30-tick cell hides
            # every misplacement inside itself.  This one prolongs the
            # estimate to the fine grid and compares it with the fine created
            # charge, so it means the same thing at every c.
            nf = min(xf.shape[1], xt_full.shape[1])
            out["l1_distance_fine_over_truth"] = float(
                np.abs(xf[:, :nf] - xt_full[:, :nf]).sum()) / J.truth_total
            for s in sigmas:
                out[f"E_rel_{s}"] = float(
                    score_rows(H, fine_xhat(H, xf, J.b0, s), s)["E_rel"])
            out["waveform_stats"] = {
                k: waveform_stats(xf[v[0] * J.ny + v[1]], v[2])
                for k, v in probe.items()}
            return out, xf

        for conv in convs:
            # "event": the event's own canonical acq_start (a genuine per-
            # event first-trigger reference, needed for a real muon track
            # whose acquisition begins at an arbitrary tick -- unlike the
            # isoline's global_tref = [0, 0], where acq_edge/acq_t0 coincide
            # with reality).  Anything not in CONVENTIONS falls through to
            # this, so a bad conv name still fails loudly, at ev.acq_start.
            if conv == "event":
                acq_val = getattr(J.ev, "acq_start", None)
                if acq_val is None:
                    raise ValueError(
                        "conventions: [event] but the event carries no "
                        "acq_start (isoline-style files use acq_edge/"
                        "acq_t0 instead)")
                conv_label = "the event's own acquisition start"
            else:
                acq_val = CONVENTIONS[conv]
                conv_label = CONV_LABEL[conv]
            for V in variants:
                split = V in ("B", "D")
                use_censor = V in ("C", "D")
                if split:
                    windows, metas = build_latch_rows(
                        J.ev.hits.location, J.ev.hits.data, J.B,
                        np.asarray(boff),
                        csa_reset_time=int(rc.csa_reset_time),
                        split_threshold=float(rc.threshold),
                        acq_start=acq_val, burst_tau=burst_tau)
                else:
                    windows, metas = zs_windows(store, conv, acq_start=acq_val)
                if kernel_bin and c > 1:
                    # kernel binned to the cell width and the sampling done
                    # on the c-tick grid (windows_to_sampling uses overlap
                    # fractions, so edges are not rounded).  Same cell grid
                    # and the same data vector as the fine-kernel operator;
                    # q_shape has one extra trailing cell ((nt - kt + 1) in
                    # bins).  Measured on iso50 d16.5 ev0: truth residual
                    # 0.427 vs 0.398, sum(A x_t) +0.16 %, forward+adjoint
                    # 13.9 ms vs 118.6 ms.
                    from ..model.operator import ZSOperator
                    K1 = np.asarray(J.K1, dtype=float)
                    if K1.shape[2] % c:
                        raise ValueError(f"kernel length {K1.shape[2]} is "
                                         f"not a multiple of c={c}")
                    Kc = K1.reshape(K1.shape[0], K1.shape[1],
                                    K1.shape[2] // c, c).sum(3)
                    op = op_cur = ZSOperator(
                        Kc, (J.nx, J.ny, J.nt_fine // c), windows, c,
                        device=J.comp.device, dtype=J.comp.dtype)
                else:
                    op = op_cur = build_zs_operator(J, c, windows,
                                                    cell_model=cell_model,
                                                    sigma_p_ticks=sig_p)
                bt_op = c if (kernel_bin and c > 1) else 1
                npad_op = max(1, int(round(npad / bt_op)))
                kinds = np.array([m.kind for m in metas])
                y = op.d.cpu().numpy().astype(float)
                supp = J.support_on_basis(op, c)
                x_truth = J.truth_on_basis(op, c, delta)
                xt_t = op.to_tensor(np.ascontiguousarray(x_truth))
                st = op.to_tensor(np.asarray(supp).astype(np.float64))
                nrm_xt = float(torch.linalg.vector_norm(xt_t))

                terms_extra, censor_info = [], []
                if use_censor:
                    # which censor families enter the objective.  Variants C
                    # and D historically meant "both", which is the only
                    # composition that was ever run -- but the post-latch and
                    # pre-trigger terms are separate physical statements
                    # ("silent after the last burst" vs "below threshold
                    # before the first trigger") with very different weights
                    # (measured on iso50 d=16.5: 2482 vs 6165 of a 9271 total
                    # curvature), so they must be separable to be ablated.
                    # iso50's own production arms B/C use post-latch ONLY.
                    fams = self.props.get("censor_terms", ["post", "pre"])
                    fams = [str(f) for f in fams]
                    unknown = set(fams) - {"post", "pre"}
                    if unknown:
                        raise ValueError(f"censor_terms: unknown {unknown}; "
                                         f"allowed are 'post' and 'pre'")
                    labels = []
                    if "post" in fams:
                        terms_extra.append(CensorRunningMax.from_hits(
                            op, hv, boff,
                            csa_reset_time=float(rc.csa_reset_time),
                            threshold=float(rc.threshold), npad_bins=npad_op,
                            beta=beta, margin=margin, norm=norm,
                            bin_ticks=bt_op))
                        labels.append("post_latch")
                    if "pre" in fams:
                        pre = list(pre_trigger_censors(
                            op, hv, boff,
                            csa_reset_time=float(rc.csa_reset_time),
                            threshold=float(rc.threshold),
                            acq_start=acq_val, npad_bins=npad_op,
                            beta=beta, margin=margin, norm=norm,
                            bin_ticks=bt_op,
                            one_tick=float(rc.one_tick),
                            close_back=close_back,
                            include_post_reset=post_reset))
                        terms_extra.extend(pre)
                        labels.extend(f"pre_trigger_ordinal{j}"
                                      for j in range(len(pre)))
                    for t, lab_t in zip(terms_extra, labels):
                        censor_info.append({
                            "term": lab_t,
                            "curvature": float(t.curvature()),
                            "truth": censor_violation(t, op, x_truth)})
                data_term = DataFidelity(op)
                all_terms = [data_term] + terms_extra
                vk = f"{conv}_{V}"

                # ---- 1. the state at the created ionisation charge --------
                pred_t = op.forward(xt_t).cpu().numpy().astype(float)
                res_t = pred_t - y
                sta = {
                    "convention": conv, "convention_label": conv_label,
                    "variant": V, "variant_label": VARIANT_LABEL[V],
                    "kernel_bin": kernel_bin, "operator_bin_ticks": bt_op,
                    "n_rows": int(op.n_data),
                    "rows_by_kind": {k: int((kinds == k).sum())
                                     for k in sorted(set(kinds))},
                    "sum_y_ke": float(y.sum()),
                    "sum_y_over_truth": float(y.sum() / J.truth_total),
                    "sum_A_x_truth_ke": float(pred_t.sum()),
                    "truth_rel_residual": float(np.linalg.norm(res_t)
                                                / np.linalg.norm(y)),
                    "truth_residual_by_row_kind": {
                        k: {"n_rows": int((kinds == k).sum()),
                            "sum_y_ke": float(y[kinds == k].sum()),
                            "sum_residual_ke": float(res_t[kinds == k].sum()),
                            "rel_residual": float(
                                np.linalg.norm(res_t[kinds == k])
                                / max(np.linalg.norm(y[kinds == k]), 1e-30))}
                        for k in sorted(set(kinds))},
                    "censor": censor_info,
                    # the prox zeroes every cell outside the support, so a
                    # flow started at the created charge starts at its
                    # support projection; this is how much that removes.
                    "n_support_cells": int(st.sum()),
                    "support_fraction": float(st.mean()),
                    "truth_charge_outside_support_ke": float(
                        (xt_t * (1.0 - st)).sum()),
                    "truth_cells_outside_support": int(
                        ((xt_t > 0) & (st == 0)).sum()),
                    "arms": {},
                }
                # which pixel each row sits on, and its Chebyshev class: a
                # data-side correction can only act on rows that exist, so
                # where the residual sits by pixel class is the question
                cls_map = np.asarray(H.chebyshev).reshape(J.nx, J.ny)
                rpx = np.array([w.px for w in windows], dtype=int)
                rpy = np.array([w.py for w in windows], dtype=int)
                rcls = cls_map[rpx, rpy]
                sta["truth_residual_by_pixel_class"] = {
                    lab: {
                        "n_rows": int(m.sum()),
                        "sum_y_ke": float(y[m].sum()),
                        "sum_residual_ke": float(res_t[m].sum()),
                        "sum_abs_residual_ke": float(np.abs(res_t[m]).sum()),
                        "rel_residual": float(
                            np.linalg.norm(res_t[m])
                            / max(np.linalg.norm(y[m]), 1e-30)),
                        "share_of_squared_residual": float(
                            (res_t[m] ** 2).sum()
                            / max((res_t ** 2).sum(), 1e-30)),
                    }
                    for lab, m in (("ionised", rcls == 0), ("plus1", rcls == 1),
                                   ("plus2", rcls == 2),
                                   ("plus3_or_more", rcls >= 3))
                    if m.any()}
                npz[f"resid_truth_{vk}"] = res_t.astype(np.float32)
                npz[f"rowkind_{vk}"] = kinds.astype("U12")
                npz[f"rowclass_{vk}"] = rcls.astype(np.int16)
                npz[f"rowpx_{vk}"] = rpx.astype(np.int32)
                npz[f"rowpy_{vk}"] = rpy.astype(np.int32)
                npz[f"y_{vk}"] = y.astype(np.float32)

                for cfg in arms_cfg:
                    lab = str(cfg["label"])
                    alpha = float(cfg.get("alpha", 0.0))
                    pos = bool(cfg.get("positivity", True))
                    alpha_t, adiag = None, None
                    if cfg.get("alpha_map"):
                        alpha_t, adiag = self._alpha_map(
                            cfg["alpha_map"], hv, boff, op, acq_val,
                            J.b0, c, delta, st)
                        alpha = adiag["alpha_median"]
                    prox, floor, group, phase, misalign = self._make_prox(
                        cfg, p, alpha_t if alpha_t is not None else alpha,
                        pos, st, c, b0=J.b0, y_sum=float(y.sum()))
                    obj = _Objective(op, all_terms, prox)
                    g_t, f_t = obj.grad(xt_t)
                    x1 = obj.step_from(xt_t, g_t)
                    d1 = x1 - xt_t
                    gmap = (xt_t - x1) / obj.step
                    gsum = g_t.sum(dim=2)
                    dsum = d1.sum(dim=2)
                    occupied = xt_t > 0
                    sta["arms"][lab] = {
                        "alpha_ke_per_cell": alpha, "positivity": pos,
                        "alpha_map": adiag,
                        "group_floor_ke": floor, "group_cells": group,
                        "group_phase_cells": phase,
                        "group_misalign_ticks": misalign,
                        "lipschitz_total": obj.L, "step": obj.step,
                        "objective_at_truth": f_t,
                        "grad_norm_at_truth": float(
                            torch.linalg.vector_norm(g_t)),
                        "gradient_mapping_norm_at_truth": float(
                            torch.linalg.vector_norm(gmap)),
                        "grad_sum_ke": float(g_t.sum()),
                        "grad_sum_on_occupied_cells_ke": float(
                            g_t[occupied].sum()),
                        "grad_min_ke": float(g_t.min()),
                        "grad_max_ke": float(g_t.max()),
                        "grad_by_class_ke": _class_sums(gsum, masks),
                        "first_step_total_ke": float(d1.sum()),
                        "first_step_total_over_truth": float(
                            d1.sum() / J.truth_total),
                        "first_step_abs_ke": float(d1.abs().sum()),
                        "first_step_by_class_ke": _class_sums(dsum, masks),
                        "first_step_on_occupied_cells_ke": float(
                            d1[occupied].sum()),
                        "n_cells_truth_occupies": int(occupied.sum()),
                        "n_cells_first_step_adds": int((d1 > 0).sum()),
                        "n_cells_first_step_removes": int((d1 < 0).sum()),
                    }
                    npz[f"grad_pad_{vk}_{lab}"] = \
                        gsum.cpu().numpy().astype(np.float32)
                    npz[f"step1_pad_{vk}_{lab}"] = \
                        dsum.cpu().numpy().astype(np.float32)
                    for k, (ip, iq, _) in probe.items():
                        npz[f"grad_cells_{vk}_{lab}_{k}"] = \
                            g_t[ip, iq].cpu().numpy().astype(np.float32)
                        npz[f"step1_cells_{vk}_{lab}_{k}"] = \
                            d1[ip, iq].cpu().numpy().astype(np.float32)
                        npz[f"truth_cells_{vk}_{k}"] = \
                            xt_t[ip, iq].cpu().numpy().astype(np.float32)
                    print(f"[ZSGradientFlow] {vk} {lab}: L {obj.L:.4g} "
                          f"step {obj.step:.3e} |grad| "
                          f"{sta['arms'][lab]['grad_norm_at_truth']:.4g} "
                          f"first step {d1.sum():+.2f} ke "
                          f"({100 * float(d1.sum()) / J.truth_total:+.4f} %)")
                rec["stationarity"][vk] = sta

                # ---- 2. the flows -----------------------------------------
                # score() Gaussian-smooths over a window padded from the
                # TRUTH's own occupied cells (margin_windows) plus the
                # sigma_H kernel reach; for a real track (not the isoline)
                # that window can exceed what this event's own fine block
                # stores, if the occupied span sits close to the block's
                # edge.  A resolution-space score is secondary to the
                # charge-ratio numbers this campaign is actually after
                # (sum_xhat_ke, class_sums_ke, neither of which touches
                # score()), so this is degraded, not fatal: the event's
                # solves still run and are still recorded.
                try:
                    base, _ = score(x_truth, x_truth)
                    rec["stationarity"][vk]["truth_scores"] = base
                except ValueError as exc:
                    print(f"[ZSGradientFlow] {vk}: truth score unavailable "
                          f"({exc}); continuing without E_rel for this "
                          f"event")
                    rec["stationarity"][vk]["truth_scores"] = None
                    rec["stationarity"][vk]["truth_scores_error"] = str(exc)
                for cfg in arms_cfg:
                    lab = str(cfg["label"])
                    alpha = float(cfg.get("alpha", 0.0))
                    pos = bool(cfg.get("positivity", True))
                    n_it = int(cfg.get("iters", iters))
                    alpha_t, adiag = None, None
                    if cfg.get("alpha_map"):
                        alpha_t, adiag = self._alpha_map(
                            cfg["alpha_map"], hv, boff, op, acq_val,
                            J.b0, c, delta, st)
                        alpha = adiag["alpha_median"]
                    prox, floor, group, phase, misalign = self._make_prox(
                        cfg, p, alpha_t if alpha_t is not None else alpha,
                        pos, st, c, b0=J.b0, y_sum=float(y.sum()))
                    obj = _Objective(op, all_terms, prox)
                    for start in starts:
                        x0 = self._start_vector(
                            start, xt_t, op, y, windows, kinds, J, c, delta,
                            float(p.get("seed_threshold_ke", 10.0)),
                            float(p.get("seed_matched_frac", 0.05)))
                        for method in methods:
                            t0 = time.time()
                            stages = None
                            stopped = n_it
                            n_restarts = 0
                            fista_extra = None
                            if method == "flow":
                                x, traj = self._flow(
                                    obj, x0, xt_t, nrm_xt, n_it, rec_every,
                                    score_every, masks, score, x_truth,
                                    sigma_key)
                            elif method == "ladder_refit":
                                # the production strategy: a decreasing alpha
                                # ladder, each stage warm-started from the
                                # previous and re-seeded from its own
                                # skeleton, then an alpha = 0 refit on the
                                # frozen strong support with the faint charge
                                # held as background
                                x, stages = self._ladder_refit(
                                    op, all_terms, st, x0, n_it, p, cfg)
                                traj = None
                            else:
                                # arm-level override, else job-level, else
                                # the plain engine (archived behaviour)
                                eng = Fista(
                                    n_iter=n_it,
                                    restart=bool(cfg.get(
                                        "fista_restart",
                                        p.get("fista_restart", False))),
                                    rel_tol=cfg.get(
                                        "fista_rel_tol",
                                        p.get("fista_rel_tol")),
                                    patience=int(cfg.get(
                                        "fista_patience",
                                        p.get("fista_patience", 5))),
                                    backtrack=bool(cfg.get(
                                        "fista_backtrack",
                                        p.get("fista_backtrack", False))),
                                    bt_L0_div=float(cfg.get(
                                        "fista_bt_L0_div",
                                        p.get("fista_bt_L0_div", 16.0))),
                                    trace_every=int(cfg.get(
                                        "fista_trace_every",
                                        p.get("fista_trace_every", 0))))
                                x = eng.minimize(op, all_terms, prox, x0)
                                stopped = int(eng.stopped_at)
                                n_restarts = int(eng.n_restarts)
                                fista_extra = {
                                    "n_backtracks": int(eng.n_backtracks),
                                    "sum_trace": [[int(i), float(s)]
                                                  for i, s in eng.trace],
                                    "step_mean": (float(np.mean(eng.steps))
                                                  if eng.steps else None),
                                    "step_min": (float(np.min(eng.steps))
                                                 if eng.steps else None),
                                    "step_max": (float(np.max(eng.steps))
                                                 if eng.steps else None),
                                    "step_fixed_bound": float(obj.step)}
                                traj = None
                                stages = None
                            dt = time.time() - t0
                            entry = self._finalise(
                                J, op, obj, x, xt_t, nrm_xt, masks, y, kinds,
                                score, x_truth, conv, V, lab, alpha, pos,
                                start, method, n_it, dt, traj, censor_info,
                                terms_extra, alpha_tensor=alpha_t,
                                alpha_map_diag=adiag)
                            if stages:
                                entry["ladder_stages"] = stages
                            entry["iters_run"] = stopped
                            entry["n_restarts"] = n_restarts
                            if fista_extra is not None:
                                entry["fista"] = fista_extra
                            rec["runs"].append(entry)
                            tag = f"{vk}_{lab}_{start}_{method}"
                            if traj is not None:
                                for key, arr in traj.items():
                                    npz[f"traj_{tag}_{key}"] = \
                                        np.asarray(arr, dtype=np.float32)
                            xf = entry.pop("_xf")
                            for k, (ip, iq, _) in probe.items():
                                npz[f"final_fine_{tag}_{k}"] = \
                                    xf[ip * J.ny + iq].astype(np.float32)
                            xnp = x.detach().cpu().numpy().astype(np.float32)
                            npz[f"final_cells_{tag}"] = xnp
                            finals[(conv, V, lab, method)] = \
                                finals.get((conv, V, lab, method), {})
                            finals[(conv, V, lab, method)][start] = (
                                xnp, entry["objective_full"],
                                entry["rel_residual"],
                                entry["sum_xhat_over_truth"],
                                entry.get(sigma_key))
                            er = entry.get(sigma_key)
                            er_s = f"{er:.4f}" if er is not None else "n/a"
                            print(f"[ZSGradientFlow] {tag}: sum/truth "
                                  f"{entry['sum_xhat_over_truth']:.4f} "
                                  f"L1/truth "
                                  f"{entry['l1_distance_over_truth']:.4f} "
                                  f"{sigma_key} {er_s} "
                                  f"{dt:.1f} s")
                if use_censor:
                    for t in terms_extra:
                        del t
                    torch.cuda.empty_cache()

        # ---- do the two starts of one objective end at the same point? -----
        # Both are feasible points of the same convex problem, so a difference
        # in the full objective F = f + alpha ||x||_1 is a convergence
        # difference, while agreement in F with a non-zero distance between
        # the iterates is the null space of A on the support.
        q = J.truth_total
        for (conv, V, lab, method), d in sorted(finals.items()):
            if "truth" not in d or "zero" not in d:
                continue
            (xa, fa, ra, sa, ea) = d["truth"]
            (xb, fb, rb, sb, eb) = d["zero"]
            diff = xa - xb
            cmp = {
                "convention": conv, "variant": V, "arm": lab,
                "method": method,
                "l1_between_starts_ke": float(np.abs(diff).sum()),
                "l1_between_starts_over_truth": float(np.abs(diff).sum()) / q,
                "l2_relative_between_starts": float(
                    np.linalg.norm(diff)
                    / max(np.linalg.norm(xb), 1e-30)),
                "max_abs_cell_difference_ke": float(np.abs(diff).max()),
                "objective_full_truth_start": fa,
                "objective_full_zero_start": fb,
                "objective_full_difference": fa - fb,
                "objective_full_relative_difference": (
                    (fa - fb) / max(abs(fb), 1e-30)),
                "rel_residual_truth_start": ra,
                "rel_residual_zero_start": rb,
                "sum_over_truth_truth_start": sa,
                "sum_over_truth_zero_start": sb,
                "E_rel_goal_truth_start": ea,
                "E_rel_goal_zero_start": eb,
            }
            rec["start_comparison"].append(cmp)
            print(f"[ZSGradientFlow] {conv}_{V} {lab} {method}: two starts "
                  f"differ by L1 {cmp['l1_between_starts_over_truth']:.4f} "
                  f"of the created charge, objective "
                  f"{fa:.6g} (truth) against {fb:.6g} (zero), "
                  f"relative {cmp['objective_full_relative_difference']:+.3e}")

        if out_npz:
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **npz)
            print(f"[ZSGradientFlow] wrote {out_npz}")
        self.put(store, "zs.gradflow", rec)
        self._emit(store, rec)


    # -- prox ----------------------------------------------------------------
    @staticmethod
    def _alpha_map(amap, hv, boff, op, acq_start_tick, b0, c, delta, st):
        """Dispatch on ``alpha_map.type``: ``exp_time`` (default) or
        ``exp_trigger``."""
        kind = str(amap.get("type", "exp_time"))
        if kind == "exp_time":
            return ZSGradientFlow._alpha_from_time(
                amap, op, acq_start_tick, b0, c, delta, st)
        if kind == "exp_trigger":
            return ZSGradientFlow._alpha_from_trigger(
                amap, hv, boff, op, acq_start_tick)
        raise ValueError(f"alpha_map.type: unknown {kind!r}; defined are "
                         f"'exp_time' and 'exp_trigger'")

    @staticmethod
    def _alpha_from_time(amap, op, acq_start_tick, b0, c, delta, st):
        """Per-cell l1 weight from the cell's OWN position on the time axis.

        The unknown grid's time axis is the arrival time: cell ``m`` holds
        the charge whose anode-arrival tick is
        ``b0 + (m + 1/2) c - delta + ARRIVAL_TICK`` (the inverse of
        ``_BasisJob.truth_on_basis``, ``m = (tick + delta - b0) // c``, with
        ``ARRIVAL_TICK`` the response's anode-arrival index).  Minus the
        event's ``acq_start`` that is the drift time of that cell, hence its
        depth -- exactly, per cell, with no trigger involved.  Checked: iso50
        d = 16.5 cm puts its truth at m ~ 321 with b0 = -3360, giving 103.0
        us against the true 103.4 us.

        This supersedes ``_alpha_from_trigger`` (first trigger per pixel)
        for the reason the user gave: the first trigger only dates the
        FIRST voxel of a pixel.  Two tracks on the same pixels at different
        depths, or one inclined track whose later cells sit deeper, need
        every cell weighted by its own depth, and the time axis already
        carries it.  Nothing here reads the truth.

            alpha(m) = alpha0 * exp(-(t_m + t_offset_us) / tau_us)

        ``diag`` reports the drift-time and alpha range over the SUPPORT
        cells, not the padded axis, so it describes where the solve lives.
        """
        from .exactrows_algs import ARRIVAL_TICK
        TICK_US = 0.05
        alpha0 = float(amap["alpha0"])
        tau_us = float(amap["tau_us"])
        t_off = float(amap.get("t_offset_us", 0.0))
        nx, ny, nt = op.q_shape
        k = np.arange(nt, dtype=float)
        tick = float(b0) + (k + 0.5) * float(c) - float(delta) + ARRIVAL_TICK
        t_us = np.clip((tick - float(acq_start_tick)) * TICK_US + t_off,
                       0.0, None)
        a_k = alpha0 * np.exp(-t_us / tau_us)
        alpha_np = np.broadcast_to(a_k[None, None, :], (nx, ny, nt))
        alpha_t = op.to_tensor(np.ascontiguousarray(alpha_np,
                                                    dtype=np.float64))
        occ = np.asarray(st.detach().cpu().numpy() > 0).any(axis=(0, 1))
        ts, as_ = t_us[occ], a_k[occ]
        diag = {
            "type": "exp_time", "alpha0": alpha0, "tau_us": tau_us,
            "t_offset_us": t_off, "n_support_time_cells": int(occ.sum()),
            "support_t_drift_us": {
                "min": float(ts.min()) if ts.size else None,
                "median": float(np.median(ts)) if ts.size else None,
                "max": float(ts.max()) if ts.size else None},
            "alpha_median": float(np.median(as_)) if as_.size else alpha0,
            "alpha_min": float(as_.min()) if as_.size else alpha0,
            "alpha_max": float(as_.max()) if as_.size else alpha0,
        }
        return alpha_t, diag

    @staticmethod
    def _alpha_from_trigger(amap, hv, boff, op, acq_start_tick):
        """Per-cell l1 weight set from each pixel's FIRST trigger time.

        The reconstruction does not know the depth.  What it does know is
        the event's acquisition start ``t0`` (``acq_start``) and, per pixel,
        the fine tick of the first threshold crossing.  Their difference is
        a drift-time estimate, hence a depth estimate, made from the data
        alone -- checked on iso50: median over pixels of
        ``first_trigger - t0`` is 23.2 / 98.9 / 174.4 us against true drift
        times 28.2 / 103.4 / 178.5 us at 4.5 / 16.5 / 28.5 cm, i.e. a
        constant ~-4.5 us offset and ~+-1 us scatter within a depth.

        ``alpha`` is a per-cell soft threshold in ke.  Deeper charge is
        spread over more cells by diffusion (measured: charge per occupied
        5-tick cell 9.12 -> 3.95 ke from 4.5 to 28.5 cm while the total
        falls only 13%), so one alpha cannot serve every depth.  The map
        tried here is the exponential the user proposed,

            alpha(t) = alpha0 * exp(-(t + t_offset_us) / tau_us),

        with ``t`` the per-pixel inferred drift time in us.  Pixels that
        never triggered have no ``t``; they take the event median of alpha
        over the triggered pixels (``fill: median``, the only option
        defined).  Nothing here reads the truth.

        Returns ``(alpha_tensor, diag)`` -- the (nx, ny, nt) tensor for the
        prox and a dict of what was inferred, so the depth-unknown run can be
        compared against the true depth afterwards.
        """
        TICK_US = 0.05
        alpha0 = float(amap["alpha0"])
        tau_us = float(amap["tau_us"])
        t_off = float(amap.get("t_offset_us", 0.0))
        fill = str(amap.get("fill", "median"))
        if fill != "median":
            raise ValueError(f"alpha_map.fill: only 'median' is defined, "
                             f"got {fill!r}")
        nx, ny, nt = op.q_shape
        px = np.asarray(hv.pixel_x).astype(int) - int(boff[0])
        py = np.asarray(hv.pixel_y).astype(int) - int(boff[1])
        trig = np.asarray(hv.trigger, dtype=float)
        first = np.full((nx, ny), np.inf)
        ok = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        np.minimum.at(first, (px[ok], py[ok]), trig[ok])
        triggered = np.isfinite(first)
        t_us = np.clip((first - float(acq_start_tick)) * TICK_US, 0.0, None)
        a_pix = np.full((nx, ny), np.nan)
        a_pix[triggered] = alpha0 * np.exp(-(t_us[triggered] + t_off) / tau_us)
        a_med = float(np.nanmedian(a_pix)) if triggered.any() else alpha0
        a_pix[~triggered] = a_med
        alpha_np = np.broadcast_to(a_pix[:, :, None], (nx, ny, nt))
        alpha_t = op.to_tensor(np.ascontiguousarray(alpha_np, dtype=np.float64))
        tt = t_us[triggered]
        diag = {
            "type": "exp_trigger", "alpha0": alpha0, "tau_us": tau_us,
            "t_offset_us": t_off, "fill": fill,
            "n_triggered_pixels": int(triggered.sum()),
            "inferred_t_drift_us": {
                "median": float(np.median(tt)) if tt.size else None,
                "p16": float(np.percentile(tt, 16)) if tt.size else None,
                "p84": float(np.percentile(tt, 84)) if tt.size else None},
            "alpha_median": a_med,
            "alpha_min": float(np.nanmin(a_pix)),
            "alpha_max": float(np.nanmax(a_pix)),
        }
        return alpha_t, diag

    @staticmethod
    def _make_prox(cfg, props, alpha, pos, st, c, b0=None, y_sum=None):
        """CoordProx, or the group-floor prox when a floor is configured.

        ``group_floor_ke`` > 0 relaxes positivity to "each cell may go down
        to -floor, each group of ``group_ticks`` fine ticks must still sum to
        at least zero".  At c = 30 the group is one cell and the two
        coincide, which is the sanity check.

        The groups are anchored to the UNIVERSAL grid (absolute tick 0, the
        ``eval/universal.py`` convention) rather than to the operator's own
        ``b0``, via :func:`universal_group_phase` -- unless
        ``group_universal: false`` is set, which restores the old
        cell-index-0 anchoring (kept only for A/B comparison against runs
        made before this was added).
        """
        from ..terms.base import (CoordProx, GroupFloorProx, SimplexProx,
                                  universal_group_phase)
        from .fixedgrid_algs import _SupportProx
        floor = float(cfg.get("group_floor_ke",
                              props.get("group_floor_ke", 0.0)))
        # pin_global_charge: "self_trigger" pins sum(x) to the recorded
        # self-triggered sum, sum(y).  This removes the fit's freedom to
        # choose its own normalisation.  l1 becomes inoperative under it
        # (see SimplexProx) so alpha is ignored and that is recorded.
        pin = cfg.get("pin_global_charge", props.get("pin_global_charge"))
        if pin:
            if str(pin) != "self_trigger":
                raise ValueError(f"pin_global_charge: only 'self_trigger' "
                                 f"is defined, got {pin!r}")
            if y_sum is None:
                raise ValueError("pin_global_charge needs the record sum")
            if not pos:
                raise ValueError("pin_global_charge requires positivity "
                                 "(the simplex lives in the non-negative "
                                 "orthant)")
            return SimplexProx(st, float(y_sum)), 0.0, 0, 0, None
        if not pos:
            return _SupportProx(st), 0.0, 0, 0, None
        if floor <= 0:
            return CoordProx(alpha, st), 0.0, 0, 0, None
        gt = int(cfg.get("group_ticks", props.get("group_ticks", 30)))
        c = int(c)
        group = max(gt // c, 1)
        universal = bool(cfg.get("group_universal",
                                 props.get("group_universal", True)))
        phase, misalign = 0, None
        if universal:
            if b0 is None:
                raise ValueError("group_universal needs the block origin "
                                 "(b0); pass it from the caller's J.b0")
            phase, misalign = universal_group_phase(b0, c, gt)
        return (GroupFloorProx(alpha, st, floor=floor, group=group,
                               group_phase=phase),
                floor, group, phase, misalign)

    # -- the production ladder + refit ---------------------------------------
    @staticmethod
    def _ladder_refit(op, terms, support, x0, n_iter, props, cfg):
        """``Ladder`` then ``FinalRefit`` from :mod:`solve.strategy`.

        The ladder's alphas default to a decade above the campaign's single
        alpha down to it, which is the "strong charge first" homotopy: each
        stage warm-starts at the previous solution and rebuilds its seed from
        that solution's own skeleton (``q > seed_cut``), so nothing about the
        seeding is tied to one topology.  ``FinalRefit`` then re-solves at
        ``alpha = 0`` on the frozen strong support with the faint charge held
        as background through the data term's target, which removes the l1
        amplitude shrinkage rather than leaving it in the charge scale.
        """
        from ..solve.engine import Fista
        from ..solve.strategy import FinalRefit, Ladder, SolveState
        alphas = [float(a) for a in cfg.get(
            "ladder_alphas", props.get("ladder_alphas", [0.3, 0.1, 0.03,
                                                         0.01]))]
        stage_iter = int(cfg.get("ladder_stage_iters",
                                 props.get("ladder_stage_iters",
                                           max(n_iter // len(alphas), 1))))
        lad = Ladder(alphas=alphas,
                     seed_cut=float(props.get("ladder_seed_cut", 0.5)),
                     soft_len=float(props.get("ladder_soft_len", 2.0)),
                     soft_exponent=float(props.get("ladder_soft_exponent",
                                                   1.0)),
                     n_iter=stage_iter)
        engine = Fista(n_iter=stage_iter)
        state = lad.run(engine, op, terms, support, SolveState(q=x0.clone()))
        if bool(props.get("refit", True)):
            state = FinalRefit(
                eps=float(props.get("refit_eps", 0.5)),
                alpha=float(props.get("refit_alpha", 0.0)),
                n_iter=int(props.get("refit_iters", stage_iter))
            ).run(engine, op, terms, support, state)
        stages = [{"label": r.label, "alpha": r.alpha, "q_sum": r.q_sum,
                   "nnz": r.nnz, "objective": r.objective, "l1": r.l1}
                  for r in state.history]
        for s in stages:
            print(f"[ZSGradientFlow]   {s['label']}: alpha={s['alpha']} "
                  f"q_sum={s['q_sum']:.1f} nnz={s['nnz']}")
        return state.q, stages

    # -- starts --------------------------------------------------------------
    @staticmethod
    def _start_vector(start, xt_t, op, y, windows, kinds, J, c, delta,
                      seed_threshold_ke, seed_matched_frac):
        """The initial iterate.

        ``truth``  the created ionisation charge on the cell grid;
        ``zero``   the production start;
        ``seed_trigger``  one cell per pixel, at the release time the pixel's
            FIRST threshold crossing implies -- ``t_trig - ARRIVAL_TICK`` --
            carrying that pixel's whole recorded charge divided by
            ``sum Kbar``.  Only pixels whose recorded total exceeds
            ``seed_threshold_ke`` are seeded: that threshold is what
            separates a pixel that collected charge from one that only saw
            its neighbour's induced current (5-6 ke per induced-only record
            against 31 ke on an ionised pixel);
        ``seed_matched``  the backprojection ``max(A^T y, 0)``, zeroed below
            ``seed_matched_frac`` of its maximum, rescaled so that the seed
            carries ``sum y / sum Kbar``.

        Both seeds are starts, not estimates: they are fed to the same
        objective and the same iteration count as the zero start.
        """
        from .exactrows_algs import ARRIVAL_TICK
        if start == "truth":
            return xt_t.clone()
        if start == "zero":
            return torch.zeros_like(xt_t)
        sumK = float(np.asarray(J.K1).sum())
        if start == "seed_matched":
            g = torch.clamp(op.adjoint(op.d), min=0.0)
            if float(g.max()) > 0:
                g = g * (g >= seed_matched_frac * float(g.max()))
            s = float(g.sum())
            if s > 0:
                g = g * (float(y.sum()) / sumK / s)
            return g
        if start == "seed_trigger":
            x0 = torch.zeros_like(xt_t)
            # per pixel: total recorded charge and the earliest latch
            tot: dict = {}
            first: dict = {}
            for w, k in zip(windows, kinds):
                if k == "pseudo":         # a threshold equality, not charge
                    continue
                key = (int(w.px), int(w.py))
                tot[key] = tot.get(key, 0.0) + float(w.value)
                first[key] = min(first.get(key, 1e18), float(w.t_hi))
            nx, ny, n = op.q_shape
            for (px, py), q in tot.items():
                if q <= seed_threshold_ke or not (0 <= px < nx
                                                  and 0 <= py < ny):
                    continue
                t_trig = first[(px, py)] - J.B      # latch is trigger + B
                m = int(np.floor((t_trig - ARRIVAL_TICK + delta) / c))
                if 0 <= m < n:
                    x0[px, py, m] += q / sumK
            return x0
        raise ValueError(f"unknown start {start!r}")

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _expand_fine(J, op, x_np):
        e = op.expand(op.to_tensor(np.ascontiguousarray(x_np))
                      ).cpu().numpy().astype(np.float64)
        e = e.reshape(-1, op.n_fine_used)
        xf = np.zeros((e.shape[0], J.nt_fine))
        xf[:, :e.shape[1]] = e
        return xf

    def _flow(self, obj, x0, xt_t, nrm_xt, n_iter, rec_every, score_every,
              masks, score, x_truth, sigma_key):
        """Projected gradient flow, with the distance to the truth recorded."""
        x = obj.prox(x0.clone(), 0.0)
        truth_total = float(xt_t.sum())
        keys = ("iter", "sum_ke", "sum_over_truth", "l1_distance_ke",
                "l1_distance_over_truth", "l2_relative_distance",
                "objective", "grad_norm", "gradient_mapping_norm",
                "E_rel_goal") + tuple(f"class_{k}_ke" for k in CLASS_LABELS)
        traj = {k: [] for k in keys}
        for k in range(n_iter + 1):
            g, val = obj.grad(x)
            if k % rec_every == 0 or k == n_iter:
                d = x - xt_t
                pad = x.sum(dim=2)
                cs = _class_sums(pad, masks)
                gm = float(torch.linalg.vector_norm(
                    (x - obj.step_from(x, g)) / obj.step))
                traj["iter"].append(k)
                traj["sum_ke"].append(float(x.sum()))
                traj["sum_over_truth"].append(float(x.sum()) / truth_total)
                traj["l1_distance_ke"].append(float(d.abs().sum()))
                traj["l1_distance_over_truth"].append(
                    float(d.abs().sum()) / truth_total)
                traj["l2_relative_distance"].append(
                    float(torch.linalg.vector_norm(d)) / nrm_xt)
                traj["objective"].append(val)
                traj["grad_norm"].append(float(torch.linalg.vector_norm(g)))
                traj["gradient_mapping_norm"].append(gm)
                for lab in CLASS_LABELS:
                    traj[f"class_{lab}_ke"].append(cs[lab])
                if score_every and (k % score_every == 0 or k == n_iter):
                    s, _ = score(x.detach().cpu().numpy().astype(np.float64),
                                 x_truth)
                    traj["E_rel_goal"].append(s[sigma_key])
                else:
                    traj["E_rel_goal"].append(float("nan"))
            if k == n_iter:
                break
            x = obj.step_from(x, g)
        return x, traj

    def _finalise(self, J, op, obj, x, xt_t, nrm_xt, masks, y, kinds, score,
                  x_truth, conv, V, lab, alpha, pos, start, method, n_it, dt,
                  traj, censor_info, terms_extra, alpha_tensor=None,
                  alpha_map_diag=None):
        # l1 price: alpha is a scalar unless an alpha_map made it a per-cell
        # tensor, in which case the price is the elementwise product summed.
        if alpha_tensor is not None:
            l1_price = float((alpha_tensor * x.abs()).sum())
        else:
            l1_price = alpha * float(x.abs().sum())
        x_np = x.detach().cpu().numpy().astype(np.float64)
        d = x - xt_t
        pred = op.forward(x).cpu().numpy().astype(float)
        res = pred - y
        g, val = obj.grad(x)
        try:
            sc, xf = score(x_np, x_truth)
        except ValueError as exc:
            # same degradation as the truth-score call above: the charge
            # numbers below (sum_xhat_ke, class_sums_ke) do not depend on
            # score() at all, only E_rel/pixels do, so this run is still
            # worth keeping.
            print(f"[ZSGradientFlow] {conv}_{V} {lab} {start} {method}: "
                  f"score unavailable ({exc}); recording the solve without "
                  f"E_rel")
            sc = {"pixels": {}, "_score_error": str(exc)}
            xf = np.zeros((J.nx * J.ny, J.nt_fine))
        entry = {
            "convention": conv, "variant": V, "variant_label": VARIANT_LABEL[V],
            "arm": lab, "alpha_ke_per_cell": alpha, "positivity": pos,
            "start": start, "start_label": ("the created ionisation charge"
                                            if start == "truth" else "zero"),
            "method": method, "iters": n_it, "wall_time_s": float(dt),
            "sum_xhat_ke": float(x.sum()),
            "sum_xhat_over_truth": float(x.sum()) / float(xt_t.sum()),
            "l1_distance_ke": float(d.abs().sum()),
            "l1_distance_over_truth": float(d.abs().sum()) / float(xt_t.sum()),
            "l2_relative_distance": float(torch.linalg.vector_norm(d)) / nrm_xt,
            "objective": val,
            # the full objective F = f + alpha ||x||_1.  The positivity and
            # support indicators are zero at any point the prox returns, so F
            # is comparable between two feasible iterates of one problem.
            "objective_full": val + l1_price,
            "alpha_map": alpha_map_diag,
            "grad_norm": float(torch.linalg.vector_norm(g)),
            "gradient_mapping_norm": float(torch.linalg.vector_norm(
                (x - obj.step_from(x, g)) / obj.step)),
            "nnz": int((x > 0).sum()),
            "rel_residual": float(np.linalg.norm(res) / np.linalg.norm(y)),
            "residual_by_row_kind": {
                k: {"n_rows": int((kinds == k).sum()),
                    "sum_residual_ke": float(res[kinds == k].sum()),
                    "rel_residual": float(
                        np.linalg.norm(res[kinds == k])
                        / max(np.linalg.norm(y[kinds == k]), 1e-30))}
                for k in sorted(set(kinds))},
            "class_sums_ke": _class_sums(x.sum(dim=2), masks),
            "censor_violation": [
                {"term": ci["term"], "solution": censor_violation(t, op, x),
                 "truth": ci["truth"]}
                for ci, t in zip(censor_info, terms_extra)],
            "_xf": xf,
        }
        entry.update({k: v for k, v in sc.items() if k != "pixels"})
        entry["pixels"] = sc["pixels"]
        if traj is not None:
            l1 = np.asarray(traj["l1_distance_over_truth"], dtype=float)
            it = np.asarray(traj["iter"], dtype=int)
            i = int(np.argmin(l1))
            entry["closest_approach"] = {
                "iteration": int(it[i]),
                "l1_distance_over_truth": float(l1[i]),
                "l1_distance_over_truth_at_end": float(l1[-1]),
                "sum_over_truth_at_closest": float(
                    np.asarray(traj["sum_over_truth"])[i]),
                "moved_away_after_closest": bool(l1[-1] > l1[i] + 1e-9)}
        return entry
