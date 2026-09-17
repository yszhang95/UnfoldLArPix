"""Residual at the created charge, and the first step, on four field responses.

Written for the ``frvariant_isoline`` campaign: one isoline at 16.5 cm
simulated with four field responses (all 25 x 25 pixels or the collection
pixel only, impact dependent or impact averaged) on 31 depths whose step is
one fine tick of drift, read out by ``nd_readout`` and by ``fixed_interval``.
The operator here always uses the IMPACT-AVERAGED 25 x 25 kernel, whatever
the sample was simulated with, because an operator can never know the impact
position -- so on the two center-pixel-only samples it additionally carries
the mismatch of expecting neighbour induced charge that the sample does not
have.

The three objectives, all with ``f(x) = 1/2 ||A x - y||^2``:

===========  =======================================  ========================
``ls``       ``min f``                                prox = support mask only
``pos``      ``min f`` s.t. ``x >= 0``                prox = clamp at 0
``pos_l1``   ``min f + alpha ||x||_1`` s.t. ``x>=0``  prox = soft-threshold
===========  =======================================  ========================

By default the support is the WHOLE block (``support: full``), so the three
arms are exactly the three objectives above and nothing is masked away; the
charge that the production support (hits + dilation) would have removed is
reported as ``truth_charge_outside_hits_support_ke`` instead of being
removed.  ``support: hits`` restores the production behaviour.

Definitions
-----------
``y``
    the recorded window integrals, ``op.d``: one row per record window.  For
    ``fixed_interval`` samples the rows come from the pseudo-hits conversion
    (``unfoldlarpix.io.pseudo_hits``), one row per (pixel, 30-tick sample).

``x_truth``
    the created ionisation charge (``effq``) summed onto the operator's own
    charge grid.  Two truth models:

    ``point``
        each effq entry goes into ONE bin, the bin whose index is
        ``m = floor((t + delta + s) / c)`` with ``t`` the effq fine tick,
        ``delta`` the registration shift (production value ``-1``) and ``s``
        the bin-index rule: ``s = 0`` is ``mode: floor``, the
        nearest-lower-edge assignment; ``s = c // 2`` is ``mode: round``,
        the bin-centre assignment.  At ``c = 1`` the two coincide exactly,
        because the fine tick IS the bin.
    ``group``
        the fine (``c = 1``) grid, but the charge is AVERAGED over groups of
        ``group_ticks`` fine ticks defined on the ABSOLUTE tick axis:
        group ``g`` holds the absolute ticks with
        ``floor((T + s) / G) = g``, its total charge is spread uniformly
        over the ticks of the group that lie inside the block, and ``s`` is
        again 0 for ``floor`` and ``G // 2`` for ``round``.  Total charge is
        conserved exactly (``group_truth_charge_ke`` is checked against the
        point truth); what changes is only how the charge is distributed in
        time inside each group.

``A``
    ``sample o conv``.  At ``c = 1`` the kernel is the impact-averaged
    response integrated at one fine tick (``prepared_raw(1)``, 25 x 25 x
    3900) and the block bin is one fine tick.  At ``c > 1`` the kernel is
    that same response summed in groups of ``c`` ticks (``K_c``) and the
    block bin is ``c`` fine ticks -- the BIN-INTEGRATED response.  The
    sampling matrix is built by ``windows_to_sampling``, which converts a
    window ``(t_lo, t_hi]`` in fine ticks into overlap FRACTIONS of the
    ``c``-tick bins it covers, so a record whose trigger stamp falls inside
    a coarse bin contributes that bin with weight ``frac < 1`` instead of
    being rounded to a bin edge.  ``row_fraction_*`` reports how much of
    the sampling weight is fractional.

``residual``
    ``r = A x_truth - y``, in ke, one entry per row.  Reported overall, by
    row kind, and by the Chebyshev distance class of the row's pixel from
    the ionised pixels (0 = ionised, +1, +2, >= +3).

``grad``
    ``nabla f(x_truth) = A^T (A x_truth - y)``, in ke per cell.

``step``
    ``1 / (1.05 L)`` with ``L = ||A^T A||`` by power iteration: the shipped
    FISTA step, so the first step below is the actual first move of the
    production solver if its iterate were the created charge.

``first_step`` / ``q_next``
    ``q_next = prox(x_truth - step * grad)`` and
    ``first_step = q_next - x_truth``, in ke per cell: what one iteration of
    each objective does to the created charge.  ``first_step`` summed is the
    charge one iteration would add (positive) or remove (negative).

``gradient_mapping_norm``
    ``||(x_truth - q_next) / step||``.  Zero if and only if the created
    charge minimises that objective, so it is the literal statement of how
    far the created charge is from being the answer, in the objective's own
    units.

Per-pixel output
----------------
For every pixel in ``probe_pixels`` the NPZ carries, per configuration and
arm, the full time series over that pixel's cells -- ``truth``, ``grad``,
``step1``, ``qnext`` -- plus the pixel's row table ``(t_lo, t_hi, y,
A x_truth)`` in ABSOLUTE fine ticks and its trigger / latch / rearm times,
so a plot can overlay the created charge, the next iterate and the records
on one absolute time axis, and the same pixel's ``fixed_interval`` rows on
top of its ``nd_readout`` rows.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..fwk.component import algorithm
from ..model.operator import ZSOperator
from ..terms.base import CoordProx, IterCtx
from ..terms.data import DataFidelity
from .fixedgrid_algs import _SupportProx
from .zsbasis_algs import CONV_LABEL, _BasisJob, _JsonAlg, zs_windows

DEFAULT_CONFIGS = [
    {"label": "c1_point", "cell_ticks": 1, "truth": "point", "mode": "floor"},
    {"label": "c1_avg5_floor", "cell_ticks": 1, "truth": "group",
     "group_ticks": 5, "mode": "floor"},
    {"label": "c1_avg5_round", "cell_ticks": 1, "truth": "group",
     "group_ticks": 5, "mode": "round"},
    {"label": "c5_floor", "cell_ticks": 5, "truth": "point", "mode": "floor"},
    {"label": "c5_round", "cell_ticks": 5, "truth": "point", "mode": "round"},
    {"label": "c30_floor", "cell_ticks": 30, "truth": "point",
     "mode": "floor"},
    {"label": "c30_round", "cell_ticks": 30, "truth": "point",
     "mode": "round"},
]
DEFAULT_ARMS = [
    {"label": "ls", "alpha": 0.0, "positivity": False},
    {"label": "pos", "alpha": 0.0, "positivity": True},
    {"label": "pos_l1", "alpha": 0.01, "positivity": True},
]
CLASS_LABELS = ("ionised", "plus1", "plus2", "plus3_or_more")


def mode_shift(mode: str, c: int) -> int:
    """The bin-index rule as an integer shift: floor -> 0, round -> c // 2."""
    if mode == "floor":
        return 0
    if mode == "round":
        return int(c) // 2
    raise ValueError(f"mode must be 'floor' or 'round', got {mode!r}")


def point_truth(J: _BasisJob, op, c: int, delta: int, shift: int):
    """Each effq entry into the single bin ``floor((t+delta+shift)/c)``."""
    nx, ny, n = op.q_shape
    m = np.floor_divide(J.truth_tick + int(delta) + int(shift) - J.b0, int(c))
    ok = ((m >= 0) & (m < n) & (J.truth_ix >= 0) & (J.truth_ix < nx)
          & (J.truth_iy >= 0) & (J.truth_iy < ny))
    out = np.zeros((nx, ny, n))
    np.add.at(out, (J.truth_ix[ok], J.truth_iy[ok], m[ok]), J.truth_q[ok])
    return out, {"n_effq_entries": int(J.truth_q.size),
                 "n_effq_entries_in_grid": int(ok.sum()),
                 "charge_in_grid_ke": float(J.truth_q[ok].sum()),
                 "charge_outside_grid_ke": float(J.truth_q[~ok].sum())}


def group_truth(J: _BasisJob, op, G: int, delta: int, shift: int):
    """Fine grid, charge averaged over absolute-origin groups of ``G`` ticks.

    The group of absolute tick ``T`` is ``g = floor((T + shift) / G)``.  The
    group index is non-decreasing in the fine bin index, so the grouping is a
    run-length partition of the time axis and ``np.add.reduceat`` bins it
    exactly.  Each group's charge is spread over the ticks of that group that
    lie INSIDE the block, so the total is conserved even where the first or
    last group is clipped by the block edge.
    """
    pt, diag = point_truth(J, op, 1, delta, 0)
    nx, ny, n = op.q_shape
    G = int(G)
    # the truth is registered with ``delta`` FIRST (that is an operator
    # convention: where the kernel releases a bin's charge), and the groups
    # are then defined on the absolute tick axis of the operator's own fine
    # bins, abs_tick(i) = i + b0.
    absolute = np.arange(n) + J.b0
    g = np.floor_divide(absolute + int(shift), G)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(g)) + 1))
    counts = np.diff(np.concatenate((starts, [n])))
    gsum = np.add.reduceat(pt, starts, axis=2)
    avg = gsum / counts[None, None, :]
    out = np.repeat(avg, counts, axis=2)
    diag = dict(diag)
    diag.update({
        "group_ticks": G, "n_groups": int(len(starts)),
        "group_shift_ticks": int(shift),
        "first_group_abs_tick": int(absolute[0]),
        "block_offset_mod_group": int((J.b0 + int(shift)) % G),
        "partial_groups": int((counts != G).sum()),
        "point_truth_charge_ke": float(pt.sum()),
        "group_truth_charge_ke": float(out.sum()),
        "charge_conservation_ke": float(out.sum() - pt.sum()),
        "n_cells_point": int((pt > 0).sum()),
        "n_cells_group": int((out > 0).sum()),
    })
    return out, diag


def build_operator(J: _BasisJob, c: int, windows, device, dtype, K1=None):
    """``c = 1``: the fine kernel on fine bins.  ``c > 1``: ``K_c`` on
    ``c``-tick bins, i.e. the BIN-INTEGRATED response, with the window
    overlap fractions carrying the misalignment between the fine-tick
    window edges and the coarse bins."""
    c = int(c)
    K1 = np.asarray(J.K1 if K1 is None else K1, dtype=float)
    if c == 1:
        return ZSOperator(K1, (J.nx, J.ny, J.nt_fine), windows, 1,
                          device=device, dtype=dtype), K1.shape
    if K1.shape[2] % c:
        raise ValueError(f"kernel length {K1.shape[2]} is not a multiple "
                         f"of c={c}")
    if J.nt_fine % c:
        raise ValueError(f"fine block {J.nt_fine} is not a multiple of c={c}")
    Kc = K1.reshape(K1.shape[0], K1.shape[1], K1.shape[2] // c, c).sum(3)
    return ZSOperator(Kc, (J.nx, J.ny, J.nt_fine // c), windows, c,
                      device=device, dtype=dtype), Kc.shape


def pixel_class(J: _BasisJob) -> np.ndarray:
    """Chebyshev distance from the ionised pixels, capped at 3."""
    ion = np.zeros((J.nx, J.ny), bool)
    ok = ((J.truth_ix >= 0) & (J.truth_ix < J.nx) & (J.truth_iy >= 0)
          & (J.truth_iy < J.ny))
    ion[J.truth_ix[ok], J.truth_iy[ok]] = True
    dist = np.full((J.nx, J.ny), 3, np.int16)
    dist[ion] = 0
    reached = ion.copy()
    for d in (1, 2):
        grown = reached.copy()
        for ax in (0, 1):
            for sh in (-1, 1):
                grown |= np.roll(reached, sh, axis=ax)
        new = grown & ~reached
        dist[new] = d
        reached = grown
    return dist


def class_masks(cls: np.ndarray) -> dict:
    return {"ionised": cls == 0, "plus1": cls == 1, "plus2": cls == 2,
            "plus3_or_more": cls >= 3}


@algorithm("FRVariantResidual")
class FRVariantResidual(_JsonAlg):
    """``A x_truth - y`` and one prox-gradient step, over a configuration
    matrix of bin size, truth model and bin-index rule.

    See the module docstring for every definition.  No solve is run: the
    cost is one operator build, one forward, one adjoint and one prox per
    (configuration, arm).
    """

    reads = ("event", "readout_config", "block", "block_offset", "op",
             "support", "hits_view")
    writes = ("frvariant.residual",)

    def execute(self, store):
        J = _BasisJob(self, store)
        p = self.props
        conv = str(p.get("convention", "acq_edge"))
        delta = int(p.get("registration_delta", -1))
        configs = list(p.get("configs", DEFAULT_CONFIGS))
        arms = list(p.get("arms", DEFAULT_ARMS))
        support_mode = str(p.get("support", "full"))
        # OPTIONAL kernel override.  The operator's kernel is normally the
        # detector service's impact-averaged response, which is also what
        # FFTWarmStart uses to build the block.  A unipolar (center-pixel
        # only) response cannot be given to the detector service at all --
        # FFTWarmStart's burst processor needs a BIPOLAR induction template
        # and raises on one that is identically zero off the collection
        # pixel.  So the block keeps the full response and only the
        # convolution kernel is swapped here, which is what "run a sample
        # against the impact average of its own field response" means.
        kres = p.get("kernel_response")
        K1_over = None
        if kres:
            from ..deconv_workflow import prepare_field_response
            K1_over = np.asarray(prepare_field_response(
                str(kres), 1, normalized=False).integrated_response)
        probes = [tuple(int(v) for v in q) for q in p.get("probe_pixels", [])]
        out_npz = p.get("out_npz")
        dev, dt = J.comp.device, J.comp.dtype

        windows, metas = zs_windows(store, conv)
        kinds = np.array([m.kind for m in metas])
        rpx = np.array([w.px for w in windows], dtype=int)
        rpy = np.array([w.py for w in windows], dtype=int)
        cls = pixel_class(J)
        rcls = cls[np.clip(rpx, 0, J.nx - 1), np.clip(rpy, 0, J.ny - 1)]
        hits_support = np.asarray(store.get("support"))

        rec = {
            "truth_total_ke": J.truth_total,
            "convention": conv, "convention_label": CONV_LABEL[conv],
            "registration_delta": delta,
            "support_mode": support_mode,
            "block_shape": [J.nx, J.ny, J.nt],
            "block_offset": [float(v) for v in J.boff],
            "fine_ticks_in_block": int(J.nt_fine),
            "adc_hold_delay_ticks": int(J.B),
            "kernel": J.kernel_report(),
            "kernel_response_override": (None if not kres else str(kres)),
            "kernel_sum_used": float((J.K1 if K1_over is None
                                      else K1_over).sum()),
            "n_rows": int(len(windows)),
            "rows_by_kind": {k: int((kinds == k).sum())
                             for k in sorted(set(kinds))},
            "rows_by_pixel_class": {lab: int(m.sum()) for lab, m
                                    in class_masks(rcls).items() if m.any()},
            "probe_pixels": [],
            "configs": {},
        }
        npz: dict = {}

        # -- the probe pixels: identity, records, trigger times -------------
        hv = store.get("hits_view")
        hloc = np.asarray(hv.location)
        for (px, py) in probes:
            ip, iq = px - int(J.boff[0]), py - int(J.boff[1])
            inb = 0 <= ip < J.nx and 0 <= iq < J.ny
            sel = (hloc[:, 0] == px) & (hloc[:, 1] == py)
            rows = np.flatnonzero((rpx == ip) & (rpy == iq)) if inb else []
            info = {
                "pixel": [px, py], "block_index": [ip, iq],
                "in_block": bool(inb),
                "pixel_class": int(cls[ip, iq]) if inb else None,
                "n_hit_records": int(sel.sum()),
                "n_operator_rows": int(len(rows)),
                "trigger_ticks": [int(v) for v in hloc[sel, 2]],
                "first_latch_ticks": [int(v) for v in hloc[sel, 3]],
                "rearm_ticks": [int(v) for v in hloc[sel, 4]],
                "created_charge_ke": float(
                    J.truth_q[(J.truth_ix == ip) & (J.truth_iy == iq)].sum())
                if inb else 0.0,
            }
            if len(rows):
                # the first window's lower edge is -inf (the acquisition
                # start); report it clipped to the block, which is where
                # windows_to_sampling clips it too
                los = [max(windows[r].t_lo, 0.0) for r in rows]
                info["row_time_range_abs_ticks"] = [
                    float(min(los) + J.b0),
                    float(max(windows[r].t_hi for r in rows) + J.b0)]
                info["n_rows_with_open_lower_edge"] = int(sum(
                    1 for r in rows if not np.isfinite(windows[r].t_lo)))
            rec["probe_pixels"].append(info)

        # -- the configuration matrix ---------------------------------------
        for cfg in configs:
            lab = str(cfg["label"])
            c = int(cfg.get("cell_ticks", 1))
            truth_model = str(cfg.get("truth", "point"))
            mode = str(cfg.get("mode", "floor"))
            G = int(cfg.get("group_ticks", 5))
            t0 = time.time()
            op, kshape = build_operator(J, c, windows, dev, dt, K1=K1_over)
            if truth_model == "point":
                x_truth, tdiag = point_truth(J, op, c, delta,
                                             mode_shift(mode, c))
            elif truth_model == "group":
                if c != 1:
                    raise ValueError("truth 'group' is defined on the fine "
                                     f"grid; got cell_ticks={c}")
                # the grouping PHASE.  ``mode`` picks 0 (floor) or G//2
                # (round); ``group_shift`` names any of the G phases
                # explicitly, which is what a scan over the phase needs.
                gs = cfg.get("group_shift")
                gs = mode_shift(mode, G) if gs is None else int(gs)
                x_truth, tdiag = group_truth(J, op, G, delta, gs)
            else:
                raise ValueError(f"unknown truth model {truth_model!r}")

            y = op.d.cpu().numpy().astype(float)
            xt = op.to_tensor(np.ascontiguousarray(x_truth))
            pred = op.forward(xt).cpu().numpy().astype(float)
            res = pred - y
            ny_ = max(np.linalg.norm(y), 1e-30)

            # how much of the sampling weight is a partial-bin fraction
            w = op._weights.cpu().numpy()
            frac = np.abs(w - np.rint(w)) > 1e-9

            if support_mode == "hits":
                supp = J.support_on_basis(op, c)
            else:
                supp = np.ones(op.q_shape)
            st = op.to_tensor(np.ascontiguousarray(supp.astype(float)))
            hs = J.support_on_basis(op, c)
            hs_t = op.to_tensor(np.ascontiguousarray(hs.astype(float)))

            cm = class_masks(cls)
            block = {
                "cell_ticks": c, "truth_model": truth_model,
                "bin_index_rule": mode,
                "bin_index_shift_ticks": mode_shift(
                    mode, G if truth_model == "group" else c),
                "round_equals_floor": bool(
                    mode_shift("round", c) == 0 and truth_model == "point"),
                "kernel_bin_ticks": c,
                "kernel_shape": [int(v) for v in kshape],
                "q_shape": [int(v) for v in op.q_shape],
                "truth": tdiag,
                "sum_y_ke": float(y.sum()),
                "sum_y_over_truth": float(y.sum() / J.truth_total),
                "sum_A_x_truth_ke": float(pred.sum()),
                "sum_A_x_truth_over_y": float(pred.sum() / max(y.sum(), 1e-30)),
                "rel_residual": float(np.linalg.norm(res) / ny_),
                "sum_residual_ke": float(res.sum()),
                "sum_abs_residual_ke": float(np.abs(res).sum()),
                "row_fraction_partial_weights": float(frac.mean()),
                "row_weight_sum": float(w.sum()),
                "residual_by_row_kind": {
                    k: {"n_rows": int((kinds == k).sum()),
                        "sum_y_ke": float(y[kinds == k].sum()),
                        "sum_residual_ke": float(res[kinds == k].sum()),
                        "rel_residual": float(
                            np.linalg.norm(res[kinds == k])
                            / max(np.linalg.norm(y[kinds == k]), 1e-30))}
                    for k in sorted(set(kinds))},
                "residual_by_pixel_class": {
                    lab2: {"n_rows": int(m.sum()),
                           "sum_y_ke": float(y[m].sum()),
                           "sum_residual_ke": float(res[m].sum()),
                           "sum_abs_residual_ke": float(np.abs(res[m]).sum()),
                           "rel_residual": float(
                               np.linalg.norm(res[m])
                               / max(np.linalg.norm(y[m]), 1e-30)),
                           "share_of_squared_residual": float(
                               (res[m] ** 2).sum()
                               / max((res ** 2).sum(), 1e-30))}
                    for lab2, m in
                    ((l2, rcls == i if i < 3 else rcls >= 3)
                     for i, l2 in enumerate(CLASS_LABELS)) if m.any()},
                "n_support_cells": int(st.sum()),
                "support_fraction": float(st.mean()),
                "truth_charge_outside_hits_support_ke": float(
                    (xt * (1.0 - hs_t)).sum()),
                "truth_cells_outside_hits_support": int(
                    ((xt > 0) & (hs_t == 0)).sum()),
                "operator_build_s": time.time() - t0,
                "collected_fraction_of_truth": None,
                "arms": {},
            }

            npz[f"{lab}__resid"] = res.astype(np.float32)
            npz[f"{lab}__y"] = y.astype(np.float32)
            npz[f"{lab}__pred_truth"] = pred.astype(np.float32)

            occupied = xt > 0
            for arm in arms:
                alab = str(arm["label"])
                alpha = float(arm.get("alpha", 0.0))
                pos = bool(arm.get("positivity", True))
                prox = CoordProx(alpha, st) if pos else _SupportProx(st)
                term = DataFidelity(op)
                L = float(term.curvature())
                step = 1.0 / (1.05 * max(L, 1e-12))
                ctx = IterCtx(xt, op)
                g = torch.zeros_like(xt)
                term.grad_into(ctx, g)
                fval = float(term.value(ctx))
                qnext = prox(xt - step * g, step)
                d1 = qnext - xt
                gmap = (xt - qnext) / step
                block["arms"][alab] = {
                    "alpha_ke_per_cell": alpha, "positivity": pos,
                    "lipschitz": L, "step": step,
                    "objective_at_truth": fval,
                    "grad_norm_at_truth": float(
                        torch.linalg.vector_norm(g)),
                    "gradient_mapping_norm_at_truth": float(
                        torch.linalg.vector_norm(gmap)),
                    "grad_sum_ke": float(g.sum()),
                    "grad_min_ke": float(g.min()),
                    "grad_max_ke": float(g.max()),
                    "grad_sum_on_occupied_cells_ke": float(g[occupied].sum()),
                    "grad_by_class_ke": {
                        lab2: float(g.sum(dim=2).cpu().numpy()[m].sum())
                        for lab2, m in cm.items() if m.any()},
                    "first_step_total_ke": float(d1.sum()),
                    "first_step_total_over_truth": float(
                        d1.sum() / J.truth_total),
                    "first_step_abs_ke": float(d1.abs().sum()),
                    "first_step_on_occupied_cells_ke": float(
                        d1[occupied].sum()),
                    "first_step_by_class_ke": {
                        lab2: float(d1.sum(dim=2).cpu().numpy()[m].sum())
                        for lab2, m in cm.items() if m.any()},
                    "q_next_total_ke": float(qnext.sum()),
                    "q_next_over_truth": float(qnext.sum() / J.truth_total),
                    "n_cells_truth_occupies": int(occupied.sum()),
                    "n_cells_first_step_adds": int((d1 > 0).sum()),
                    "n_cells_first_step_removes": int((d1 < 0).sum()),
                    "n_cells_q_next_nonzero": int((qnext > 0).sum()),
                }
                npz[f"{lab}__{alab}__grad_pad"] = \
                    g.sum(dim=2).cpu().numpy().astype(np.float32)
                npz[f"{lab}__{alab}__step1_pad"] = \
                    d1.sum(dim=2).cpu().numpy().astype(np.float32)
                for (px, py) in probes:
                    ip, iq = px - int(J.boff[0]), py - int(J.boff[1])
                    if not (0 <= ip < J.nx and 0 <= iq < J.ny):
                        continue
                    k = f"{px}_{py}"
                    npz[f"{lab}__{alab}__grad_cells__{k}"] = \
                        g[ip, iq].cpu().numpy().astype(np.float32)
                    npz[f"{lab}__{alab}__step1_cells__{k}"] = \
                        d1[ip, iq].cpu().numpy().astype(np.float32)
                    npz[f"{lab}__{alab}__qnext_cells__{k}"] = \
                        qnext[ip, iq].cpu().numpy().astype(np.float32)
                print(f"[FRVariantResidual] {lab} {alab}: "
                      f"rel_resid {block['rel_residual']:.4g} L {L:.4g} "
                      f"|grad| {block['arms'][alab]['grad_norm_at_truth']:.4g} "
                      f"first step {float(d1.sum()):+.2f} ke "
                      f"({100 * float(d1.sum()) / J.truth_total:+.4f} %)")
                del g, qnext, d1, gmap, ctx, prox, term

            # the per-cell measurement gain c = A^T 1: the fraction of a
            # cell's charge that the recorded windows actually collect.  It
            # is the honest answer to "does this window see this charge",
            # which the cell's POSITION on a peak-aligned axis does not give:
            # the kernel delivers 87.4% of a charge's signal at or before its
            # peak, so a window can collect most of the charge of a cell
            # drawn to the right of its edge.
            gain = op.measurement_gain()

            # per-probe truth series and row tables, on the absolute axis
            for (px, py) in probes:
                ip, iq = px - int(J.boff[0]), py - int(J.boff[1])
                if not (0 <= ip < J.nx and 0 <= iq < J.ny):
                    continue
                k = f"{px}_{py}"
                npz[f"{lab}__truth_cells__{k}"] = \
                    x_truth[ip, iq].astype(np.float32)
                npz[f"{lab}__gain_cells__{k}"] = \
                    gain[ip, iq].cpu().numpy().astype(np.float32)
                sel = np.flatnonzero((rpx == ip) & (rpy == iq))
                # split the prediction of this pixel's rows into the part
                # driven by the pixel's OWN created charge and the part
                # induced by every other pixel's electrons.  Without this a
                # "recorded / created" ratio for one pixel divides charge
                # recorded from one set of electrons by the charge created
                # by a different set.
                xl = np.zeros_like(x_truth)
                xl[ip, iq] = x_truth[ip, iq]
                pred_local = op.forward(
                    op.to_tensor(np.ascontiguousarray(xl))
                ).cpu().numpy().astype(float)
                tab = np.array(
                    [[windows[r].t_lo + J.b0, windows[r].t_hi + J.b0,
                      y[r], pred[r], pred_local[r]] for r in sel],
                    dtype=np.float64) if len(sel) else np.zeros((0, 5))
                npz[f"{lab}__rows__{k}"] = tab.astype(np.float64)
            # the absolute tick of cell 0 and the cell width, so every series
            # above can be put on an absolute time axis without guessing
            npz[f"{lab}__cell_axis"] = np.array(
                [J.b0, c, op.q_shape[2]], dtype=np.int64)
            rec["configs"][lab] = block
            # each configuration builds its own operator; the kernel FFT and
            # the block-sized temporaries are ~1 GB apiece on a dense
            # fixed_interval sample, so the previous one must be released
            # before the next is built or a 12 GB card runs out
            block["collected_fraction_of_truth"] = float(
                (gain * xt).sum() / max(float(xt.sum()), 1e-30))
            del op, xt, pred, st, hs_t, occupied, gain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[FRVariantResidual] {lab} done in "
                  f"{time.time() - t0:.1f} s")

        if out_npz:
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **npz)
            print(f"[FRVariantResidual] wrote {out_npz}")
        self._emit(store, rec)
        store.put("frvariant.residual", rec)
        return rec
