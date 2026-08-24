"""Fit-grid diagnostics: allocation, stationarity, spectrum, gain, geometry.

Ports the reusable half of the ``noiseless_closure_round`` and
``operator_mechanism`` probes.  Everything here works from store products --
``op``, ``row_meta``, ``solve.q``, ``truth.q`` -- so no window list is rebuilt
and every quantity is of the operator the solver used.
"""
from __future__ import annotations

import numpy as np
import torch

from ..fwk.component import Algorithm, algorithm
from .truth_algs import _resid_conventions

LATCH_KINDS = ("lumped", "remainder", "diff")


def _pkq_scan(store, op, rm, t, lo, hi, boff, B, kinds=LATCH_KINDS):
    """The kernel-peak offset, chosen not assumed.

    A latch instant maps to a q bin through the response peak; the offset is a
    property of the kernel, but which integer it is was settled separately
    (``pkq_check/``) and the archived scripts each carried their own literal.
    Scanning it here and reporting the winner makes the choice visible.

    ``kinds`` MUST be the same row set the caller then analyses.  Scanning over
    a wider set picks a different winner -- measured: over all latch kinds the
    winner is 125 where over ``lumped`` alone it is 126, and that one bin moves
    the usable window count from 170 to 177.
    """
    best, best_s = None, -1.0
    nt = t.shape[2]
    spans = []
    for r in range(op.n_data):
        if rm["kind"][r] not in kinds:
            continue
        spans.append((int(rm["px"][r]), int(rm["py"][r]),
                      int(np.floor(max(float(rm["t_lo"][r]), 0.0) / B)),
                      int(np.floor((float(rm["t_hi"][r]) - 1) / B))))
    for pkq in range(lo, hi + 1):
        s = 0.0
        for X, Y, j0, j1 in spans:
            a, b = j0 - pkq, j1 - pkq
            # SKIP a window whose span leaves the grid; clipping it instead
            # scores a partial span and biases the winner
            if a >= 0 and b < nt and b >= a:
                s += float(t[X, Y, a:b + 1].sum())
        if s > best_s:
            best, best_s = pkq, s
    if best in (lo, hi):
        print("[pkq_scan] WARNING: winner %d is at the scan boundary [%d, %d]"
              % (best, lo, hi))
    return best, best_s


class _FitGridAlg(Algorithm):
    """Shared setup for the fit-grid diagnostics."""

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",)

    def _grids(self, store):
        op = store.get("op")
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        t = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        if q.shape != t.shape:
            raise ValueError(f"solve.q {q.shape} vs {self.prefix}.q {t.shape}")
        return op, q, t, q - t


@algorithm("WindowAllocation")
class WindowAllocation(_FitGridAlg):
    """Is the charge conserved over the bins a window covers, and displaced?

    Ports ``lumped_span{,_bykind}.py``, ``lumped_alloc.py`` and
    ``window_pair.py`` -- four scripts asking one question on the fit grid.

    For each latch-bounded row the window's span in q bins is
    ``[floor(t_lo/B), floor(t_hi/B)] - PKQ``.  Two statements are reported:

    * **span conservation** -- reco/truth summed over the span, so a solution
      that moves charge *within* the window scores 1;
    * **end allocation** -- the pair (end bin, end bin - 1), where the
      within-bin model has to guess, with the correlation between the two
      deltas.  A near -1 correlation is charge moved between the pair rather
      than created.

    Props: ``pkq`` (int, or ``scan`` with ``pkq_range``), ``kinds``,
    ``truth_floor`` (0.5 ke), ``truth_prefix``.
    """

    reads = ("op", "row_meta", "readout_config", "block_offset", "solve.q")
    writes = ("windowalloc.summary",)

    def execute(self, store):
        op, q, t, delta = self._grids(store)
        rm = store.get("row_meta")
        rc = store.get("readout_config")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        B = float(int(rc.adc_hold_delay) // S)
        kinds = tuple(self.props.get("kinds", LATCH_KINDS))
        floor = float(self.props.get("truth_floor", 0.5))
        pkq_cfg = self.props.get("pkq", "scan")
        if pkq_cfg == "scan":
            lo, hi = self.props.get("pkq_range", (118, 135))
            pkq, pkq_score = _pkq_scan(store, op, rm, t, int(lo), int(hi),
                                       boff, B, kinds)
        else:
            pkq, pkq_score = int(pkq_cfg), None

        nt = t.shape[2]
        spans, ratios, cshift = [], [], []
        lags = {m: ([], []) for m in range(-3, 4)}
        pair_d, pair_p, pair_te = [], [], []
        n_kind = 0
        for r in range(op.n_data):
            if rm["kind"][r] not in kinds:
                continue
            n_kind += 1
            X, Y = int(rm["px"][r]), int(rm["py"][r])
            j0 = int(np.floor(max(float(rm["t_lo"][r]), 0.0) / B)) - pkq
            # the upper edge is EXCLUSIVE: floor((t_hi - 1)/B), so a window
            # ending exactly on a boundary does not claim the next bin
            j1 = int(np.floor((float(rm["t_hi"][r]) - 1) / B)) - pkq
            if j0 < 0 or j1 >= nt or j1 < j0:
                continue
            tr = t[X, Y, j0:j1 + 1]
            re = q[X, Y, j0:j1 + 1]
            if tr.sum() + re.sum() < 1.0:
                continue
            spans.append(j1 - j0 + 1)
            if tr.sum() > floor:
                ratios.append(float(re.sum() / tr.sum()))
            if tr.sum() > floor and re.sum() > floor and j1 > j0:
                pos = np.arange(j1 - j0 + 1)
                cshift.append(float((re * pos).sum() / re.sum()
                                    - (tr * pos).sum() / tr.sum()))
            for m in range(-3, 4):
                jm = j0 + m
                if 0 <= jm < nt:
                    lags[m][0].append(delta[X, Y, j0])
                    lags[m][1].append(delta[X, Y, jm])
            if 1 <= j1 < nt:
                pair_d.append(delta[X, Y, j1])
                pair_p.append(delta[X, Y, j1 - 1])
                pair_te.append(t[X, Y, j1])
        rec = {"pkq": pkq, "PKQ": pkq,
               "pkq_chosen_by": ("scan" if pkq_cfg == "scan" else "prop"),
               "pkq_truth_on_spans_ke": pkq_score,
               "kinds": list(kinds), "truth_floor_ke": floor,
               "n_kind": n_kind, "n_lumped": n_kind, "n_used": len(spans),
               "n_windows": len(spans),
               "span_bins_mean": float(np.mean(spans)) if spans else None,
               "sum_ratio_median": float(np.median(ratios)) if ratios else None,
               "sum_ratio_mean": float(np.mean(ratios)) if ratios else None,
               "centroid_shift_mean_bins": (float(np.mean(cshift))
                                            if cshift else None),
               "lag_corr": {}}
        for m, (a_, b_) in lags.items():
            a_, b_ = np.asarray(a_), np.asarray(b_)
            rec["lag_corr"][str(m)] = (
                float(np.corrcoef(a_, b_)[0, 1])
                if a_.size > 5 and a_.std() > 0 and b_.std() > 0
                else None)
        if spans:
            rec["span_conservation"] = {
                "n": len(ratios),
                "median": rec["sum_ratio_median"], "mean": rec["sum_ratio_mean"],
                "frac_gt_1.05": float(np.mean(np.asarray(ratios) > 1.05))
                if ratios else None}
        if len(pair_d) >= 3:
            de, dp = np.asarray(pair_d), np.asarray(pair_p)
            rec["end_pair"] = {
                "n": de.size, "sum_end_ke": float(de.sum()),
                "sum_prev_ke": float(dp.sum()),
                "sum_pair_ke": float((de + dp).sum()),
                "corr_end_prev": (float(np.corrcoef(de, dp)[0, 1])
                                  if de.std() and dp.std() else None),
                "median_truth_end_ke": float(np.median(pair_te))}
        sc = rec.get("span_conservation")
        ep = rec.get("end_pair")
        print("[WindowAllocation] PKQ=%s  %d/%d used  span ratio med %s  "
              "centroid shift %s  end pair corr %s  pair sum %s ke"
              % (pkq, rec["n_used"], n_kind,
                 "n/a" if not sc else "%.4f" % sc["median"],
                 _f(rec["centroid_shift_mean_bins"]),
                 "n/a" if not ep else "%+.3f" % (ep["corr_end_prev"] or 0),
                 "n/a" if not ep else "%+.2f" % ep["sum_pair_ke"]))
        self.put(store, "windowalloc.summary", rec)


@algorithm("Stationarity")
class Stationarity(_FitGridAlg):
    """Is the shipped solution stationary?  The KKT test at q_hat.

    Ports ``kkt_check.py``.  For
    ``min 1/2||A q - d||^2 + sum_v alpha_v q_v`` over ``q >= 0`` supported on
    ``S``, with ``g = -A^T(A q_hat - d) = A^T e``:

    * on the ACTIVE set (``q_hat > 0``) stationarity needs ``g_v = alpha_v``;
    * on the INACTIVE part of the support it needs ``g_v <= alpha_v``.

    Note what this is NOT: ``A^T e`` evaluated at the TRUTH is not any solver
    iterate, so its size says nothing about convergence.  This is evaluated at
    the solution, which is the only place the question has an answer.

    Props: ``alpha`` (the final ladder weight, default 0.3), ``eps``
    (positivity threshold, 1e-6), ``truth_prefix``.
    """

    reads = ("op", "support", "solve.q")
    writes = ("kkt.summary",)

    def __init__(self, **props):
        Algorithm.__init__(self, **props)   # no truth needed
        self.prefix = str(props.get("truth_prefix", "truth"))

    def execute(self, store):
        op = store.get("op")
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        sup = np.asarray(store.get("support")).astype(bool)
        alpha = float(self.props.get("alpha", 0.3))
        eps = float(self.props.get("eps", 1e-6))
        r = op.forward(op.to_tensor(q)).detach() - op.d
        g = -op.adjoint(r).detach().cpu().numpy().astype(np.float64)
        act = (q > eps) & sup
        ina = (~(q > eps)) & sup
        rec = {"alpha": alpha, "n_active": int(act.sum()),
               "n_inactive_in_support": int(ina.sum()),
               "n_support": int(sup.sum())}
        rec["nnz"] = int(act.sum())
        if act.any():
            a = np.abs(g[act] - alpha)
            rec["active_absdev"] = {"median": float(np.median(a)),
                                    "p90": float(np.percentile(a, 90)),
                                    "max": float(a.max())}
        if ina.any():
            v = np.clip(g[ina] - alpha, 0.0, None)   # violation only
            rec["inactive_violation"] = {"frac_gt_0": float((v > 0).mean()),
                                         "p90": float(np.percentile(v, 90)),
                                         "max": float(v.max())}
        print("[Stationarity] alpha=%.2f  nnz %d  active |g-a| med %s  "
              "inactive %d, %s violating"
              % (alpha, rec["nnz"],
                 "n/a" if "active_absdev" not in rec
                 else "%.4g" % rec["active_absdev"]["median"],
                 rec["n_inactive_in_support"],
                 "n/a" if "inactive_violation" not in rec
                 else "%.3f%%" % (100 * rec["inactive_violation"]["frac_gt_0"])))
        self.put(store, "kkt.summary", rec)


@algorithm("ColumnGain")
class ColumnGain(_FitGridAlg):
    """The measurement gain per voxel, and what it does to the total.

    Ports ``gain_from_colsum.py`` and the ``gain_audit`` family.  With
    ``c_v = sum_r A_rv = (A^T 1)_v``, for ANY q

        sum_r (A q)_r  =  sum_v c_v q_v

    exactly -- asserted here.  So the data total is the c-weighted charge, and
    a row residual can raise the booked total either by adding charge or by
    moving it to voxels of lower ``c``.  The two are separated by comparing
    ``sum delta`` with ``sum c*delta / c_bar``.

    Props: ``truth_prefix``, ``atol`` (1e-3 ke for the identity).
    """

    reads = ("op", "support", "solve.q")
    writes = ("colgain.summary",)

    def execute(self, store):
        op, q, t, delta = self._grids(store)
        sup = np.asarray(store.get("support")).astype(bool)
        atol = float(self.props.get("atol", 1e-3))
        ones = torch.ones(op.n_data, dtype=op.dtype, device=op.device)
        c = op.adjoint(ones).detach().cpu().numpy().astype(np.float64)
        # the identity, on the truth: both sides are exact sums
        lhs = float(op.forward(op.to_tensor(t)).detach().sum())
        rhs = float((c * t).sum())
        if abs(lhs - rhs) > atol * max(abs(lhs), 1.0):
            raise AssertionError(
                f"sum(A q) = {lhs:.6g} against sum(c q) = {rhs:.6g}")
        has_t = t > 0
        cw = (float((c * t).sum() / t.sum()) if t.sum() else float("nan"))
        on = c[has_t]
        rec = {"identity": "sum_r (A q)_r == sum_v c_v q_v",
               "identity_abs_ke": abs(lhs - rhs),
               # the charge-weighted mean gain: the denominator that turns a row
               # residual into a charge, and it is NOT the median
               "c_bar_qweighted": cw,
               "c_median": float(np.median(on)) if on.size else None,
               "c_q25": float(np.percentile(on, 25)) if on.size else None,
               "c_q75": float(np.percentile(on, 75)) if on.size else None,
               "n_voxels_with_truth": int(has_t.sum()),
               "Rres": float((np.asarray(op.d.detach().cpu().numpy(),
                                         np.float64).ravel()
                              - np.asarray(op.forward(op.to_tensor(t)).detach()
                                           .cpu().numpy(), np.float64).ravel()
                              ).sum()),
               "sum_delta_ke": float(delta.sum()),
               "c_weighted_delta_ke": float((c * delta).sum())}
        # If c were constant at c_bar, a residual R_res could only be absorbed
        # by adding R_res / c_bar of total charge, so the predicted gain is
        # 1 / c_bar in ke of charge per ke of residual.
        rec["G_pred"] = (1.0 / cw) if cw else None
        # The MEASURED gain is (Q_real - Q_synth) / R_res, which needs BOTH
        # arms of the campaign.  A single-event algorithm cannot form it, so
        # Q_hat is published and the caller divides.  Inventing a one-arm
        # substitute here would give a number that looks like G_meas and is not.
        rec["Q_hat_ke"] = float(q.sum())
        rec["Q_truth_ke"] = float(t.sum())
        rec["G_meas"] = None
        rec["G_meas_note"] = ("(Q_real - Q_synth) / Rres; needs the synth arm, "
                             "form it from two runs' Q_hat_ke")
        cbar = cw
        m = sup & (np.abs(delta) > 0)
        if m.sum() >= 3 and c[m].std() and delta[m].std():
            rec["corr_delta_c"] = float(np.corrcoef(c[m], delta[m])[0, 1])
            rec["n_corr"] = int(m.sum())
        print("[ColumnGain] c_bar(q) %.4g  c_med %.4g  Rres %+.2f ke  "
              "G_pred %s  G_meas %s  corr(delta,c) %s"
              % (cbar, rec["c_median"] or float("nan"), rec["Rres"],
                 _f(rec["G_pred"]), "two-arm",
                 "n/a" if "corr_delta_c" not in rec
                 else "%+.3f" % rec["corr_delta_c"]))
        self.put(store, "colgain.summary", rec)


@algorithm("ResidualSpectrum")
class ResidualSpectrum(_FitGridAlg):
    """Where does the row residual sit in the operator's spectrum?

    Ports ``resid_spectrum.py`` and the mode-projection half of
    ``operator_mechanism/{projection_test,spectrum_test}.py``.  The row Gram
    ``G = A A^T`` (or restricted) is eigen-decomposed and the residual
    ``e = d - A q_truth`` is projected onto its eigenvectors, so the question
    "is the error in directions the data constrain, or in the weak ones" gets
    a number instead of an argument.

    Reports the cumulative share of ``||e||^2`` against the cumulative share of
    ``tr G``: if the error concentrates where the spectrum is weak, the first
    runs ahead of the second.

    Props: ``restrict`` (``free``|``support``), ``max_dim`` (4000),
    ``quantiles``, ``truth_prefix``.
    """

    reads = ("op", "solve.q")
    writes = ("residspec.summary",)

    def execute(self, store):
        op, q, t, _ = self._grids(store)
        restrict = str(self.props.get("restrict", "free"))
        max_dim = int(self.props.get("max_dim", 4000))
        n = int(op.n_data)
        if n > max_dim:
            raise ValueError(f"{n} rows above max_dim {max_dim}; this is the "
                             "dense route by design")
        P = None
        if restrict == "support":
            P = np.asarray(store.get("support")).astype(np.float64)
        G = np.zeros((n, n))
        for r in range(n):
            b = np.zeros(n)
            b[r] = 1.0
            adj = op.adjoint(op.to_tensor(b)).detach().cpu().numpy()
            if P is not None:
                adj = adj * P
            G[:, r] = op.forward(op.to_tensor(adj)).detach().cpu().numpy()
        G = 0.5 * (G + G.T)
        w, V = np.linalg.eigh(G)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        sw, weighted = _resid_conventions(op)
        d = np.asarray(op.d.detach().cpu().numpy(), np.float64).ravel()
        Aq = np.asarray(op.forward(op.to_tensor(t)).detach().cpu().numpy(),
                        np.float64).ravel()
        if weighted:
            d, Aq = d / sw, Aq / sw
        e = d - Aq
        proj = V.T @ e
        p2 = proj ** 2
        tot = float(p2.sum())
        wpos = np.clip(w, 0.0, None)
        cw = np.cumsum(wpos) / max(wpos.sum(), 1e-300)
        ce = np.cumsum(p2) / max(tot, 1e-300)
        A1 = np.asarray(op.forward(op.to_tensor(np.ones(op.q_shape)))
                        .detach().cpu().numpy(), np.float64).ravel()
        a1u = V.T @ A1
        qs = [float(x) for x in self.props.get("quantiles", (0.5, 0.9, 0.99))]
        rec = {"restrict": restrict, "n_rows": n,
               "lambda_max": float(w[0]), "lambda_min": float(w[-1]),
               "trace": float(wpos.sum()),
               "R_res_ke": float(e.sum()), "e_norm_sq": tot,
               "norm_e": float(np.linalg.norm(e)),
               "R_res": float(e.sum()), "lam_max": float(w[0]),
               "check_sum_proj2_over_norme2": float(tot / max(
                   float(e @ e), 1e-300)),
               "tol_scan": {}, "modes_for_trace_frac": {},
               "modes_for_error_frac": {}, "error_share_at_trace_frac": {}}
        for tol in [float(x) for x in self.props.get(
                "tolerances", (1e-6, 1e-8, 1e-10))]:
            keep = w > tol * w[0]
            rec["tol_scan"]["%g" % tol] = {
                "n_modes_kept": int(keep.sum()),
                "n_null": int((~keep).sum()),
                "frac_e2_in_null": float(p2[~keep].sum() / max(tot, 1e-300)),
                "T_ke": float((proj[keep] * a1u[keep] / w[keep]).sum())}
        for f in qs:
            rec["modes_for_trace_frac"][str(f)] = int(np.searchsorted(cw, f) + 1)
            rec["modes_for_error_frac"][str(f)] = int(np.searchsorted(ce, f) + 1)
            k = int(np.searchsorted(cw, f))
            rec["error_share_at_trace_frac"][str(f)] = float(ce[min(k, n - 1)])
        # the blunt version of the same statement
        half = n // 2
        rec["error_share_strong_half"] = float(ce[half - 1]) if half else None
        print("[ResidualSpectrum] %s  n=%d  lmax %.4g lmin %.4g  "
              "modes for 90%% of trace %d, for 90%% of error %d  "
              "(error in strong half %.3f)"
              % (restrict, n, rec["lambda_max"], rec["lambda_min"],
                 rec["modes_for_trace_frac"]["0.9"],
                 rec["modes_for_error_frac"]["0.9"],
                 rec["error_share_strong_half"] or float("nan")))
        self.put(store, "residspec.summary", rec)


@algorithm("LeadBinLag")
class LeadBinLag(Algorithm):
    """Kernel lag and within-bin rise at a window's partial bins.

    Ports ``lead_bin_lag.py``, which the note cites as ``app:repro:leadlag``.
    For each row's leading and trailing partial bin, and for the point charges
    contributing to it, the kernel lag ``m = b - k_j`` and the within-bin rise
    -- the response integral over the last third of the bin minus the first
    third, for a unit charge released at ``k_j B`` -- are charge-weighted over
    all contributing charges.  A positive rise is exactly the condition that
    makes the exact integral over the covered piece exceed the length fraction,
    which is why the leading partial bin's sign is fixed by the kernel and not
    by a convention.

    Props: ``convention`` (must match the truth being explained),
    ``max_lag_bins`` (default: the kernel's coarse-bin extent),
    ``truth_prefix``.
    """

    reads = ("event", "readout_config", "op", "block_offset", "row_meta")
    writes = ("leadlag.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.convention = str(props.get("convention", "round"))
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",)

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        ev = store.get("event")
        rm = store.get("row_meta")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        B = int(rc.adc_hold_delay) // S
        prep = self.services["detector"].prepared(B)
        R = np.asarray(prep.full_response, dtype=np.float64)
        Kint = np.asarray(prep.integrated_response, dtype=np.float64)
        # the archived script used a hard 130 here; the kernel's own coarse-bin
        # extent is the principled bound and is the default, with the literal
        # reachable so an archived number can be reproduced exactly
        kt = int(self.props.get("max_lag_bins", Kint.shape[2]))
        kx, ky, nfine = R.shape
        cx, cy = (kx - 1) // 2, (ky - 1) // 2
        CS = np.concatenate([np.zeros((kx, ky, 1)), np.cumsum(R, axis=-1)], -1)

        nxq, nyq, ntq = op.q_shape
        el = np.asarray(ev.effq.location)
        eq = np.asarray(ev.effq.data, dtype=np.float64)[:, -1]
        jx = el[:, 0].astype(np.int64) - int(boff[0])
        jy = el[:, 1].astype(np.int64) - int(boff[1])
        tf = el[:, 2] - boff[2]
        off = {"round": 0.5, "floor": 0.0, "shift": -0.5}[self.convention]
        jk = np.floor(tf / B + off).astype(np.int64)
        keep = ((jx >= 0) & (jx < nxq) & (jy >= 0) & (jy < nyq)
                & (jk >= 0) & (jk < ntq))
        jx, jy, jk, Q = jx[keep], jy[keep], jk[keep], eq[keep]
        by_pix: dict[tuple[int, int], list[int]] = {}
        for i in range(jx.size):
            by_pix.setdefault((int(jx[i]), int(jy[i])), []).append(i)

        rows = op._rows.cpu().numpy()
        cols = op._cols.cpu().numpy()
        wts = op._weights.cpu().numpy().astype(np.float64)
        sw, weighted = _resid_conventions(op)
        if weighted:
            wts = wts / sw[rows]
        ntb = op.block_shape[2]
        per_row: dict[int, dict[int, float]] = {}
        for r, c, w in zip(rows, cols, wts):
            per_row.setdefault(int(r), {})[int(c) % ntb] = float(w)

        def third(dx, dy, lo, hi):
            a = np.clip(lo, 0, nfine).astype(int)
            b = np.clip(hi, 0, nfine).astype(int)
            return CS[dx + cx, dy + cy, b] - CS[dx + cx, dy + cy, a]

        out = {}
        for which, pick in (("lead", min), ("trail", max)):
            lag_w, rise_w, wsum = [], [], []
            for r in range(op.n_data):
                bl = per_row.get(r)
                if not bl or rm["kind"][r] != "lumped":
                    continue
                b0 = pick(bl)
                # only a PARTIALLY covered bin is the subject: a fully covered
                # bin has weight 1 and the model guesses nothing there
                if abs(bl[b0] - 1.0) < 1e-9:
                    continue
                sel: list[int] = []
                for ddx in range(-cx, cx + 1):
                    for ddy in range(-cy, cy + 1):
                        got = by_pix.get((int(rm["px"][r]) - ddx,
                                          int(rm["py"][r]) - ddy))
                        if got:
                            sel.extend(got)
                if not sel:
                    continue
                s = np.asarray(sel)
                dx, dy = int(rm["px"][r]) - jx[s], int(rm["py"][r]) - jy[s]
                m = b0 - jk[s]
                good = (m >= 0) & (m < kt)      # the kernel reaches these only
                if not good.any():
                    continue
                s = s[good]
                dx, dy, m = dx[good], dy[good], m[good]
                t0 = jk[s] * float(B)
                lo = b0 * float(B) - t0
                first = third(dx, dy, lo, lo + B / 3.0)
                last = third(dx, dy, lo + 2.0 * B / 3.0, lo + B)
                q = Q[s]
                lag_w.append(float((q * m).sum()))
                rise_w.append(float((q * (last - first)).sum()))
                wsum.append(float(q.sum()))
            ws = np.asarray(wsum)
            tot = float(ws.sum())
            out[which] = {
                "n_rows": int(ws.size),
                "charge_ke": tot,
                "mean_lag_bins": (float(np.sum(lag_w) / tot) if tot else None),
                "mean_rise_ke": (float(np.sum(rise_w) / tot) if tot else None),
                "rise_positive_frac": (float(np.mean(np.asarray(rise_w) > 0))
                                       if ws.size else None)}
        rec = {"convention": self.convention, "bin_ticks": B,
               "max_lag_bins": kt, "kernel_bins": int(Kint.shape[2]), **out}
        print("[LeadBinLag] lead: lag %s rise %s (%.0f%% positive) | "
              "trail: lag %s rise %s"
              % (_f(out["lead"]["mean_lag_bins"]), _f(out["lead"]["mean_rise_ke"]),
                 100 * (out["lead"]["rise_positive_frac"] or 0),
                 _f(out["trail"]["mean_lag_bins"]),
                 _f(out["trail"]["mean_rise_ke"])))
        self.put(store, "leadlag.summary", rec)


def _f(x):
    return "n/a" if x is None else "%+.4f" % x


@algorithm("ChainPosition")
class ChainPosition(_FitGridAlg):
    """Mid-chain versus final latched bins.

    Ports ``chain_position.py``.  A bin whose pixel has a latch both before and
    after it is constrained from both sides; the last latched bin of a sequence
    is constrained from one.  If the over-book is an end effect the two
    populations must differ, and if it is not they must not.

    Props: ``pkq``, ``truth_floor``, ``truth_prefix``.
    """

    reads = ("op", "row_meta", "readout_config", "block_offset", "solve.q")
    writes = ("chainpos.summary",)

    def execute(self, store):
        op, q, t, delta = self._grids(store)
        rm = store.get("row_meta")
        rc = store.get("readout_config")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        B = float(int(rc.adc_hold_delay) // S)
        floor = float(self.props.get("truth_floor", 0.5))
        pkq_cfg = self.props.get("pkq", "scan")
        if pkq_cfg == "scan":
            lo, hi = self.props.get("pkq_range", (118, 135))
            pkq, _ = _pkq_scan(store, op, rm, t, int(lo), int(hi), boff, B)
        else:
            pkq = int(pkq_cfg)

        latch: dict[tuple[int, int], list[int]] = {}
        for r in range(op.n_data):
            if rm["kind"][r] not in LATCH_KINDS:
                continue
            j = int(np.floor(float(rm["t_hi"][r]) / B)) - pkq
            latch.setdefault((int(rm["px"][r]), int(rm["py"][r])), []).append(j)
        groups = {"mid": [], "final": []}
        for (X, Y), js in latch.items():
            js = sorted(set(js))
            for i, j in enumerate(js):
                if not (0 <= j < t.shape[2]) or t[X, Y, j] < floor:
                    continue
                key = "final" if i == len(js) - 1 else "mid"
                groups[key].append(float(q[X, Y, j] / t[X, Y, j]))
        rec = {"pkq": pkq, "truth_floor_ke": floor, "groups": {}}
        for k, v in groups.items():
            if not v:
                rec["groups"][k] = None
                continue
            a = np.asarray(v)
            rec["groups"][k] = {"n": a.size, "median": float(np.median(a)),
                                "mean": float(a.mean()),
                                "frac_gt_1.05": float((a > 1.05).mean())}
        g = rec["groups"]
        if g.get("mid") and g.get("final"):
            rec["final_minus_mid_median"] = (g["final"]["median"]
                                             - g["mid"]["median"])
        print("[ChainPosition] mid n=%s med=%s | final n=%s med=%s"
              % (g["mid"]["n"] if g.get("mid") else "-",
                 "%.4f" % g["mid"]["median"] if g.get("mid") else "-",
                 g["final"]["n"] if g.get("final") else "-",
                 "%.4f" % g["final"]["median"] if g.get("final") else "-"))
        self.put(store, "chainpos.summary", rec)


@algorithm("GeometryOverestimate")
class GeometryOverestimate(_FitGridAlg):
    """Does the over- or under-book follow the readout geometry?

    Ports ``operator_mechanism/geom_overest2.py`` and ``trigdist.py``.  For
    every support voxel, the signed distance in bins to its pixel's nearest
    latch instant, against ``delta = q_hat - q_truth``.  A correlation says the
    error is placed by where the readout sampled, not by the charge.

    Props: ``pkq``, ``truth_prefix``, ``max_dist`` (bins to profile, 6).
    """

    reads = ("op", "row_meta", "readout_config", "block_offset", "support",
             "solve.q")
    writes = ("geomover.summary",)

    def execute(self, store):
        op, q, t, delta = self._grids(store)
        rm = store.get("row_meta")
        rc = store.get("readout_config")
        sup = np.asarray(store.get("support")).astype(bool)
        boff = np.asarray(store.get("block_offset"), dtype=float)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        B = float(int(rc.adc_hold_delay) // S)
        maxd = int(self.props.get("max_dist", 6))
        pkq_cfg = self.props.get("pkq", "scan")
        if pkq_cfg == "scan":
            lo, hi = self.props.get("pkq_range", (118, 135))
            pkq, _ = _pkq_scan(store, op, rm, t, int(lo), int(hi), boff, B)
        else:
            pkq = int(pkq_cfg)

        nt = t.shape[2]
        per_pix: dict[tuple[int, int], list[int]] = {}
        for r in range(op.n_data):
            if rm["kind"][r] not in LATCH_KINDS:
                continue
            j = int(np.floor(float(rm["t_hi"][r]) / B)) - pkq
            per_pix.setdefault((int(rm["px"][r]), int(rm["py"][r])), []).append(j)
        dist = np.full(t.shape, np.nan)
        for (X, Y), js in per_pix.items():
            if not (0 <= X < t.shape[0] and 0 <= Y < t.shape[1]):
                continue
            js = np.asarray(sorted(set(js)))
            k = np.arange(nt)
            i = np.searchsorted(js, k)
            i = np.clip(i, 1, js.size - 1) if js.size > 1 else np.zeros_like(i)
            lo_j = js[np.clip(i - 1, 0, js.size - 1)]
            hi_j = js[np.clip(i, 0, js.size - 1)]
            pick = np.where(np.abs(k - lo_j) <= np.abs(k - hi_j), lo_j, hi_j)
            dist[X, Y, :] = k - pick
        m = sup & np.isfinite(dist)
        rec = {"pkq": pkq, "n_voxels": int(m.sum()), "profile": []}
        if m.sum() >= 3:
            dd, de = dist[m], delta[m]
            if dd.std() and de.std():
                rec["corr_delta_dist"] = float(np.corrcoef(dd, de)[0, 1])
            rec["corr_delta_absdist"] = (
                float(np.corrcoef(np.abs(dd), de)[0, 1])
                if np.abs(dd).std() and de.std() else None)
            for k in range(-maxd, maxd + 1):
                sel = dd == k
                if sel.sum() < 3:
                    continue
                rec["profile"].append({"dist_bins": k, "n": int(sel.sum()),
                                       "mean_delta_ke": float(de[sel].mean()),
                                       "median_delta_ke": float(
                                           np.median(de[sel]))})
        print("[GeometryOverestimate] %d voxels  corr(delta, signed dist) %s  "
              "corr(delta, |dist|) %s"
              % (rec["n_voxels"],
                 _f(rec.get("corr_delta_dist")), _f(rec.get("corr_delta_absdist"))))
        self.put(store, "geomover.summary", rec)


@algorithm("CentroidError")
class CentroidError(_FitGridAlg):
    """Per-pixel time-centroid error, and whether neighbours share it.

    Ports the centroid half of ``noiseless_closure/precensor_eval.py`` and
    supplies the two arrival-time rows of `tab:isores-isoline`.

    For each pixel holding enough truth, the charge-weighted time centroid is
    formed on the fit grid for the reconstruction and for the truth, and

        dt(pixel) = centroid(reco) - centroid(truth)          [bins]

    Its spread says how far the solve moves charge in time. The **neighbour
    correlation** of ``dt`` is the entanglement test: near zero means each
    pixel is misplaced independently, and a large positive value means
    neighbouring pixels are misplaced *together*, which is the signature of a
    shared charge lump moved coherently through the response.

    Props: ``pixel_floor`` (truth charge a pixel needs to be scored, 1.0 ke),
    ``neighbours`` (``4`` or ``8``; default 4), ``truth_prefix``.
    """

    reads = ("op", "solve.q")
    writes = ("centroiderr.summary",)

    def execute(self, store):
        op, q, t, _ = self._grids(store)
        floor = float(self.props.get("pixel_floor", 1.0))
        nb = int(self.props.get("neighbours", 4))
        if nb not in (4, 8):
            raise ValueError("neighbours must be 4 or 8")
        k = np.arange(t.shape[2], dtype=np.float64)
        tw = t.sum(axis=2)
        qw = q.sum(axis=2)
        live = (tw > floor) & (qw > 0)
        ct = np.where(tw > 0, (t * k).sum(axis=2) / np.where(tw > 0, tw, 1), np.nan)
        cq = np.where(qw > 0, (q * k).sum(axis=2) / np.where(qw > 0, qw, 1), np.nan)
        dt = np.where(live, cq - ct, np.nan)
        v = dt[np.isfinite(dt)]
        rec = {"pixel_floor_ke": floor, "neighbours": nb,
               "n_pixels": int(v.size)}
        if v.size:
            rec["dt_bins"] = {"mean": float(v.mean()), "sd": float(v.std()),
                              "median": float(np.median(v)),
                              "abs_mean": float(np.abs(v).mean()),
                              "p90_abs": float(np.percentile(np.abs(v), 90))}
        # neighbour pairs, each counted once
        offs = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if nb == 8 else [])
        a, b = [], []
        for dx, dy in offs:
            s1 = dt[max(0, -dx):dt.shape[0] - max(0, dx),
                    max(0, -dy):dt.shape[1] - max(0, dy)]
            s2 = dt[max(0, dx):dt.shape[0] - max(0, -dx),
                    max(0, dy):dt.shape[1] - max(0, -dy)]
            m = np.isfinite(s1) & np.isfinite(s2)
            a.append(s1[m]); b.append(s2[m])
        a = np.concatenate(a) if a else np.array([])
        b = np.concatenate(b) if b else np.array([])
        if a.size >= 3 and a.std() > 0 and b.std() > 0:
            rec["neighbour"] = {"n_pairs": int(a.size),
                                "corr": float(np.corrcoef(a, b)[0, 1])}
        else:
            rec["neighbour"] = {"n_pairs": int(a.size), "corr": None}
        print("[CentroidError] %d pixels  sd(dt) %s bins  neighbour corr %s "
              "(%d pairs)"
              % (rec["n_pixels"],
                 "n/a" if "dt_bins" not in rec else "%.4f" % rec["dt_bins"]["sd"],
                 _f(rec["neighbour"]["corr"]), rec["neighbour"]["n_pairs"]))
        self.put(store, "centroiderr.summary", rec)
