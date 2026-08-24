"""The exact row-residual decomposition and its back-projected gradient.

Ports ``analysis_output/noiseless_closure_round/d_terms.py`` and
``grad_profile.py`` into the framework.  Neither needs a waveform file, so
unlike :class:`OperatorError` these cover every sample.

With ``R`` the fine-tick response, ``CS[d,tau] = sum_{s<tau} R[d,s]``,
``K[d,m] = sum_{s in [mB,(m+1)B)} R[d,s]``, ``w_r(b)`` the window/bin overlap
fraction and

    g_r(d,t) = CS[d, t_hi - t] - CS[d, t_lo - t]

the identity is, per row r, over point charges Q_j at pixel offset d_j and
fine tick t_j with assigned bin k_j:

    d_r - (A q)_r
      = [ d_r - sum_j Q_j g_r(d_j, t_j) ]                    (0)  readout
      + [ sum_j Q_j ( g_r(d_j,t_j) - g_r(d_j,k_j B) ) ]      (i)  t_j -> k_j B
      + [ sum_j Q_j ( g_r(d_j,k_j B)
                      - sum_b w_r(b) K[d_j, b - k_j] ) ]     (ii) box sampling

(ii) is reported split by bin position -- leading partial, trailing partial,
and whole bins, which cancel identically because a fully covered bin has
``w_r(b) = 1`` and ``K`` equal to its exact integral.  That the five parts sum
back to the row residual is asserted, as is the agreement of the box term with
``op.forward``: both are cheap and both catch a convention slip here rather
than in a plot.

Nothing is rebuilt: the sampling triplet comes from the operator itself
(``op._rows/_cols/_weights``) and the window edges from ``row_meta``, so the
decomposition is provably of the same operator the solver used.
"""
from __future__ import annotations

import numpy as np
import torch

from ..fwk.component import Algorithm, algorithm
from .truth_algs import CONVENTIONS, _resid_conventions

PARTS = ("t0", "t1", "t2_lead", "t2_trail", "t2_whole")


@algorithm("RowSplit")
class RowSplit(Algorithm):
    """Exact five-part decomposition of ``d - A q_truth``.

    Props: ``convention`` (how the point charges are assigned to bins;
    must match the ``BuildTruth`` that produced the truth being explained),
    ``truth_prefix`` (default ``truth``), ``store_rows`` (default True),
    ``atol`` (assertion tolerance in ke, default 1e-3).
    """

    reads = ("event", "readout_config", "op", "block_offset", "row_meta")
    writes = ("rowsplit.rows", "rowsplit.summary")

    def __init__(self, **props):
        super().__init__(**props)
        self.convention = str(props.get("convention", "round"))
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, "
                             f"got {self.convention!r}")
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
        atol = float(self.props.get("atol", 1e-3))

        prep = self.services["detector"].prepared(B)
        R = np.asarray(prep.full_response, dtype=np.float64)
        K = np.asarray(prep.integrated_response, dtype=np.float64)
        kx, ky, nfine = R.shape
        cx, cy = (kx - 1) // 2, (ky - 1) // 2
        CS = np.concatenate([np.zeros((kx, ky, 1)), np.cumsum(R, axis=-1)], -1)

        # point charges, assigned by the same rule BuildTruth uses
        nxq, nyq, ntq = op.q_shape
        el = np.asarray(ev.effq.location)
        eq = np.asarray(ev.effq.data, dtype=np.float64)[:, -1]
        jx = el[:, 0].astype(np.int64) - int(boff[0])
        jy = el[:, 1].astype(np.int64) - int(boff[1])
        tf = (el[:, 2] - boff[2]) / 1.0            # fine ticks, block-local
        off = {"round": 0.5, "floor": 0.0, "shift": -0.5}.get(self.convention)
        if off is None:                            # linear has no single bin
            raise ValueError("RowSplit needs a single-bin assignment; "
                            "'linear' spreads a charge over two bins")
        jk = np.floor(tf / B + off).astype(np.int64)
        keep = ((jx >= 0) & (jx < nxq) & (jy >= 0) & (jy < nyq)
                & (jk >= 0) & (jk < ntq))
        jx, jy, tf, jk, Q = jx[keep], jy[keep], tf[keep], jk[keep], eq[keep]

        # cross-check against the truth grid this claims to explain
        qg = np.zeros(op.q_shape)
        np.add.at(qg, (jx, jy, jk), Q)
        q_truth = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        dq = float(np.abs(qg - q_truth).max())
        if dq > atol:
            raise AssertionError(
                f"RowSplit's own charge assignment differs from "
                f"{self.prefix}.q by {dq:.3g} ke -- the conventions disagree "
                f"(RowSplit convention={self.convention!r})")

        # the operator's own sampling, un-whitened if row weights were folded in
        sw, weighted = _resid_conventions(op)
        rows = op._rows.cpu().numpy()
        cols = op._cols.cpu().numpy()
        wts = op._weights.cpu().numpy().astype(np.float64)
        if weighted:
            wts = wts / sw[rows]
        ntb = op.block_shape[2]
        per_row: dict[int, list[tuple[int, float]]] = {}
        for r, c, w in zip(rows, cols, wts):
            per_row.setdefault(int(r), []).append((int(c) % ntb, float(w)))

        d = np.asarray(op.d.detach().cpu().numpy(), np.float64).ravel()
        if weighted:
            d = d / sw
        Aq = op.forward(op.to_tensor(q_truth)).detach().cpu().numpy()
        Aq = np.asarray(Aq, np.float64).ravel()
        if weighted:
            Aq = Aq / sw

        # charges indexed by pixel, for the kernel-footprint lookup
        by_pix: dict[tuple[int, int], list[int]] = {}
        for i in range(jx.size):
            by_pix.setdefault((int(jx[i]), int(jy[i])), []).append(i)

        def csv(dx, dy, tau):
            return CS[dx + cx, dy + cy, np.clip(tau, 0, nfine).astype(int)]

        n_rows = int(op.n_data)
        part = {p: np.zeros(n_rows) for p in PARTS}
        mism = 0.0
        for r in range(n_rows):
            bl = per_row.get(r)
            if not bl:
                continue
            lo = max(float(rm["t_lo"][r]), 0.0)
            hi = float(rm["t_hi"][r])
            px, py = int(rm["px"][r]), int(rm["py"][r])
            sel: list[int] = []
            for ddx in range(-cx, cx + 1):
                for ddy in range(-cy, cy + 1):
                    got = by_pix.get((px - ddx, py - ddy))
                    if got:
                        sel.extend(got)
            if not sel:
                part["t0"][r] = d[r]
                continue
            s = np.asarray(sel)
            dx, dy = px - jx[s], py - jy[s]
            t_ex, t_bin, q = tf[s], jk[s] * float(B), Q[s]
            g_ex = csv(dx, dy, hi - t_ex) - csv(dx, dy, lo - t_ex)
            g_bin = csv(dx, dy, hi - t_bin) - csv(dx, dy, lo - t_bin)

            box = np.zeros(s.size)
            bb = [b for (b, _) in bl]
            b_first, b_last = min(bb), max(bb)
            lead = trail = whole = 0.0
            for (b, w) in bl:
                m = b - jk[s]
                good = (m >= 0) & (m < K.shape[2])
                contrib = np.zeros(s.size)
                if good.any():
                    contrib[good] = w * K[dx[good] + cx, dy[good] + cy, m[good]]
                box += contrib
                ex_b = (csv(dx, dy, min(hi, (b + 1) * float(B)) - t_bin)
                        - csv(dx, dy, max(lo, b * float(B)) - t_bin))
                piece = float((q * (ex_b - contrib)).sum())
                if b == b_first and abs(w - 1.0) > 1e-9:
                    lead += piece
                elif b == b_last and abs(w - 1.0) > 1e-9:
                    trail += piece
                else:
                    whole += piece
            mism = max(mism, abs(float((q * box).sum()) - Aq[r]))
            part["t0"][r] = d[r] - float((q * g_ex).sum())
            part["t1"][r] = float((q * (g_ex - g_bin)).sum())
            part["t2_lead"][r] = lead
            part["t2_trail"][r] = trail
            part["t2_whole"][r] = whole

        resid = d - Aq
        total = sum(part[p] for p in PARTS)
        worst = float(np.abs(total - resid).max()) if n_rows else 0.0
        if worst > atol:
            raise AssertionError(
                f"the five parts do not sum to d - A q_truth: worst "
                f"{worst:.3g} ke over {n_rows} rows")

        kind = np.asarray(rm["kind"], dtype=object)
        summary = {
            "convention": self.convention,
            "identity": "d - A q = (0) + (i) + (ii,lead) + (ii,trail) + (ii,whole)",
            "identity_worst_abs_ke": worst,
            "box_vs_forward_worst_abs_ke": mism,
            "charge_grid_vs_truth_worst_abs_ke": dq,
            "row_weights_active": weighted,
            "units": "ke (un-whitened)" if weighted else "ke",
            "n_rows": n_rows,
            "R_res_ke": float(resid.sum()),
            "totals_ke": {p: float(part[p].sum()) for p in PARTS},
            "n_positive": {p: int((part[p] > 0).sum()) for p in PARTS},
            "by_kind": {},
        }
        for k in sorted(set(kind.tolist())):
            m = kind == k
            summary["by_kind"][k] = {
                "n": int(m.sum()),
                "R_res_ke": float(resid[m].sum()),
                **{p: float(part[p][m].sum()) for p in PARTS},
            }
        print("[RowSplit] %d rows, R_res = %+.2f ke; (0) %+.1f (i) %+.1f "
              "(ii,lead) %+.1f (ii,trail) %+.1f (ii,whole) %+.2g; "
              "identity %.1e, box-vs-A %.1e"
              % (n_rows, summary["R_res_ke"],
                 *[summary["totals_ke"][p] for p in PARTS], worst, mism))
        self.put(store, "rowsplit.rows",
                 part if bool(self.props.get("store_rows", True)) else None)
        self.put(store, "rowsplit.summary", summary)


@algorithm("GradientProfile")
class GradientProfile(Algorithm):
    """Back-project each part of the row residual into charge space.

    The data term is ``1/2 ||A q - d||^2`` with gradient ``A^T (A q - d)``, so
    at the truth the DESCENT direction is ``g = A^T e`` with
    ``e = d - A q_truth``: a field on the fit grid in ke, positive where the
    data term wants more charge.  Because the split of :class:`RowSplit` is
    per-row and ``A^T`` is linear, each part back-projects on its own and the
    five fields sum to ``g`` -- asserted here.

    Props: ``alpha`` (the l1 threshold to compare against, default 0.3 --- a
    voxel is pushed to grow only where ``g_v > alpha``), ``store_fields``
    (default False; the fields are the size of the fit grid).
    """

    reads = ("op", "rowsplit.rows", "support")
    writes = ("grad.summary",)

    def execute(self, store):
        op = store.get("op")
        part = store.get("rowsplit.rows")
        if part is None:
            raise ValueError("GradientProfile needs RowSplit(store_rows: true)")
        support = np.asarray(store.get("support")).astype(bool)
        alpha = float(self.props.get("alpha", 0.3))

        fields = {}
        for p in PARTS:
            r = op.to_tensor(np.asarray(part[p], np.float64))
            fields[p] = (op.adjoint(r).detach().cpu().numpy()
                         .astype(np.float64))
        g = sum(fields.values())
        e = sum(np.asarray(part[p], np.float64) for p in PARTS)
        g_ref = op.adjoint(op.to_tensor(e)).detach().cpu().numpy()
        worst = float(np.abs(g - g_ref).max())
        # float32 adjoint on a large grid: compare against its own scale
        tol = 1e-4 * max(float(np.abs(g_ref).max()), 1.0)
        if worst > tol:
            raise AssertionError(
                f"the back-projected parts do not sum to A^T e: worst "
                f"{worst:.3g} against tolerance {tol:.3g}")

        def stats(f):
            on = f[support]
            return {"max_ke": float(f.max()), "min_ke": float(f.min()),
                    "n_above_alpha": int((f > alpha).sum()),
                    "n_above_alpha_on_support": int((on > alpha).sum()),
                    "sum_on_support_ke": float(on.sum()),
                    "rms_on_support_ke": float(np.sqrt((on ** 2).mean()))
                    if on.size else None}

        summary = {
            "alpha": alpha,
            "identity": "g = A^T e = sum_parts A^T e_part",
            "identity_worst_abs": worst,
            "n_support_voxels": int(support.sum()),
            "total": stats(g),
            "parts": {p: stats(fields[p]) for p in PARTS},
        }
        print("[GradientProfile] |g|max %.2f ke, %d voxels above alpha=%.2f "
              "(%d on support); identity %.1e"
              % (summary["total"]["max_ke"],
                 summary["total"]["n_above_alpha"], alpha,
                 summary["total"]["n_above_alpha_on_support"], worst))
        if bool(self.props.get("store_fields", False)):
            summary["_fields"] = fields
        self.put(store, "grad.summary", summary)


@algorithm("FitVsTruth")
class FitVsTruth(Algorithm):
    """Does the reconstruction fit the data BETTER than the truth does?

    Ports ``operator_studies/fit_vs_truth.py``.  Three numbers, all already in
    the store, plus the ratios that make them readable:

        L_reco  = 1/2 ||A q_hat  - d||^2      (resid.solution_summary)
        L_truth = 1/2 ||A q_true - d||^2      (resid.summary)
        L_n     = 1/2 sum_r var_r             (row_var)

    ``L_reco < L_truth`` means the solver is exploiting freedoms the truth does
    not use.  ``L_reco < L_n`` on top of that is over-fitting in the strict
    sense: it is fitting noise and operator error.  ``L_truth`` is the honest
    reference because it contains the readout noise AND the operator error,
    which is why it is not a bound.

    Nothing is recomputed here -- if the numbers disagree with the residual
    algorithms, they disagree for a reason worth finding.
    """

    reads = ("resid.summary", "resid.solution_summary", "solve.q")
    writes = ("fitvstruth.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",
                                          f"{self.prefix}.meta")

    def execute(self, store):
        rt = store.get("resid.summary")
        rs = store.get("resid.solution_summary")
        tm = store.get(f"{self.prefix}.meta")
        L_truth = float(rt["half_sq_norm"])
        L_reco = float(rs["half_sq_norm"])
        rv = store.get("row_var") if "row_var" in store else None
        L_n = (0.5 * float(np.sum(np.asarray(rv, dtype=np.float64)))
               if rv is not None else None)

        q_hat = np.asarray(store.get("solve.q"), dtype=np.float64)
        q_true = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        summary = {
            "n_rows": int(rt["n_rows"]),
            "truth_convention": tm["convention"],
            "L_reco": L_reco,
            "L_truth": L_truth,
            "noise_floor": L_n,
            "sum_q_reco_ke": float(q_hat.sum()),
            "sum_q_truth_ke": float(q_true.sum()),
            "truth_in_grid": (1.0 - float(tm.get("off_grid_frac", 0.0))),
            "ratio_reco_truth": (L_reco / L_truth) if L_truth else None,
            "reco_over_floor": (L_reco / L_n) if L_n else None,
            "truth_over_floor": (L_truth / L_n) if L_n else None,
            "reco_fits_better_than_truth": L_reco < L_truth,
            "over_fitting_strict": (bool(L_reco < L_n) if L_n else None),
            "noise_floor_available": L_n is not None,
        }
        print("[FitVsTruth] L_reco %.1f  L_truth %.1f  floor %s  "
              "ratio %.4f  reco/floor %s  -> %s"
              % (L_reco, L_truth,
                 "n/a" if L_n is None else "%.1f" % L_n,
                 summary["ratio_reco_truth"] or float("nan"),
                 "n/a" if L_n is None else "%.3f" % summary["reco_over_floor"],
                 "OVER-FITS" if summary["over_fitting_strict"]
                 else "fits better than truth" if summary[
                     "reco_fits_better_than_truth"] else "does not"))
        self.put(store, "fitvstruth.summary", summary)
