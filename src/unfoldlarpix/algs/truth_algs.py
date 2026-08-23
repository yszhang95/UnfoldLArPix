"""Fit-grid truth and the row residual, as store products.

Why these exist.  Twenty-two scripts under ``analysis_output`` compute
``d - A q_truth``, and the fit-grid truth they need is a copy-pasted one-liner

    it = np.floor((el[:, 2] - boff[2]) / B + 0.5)          # 28 instances

with no shared helper anywhere -- ``q_truth`` appears nowhere in ``src/``.  So
the note's central residual identity is re-derived from scratch every time it
is used, and the truth-binning CONVENTION (round, floor, linear) is a literal
inside each script rather than a declared property of the job.  That is the
same class of defect as a table typed by hand: nothing connects the number to
a recipe, and a convention change has to be found by grep.

    BuildTruth    the effective charge binned onto the operator's own fit grid,
                  under a declared convention, with the off-grid charge
                  accounted for rather than silently dropped.
    RowResidual   d - A q_truth per row, its per-row-kind sums, and the signed
                  and unsigned totals the note quotes as R_res.

Both are ordinary store products, so a downstream study reads them instead of
rebuilding them, and ``WriteCharges`` can embed them for offline work.

Definitions, stated once here instead of in each consumer
--------------------------------------------------------
``q_truth[ix, iy, k]`` is the sum of effective charge released on pixel
``(ix, iy)`` whose release time falls in fit-grid bin ``k``.  The fit grid has
the operator's own ``q_shape`` and shares the block grid's origin
``block_offset``; ``B = adc_hold_delay`` fine ticks per bin.  The convention
fixes which bin a release time ``t_f`` goes to:

    round   k = floor((t_f - boff_z)/B + 0.5)    ADOPTED.  Nearest bin; mean
                                                 release error 0 by symmetry.
    floor   k = floor((t_f - boff_z)/B)          SUPERSEDED.  Mean release
                                                 error -0.47 B, which is what
                                                 produced the withdrawn
                                                 tred-vs-synth gaps.
    linear  charge split between k and k+1 in    charge-conserving, and the
            proportion to the sub-bin phase      only one with no binning bias
    shift   k = floor((t_f - boff_z)/B - 0.5)    CONTROL, deliberately wrong.
                                                 Displaces the release another
                                                 half bin EARLIER than floor,
                                                 which is already ~0.47 B early,
                                                 so it bounds the scale of the
                                                 effect from the wrong side.
                                                 Faithful to `shift_half` in
                                                 noiseless_closure/rowbias_decomp.py.

Charge that lands outside the fit grid in any axis is NOT dropped silently:
``off_grid_ke`` and ``off_grid_frac`` are reported, because a truth sum that
quietly differs from the event's own total is how a closure test goes wrong
without failing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..fwk.component import Algorithm, algorithm

CONVENTIONS = ("round", "floor", "linear", "shift")


def _resid_conventions(op):
    """The two conventions a row residual carries, and the un-whitening factor.

    SIGN.  These algorithms use ``d - A q``, which is the note's convention for
    ``R_res``: positive means the readout recorded more than the operator says
    it should have.  ``terms.data.DataFidelity`` uses the OPPOSITE order,
    ``A q - target``, because it is minimising.  The squared norms agree; the
    signed sums do not, so both are labelled.

    WEIGHTING.  When ``BuildMeasurement`` was given ``row_weights``, the
    operator folds ``sqrt(w)`` into BOTH ``op.d`` and the sampling weights, so
    ``op.d - A q`` is the WHITENED residual and its sum is not a charge.  The
    factor is returned so the charge-unit residual can be recovered; without it
    a summary would report whitened numbers labelled ``ke``, which is wrong
    wherever ``row_weights: diag`` was used (the ``nb1_diag`` campaign).
    """
    w = getattr(op, "row_weights", None)
    if w is None:
        return None, False
    import numpy as _np
    sw = _np.sqrt(_np.asarray(w, dtype=_np.float64)).ravel()
    return sw, True


@algorithm("BuildTruth")
class BuildTruth(Algorithm):
    """Bin the event's effective charge onto the operator's fit grid.

    Props: ``convention`` (round|floor|linear|shift, default round) and
    ``out_prefix`` (default ``truth``).  The prefix makes the write keys
    instance-level, so several conventions can be built in one sequence --
    the store is write-once, so a fixed key would collide.
    """

    reads = ("event", "readout_config", "block_offset", "op")
    writes = ("truth.q", "truth.meta")

    def __init__(self, **props):
        super().__init__(**props)
        # fail here, not after LoadEvent and BuildMeasurement have run
        self.convention = str(props.get("convention", "round"))
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, "
                             f"got {self.convention!r}")
        self.prefix = str(props.get("out_prefix", "truth"))
        self.writes = (f"{self.prefix}.q", f"{self.prefix}.meta")

    def execute(self, store):
        conv = self.convention
        ev = store.get("event")
        rc = store.get("readout_config")
        op = store.get("op")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        B = float(rc.adc_hold_delay)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        Bq = B / S                      # the operator's own bin, if subbinned

        el = np.asarray(ev.effq.location)
        eq = np.asarray(ev.effq.data, dtype=np.float64)[:, -1]
        nx, ny, nt = op.q_shape
        ix = el[:, 0].astype(np.int64) - int(boff[0])
        iy = el[:, 1].astype(np.int64) - int(boff[1])
        tf = (el[:, 2] - boff[2]) / Bq

        q = np.zeros(op.q_shape, dtype=np.float64)
        in_px = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if conv == "linear":
            k0 = np.floor(tf).astype(np.int64)
            w1 = tf - k0                          # phase inside the bin
            for kk, ww in ((k0, 1.0 - w1), (k0 + 1, w1)):
                ok = in_px & (kk >= 0) & (kk < nt)
                np.add.at(q, (ix[ok], iy[ok], kk[ok]), eq[ok] * ww[ok])
            placed = eq * 0.0
            for kk, ww in ((k0, 1.0 - w1), (k0 + 1, w1)):
                ok = in_px & (kk >= 0) & (kk < nt)
                placed[ok] += eq[ok] * ww[ok]
        else:
            off = {"round": 0.5, "floor": 0.0, "shift": -0.5}[conv]
            k = np.floor(tf + off).astype(np.int64)
            ok = in_px & (k >= 0) & (k < nt)
            np.add.at(q, (ix[ok], iy[ok], k[ok]), eq[ok])
            placed = np.where(ok, eq, 0.0)

        total = float(eq.sum())
        off = total - float(placed.sum())
        meta = {
            "convention": conv,
            "q_shape": [int(v) for v in op.q_shape],
            "bin_fine_ticks": Bq,
            "time_subbin": S,
            "block_offset": [float(v) for v in boff],
            "n_deposits": int(eq.size),
            "sum_effq_ke": total,
            "sum_truth_grid_ke": float(q.sum()),
            # charge that fell outside the fit grid, reported not dropped
            "off_grid_ke": off,
            "off_grid_frac": (off / total) if total else 0.0,
            "outside_pixels": int((~in_px).sum()),
        }
        print(f"[BuildTruth] {conv}: {meta['sum_truth_grid_ke']:.1f} of "
              f"{total:.1f} ke on the fit grid "
              f"({100 * meta['off_grid_frac']:.3f}% off grid, "
              f"{meta['outside_pixels']} deposits outside the pixel range)")
        self.put(store, f"{self.prefix}.q", q)
        self.put(store, f"{self.prefix}.meta", meta)


@algorithm("RowResidual")
class RowResidual(Algorithm):
    """``d - A q_truth`` per row, with the per-kind and signed/unsigned sums.

    This is the quantity the note calls the row residual, and the signed sum
    over rows is ``R_res``.  It is a property of the operator and the truth
    alone -- no solution enters -- so it is available before Solve.

    Props: ``store_rows`` (bool, default True) keep the per-row vector.
    """

    reads = ("op", "truth.q", "truth.meta")
    writes = ("resid.rows", "resid.summary")

    def execute(self, store):
        op = store.get("op")
        q = store.get("truth.q")
        tm = store.get("truth.meta")
        Aq = op.forward(op.to_tensor(q)).detach()
        d = op.d.detach()
        r = (d - Aq).cpu().numpy().astype(np.float64).ravel()
        dv = d.cpu().numpy().astype(np.float64).ravel()
        av = Aq.cpu().numpy().astype(np.float64).ravel()
        sw, weighted = _resid_conventions(op)

        summary = {
            "truth_convention": tm["convention"],
            "sign_convention": "d - A q  (DataFidelity uses A q - target)",
            "row_weights_active": weighted,
            "units": "whitened (sqrt(w) folded in)" if weighted else "ke",
            "n_rows": int(r.size),
            "sum_d_ke": float(dv.sum()),
            "sum_Aq_truth_ke": float(av.sum()),
            # R_res: the SIGNED sum, always quoted beside the unsigned norm
            # because a small signed sum can hide large cancelling rows
            "R_res_ke": float(r.sum()),
            "abs_norm_ke": float(np.linalg.norm(r)),
            "sq_norm": float(r @ r),
            # the objective's data term carries a 1/2, so this is the number
            # a loss ledger reports and it is HALF the squared norm above.
            # Naming both is cheaper than rediscovering the factor.
            "half_sq_norm": float(0.5 * (r @ r)),
            "mean_row_ke": float(r.mean()),
            "rms_row_ke": float(np.sqrt((r ** 2).mean())),
            "row_bias_frac": (float(dv.sum() / av.sum() - 1.0)
                              if av.sum() else None),
            "n_rows_positive": int((r > 0).sum()),
        }
        if weighted:
            # recover charge units: op.d = sqrt(w) d_raw and A = sqrt(w) A_raw
            rc = r / sw
            summary["charge_units"] = {
                "R_res_ke": float(rc.sum()),
                "abs_norm_ke": float(np.linalg.norm(rc)),
                "rms_row_ke": float(np.sqrt((rc ** 2).mean())),
            }
        if "row_meta" in store:
            kind = np.asarray(store.get("row_meta")["kind"], dtype=object)
            summary["by_kind"] = {
                k: {"n": int((kind == k).sum()),
                    "sum_d_ke": float(dv[kind == k].sum()),
                    "sum_Aq_ke": float(av[kind == k].sum()),
                    "R_res_ke": float(r[kind == k].sum()),
                    "n_positive": int((r[kind == k] > 0).sum())}
                for k in sorted(set(kind.tolist()))}
        print(f"[RowResidual] {summary['n_rows']} rows, R_res = "
              f"{summary['R_res_ke']:+.2f} ke, |r| = "
              f"{summary['abs_norm_ke']:.2f} ke, row bias = "
              f"{100 * (summary['row_bias_frac'] or 0):+.2f}%, "
              f"{summary['n_rows_positive']}/{summary['n_rows']} rows positive")
        self.put(store, "resid.rows",
                 r if bool(self.props.get("store_rows", True)) else None)
        self.put(store, "resid.summary", summary)


def _rank(x):
    """Ranks with ties broken by position -- enough for a rank correlation
    on continuous quantities, and avoids a scipy dependency."""
    o = np.argsort(x, kind="stable")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(x.size, dtype=np.float64)
    return r


def _spearman(x, y):
    if x.size < 3:
        return None
    a, b = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / den) if den else None


_CUR = "current_tpc0_batch0"

# Index alignment between a window's tick range and the stored current array.
# The archived construction (examples/operator_studies/anisotropy.py) used 0.
# CALIBRATED, not assumed: on the one sample with a NOISELESS readout whose
# current is in the data file (pgun_positron_3gev_ang50_tred_nb4_wf_nonoise),
# physics fixes the answer -- with no noise the recorded value must equal the
# integral of the simulated current over the window.  Scanning the offset:
#
#     offset   -2      -1       0      +1      +2
#     |n| ke   46.98   31.21   15.68    3.84   16.15
#
# a sharp minimum at +1, and there the two row kinds that integrate a whole
# window close EXACTLY: d/E_1 = 1.0000 for both `diff` and `lumped`, against
# 0.9994 and 0.9928 at offset 0.  A scale factor was tested at the same time
# (one_tick = 2) and rejected: scale 2 is worse by two orders of magnitude.
#
# What does NOT go away at +1 is the `pseudo` (0.9907) and `remainder`
# (1.0281) discrepancy -- the two kinds the trigger split creates.  So the
# note's unexplained pair-sum excess is real and is now isolated to them.
_CUR_TICK_OFFSET = 1


def _resolve_current(src, wf_prop):
    """Locate the noiseless induced current, and say where it came from.

    The simulation stores it only when ``save_waveform`` is on, and then it
    may sit EITHER in the data file itself OR in a companion written by a
    separate noiseless run.  Both occur in this campaign, so both are tried:

    1. an explicit ``wf`` prop (override, for a hand-picked companion);
    2. the ``current_tpc0_batch0`` field of the data file itself;
    3. the ``<input>_wf.npz`` companion beside it.

    Raises with ``save_waveform`` quoted, so "no decomposition possible" is
    distinguishable from "looked in the wrong place".
    """
    tried = []
    for cand, label in ((wf_prop, "wf prop"), (src, "data file"),
                        (str(src).replace(".npz", "_wf.npz"), "companion")):
        if cand is None:
            continue
        pth = Path(cand)
        tried.append(f"{label}={pth.name}")
        if not pth.exists():
            continue
        z = np.load(pth, allow_pickle=True)
        if _CUR in z.files:
            return z, str(pth), label
    sw = "unknown"
    if Path(src).exists():
        zz = np.load(src, allow_pickle=True)
        if "save_waveform" in zz.files:
            sw = str(zz["save_waveform"].item())
    raise FileNotFoundError(
        f"OperatorError needs the noiseless current and found none. "
        f"Tried: {', '.join(tried)}. The data file reports "
        f"save_waveform={sw} -- if that is False and no companion exists, the "
        f"current was never stored and this sample cannot be decomposed.")


@algorithm("OperatorError")
class OperatorError(Algorithm):
    """Split the row residual into OPERATOR error and READOUT error.

    With the noiseless induced current available as a companion waveform
    file, every row's exact window integral is known, so the two error
    sources separate:

        d_exact_r                            exact charge in the window
        e_r = (A q_truth)_r - d_exact_r      operator model error
        n_r = d_r          - d_exact_r       readout error

    and :class:`RowResidual`'s ``(d - A q_truth)_r`` is identically
    ``n_r - e_r``.  That identity is ASSERTED here, so a convention slip in
    either path fails at this algorithm instead of propagating into a plot.

    Also reported per row, because the operator's error is expected to
    depend on them: ``dt`` (window length in ticks) and ``q_part`` (the
    charge sitting in the two partially-covered fit bins at the window
    edges -- the part the within-bin model has to guess).  The summary gives
    the rank correlation of the relative error against both, which is the
    measurement that distinguishes "long windows are more accurate" from
    "the error is set by partial-bin charge".

    MEASURED, so that these are not re-derived (6 samples with waveforms,
    setup B, ``round`` truth):

    * ``|e|`` exceeds ``|n|`` in 5 of 6 -- the operator error dominates the
      readout error, by 2.7x on ``pos_a50_nb4``.  This is the reason a
      residual target set by the noise model does not help.
    * ``dt`` is CONFOUNDED with the row kind: ``remainder`` and ``diff``
      windows are one bin long by construction (30 ticks), ``lumped`` runs
      190-470 and ``pseudo`` 800-1860.  A correlation of error against ``dt``
      is therefore a comparison between kinds, not a trend within one, and
      its sign flips across samples (-0.30 to +0.15).
    * ``part_frac`` is DEGENERATE: it is identically 1 for ``remainder`` and
      ``diff``, which are most of the rows, so it cannot discriminate.  Do
      not read its correlation as a mechanism.
    * What is stable is the per-kind median relative error: ``pseudo`` and
      ``lumped`` low (0.04-0.15), ``remainder`` and ``diff`` high
      (0.07-0.26).  The operator error is a per-kind property; no smooth
      function of window length or partial-bin charge was found that
      explains it.

    Props: ``wf`` (explicit path to a file carrying the current; by default
    it is looked up, see below), ``truth_prefix`` (default ``truth``),
    ``store_rows`` (bool, default True).

    The current is stored only when the simulation ran with
    ``save_waveform``, and then it may sit either in the data file itself or
    in a companion from a separate noiseless run.  Both are tried, in that
    order, and ``current_from`` records which was used.  In the present
    campaign 1 of 121 data files carries it in place and 7 have a companion,
    so 113 samples cannot be decomposed at all -- the error message quotes
    ``save_waveform`` so that case is distinguishable from a wrong path.
    """

    reads = ("event", "readout_config", "op", "block_offset", "row_meta")
    writes = ("error.rows", "error.summary")

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",)

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        rm = store.get("row_meta")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        S = int(store.get("time_subbin")) if "time_subbin" in store else 1
        B = int(rc.adc_hold_delay) // S

        src = getattr(store.get("event"), "source", None) or \
            store.get("job.config")["sequence"][0]["LoadEvent"]["input"]
        z, wf, cur_from = _resolve_current(src, self.props.get("wf"))
        cur = np.asarray(z[_CUR])
        cur = cur.reshape(-1, cur.shape[-1])
        cl = np.asarray(z[_CUR + "_location"])
        Nt = cur.shape[1]
        # cumulative current per pixel, with a leading zero so that
        # cs[b] - cs[a] is the integral over [a, b)
        cs = {(int(a), int(b)): np.concatenate([[0.0], np.cumsum(cur[i])])
              for i, (a, b) in enumerate(cl[:, :2])}

        n_rows = int(op.n_data)
        d_ex = np.zeros(n_rows)
        q_part = np.zeros(n_rows)
        dt = np.zeros(n_rows)
        t0 = int(boff[2])
        missing = 0
        for r in range(n_rows):
            t_lo = max(float(rm["t_lo"][r]), 0.0)
            t_hi = float(rm["t_hi"][r])
            k = (int(rm["px"][r] + boff[0]), int(rm["py"][r] + boff[1]))
            a = int(np.clip(t_lo + t0 + _CUR_TICK_OFFSET, 0, Nt))
            b = int(np.clip(t_hi + t0 + _CUR_TICK_OFFSET, 0, Nt))
            dt[r] = t_hi - t_lo
            if k not in cs:
                missing += 1
                continue
            if b > a:
                d_ex[r] = cs[k][b] - cs[k][a]
                t0o = t0 + _CUR_TICK_OFFSET
                lo_e = t0o + ((a - t0o) // B + 1) * B
                hi_e = t0o + ((b - t0o) // B) * B
                if hi_e <= lo_e:              # window inside a single fit bin
                    q_part[r] = d_ex[r]
                else:
                    q_part[r] = ((cs[k][min(lo_e, Nt)] - cs[k][a])
                                 + (cs[k][b] - cs[k][max(hi_e, 0)]))

        qt = store.get(f"{self.prefix}.q")
        Aqt = op.forward(op.to_tensor(qt)).detach().cpu().numpy()
        Aqt = np.asarray(Aqt, np.float64).ravel()
        d = np.asarray(op.d.detach().cpu().numpy(), np.float64).ravel()
        sw, weighted = _resid_conventions(op)
        if weighted:                  # decompose in CHARGE units, always
            d = d / sw
            Aqt = Aqt / sw

        e = Aqt - d_ex                # operator model error
        n = d - d_ex                  # readout error
        # RowResidual computes d - A q_truth; it must equal n - e exactly
        resid = d - Aqt
        worst = float(np.max(np.abs(resid - (n - e)))) if n_rows else 0.0
        if not np.allclose(resid, n - e, rtol=0, atol=1e-6):
            raise AssertionError(
                f"(d - A q_truth) != n - e, worst {worst:.3g} -- the two "
                "paths disagree on a convention")

        kind = np.asarray(rm["kind"], dtype=object)
        rel = np.where(np.abs(d_ex) > 0, np.abs(e) / np.abs(d_ex), np.nan)
        # the mechanism-level variable: what FRACTION of the window's charge
        # sits in the two partially-covered fit bins.  dt is a poor proxy for
        # it because the row kind fixes both (remainder/diff windows are one
        # bin long by construction), so a correlation against dt is a
        # comparison BETWEEN kinds, not a trend within one.
        frac = np.where(np.abs(d_ex) > 0, q_part / np.abs(d_ex), np.nan)
        ok = np.isfinite(rel) & np.isfinite(frac)
        summary = {
            "current_file": str(wf),
            "current_from": cur_from,
            "current_tick_offset": _CUR_TICK_OFFSET,
            "truth_convention": store.get(f"{self.prefix}.meta")["convention"],
            "identity_checked": "(d - A q_truth) == n - e",
            "identity_worst_abs": worst,
            "row_weights_active": weighted,
            "units": "ke (un-whitened)" if weighted else "ke",
            "n_rows": n_rows,
            "n_rows_no_current": int(missing),
            "sum_d_exact_ke": float(d_ex.sum()),
            "operator_error": {
                "sum_ke": float(e.sum()), "abs_norm_ke": float(np.linalg.norm(e)),
                "mean_abs_ke": float(np.abs(e).mean()),
                "rel_median": (float(np.nanmedian(rel)) if ok.any() else None)},
            "readout_error": {
                "sum_ke": float(n.sum()), "abs_norm_ke": float(np.linalg.norm(n)),
                "mean_abs_ke": float(np.abs(n).mean())},
            # the measurement that separates the two candidate explanations
            "spearman_rel_err_vs_dt": _spearman(dt[ok], rel[ok]),
            "spearman_rel_err_vs_q_part": _spearman(q_part[ok], rel[ok]),
            "spearman_abs_err_vs_q_part": _spearman(q_part, np.abs(e)),
            "spearman_rel_err_vs_part_frac": _spearman(frac[ok], rel[ok]),
            "part_frac_median": float(np.nanmedian(frac[ok])) if ok.any() else None,
            "by_kind": {},
        }
        for k in sorted(set(kind.tolist())):
            m = kind == k
            mo = m & ok
            summary["by_kind"][k] = {
                "n": int(m.sum()),
                "mean_dt_ticks": float(dt[m].mean()) if m.any() else None,
                "sum_e_ke": float(e[m].sum()),
                "mean_abs_e_ke": float(np.abs(e[m]).mean()) if m.any() else None,
                "rel_median": (float(np.nanmedian(rel[mo]))
                               if mo.any() else None),
                "mean_q_part_ke": float(q_part[m].mean()) if m.any() else None,
                # d/E_1 per kind: the ratio tab:truecurrent quotes, so it is
                # reproducible from a store product instead of a script
                "sum_d_ke": float(d[m].sum()),
                "sum_d_exact_ke": float(d_ex[m].sum()),
                "d_over_exact": (float(d[m].sum() / d_ex[m].sum())
                                 if d_ex[m].sum() else None),
                "spearman_rel_vs_dt": _spearman(dt[mo], rel[mo]),
                "spearman_rel_vs_q_part": _spearman(q_part[mo], rel[mo]),
                "part_frac_median": (float(np.nanmedian(frac[mo]))
                                     if mo.any() else None),
                "spearman_rel_vs_part_frac": _spearman(frac[mo], rel[mo]),
            }
        print(f"[OperatorError] {n_rows} rows, |e| = "
              f"{summary['operator_error']['abs_norm_ke']:.2f} ke, |n| = "
              f"{summary['readout_error']['abs_norm_ke']:.2f} ke, "
              f"rho(rel err, dt) = {summary['spearman_rel_err_vs_dt']}, "
              f"rho(rel err, part frac) = "
              f"{summary['spearman_rel_err_vs_part_frac']}")
        self.put(store, "error.rows",
                 {"e": e, "n": n, "d_exact": d_ex, "q_part": q_part,
                  "part_frac": frac, "dt": dt}
                 if bool(self.props.get("store_rows", True)) else None)
        self.put(store, "error.summary", summary)


@algorithm("SolutionResidual")
class SolutionResidual(Algorithm):
    """``d - A q_hat`` per row at the solution -- the measurement residual.

    The counterpart of :class:`RowResidual`, and the one a downstream study
    actually needs: ``RowResidual`` is a property of the operator and the truth
    with no solver in it, this one is what the fit left on the table.  Seven
    scripts under ``analysis_output`` recompute it; ``Solve`` publishes only
    ``solve.loss``, in which ``DataFidelity`` has already collapsed it to the
    scalar :math:`\\|Aq-d\\|^2`, so the per-row vector was nowhere.

    Reported against three references, because they answer different questions
    and the note keeps them apart:

    ``d``                the fit's own target.  ``R_fit`` is the signed sum;
                         a solver that is converged and unconstrained would
                         drive it to zero, and the amount by which it does not
                         is what positivity, the l1 and the censor hold back.
    ``A q_truth``        available when BuildTruth ran: the part of the
                         residual that is the operator's own error rather than
                         the fit's, so ``(d - A q_hat) - (d - A q_truth)
                         = A(q_truth - q_hat)`` isolates the charge the fit
                         moved from where the truth put it.
    ``row_var``          the analytic per-row readout variance, when
                         BuildMeasurement could compute it.  ``chi2_per_row``
                         is the only form in which the residual can be called
                         large or small without a further convention.

    The reference is ``op.d``, recorded as ``reference`` in the summary, and
    that is deliberately the PHYSICAL residual rather than whatever the solver
    internally minimised.  Two jobs differ from the naive reading:

    * the amplitude refit overrides ``DataFidelity``'s target with
      ``d - A q_faint`` (``solve/strategy.py``).  Since the reported solution is
      ``q_strong + q_faint``, ``op.d - A q_hat`` is the same residual --- which
      is what that code's own comment asserts and what makes the caller's
      noise-floor target still apply.
    * the synthetic arm of the closure test overrides the target with
      ``A q_truth``.  There ``op.d`` is still the tred data, so this algorithm
      reports the residual against the DATA, not the residual the solver drove
      to zero.  Both are wanted; they are not the same number, and the
      ``reference`` field says which one this is.

    Props: ``store_rows`` (bool, default True).
    """

    reads = ("op", "solve.q")
    writes = ("resid.solution", "resid.solution_summary")

    def execute(self, store):
        op = store.get("op")
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        Aq = op.forward(op.to_tensor(q)).detach()
        d = op.d.detach()
        r = (d - Aq).cpu().numpy().astype(np.float64).ravel()
        dv = d.cpu().numpy().astype(np.float64).ravel()
        av = Aq.cpu().numpy().astype(np.float64).ravel()
        sw, weighted = _resid_conventions(op)

        s = {
            "n_rows": int(r.size),
            "sign_convention": "d - A q  (DataFidelity uses A q - target)",
            "reference": "op.d",
            "row_weights_active": weighted,
            "units": "whitened (sqrt(w) folded in)" if weighted else "ke",
            "sum_d_ke": float(dv.sum()),
            "sum_Aq_hat_ke": float(av.sum()),
            "R_fit_ke": float(r.sum()),
            "abs_norm_ke": float(np.linalg.norm(r)),
            "sq_norm": float(r @ r),
            # what DataFidelity and the loss ledger report: the objective's
            # data term carries a 1/2.  The archived analyze_round.py's
            # `L_qhat` is this number, not sq_norm.
            "half_sq_norm": float(0.5 * (r @ r)),
            "mean_row_ke": float(r.mean()),
            "rms_row_ke": float(np.sqrt((r ** 2).mean())),
            "n_rows_positive": int((r > 0).sum()),
        }
        if weighted:
            rc = r / sw
            s["charge_units"] = {
                "R_fit_ke": float(rc.sum()),
                "abs_norm_ke": float(np.linalg.norm(rc)),
                "rms_row_ke": float(np.sqrt((rc ** 2).mean())),
            }
        # against the truth's own forward image: what the fit moved
        if "resid.rows" in store and store.get("resid.rows") is not None:
            rt = np.asarray(store.get("resid.rows"), dtype=np.float64).ravel()
            moved = r - rt                    # = A(q_truth - q_hat)
            s["vs_truth"] = {
                "R_res_ke": float(rt.sum()),
                "sum_A_dq_ke": float(moved.sum()),
                "abs_norm_A_dq_ke": float(np.linalg.norm(moved)),
                # how much of the operator's row residual the fit absorbed
                "absorbed_frac": (float(1.0 - r.sum() / rt.sum())
                                  if rt.sum() else None),
            }
        # against the noise model: the only scale-free statement available
        rv = store.get("row_var") if "row_var" in store else None
        if rv is not None:
            v = np.asarray(rv, dtype=np.float64).ravel()
            ok = v > 0
            if ok.any():
                s["chi2_per_row"] = float((r[ok] ** 2 / v[ok]).mean())
                s["n_rows_with_var"] = int(ok.sum())
        if "row_meta" in store:
            kind = np.asarray(store.get("row_meta")["kind"], dtype=object)
            s["by_kind"] = {
                k: {"n": int((kind == k).sum()),
                    "R_fit_ke": float(r[kind == k].sum()),
                    "rms_row_ke": float(np.sqrt((r[kind == k] ** 2).mean())),
                    "n_positive": int((r[kind == k] > 0).sum())}
                for k in sorted(set(kind.tolist()))}
        print(f"[SolutionResidual] {s['n_rows']} rows, R_fit = "
              f"{s['R_fit_ke']:+.2f} ke, |r| = {s['abs_norm_ke']:.2f} ke"
              + (f", chi2/row = {s['chi2_per_row']:.3g}"
                 if "chi2_per_row" in s else ", no row_var")
              + (f", absorbed {100 * s['vs_truth']['absorbed_frac']:.1f}% of "
                 f"R_res" if "vs_truth" in s
                 and s["vs_truth"]["absorbed_frac"] is not None else ""))
        self.put(store, "resid.solution",
                 r if bool(self.props.get("store_rows", True)) else None)
        self.put(store, "resid.solution_summary", s)
