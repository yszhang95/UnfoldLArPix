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

Charge that lands outside the fit grid in any axis is NOT dropped silently:
``off_grid_ke`` and ``off_grid_frac`` are reported, because a truth sum that
quietly differs from the event's own total is how a closure test goes wrong
without failing.
"""
from __future__ import annotations

import numpy as np

from ..fwk.component import Algorithm, algorithm

CONVENTIONS = ("round", "floor", "linear")


@algorithm("BuildTruth")
class BuildTruth(Algorithm):
    """Bin the event's effective charge onto the operator's fit grid.

    Props: ``convention`` (round|floor|linear, default round).
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
            k = (np.floor(tf + 0.5) if conv == "round"
                 else np.floor(tf)).astype(np.int64)
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
        self.put(store, "truth.q", q)
        self.put(store, "truth.meta", meta)


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

        summary = {
            "truth_convention": tm["convention"],
            "n_rows": int(r.size),
            "sum_d_ke": float(dv.sum()),
            "sum_Aq_truth_ke": float(av.sum()),
            # R_res: the SIGNED sum, always quoted beside the unsigned norm
            # because a small signed sum can hide large cancelling rows
            "R_res_ke": float(r.sum()),
            "abs_norm_ke": float(np.linalg.norm(r)),
            "mean_row_ke": float(r.mean()),
            "rms_row_ke": float(np.sqrt((r ** 2).mean())),
            "row_bias_frac": (float(dv.sum() / av.sum() - 1.0)
                              if av.sum() else None),
            "n_rows_positive": int((r > 0).sum()),
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
