"""The evaluation layer as framework algorithms.

``eval/universal.py`` was called directly by dozens of scripts, each choosing
its own widths, deposit shape and threshold, and each recomputing the
universal grid's alignment.  That is how the same table came to exist at two
deposit protocols and how two callers came to use two formulas for the grid
origin.  Here the protocol is declared once, recorded in the store, and the
grid origin is published so a consumer maps onto the blocks instead of
re-deriving them.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..deconv_workflow import smear_effective_charge
from ..eval.universal import metrics_from_blocks, universal_rebin
from ..fwk.component import Algorithm, algorithm


@algorithm("Evaluate")
class Evaluate(Algorithm):
    """Rebin truth and reconstruction onto the universal grid, and score.

    Props:
      ``from``       ``solution`` (default; the NPZ behind ``solve.provenance``),
                     ``output`` (what WriteCharges just wrote), or ``path``.
      ``path``       explicit NPZ, required when ``from: path``.
      ``truth``      ``recompute`` (default) smears the event's effective
                     charge at THESE widths; ``embedded`` uses the
                     ``smeared_true`` field carried by the solve NPZ.  They are
                     NOT interchangeable: the embedded field was smeared at the
                     warm start's ``sigma_pixel`` (0.2 in the shipped configs),
                     so scoring against it at the evaluation width silently
                     changes every truth-dependent metric -- measured, up to
                     10x on ``true_killed``.  ``recompute`` is what the
                     centre-scored evaluation of record does.
      ``truth_npz``  explicit truth file; overrides ``truth``.
      ``deposit``    ``gaussian`` (default) | ``linear``.
      ``sigma_time``, ``sigma_pixel``  the analysis widths; the adopted pair is
                     0.005 and 0.5 (sec:eval:grid).
      ``corr_threshold``  0.5 by default.
      ``time_offsets``  ``none`` (default, the adopted bin-centre deposit) or
                     ``file`` to take ``deconv_q_offsets`` from the NPZ -- the
                     RETIRED sub-bin offsets protocol, kept only so an archived
                     number can be reproduced.  Which one was used is recorded.

    Writes ``eval.truth``, ``eval.reco`` (universal-grid blocks),
    ``eval.origin`` and ``eval.metrics``.
    """

    writes = ("eval.truth", "eval.reco", "eval.origin", "eval.metrics",
              "eval.protocol")

    def __init__(self, **props):
        super().__init__(**props)
        self.truth_mode = str(props.get("truth", "recompute"))
        if self.truth_mode not in ("recompute", "embedded"):
            raise ValueError("truth must be recompute|embedded")
        self.reads = (("event",) if self.truth_mode == "recompute"
                      and not props.get("truth_npz") else ())
        self.source = str(props.get("from", "solution"))
        if self.source not in ("solution", "output", "path"):
            raise ValueError("from must be solution|output|path, got "
                             f"{self.source!r}")
        if self.source == "solution":
            self.reads = self.reads + ("solve.provenance",)
        elif self.source == "output":
            self.reads = self.reads + ("output.path",)
        elif "path" not in props:
            raise ValueError("from: path needs a path prop")

    def execute(self, store):
        if self.source == "solution":
            npz = Path(store.get("solve.provenance")["path"])
        elif self.source == "output":
            npz = Path(store.get("output.path"))
        else:
            npz = Path(str(self.props["path"]))

        dep = str(self.props.get("deposit", "gaussian"))
        st = float(self.props.get("sigma_time", 0.005))
        sp = float(self.props.get("sigma_pixel", 0.5))
        thr = float(self.props.get("corr_threshold", 0.5))
        anchor = str(self.props.get("edge_anchor", "universal"))
        tnpz = self.props.get("truth_npz")
        which = str(self.props.get("time_offsets", "none"))
        if which not in ("none", "file"):
            raise ValueError("time_offsets must be none|file")
        to = None
        if which == "file":
            z = np.load(npz, allow_pickle=True)
            if "deconv_q_offsets" not in z.files:
                raise KeyError(f"{npz.name} carries no deconv_q_offsets")
            to = np.asarray(z["deconv_q_offsets"], dtype=np.float64)

        tmp = None
        try:
            if tnpz is None and self.truth_mode == "recompute":
                # the embedded smeared_true is at the WARM START's width, not
                # this one; scoring against it changes every truth-dependent
                # metric.  Recompute at the widths actually declared here.
                ev = store.get("event")
                off, smt = smear_effective_charge(ev, sigma_time=st,
                                                  sigma_pixel=sp)
                tmp = tempfile.TemporaryDirectory(prefix="eval_truth_")
                tnpz = str(Path(tmp.name) / "truth.npz")
                np.savez(tnpz, smeared_true=smt,
                         smear_offset=np.asarray(off))
            truth, reco, origin = universal_rebin(
                npz, truth_npz=(Path(tnpz) if tnpz else None),
                deposit_shape=dep, sigma_time=st, sigma_pxl=sp,
                time_offsets=to, edge_anchor=anchor, return_origin=True)
        finally:
            if tmp is not None:
                tmp.cleanup()
        metrics = metrics_from_blocks(truth, reco, corr_threshold=thr)
        protocol = {"npz": str(npz), "deposit_shape": dep,
                    "sigma_time": st, "sigma_pixel": sp,
                    "corr_threshold": thr, "edge_anchor": anchor,
                    "time_offsets": which,
                    "truth": self.truth_mode,
                    "truth_npz": (str(tnpz) if tnpz else None)}
        print("[Evaluate] %s deposit, sigma=(%.3f, %.2f), offsets=%s -> "
              "integral %+.2f%%  r %.4f  slope %.3f  ghost %.2f%%  killed %.1f"
              % (dep, st, sp, which, metrics["integral_pct"],
                 metrics["pearson_r"], metrics["slope"],
                 100 * metrics["ghost_frac"], metrics["true_killed"]))
        self.put(store, "eval.truth", truth)
        self.put(store, "eval.reco", reco)
        self.put(store, "eval.origin", origin)
        self.put(store, "eval.metrics", metrics)
        self.put(store, "eval.protocol", protocol)


@algorithm("LumpedAllocation")
class LumpedAllocation(Algorithm):
    """Is the over-book concentrated in the bin a long window ends in?

    Ports the four copies of the lumped-window allocation test
    (``operator_mechanism/lastlatch.py``, ``pkq_check/lastlatch_pkq{,_note}.py``,
    ``eval_centers/lastlatch_centers.py``), which differ only in the deposit
    protocol and in the ``PKQ`` constant.

    Each latch instant is mapped to its universal-grid bin through the kernel
    peak offset ``PKQ``, and the ratio reco/truth in that bin is stratified by
    the length of the window ending there.  Only rows that end at a latch are
    used (``remainder``, ``lumped``, ``diff``); a bin reached by several windows
    takes the longest.

    Props: ``pkq`` (default 127 -- settled in ``pkq_check/``), ``truth_floor``
    (default 2.0 ke, the bin must hold this much truth to be scored),
    ``high_floor`` (default 5.0 ke for the high-charge subset), ``edges``
    (window-length strata in bins, default [1.2, 2.5]).
    """

    reads = ("eval.truth", "eval.reco", "eval.origin", "row_meta",
             "readout_config", "block_offset")
    writes = ("lumped.summary",)

    def execute(self, store):
        truth = np.asarray(store.get("eval.truth"), dtype=np.float64)
        reco = np.asarray(store.get("eval.reco"), dtype=np.float64)
        org = store.get("eval.origin")
        rm = store.get("row_meta")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        B = float(org["bin_ticks"])
        pkq = int(self.props.get("pkq", 127))
        floor = float(self.props.get("truth_floor", 2.0))
        hi_floor = float(self.props.get("high_floor", 5.0))
        edges = [float(x) for x in self.props.get("edges", [1.2, 2.5])]
        u_min = int(org["u_min"])
        px0, py0 = org["p_min"]

        # bin reached by each latch, and the longest window ending there
        best: dict[tuple[int, int, int], float] = {}
        n_rows = len(rm["kind"])
        for r in range(n_rows):
            if rm["kind"][r] not in ("remainder", "lumped", "diff"):
                continue
            X = int(rm["px"][r] + boff[0]) - int(px0)
            Y = int(rm["py"][r] + boff[1]) - int(py0)
            jb = int(np.floor((float(rm["t_hi"][r]) + boff[2]) / B)) - pkq - u_min
            wl = (float(rm["t_hi"][r]) - max(float(rm["t_lo"][r]), 0.0)) / B
            k = (X, Y, jb)
            best[k] = max(best.get(k, 0.0), wl)

        def stratum(wl):
            for i, e in enumerate(edges):
                if wl <= e:
                    return i
            return len(edges)

        names = ([f"win<={edges[0]}"]
                 + [f"win {edges[i]}-{edges[i + 1]}"
                    for i in range(len(edges) - 1)]
                 + [f"win >{edges[-1]}"])
        acc: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
        n_out = 0
        for (X, Y, j), wl in best.items():
            if not (0 <= X < truth.shape[0] and 0 <= Y < truth.shape[1]
                    and 0 <= j < truth.shape[2]):
                n_out += 1
                continue
            t = truth[X, Y, j]
            if t < floor:
                continue
            acc[names[stratum(wl)]].append((float(t), float(reco[X, Y, j] / t)))

        summary = {"pkq": pkq, "truth_floor_ke": floor,
                   "high_floor_ke": hi_floor, "edges_bins": edges,
                   "n_latch_bins": len(best), "n_outside_grid": n_out,
                   "protocol": store.get("eval.protocol"), "strata": {}}
        for nm in names:
            v = acc[nm]
            if not v:
                summary["strata"][nm] = None
                continue
            t = np.array([a for a, _ in v])
            r = np.array([b for _, b in v])
            hi = t >= hi_floor
            summary["strata"][nm] = {
                "n": len(v), "median": float(np.median(r)),
                "mean": float(r.mean()),
                "frac_gt_1.1": float((r > 1.1).mean()),
                "n_high": int(hi.sum()),
                "median_high": (float(np.median(r[hi])) if hi.sum() > 2
                                else None)}
        for nm in names:
            v = summary["strata"][nm]
            if v:
                print("[LumpedAllocation] %-14s N=%4d med=%.3f frac>1.1=%.2f"
                      " | truth>=%.0f: N=%3d med=%s"
                      % (nm, v["n"], v["median"], v["frac_gt_1.1"], hi_floor,
                         v["n_high"],
                         "n/a" if v["median_high"] is None
                         else "%.3f" % v["median_high"]))
        self.put(store, "lumped.summary", summary)
