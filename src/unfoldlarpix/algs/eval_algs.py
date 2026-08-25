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
      ``out_prefix``  default ``eval``; set it to run two protocols in one
                     sequence, which is what a deposit A/B needs.
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
        # instance-level writes: the store is write-once, so two protocols in
        # one sequence need two prefixes.  Same pattern as BuildTruth.
        self.prefix = str(props.get("out_prefix", "eval"))
        self.writes = tuple(f"{self.prefix}.{k}" for k in
                            ("truth", "reco", "origin", "metrics", "protocol"))
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
        tnpz_declared = tnpz          # what the CALLER asked for; a temp file
                                      # made here is an implementation detail
                                      # and must not enter the protocol record,
                                      # or two identical runs look different
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
                    "truth_npz": (str(tnpz_declared) if tnpz_declared
                                  else None)}
        print("[Evaluate] %s deposit, sigma=(%.3f, %.2f), offsets=%s -> "
              "integral %+.2f%%  r %.4f  slope %.3f  ghost %.2f%%  killed %.1f"
              % (dep, st, sp, which, metrics["integral_pct"],
                 metrics["pearson_r"], metrics["slope"],
                 100 * metrics["ghost_frac"], metrics["true_killed"]))
        self.put(store, f"{self.prefix}.truth", truth)
        self.put(store, f"{self.prefix}.reco", reco)
        self.put(store, f"{self.prefix}.origin", origin)
        self.put(store, f"{self.prefix}.metrics", metrics)
        self.put(store, f"{self.prefix}.protocol", protocol)


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
             "readout_config")
    writes = ("lumped.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        # follow whichever Evaluate produced the blocks, so an A/B can score
        # both protocols in one sequence
        self.src = str(props.get("eval_prefix", "eval"))
        self.reads = ((f"{self.src}.truth", f"{self.src}.reco",
                       f"{self.src}.origin", f"{self.src}.protocol")
                      + ("row_meta", "readout_config"))
        self.writes = (f"{str(props.get('out_prefix', 'lumped'))}.summary",)

    def execute(self, store):
        truth = np.asarray(store.get(f"{self.src}.truth"), dtype=np.float64)
        reco = np.asarray(store.get(f"{self.src}.reco"), dtype=np.float64)
        org = store.get(f"{self.src}.origin")
        rm = store.get("row_meta")
        # the UNIVERSAL grid's own offset, not the fit grid's: the NPZ carries
        # boffset (charge centre) and boffset_raw (corner) and they differ by
        # B/2, so taking block_offset here shifts every latch bin by half a bin
        boff = np.asarray(org["b_off"], dtype=float)
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
                   "protocol": store.get(f"{self.src}.protocol"), "strata": {}}
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


@algorithm("PooledCorrelation")
class PooledCorrelation(Algorithm):
    """Fit-grid correlation against the truth, at several pooling scales.

    Ports the ``fit_pooling`` family (one copy per truth convention).  The
    question it answers is whether a low bin-level correlation is mostly
    plus-or-minus-one-bin misplacement: if it is, pooling in time must raise
    it sharply, and pooling in pixels must not.

    Everything is on the FIT grid with no post-processing -- ``solve.q``
    against ``truth.q`` -- so this is deliberately not the universal-grid
    evaluation of :class:`Evaluate` and the two are not comparable.

    Two selections per pooling, both reported because they answer different
    questions: ``cut`` keeps pooled voxels the reconstruction put charge in
    (``R > cut``), ``nc`` keeps the union of either side above ``eps``.

    Props: ``pools`` (list of ``[name, group_pixels, group_time]``; default
    the five of the archived study), ``cut`` (0.5 ke), ``eps`` (0.01 ke),
    ``truth_prefix``.
    """

    reads = ("solve.q",)
    writes = ("pooled.summary",)

    _POOLS = (("1x1x1", 1, 1), ("1x1x2", 1, 2), ("1x1x4", 1, 4),
              ("2x2x2", 2, 2), ("3x3x3", 3, 3))

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",)

    @staticmethod
    def _rs(x, y, m):
        x, y = x[m], y[m]
        if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan"), float("nan"), int(m.sum())
        return (float(np.corrcoef(x, y)[0, 1]),
                float(np.polyfit(x, y, 1)[0]), int(m.sum()))

    def execute(self, store):
        from ..eval.universal import pool_block
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        t = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        if q.shape != t.shape:
            raise ValueError(f"solve.q {q.shape} and {self.prefix}.q "
                             f"{t.shape} are on different grids")
        cut = float(self.props.get("cut", 0.5))
        eps = float(self.props.get("eps", 0.01))
        pools = [tuple(p) for p in self.props.get("pools", self._POOLS)]

        rec = {"cut_ke": cut, "eps_ke": eps, "grid": "fit",
               "truth_convention": store.get(f"{self.prefix}.meta")["convention"]
               if f"{self.prefix}.meta" in store else None,
               "pools": {}}
        for name, gp, gt in pools:
            T = pool_block(t, int(gp), int(gt)).ravel()
            R = pool_block(q, int(gp), int(gt)).ravel()
            r_c, s_c, n_c = self._rs(T, R, R > cut)
            r_n, s_n, n_n = self._rs(T, R, (R > eps) | (T > eps))
            rec["pools"][name] = {"r_cut": r_c, "sl_cut": s_c, "n": n_c,
                                  "r_nc": r_n, "sl_nc": s_n, "n_nc": n_n}
        print("[PooledCorrelation] r_cut " + " ".join(
            "%s=%.3f" % (k, v["r_cut"]) for k, v in rec["pools"].items()))
        self.put(store, "pooled.summary", rec)


@algorithm("ResidualTimeCorrelation")
class ResidualTimeCorrelation(Algorithm):
    """Is the fit-grid residual anti-correlated along time?

    Ports the ``fitgrid_resid_corr`` family.  An alternating (zero-sum) error
    -- charge taken from one bin and put in its neighbour -- shows up as a
    negative lag-1 correlation of ``q_hat - q_truth`` along the time axis
    within a pixel, which is what the charge-space weak modes are made of.

    Props: ``eps`` (0.01 ke; a voxel enters if either side is above it),
    ``max_lag`` (default 3), ``truth_prefix``.
    """

    reads = ("solve.q",)
    writes = ("residcorr.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",)

    def execute(self, store):
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        t = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        eps = float(self.props.get("eps", 0.01))
        max_lag = int(self.props.get("max_lag", 3))
        e = q - t
        live = (np.abs(q) > eps) | (np.abs(t) > eps)
        rec = {"eps_ke": eps, "grid": "fit",
               "n_live_voxels": int(live.sum()), "lags": {}}
        for lag in range(1, max_lag + 1):
            a = e[:, :, :-lag].ravel()
            b = e[:, :, lag:].ravel()
            m = (live[:, :, :-lag] & live[:, :, lag:]).ravel()
            x, y = a[m], b[m]
            if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
                rec["lags"][lag] = None
                continue
            rec["lags"][lag] = {"n": int(m.sum()),
                                "corr": float(np.corrcoef(x, y)[0, 1])}
        s = e[live]
        rec["residual"] = {"sum_ke": float(e.sum()), "mean_ke": float(s.mean()),
                           "rms_ke": float(np.sqrt((s ** 2).mean()))}
        print("[ResidualTimeCorrelation] " + " ".join(
            "lag%d=%s" % (k, "n/a" if v is None else "%+.3f" % v["corr"])
            for k, v in rec["lags"].items())
            + "  sum %+.1f ke" % rec["residual"]["sum_ke"])
        self.put(store, "residcorr.summary", rec)


@algorithm("ChargeProfile")
class ChargeProfile(Algorithm):
    """Charge-weighted reco/truth ratio in bins of TRUTH charge.

    Ports ``eval_centers/highq_profile_centers.py`` and the ``highq_all.py``
    family.  On the universal grid, voxels are binned by how much truth charge
    they hold and the ratio ``sum(reco)/sum(truth)`` is formed per bin.  This
    is the quantity that showed a high-q over-book at the retired offsets
    deposit and none at bin centres, so the deposit protocol is part of the
    answer and :class:`Evaluate` records which one produced these blocks.

    Consumes ``eval.truth``/``eval.reco`` -- it does not rebin -- so a profile
    and the scalar metrics beside it are provably of the same evaluation.

    Props: ``edges`` (bin edges in ke; default the archived
    ``[0.5, 1, 2, 3, 4, 5, 7, 10, 100]``), ``min_voxels`` (5), ``grid``
    (``universal`` by default, or ``fit`` to profile ``solve.q`` against
    ``truth.q`` with no post-processing -- that is
    ``noiseless_closure_round/highq_fitgrid.py``), ``select_by`` (``truth``
    by default, or ``reco``).

    ``select_by`` matters and the archived script says why: binning by RECO
    regresses to the mean and MANUFACTURES an overestimate in the top bin,
    because a voxel lands there partly by having been over-booked. Binning by
    truth is the honest choice; both are offered so the artefact stays
    visible rather than being an unstated convention.
    """

    reads = ("eval.truth", "eval.reco", "eval.protocol")
    writes = ("chargeprofile.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.grid = str(props.get("grid", "universal"))
        if self.grid not in ("universal", "fit"):
            raise ValueError("grid must be universal|fit")
        self.select_by = str(props.get("select_by", "truth"))
        if self.select_by not in ("truth", "reco"):
            raise ValueError("select_by must be truth|reco")
        self.src = str(props.get("eval_prefix", "eval"))
        self.tp = str(props.get("truth_prefix", "truth"))
        self.reads = (tuple(f"{self.src}.{k}" for k in
                            ("truth", "reco", "protocol"))
                      if self.grid == "universal"
                      else ("solve.q", f"{self.tp}.q"))
        self.writes = (f"{str(props.get('out_prefix', 'chargeprofile'))}"
                       ".summary",)

    def execute(self, store):
        if self.grid == "universal":
            t = np.asarray(store.get(f"{self.src}.truth"), np.float64).ravel()
            r = np.asarray(store.get(f"{self.src}.reco"), np.float64).ravel()
        else:
            t = np.asarray(store.get(f"{self.tp}.q"), np.float64).ravel()
            r = np.asarray(store.get("solve.q"), np.float64).ravel()
        edges = [float(x) for x in self.props.get(
            "edges", [0.5, 1, 2, 3, 4, 5, 7, 10, 100])]
        floor = edges[0]
        nmin = int(self.props.get("min_voxels", 5))
        key = t if self.select_by == "truth" else r
        m = key > floor
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = m & (key >= lo) & (key < hi)
            n = int(sel.sum())
            if n < nmin:
                continue
            ts = float(t[sel].sum())
            rows.append({"lo_ke": lo, "hi_ke": hi, "n": n,
                         "sum_truth_ke": ts,
                         "sum_reco_ke": float(r[sel].sum()),
                         "ratio": float(r[sel].sum() / ts) if ts else None})
        rec = {"edges_ke": edges, "truth_floor_ke": floor,
               "protocol": (store.get(f"{self.src}.protocol")
                            if self.grid == "universal" else None),
               "n_voxels_above_floor": int(m.sum()), "bins": rows}
        print("[ChargeProfile] " + "  ".join(
            "[%g,%g) %.3f" % (b["lo_ke"], b["hi_ke"], b["ratio"])
            for b in rows))
        self.put(store, self.writes[0], rec)


@algorithm("ProtocolAB")
class ProtocolAB(Algorithm):
    """Compare two evaluations of the same solve, metric by metric.

    Ports `offsets_ab{,_noisy}.py`, `uni_centers{,_round}.py` and
    `uniform_ab.py`, which all do one thing: score a solve twice and look at
    what moved.  The two evaluations must differ in exactly one declared
    setting -- that is checked here, so an A/B cannot silently compare two
    things that differ in two ways.

    Props: ``a``, ``b`` (the two `Evaluate` prefixes), ``keys`` (metrics to
    compare; default all shared numeric ones).
    """

    writes = ("protocolab.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.a = str(props.get("a", "eval_a"))
        self.b = str(props.get("b", "eval_b"))
        self.reads = (f"{self.a}.metrics", f"{self.a}.protocol",
                      f"{self.b}.metrics", f"{self.b}.protocol")

    def execute(self, store):
        pa = store.get(f"{self.a}.protocol")
        pb = store.get(f"{self.b}.protocol")
        diff = {k: (pa.get(k), pb.get(k)) for k in set(pa) | set(pb)
                if pa.get(k) != pb.get(k)}
        if len(diff) != 1:
            raise ValueError(
                f"an A/B must differ in exactly one setting; these differ in "
                f"{len(diff)}: {diff}")
        (knob, (va, vb)), = diff.items()
        ma, mb = store.get(f"{self.a}.metrics"), store.get(f"{self.b}.metrics")
        keys = self.props.get("keys") or sorted(
            k for k in set(ma) & set(mb)
            if isinstance(ma[k], (int, float)) and not isinstance(ma[k], bool))
        rows = {}
        for k in keys:
            x, y = float(ma[k]), float(mb[k])
            rows[k] = {"a": x, "b": y, "delta": y - x,
                       "rel": ((y - x) / x) if x else None}
        rec = {"knob": knob, "a_value": va, "b_value": vb,
               "a_prefix": self.a, "b_prefix": self.b, "metrics": rows}
        print("[ProtocolAB] %s: %s -> %s  " % (knob, va, vb) + "  ".join(
            "%s %+.4g" % (k, rows[k]["delta"]) for k in
            ("integral_pct", "pearson_r", "slope", "true_killed")
            if k in rows))
        self.put(store, "protocolab.summary", rec)
