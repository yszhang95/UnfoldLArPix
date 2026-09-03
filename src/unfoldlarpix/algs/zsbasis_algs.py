"""Zero-suppressed self-triggered readout on three time bases.

The campaigns before this one all read a ``fixed_interval`` file: every
channel is sampled every 30 fine ticks, no threshold, no reset, and the
per-channel acceptance weight is 1 by construction.  This module reads a
``nd_readout`` file, where the sampling instants are a functional of the
charge (``FORWARD_MODEL_revised.md`` Sec. 4.2) and most of the record grid
does not exist.

Definitions
-----------
Everything from :mod:`unfoldlarpix.algs.evalharness_algs` (``c_k``, ``R``,
``P_0``, ``H``, ``E_rel``, the per-ring sums), from
:mod:`unfoldlarpix.algs.finebasis_algs` (``CellGrid``, ``score_rows``,
``fine_xhat``) and from :mod:`unfoldlarpix.algs.fixedgrid_algs`
(``solve_arm``, ``resolve_support``, ``_SupportProx``) is used with the same
meaning.  The names introduced HERE are:

``the c-tick cell basis``
    unknown ``x_p[m]``, one number per pad ``p`` and per cell
    ``m`` of ``c`` consecutive fine ticks starting at ``b + c m`` with
    ``b = block_offset[2]``.  ``c = 1`` is the fine 50 ns tick.  The charge
    of a cell is released UNIFORMLY over its ``c`` ticks (``P_0``), which is
    what :class:`~unfoldlarpix.model.subbin_operator.ZSOperatorUniform`
    implements and what the scoring prolongs with.

``first-window lower edge at acquisition start``
    ``build_latch_rows(..., acq_start=None)``: the lower edge of a pixel's
    FIRST integration window is ``-inf``, clipped by
    ``windows_to_sampling`` to the start of the block.  The row then
    integrates the whole kernel that the block carries.

``first-window lower edge at the event t0``
    ``build_latch_rows(..., acq_start=0)``: the lower edge is absolute fine
    tick 0, the creation time.  ``graph_effq.py:148-159`` zeroes every
    current sample before ``t0``, so this is the edge the simulation used.

``exact-row condition``
    the window edges of a ``nd_readout`` record are integer FINE ticks that
    are NOT multiples of 30, so a row is an exact functional of the charge
    only when the sampling grid is the fine tick.  Every operator here
    therefore samples at one fine tick (``ZSOperator`` with
    ``adc_hold_delay = 1``, or ``ZSOperatorUniform`` with
    ``adc_hold_delay = c, subbin = c``, whose sampling bin is ``c/c = 1``);
    the coarseness is carried by the UNKNOWN alone.

``registration shift delta``
    the integer offset used to place a truth deposit at absolute fine tick
    ``T`` into cell ``m = floor((T + delta - b)/c)``.  The shipped operator
    releases the charge of fine index ``j`` one tick after ``b + j``
    (``STUDIES_isoline_d16p5.md`` Sec. 4.1), so ``delta = -1`` is the
    convention-consistent value; it is SCANNED and reported, never assumed.

``E_rel``
    ``sum |H P x_hat - H x| / sum |H x|`` of ``score_rows``, with ``P = P_0``
    and ``H`` the Gaussian of width ``sigma_H``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..constrained_solver import build_latch_rows
from ..fwk.component import Algorithm, algorithm
from ..model.operator import ZSOperator
from ..model.subbin_operator import ZSOperatorUniform
from .evalharness_algs import EvalHarness
from .finebasis_algs import fine_xhat, score_rows
from .fixedgrid_algs import _SupportProx, solve_arm  # noqa: F401

CONVENTIONS = {"acq_edge": None, "acq_t0": 0.0}
CONV_LABEL = {"acq_edge": "first-window lower edge at acquisition start",
              "acq_t0": "first-window lower edge at the event t0"}


# ---------------------------------------------------------------------------
# windows and operators
# ---------------------------------------------------------------------------
def zs_windows(store, conv: str, acq_start="convention"):
    """``build_latch_rows`` for one first-window convention.

    ``B = 30`` fine ticks, ``csa_reset_time`` from the file (2 ticks),
    ``split_threshold = None`` (no trigger split), ``burst_tau = None``.
    ``acq_start`` overrides the convention table with an explicit edge in
    absolute fine ticks; it exists so a cross-check can be built on the SAME
    first-window edge as the operator it is compared with.
    """
    ev = store.get("event")
    rc = store.get("readout_config")
    boff = np.asarray(store.get("block_offset"))
    acq = CONVENTIONS[conv] if acq_start == "convention" else acq_start
    return build_latch_rows(
        ev.hits.location, ev.hits.data, int(rc.adc_hold_delay), boff,
        csa_reset_time=int(rc.csa_reset_time),
        split_threshold=None, acq_start=acq, burst_tau=None)


class _BasisJob:
    """Shared machinery: kernel, operators, truth, support, scoring."""

    def __init__(self, alg, store):
        self.alg = alg
        self.store = store
        self.comp = alg.services["compute"]
        self.det = alg.services["detector"]
        self.rc = store.get("readout_config")
        self.ev = store.get("event")
        self.block = np.asarray(store.get("block"))
        self.boff = np.asarray(store.get("block_offset"), dtype=float)
        self.B = int(self.rc.adc_hold_delay)
        self.nx, self.ny, self.nt = (int(v) for v in self.block.shape)
        self.b0 = int(self.boff[2])
        self.fine = self.det.prepared_raw(1)
        self.K1 = np.asarray(self.fine.integrated_response)
        self.kt = int(self.K1.shape[2])
        self.nt_fine = self.nt * self.B
        # the effq truth of the event, in absolute fine ticks
        el = np.asarray(self.ev.effq.location)
        eq = np.asarray(self.ev.effq.data, dtype=float)[:, -1]
        self.truth_ix = el[:, 0].astype(int) - int(self.boff[0])
        self.truth_iy = el[:, 1].astype(int) - int(self.boff[1])
        self.truth_tick = el[:, 2].astype(np.int64)
        self.truth_q = eq
        self.truth_total = float(eq.sum())
        self._ops: dict = {}

    # -- kernel identity ---------------------------------------------------
    def kernel_report(self) -> dict:
        from ..deconv_workflow import integrate_kernel_over_time
        one = integrate_kernel_over_time(self.fine.full_response, 1)
        return {
            "kernel_shape": list(self.K1.shape),
            "sum_K1": float(self.K1.sum()),
            "integrate_over_1_tick_is_identity_max_abs_diff":
                float(np.abs(one - self.fine.full_response[..., :one.shape[-1]]
                             ).max()),
            "integrate_over_1_tick_is_prepared_max_abs_diff":
                float(np.abs(one - self.K1).max()),
        }

    # -- operators ----------------------------------------------------------
    def operator(self, c: int, conv: str, windows=None):
        key = (int(c), conv)
        if key in self._ops:
            return self._ops[key]
        if windows is None:
            windows, _ = zs_windows(self.store, conv)
        c = int(c)
        if c == 1:
            op = ZSOperator(self.K1, (self.nx, self.ny, self.nt_fine),
                            windows, 1, device=self.comp.device,
                            dtype=self.comp.dtype)
        else:
            if self.nt_fine % c:
                raise ValueError(f"fine block {self.nt_fine} is not a "
                                 f"multiple of the cell width {c}")
            op = ZSOperatorUniform(self.K1,
                                   (self.nx, self.ny, self.nt_fine // c),
                                   windows, c, c, device=self.comp.device,
                                   dtype=self.comp.dtype)
        self._ops[key] = op
        return op

    # -- truth on the c-tick basis -----------------------------------------
    def truth_on_basis(self, op, c: int, delta: int = -1) -> np.ndarray:
        """``R_c x`` with the registration shift ``delta``."""
        nx, ny, n = op.q_shape
        m = (self.truth_tick + int(delta) - self.b0) // int(c)
        ok = ((m >= 0) & (m < n) & (self.truth_ix >= 0) & (self.truth_ix < nx)
              & (self.truth_iy >= 0) & (self.truth_iy < ny))
        out = np.zeros((nx, ny, n))
        np.add.at(out, (self.truth_ix[ok], self.truth_iy[ok], m[ok]),
                  self.truth_q[ok])
        return out

    # -- support ------------------------------------------------------------
    def support_on_basis(self, op, c: int) -> np.ndarray:
        """The store's ``BuildSupport source: hits`` mask lifted to the basis.

        The stored mask lives on the production 30-tick fit grid.  Cell ``m``
        of the ``c``-tick basis starts at fine index ``c m`` of the fine
        charge grid, which lies in coarse bin ``(c m) // 30``.
        """
        base = np.asarray(self.store.get("support"))
        nx, ny, n = op.q_shape
        kk = np.clip((np.arange(n) * int(c)) // self.B, 0, base.shape[2] - 1)
        return np.ascontiguousarray(base[:nx, :ny, :][:, :, kk])

    # -- prolongation to the fine grid, for scoring -------------------------
    def to_fine(self, x: np.ndarray, c: int) -> np.ndarray:
        """``P_0 x`` on ``(n_pads, n_fine_ticks)``, absolute origin ``b0``."""
        c = int(c)
        if c == 1:
            f = np.asarray(x, dtype=float)
        else:
            f = np.repeat(np.asarray(x, dtype=float), c, axis=2) / c
        return f.reshape(-1, f.shape[-1])

    def rel_residual(self, op, x) -> float:
        xt = op.to_tensor(np.ascontiguousarray(x))
        r = op.forward(xt) - op.d
        return float(torch.linalg.vector_norm(r)
                     / torch.linalg.vector_norm(op.d))


def ring_sums(H: EvalHarness, x_fine: np.ndarray) -> dict:
    """Signed charge per pixel, grouped by Chebyshev distance from the
    ionised pixel row: 0 (ionised), +1, +2 and >= +3 pixels."""
    tot = x_fine.sum(axis=1)
    out = {}
    for lab, sel in (("ionised", H.chebyshev == 0), ("plus1", H.chebyshev == 1),
                     ("plus2", H.chebyshev == 2),
                     ("plus3_or_more", H.chebyshev >= 3)):
        v = tot[sel]
        out[lab] = {"n_pixels": int(sel.sum()),
                    "sum_ke": float(v.sum()),
                    "sum_positive_ke": float(v[v > 0].sum()),
                    "sum_negative_ke": float(v[v < 0].sum())}
    return out


class _JsonAlg(Algorithm):
    """Per-event records dumped at ``finalize()`` with recipe + provenance."""

    def initialize(self, services):
        super().initialize(services)
        self._records: list[dict] = []
        self._recipe: dict = {}
        self.out_path = self.props.get("out_json")

    def _emit(self, store, rec):
        self._records.append(rec)
        if not self._recipe:
            try:
                self._recipe = {"job_config": store.get("job.config"),
                                "provenance": store.provenance()}
            except Exception as exc:
                self._recipe = {"job_config_error": str(exc)}

    def finalize(self):
        if not self._records:
            return {}
        body = (self._records[0] if len(self._records) == 1
                else {"events": self._records})
        if self.out_path:
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "w") as fh:
                json.dump({"algorithm": self.name, "result": body,
                           **self._recipe}, fh, indent=1, default=str)
            print(f"[{self.name}] wrote {self.out_path}")
        return body


# ---------------------------------------------------------------------------
@algorithm("ZSSampleFacts")
class ZSSampleFacts(_JsonAlg):
    """What the zero-suppressed file contains, measured, not assumed.

    Records per pixel, the trigger and hold times, the recorded total against
    the created charge, and the pixels with records but no ionisation.
    """

    reads = ("event", "readout_config", "hits_view")
    writes = ("zs.sample",)

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        loc = np.asarray(ev.hits.location)
        dat = np.asarray(ev.hits.data, dtype=float)
        el = np.asarray(ev.effq.location)
        eq = np.asarray(ev.effq.data, dtype=float)[:, -1]
        px, py = loc[:, 0].astype(int), loc[:, 1].astype(int)
        q = dat[:, 3:]
        created = {}
        for a, b, v in zip(el[:, 0].astype(int), el[:, 1].astype(int), eq):
            created[(a, b)] = created.get((a, b), 0.0) + float(v)
        ion = np.array(sorted(created)) if created else np.zeros((0, 2), int)

        def cheb(p):
            return int(np.min(np.max(np.abs(ion - np.asarray(p)), axis=1)))

        per = {}
        for i in range(len(px)):
            per.setdefault((px[i], py[i]), []).append(i)
        hist = {}
        for v in per.values():
            hist[len(v)] = hist.get(len(v), 0) + 1
        groups = {}
        for p, idx in per.items():
            d = cheb(p)
            lab = ("ionised" if d == 0 else "plus1" if d == 1
                   else "plus2" if d == 2 else "plus3_or_more")
            g = groups.setdefault(lab, {"n_pixels": 0, "n_records": 0,
                                        "recorded_ke": 0.0, "created_ke": 0.0,
                                        "pixels": []})
            g["n_pixels"] += 1
            g["n_records"] += len(idx)
            g["recorded_ke"] += float(q[idx].sum())
            g["created_ke"] += float(created.get(p, 0.0))
            if lab != "ionised":
                g["pixels"].append({"pixel_x": int(p[0]), "pixel_y": int(p[1]),
                                    "n_records": len(idx),
                                    "recorded_ke": float(q[idx].sum()),
                                    "created_ke": float(created.get(p, 0.0)),
                                    "trigger": [int(t) for t in loc[idx, 2]],
                                    "hold": [int(t) for t in loc[idx, 3]]})
        probe = [int(v) for v in self.props.get("probe_pixel", [141, 68])]
        pk = (probe[0], probe[1])
        pinfo = {"pixel": probe, "n_records": 0}
        if pk in per:
            idx = per[pk]
            pinfo = {"pixel": probe, "n_records": len(idx),
                     "trigger_ticks": [int(t) for t in loc[idx, 2]],
                     "hold_ticks": [int(t) for t in loc[idx, 3]],
                     "next_start_ticks": [int(t) for t in loc[idx, 4]],
                     "recorded_ke": [float(v) for v in q[idx].ravel()],
                     "created_ke": float(created.get(pk, 0.0))}
        rec = {
            "readout_model": str(store.get("event").__class__.__name__),
            "n_records": int(len(px)),
            "n_pixels_with_records": int(len(per)),
            "records_per_pixel_histogram": {str(k): int(v)
                                            for k, v in sorted(hist.items())},
            "n_ionised_pixels": int(len(created)),
            "sum_effq_ke": float(eq.sum()),
            "sum_recorded_ke": float(q.sum()),
            "recorded_over_created": float(q.sum() / eq.sum()),
            "by_distance": groups,
            "probe_pixel": pinfo,
            "trigger_tick_min": int(loc[:, 2].min()),
            "trigger_tick_max": int(loc[:, 2].max()),
            "readout_config": {"adc_hold_delay": int(rc.adc_hold_delay),
                               "adc_down_time": int(rc.adc_down_time),
                               "csa_reset_time": int(rc.csa_reset_time),
                               "one_tick": int(rc.one_tick),
                               "nburst": int(rc.nburst),
                               "threshold_ke": float(rc.threshold)},
        }
        # per-pixel table for the figures
        rec["_table"] = {
            "pixel_x": [int(p[0]) for p in per],
            "pixel_y": [int(p[1]) for p in per],
            "n_records": [len(v) for v in per.values()],
            "first_trigger": [int(loc[v[0], 2]) for v in per.values()],
            "recorded_ke": [float(q[v].sum()) for v in per.values()],
            "created_ke": [float(created.get(p, 0.0)) for p in per],
        }
        print(f"[ZSSampleFacts] {rec['n_records']} records on "
              f"{rec['n_pixels_with_records']} pixels; sum y "
              f"{rec['sum_recorded_ke']:.2f} ke = "
              f"{rec['recorded_over_created']:.5f} of the created charge")
        self.put(store, "zs.sample", rec)
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("ZSOperatorCheck")
class ZSOperatorCheck(_JsonAlg):
    """Step 2: the three bases and the two first-window conventions.

    Reports the kernel identity, the row counts, the registration scan and
    ``||A x_truth - y|| / ||y||`` overall, on the first-window rows and on the
    later rows, for every (basis, convention).
    """

    reads = ("event", "readout_config", "block", "block_offset", "op")
    writes = ("zs.opcheck",)

    def execute(self, store):
        J = _BasisJob(self, store)
        cells = [int(c) for c in self.props.get("cell_ticks", [1, 5, 30])]
        deltas = [int(d) for d in self.props.get("registration_scan",
                                                 [-2, -1, 0, 1])]
        rec = {"kernel": J.kernel_report(),
               "block_shape": [J.nx, J.ny, J.nt],
               "block_offset": [float(v) for v in J.boff],
               "fine_block_ticks": int(J.nt_fine),
               "truth_total_ke": J.truth_total,
               "conventions": {}}
        for conv in self.props.get("conventions", ["acq_edge", "acq_t0"]):
            windows, metas = zs_windows(store, conv)
            kinds = [m.kind for m in metas]
            first = np.array([not m.post_reset for m in metas])
            cr = {"label": CONV_LABEL[conv],
                  "n_windows": int(len(windows)),
                  "n_first_windows": int(first.sum()),
                  "row_kinds": {k: int(kinds.count(k)) for k in set(kinds)},
                  "bases": {}}
            for c in cells:
                op = J.operator(c, conv, windows)
                y = op.d.cpu().numpy().astype(float)
                scan = {}
                best, best_v = None, np.inf
                for d in deltas:
                    xt = J.truth_on_basis(op, c, d)
                    v = J.rel_residual(op, xt)
                    scan[str(d)] = float(v)
                    if v < best_v:
                        best, best_v = d, v
                delta = int(self.props.get("registration_delta", -1))
                xt = J.truth_on_basis(op, c, delta)
                pred = op.forward(op.to_tensor(xt)).cpu().numpy().astype(float)
                res = pred - y
                fw = first[:op.n_data] if len(first) >= op.n_data else first
                br = {
                    "q_shape": [int(v) for v in op.q_shape],
                    "n_data": int(op.n_data),
                    "n_first_window_rows": int(fw.sum()),
                    "truth_on_basis_ke": float(xt.sum()),
                    "registration_scan_rel_residual": scan,
                    "registration_best_delta": int(best),
                    "registration_delta_used": delta,
                    "rel_residual": float(np.linalg.norm(res)
                                          / np.linalg.norm(y)),
                    "sum_pred_ke": float(pred.sum()),
                    "sum_y_ke": float(y.sum()),
                    "row_residual_mean_ke": float(res.mean()),
                    "row_residual_rms_ke": float(np.sqrt((res ** 2).mean())),
                    "row_residual_max_abs_ke": float(np.abs(res).max()),
                }
                for lab, sel in (("first_window", fw), ("later", ~fw)):
                    if sel.sum():
                        br[lab] = {
                            "n_rows": int(sel.sum()),
                            "sum_y_ke": float(y[sel].sum()),
                            "sum_pred_ke": float(pred[sel].sum()),
                            "sum_residual_ke": float(res[sel].sum()),
                            "mean_residual_ke": float(res[sel].mean()),
                            "rms_residual_ke":
                                float(np.sqrt((res[sel] ** 2).mean())),
                            "rel_residual": float(
                                np.linalg.norm(res[sel])
                                / max(np.linalg.norm(y[sel]), 1e-30)),
                        }
                cr["bases"][str(c)] = br
                print(f"[ZSOperatorCheck] {conv} c={c}: rows {op.n_data} "
                      f"(first {int(fw.sum())}), q {tuple(op.q_shape)}, "
                      f"||A x_truth - y||/||y|| = {br['rel_residual']:.4f}")
            rec["conventions"][conv] = cr
        self.put(store, "zs.opcheck", rec)
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("ZSCoarseCrossCheck")
class ZSCoarseCrossCheck(_JsonAlg):
    """Validation (i): on a FIXED-INTERVAL event the ``c = 30`` cell operator
    reproduces the production coarse operator with ``within_bin: uniform,
    subbin: 30``.

    ``store['op']`` is the production operator built by ``BuildMeasurement``;
    this algorithm builds the cell operator on the same windows and compares
    ``d`` and ``A x`` on random and on truth-shaped ``x``.
    """

    reads = ("event", "readout_config", "block", "block_offset", "op")
    writes = ("zs.crosscheck",)

    def execute(self, store):
        J = _BasisJob(self, store)
        prod = store.get("op")
        conv = str(self.props.get("convention", "acq_edge"))
        acq = self.props.get("acq_start", "convention")
        if acq != "convention" and acq is not None:
            acq = float(acq)
        windows, _ = zs_windows(store, conv, acq_start=acq)
        c = int(self.props.get("cell_ticks", 30))
        op = J.operator(c, conv, windows)
        dp = prod.d.cpu().numpy().astype(float)
        dc = op.d.cpu().numpy().astype(float)
        n = min(len(dp), len(dc))
        nq = min(int(prod.q_shape[2]), int(op.q_shape[2]))
        rng = np.random.default_rng(0)
        xr = rng.random((prod.q_shape[0], prod.q_shape[1], nq))
        xt = J.truth_on_basis(op, c, int(self.props.get(
            "registration_delta", -1)))[:, :, :nq]
        out = {"convention": CONV_LABEL[conv], "cell_ticks": c,
               "acq_start_ticks": (None if acq in (None, "convention")
                                   else float(acq)),
               "n_data_production": int(prod.n_data), "n_data_cell": int(op.n_data),
               "q_shape_production": [int(v) for v in prod.q_shape],
               "q_shape_cell": [int(v) for v in op.q_shape],
               "d_max_abs_diff": float(np.abs(dp[:n] - dc[:n]).max()),
               "d_sum_production_ke": float(dp.sum()),
               "d_sum_cell_ke": float(dc.sum())}
        for lab, x in (("random", xr), ("truth", xt)):
            xp = np.zeros(prod.q_shape)
            xp[:, :, :nq] = x
            xc = np.zeros(op.q_shape)
            xc[:, :, :nq] = x
            a = prod.forward(prod.to_tensor(xp)).cpu().numpy().astype(float)
            bpred = op.forward(op.to_tensor(xc)).cpu().numpy().astype(float)
            scale = max(float(np.abs(a).max()), 1e-30)
            out[f"forward_{lab}_max_abs_diff"] = float(np.abs(a[:n] - bpred[:n]).max())
            out[f"forward_{lab}_max_rel_diff"] = float(
                np.abs(a[:n] - bpred[:n]).max() / scale)
            out[f"forward_{lab}_scale"] = scale
        print(f"[ZSCoarseCrossCheck] d max|diff| {out['d_max_abs_diff']:.3e}; "
              f"forward truth max rel diff "
              f"{out['forward_truth_max_rel_diff']:.3e}")
        self.put(store, "zs.crosscheck", out)
        self._emit(store, out)


# ---------------------------------------------------------------------------
@algorithm("ZSBasisArms")
class ZSBasisArms(_JsonAlg):
    """Step 3: LS, positivity and positivity + l1 on each (basis, convention).

    ``arms`` entries are ``{label, alpha, positivity, iters}``.  ``alpha`` is
    in ke per cell and does NOT scale with ``c`` (column sums are ``sum Kbar``
    at every ``c``; measured in ``STUDIES_isoline_d16p5.md`` Sec. 4.3).
    """

    reads = ("event", "readout_config", "block", "block_offset", "op",
             "support")
    writes = ("zs.arms",)

    def execute(self, store):
        J = _BasisJob(self, store)
        cells = [int(c) for c in self.props.get("cell_ticks", [1, 5, 30])]
        convs = list(self.props.get("conventions", ["acq_edge", "acq_t0"]))
        sigmas = [float(s) for s in self.props.get("sigma_H_us", [1.5, 2.0])]
        delta = int(self.props.get("registration_delta", -1))
        iters_by_c = {int(k): int(v) for k, v in
                      (self.props.get("iters_by_cell") or {}).items()}
        arms_cfg = list(self.props.get("arms", []))
        out_npz = self.props.get("out_npz")
        wave_pixels = [list(map(int, p)) for p in
                       self.props.get("waveform_pixels", [[141, 68],
                                                          [140, 68],
                                                          [142, 68]])]

        H = EvalHarness(store, store.get("op"),
                        margin_windows=int(self.props.get("margin_windows", 40)),
                        line_pixel_y_range=self.props.get(
                            "line_pixel_y_range", (5, 131)),
                        segment_pixels=int(self.props.get("segment_pixels", 7)),
                        segment_edge_exclude=int(
                            self.props.get("segment_edge_exclude", 3)))
        rec = {"truth_total_ke": J.truth_total,
               "sigma_H_us": sigmas,
               "registration_delta": delta,
               "n_pads": int(H.n_pads),
               "arms": [], "data": {}}
        store_npz: dict = {}

        # truth scored the same way, once
        truth_scores = {}
        xt_fine_full = np.zeros((J.nx * J.ny, J.nt_fine))
        ok = ((J.truth_ix >= 0) & (J.truth_ix < J.nx)
              & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
        jj = J.truth_tick[ok] + delta - J.b0
        okj = (jj >= 0) & (jj < J.nt_fine)
        np.add.at(xt_fine_full,
                  ((J.truth_ix[ok][okj] * J.ny + J.truth_iy[ok][okj]),
                   jj[okj]), J.truth_q[ok][okj])
        for s in sigmas:
            truth_scores[str(s)] = {
                k: v for k, v in
                score_rows(H, fine_xhat(H, xt_fine_full, J.b0, s), s).items()
                if not k.startswith("_")}
        rec["truth_scores"] = truth_scores

        for conv in convs:
            windows, metas = zs_windows(store, conv)
            first = np.array([not m.post_reset for m in metas])
            for c in cells:
                op = J.operator(c, conv, windows)
                y = op.d.cpu().numpy().astype(float)
                supp = J.support_on_basis(op, c)
                L = float(op.lipschitz)
                rec["data"][f"{conv}_c{c}"] = {
                    "n_data": int(op.n_data),
                    "n_first_window_rows": int(first[:op.n_data].sum()),
                    "q_shape": [int(v) for v in op.q_shape],
                    "lipschitz": L,
                    "support_fraction": float(supp.mean()),
                    "support_cells": int(supp.sum()),
                    "sum_y_ke": float(y.sum()),
                    "sum_y_over_truth": float(y.sum() / J.truth_total),
                }
                for cfg in arms_cfg:
                    if "cells" in cfg and c not in [int(v) for v in cfg["cells"]]:
                        continue
                    if ("conventions" in cfg
                            and conv not in list(cfg["conventions"])):
                        continue
                    lab = str(cfg["label"])
                    alpha = float(cfg.get("alpha", 0.0))
                    pos = bool(cfg.get("positivity", True))
                    it = int(cfg.get("iters", iters_by_c.get(c, 1000)))
                    if "iters_by_cell" in cfg:
                        it = int(cfg["iters_by_cell"].get(str(c), it))
                    t0 = time.time()
                    x = solve_arm(op, supp, alpha, pos, it)
                    dt = time.time() - t0
                    xf = J.to_fine(x, c)
                    entry = {
                        "convention": conv, "convention_label": CONV_LABEL[conv],
                        "cell_ticks": c, "arm": lab, "alpha_ke_per_cell": alpha,
                        "positivity": pos, "iters": it,
                        "lipschitz": L,
                        "wall_time_s": float(dt),
                        "sum_xhat_ke": float(x.sum()),
                        "sum_xhat_over_truth": float(x.sum() / J.truth_total),
                        "sum_xhat_positive_ke": float(x[x > 0].sum()),
                        "sum_xhat_negative_ke": float(x[x < 0].sum()),
                        "nnz": int((x > 0).sum()),
                        "nnz_1e-3": int((x > 1e-3).sum()),
                        "rel_residual": J.rel_residual(op, x),
                        "pixels": ring_sums(H, xf),
                    }
                    for s in sigmas:
                        m = score_rows(H, fine_xhat(H, xf, J.b0, s), s)
                        entry[f"E_rel_{s}"] = float(m["E_rel"])
                        entry[f"conservation_rel_{s}"] = float(
                            m["conservation_rel"])
                        entry[f"zero_preservation_{s}"] = m["zero_preservation"]
                    rec["arms"].append(entry)
                    print(f"[ZSBasisArms] {conv} c={c} {lab}: "
                          f"sum/truth {entry['sum_xhat_over_truth']:.4f} "
                          f"E_rel(1.5) {entry.get('E_rel_1.5', float('nan')):.4f} "
                          f"res {entry['rel_residual']:.4f} "
                          f"nnz {entry['nnz']} {dt:.1f} s")
                    if out_npz:
                        for pxy in wave_pixels:
                            ip = pxy[0] - int(J.boff[0])
                            iq = pxy[1] - int(J.boff[1])
                            if 0 <= ip < J.nx and 0 <= iq < J.ny:
                                store_npz[
                                    f"wave_{conv}_c{c}_{lab}_"
                                    f"{pxy[0]}_{pxy[1]}"] = \
                                    xf[ip * J.ny + iq].astype(np.float32)
        if out_npz:
            for pxy in wave_pixels:
                ip = pxy[0] - int(J.boff[0])
                iq = pxy[1] - int(J.boff[1])
                if 0 <= ip < J.nx and 0 <= iq < J.ny:
                    store_npz[f"truth_{pxy[0]}_{pxy[1]}"] = \
                        xt_fine_full[ip * J.ny + iq].astype(np.float32)
            store_npz["fine_origin_tick"] = np.array([J.b0])
            store_npz["nt_fine"] = np.array([J.nt_fine])
            # the records of the probe pixels, with their window edges
            ev = store.get("event")
            loc = np.asarray(ev.hits.location)
            dat = np.asarray(ev.hits.data, dtype=float)
            for pxy in wave_pixels:
                m = (loc[:, 0] == pxy[0]) & (loc[:, 1] == pxy[1])
                store_npz[f"rec_{pxy[0]}_{pxy[1]}_loc"] = loc[m].astype(np.int64)
                store_npz[f"rec_{pxy[0]}_{pxy[1]}_val"] = \
                    dat[m][:, 3:].astype(np.float64)
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **store_npz)
            print(f"[ZSBasisArms] wrote {out_npz}")
        self.put(store, "zs.arms", rec)
        self._emit(store, rec)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
OKABE = {"truth": "#000000", "ls": "#D55E00", "pos_a0": "#009E73",
         "pos_l1": "#CC79A7", "coarse": "#0072B2"}
ARM_LABEL = {"ls": "least squares", "pos_a0": "positivity",
             "pos_l1": r"positivity + $\ell_1$"}
BASIS_COLOR = {1: "#E69F00", 5: "#009E73", 30: "#0072B2"}
BASIS_LABEL = {1: "fine (1 tick)", 5: "5 ticks", 30: "30 ticks"}
CONV_STYLE = {"acq_edge": "-", "acq_t0": "--"}
CONV_SHORT = {"acq_edge": "lower edge at acquisition start",
              "acq_t0": r"lower edge at the event $t_0$"}


def _ieee_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.grid(False)


@algorithm("ZSFigures")
class ZSFigures(_JsonAlg):
    """Step 4: Z1-Z5 from the arms record, the sample record and the event."""

    reads = ("event", "readout_config", "block", "block_offset", "op")
    writes = ("zs.figs",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        J = _BasisJob(self, store)
        figdir = Path(str(self.props["fig_dir"]))
        figdir.mkdir(parents=True, exist_ok=True)
        arms = json.load(open(self.props["arms_json"]))["result"]
        sample = json.load(open(self.props["sample_json"]))["result"]
        wav = np.load(self.props["arms_npz"])
        truth_total = float(arms["truth_total_ke"])
        cells = sorted({int(a["cell_ticks"]) for a in arms["arms"]})
        convs = [c for c in ("acq_edge", "acq_t0")
                 if any(a["convention"] == c for a in arms["arms"])]
        by = {(a["convention"], int(a["cell_ticks"]), a["arm"]): a
              for a in arms["arms"]}
        y_frac = np.mean([v["sum_y_over_truth"]
                          for v in arms["data"].values()])
        made = []

        # ---- Z1: sum xhat / sum q_truth ----------------------------------
        base_arms = [a for a in ("ls", "pos_a0", "pos_l1")
                     if any(k[2] == a for k in by)]
        fig, axes = plt.subplots(1, len(convs), figsize=(7.0, 3.0),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        w = 0.8 / max(len(base_arms), 1)
        for ax, conv in zip(axes, convs):
            for i, arm in enumerate(base_arms):
                xs, hs = [], []
                for k, c in enumerate(cells):
                    e = by.get((conv, c, arm))
                    if e is None:
                        continue
                    xs.append(k - 0.4 + (i + 0.5) * w)
                    hs.append(e["sum_xhat_over_truth"])
                ax.bar(xs, hs, width=w * 0.92, color=OKABE[arm],
                       label=ARM_LABEL[arm], edgecolor="none")
                for xx, hh in zip(xs, hs):
                    ax.text(xx, hh + 0.01, f"{hh:.3f}", ha="center",
                            va="bottom", fontsize=5.5, rotation=90)
            ax.axhline(1.0, color="k", lw=0.9,
                       label=r"created charge, $\Sigma\hat{x}=\Sigma q_{\rm truth}$")
            ax.axhline(y_frac, color="0.45", lw=0.9, ls=":",
                       label=r"recorded charge, $\Sigma y/\Sigma q_{\rm truth}$")
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([BASIS_LABEL[c] for c in cells])
            ax.set_title(CONV_SHORT[conv], fontsize=8)
            ax.set_xlabel("unknown time basis")
            _ieee_axes(ax)
        axes[0].set_ylabel(r"$\Sigma\hat{x}\,/\,\Sigma q_{\rm truth}$")
        axes[0].legend(fontsize=7, frameon=False)
        axes[0].set_ylim(0.0, 1.30)
        fig.tight_layout()
        p = figdir / "Z1_sum_ratio.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- accumulator prediction on the fine grid ---------------------
        op1 = J.operator(1, convs[0])
        xt = np.zeros(op1.q_shape)
        jj = J.truth_tick + int(arms["registration_delta"]) - J.b0
        ok = ((jj >= 0) & (jj < op1.q_shape[2]) & (J.truth_ix >= 0)
              & (J.truth_ix < J.nx) & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
        np.add.at(xt, (J.truth_ix[ok], J.truth_iy[ok], jj[ok]), J.truth_q[ok])
        cur = op1.conv(op1.to_tensor(xt)).cpu().numpy()
        acc = np.cumsum(cur, axis=2)
        us = (J.b0 + np.arange(J.nt_fine)) * 0.05

        def _us(a):
            """Absolute time axis in us for an array on the fine grid."""
            return (J.b0 + np.arange(len(a))) * 0.05

        # ---- Z2/Z3: waveforms --------------------------------------------
        # ONE common window for every probe pixel, taken from the ionised one:
        # the anode panel runs 0 .. last hold + 10 us, the charge panels run
        # over the ionised pixel's truth support +- 5 us in release time.
        pix_a = [int(v) for v in self.props.get("pixel_a", [141, 68])]
        rl_a = wav.get(f"rec_{pix_a[0]}_{pix_a[1]}_loc")
        tr_a = wav.get(f"truth_{pix_a[0]}_{pix_a[1]}")
        t_arr = (float(rl_a[:, 3].max()) * 0.05 if rl_a is not None
                 and len(rl_a) else us[-1])
        lo, hi = -0.5, t_arr + 10.0
        if tr_a is not None and float(np.abs(tr_a).sum()) > 0:
            nzt = np.nonzero(tr_a)[0]
            q_lo = (J.b0 + nzt[0]) * 0.05 - 5.0
            q_hi = (J.b0 + nzt[-1]) * 0.05 + 5.0
        else:
            q_lo, q_hi = t_arr - 195.6, t_arr - 185.6
        for tag, pxy in (("Z2", pix_a),
                         ("Z3", self.props.get("pixel_b", [140, 68])),
                         ("Z3b", self.props.get("pixel_c", [142, 68]))):
            pxy = [int(v) for v in pxy]
            ip, iq = pxy[0] - int(J.boff[0]), pxy[1] - int(J.boff[1])
            if not (0 <= ip < J.nx and 0 <= iq < J.ny):
                continue
            rl = wav.get(f"rec_{pxy[0]}_{pxy[1]}_loc")
            rv = wav.get(f"rec_{pxy[0]}_{pxy[1]}_val")
            tr = wav.get(f"truth_{pxy[0]}_{pxy[1]}")
            has_rec = rl is not None and len(rl) > 0
            ionised = tr is not None and float(np.abs(tr).sum()) > 0
            fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.8))
            ax[0].plot(us, acc[ip, iq], color="k", lw=1.0,
                       label=r"predicted accumulator $\overline{K}^{\rm cum}\star x$")
            if has_rec:
                ax[0].plot(rl[:, 3] * 0.05, rv.ravel(), "o", ms=5,
                           color=OKABE["coarse"], label="records at their hold")
                for k in range(len(rl)):
                    ax[0].axvline(rl[k, 2] * 0.05, color=OKABE["ls"],
                                  lw=0.8, ls=":")
                    ax[0].axvspan(rl[k, 3] * 0.05,
                                  (rl[k, 3] + int(J.rc.csa_reset_time)) * 0.05,
                                  color="0.75", alpha=0.6, lw=0)
            ax[0].axhline(float(J.rc.threshold), color="0.4", lw=0.7, ls="-.",
                          label=f"threshold {float(J.rc.threshold):.0f} ke")
            ax[0].set_ylabel("accumulator [ke]")
            ax[0].legend(fontsize=6, frameon=False, loc="upper left")
            note = ("" if ionised else "  no ionisation")
            if not has_rec:
                note += ("," if note else "  ") + " no record"
            ax[0].set_title(f"pixel ({pxy[0]}, {pxy[1]})" + note, fontsize=9)
            for row, conv in zip((1, 2), convs):
                a = ax[row]
                if tr is not None:
                    a.step(_us(tr), tr, where="post", color="k", lw=1.0,
                           label="truth")
                for c in cells:
                    key = f"wave_{conv}_c{c}_pos_a0_{pxy[0]}_{pxy[1]}"
                    if key not in wav:
                        continue
                    v = wav[key]
                    a.step(_us(v), v, where="post", lw=1.0,
                           color=BASIS_COLOR[c], label=BASIS_LABEL[c])
                a.set_ylabel("charge per fine tick [ke]")
                a.set_title("positivity, " + CONV_SHORT[conv]
                            + "  (colours by unknown basis)", fontsize=8)
                a.legend(fontsize=6, frameon=False)
            ax[0].set_xlim(lo, hi)
            ax[0].set_xlabel(r"anode time relative to $t_0$ [$\mu$s]")
            for a in ax[1:]:
                a.set_xlim(q_lo, q_hi)
                a.set_xlabel(r"release time at the response plane, "
                             r"relative to $t_0$ [$\mu$s]")
            for a in ax:
                _ieee_axes(a)
            fig.tight_layout()
            p = figdir / f"{tag}_waveform_{pxy[0]}_{pxy[1]}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            made.append(str(p))

        # ---- Z4: E_rel and the +1-pixel signed charge ---------------------
        fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
        for j, (key, ylab) in enumerate((
                ("E_rel_1.5", r"$E_{\rm rel}$, $\sigma_H = 1.5\ \mu$s"),
                ("E_rel_2.0", r"$E_{\rm rel}$, $\sigma_H = 2.0\ \mu$s"))):
            for arm in base_arms:
                for conv in convs:
                    xs, ys = [], []
                    for k, c in enumerate(cells):
                        e = by.get((conv, c, arm))
                        if e is None or key not in e:
                            continue
                        xs.append(k)
                        ys.append(e[key])
                    ax[j].plot(xs, ys, CONV_STYLE[conv], marker="o",
                               ms=4.5, lw=1.0, color=OKABE[arm],
                               mfc=(OKABE[arm] if conv == "acq_edge" else "w"),
                               label=(f"{ARM_LABEL[arm]}, {CONV_SHORT[conv]}"
                                      if j == 0 else None))
            ax[j].set_xticks(range(len(cells)))
            ax[j].set_xticklabels([BASIS_LABEL[c] for c in cells])
            ax[j].set_ylabel(ylab)
            ax[j].set_xlabel("unknown time basis")
            _ieee_axes(ax[j])
        for arm in base_arms:
            for conv in convs:
                xs, ys = [], []
                for k, c in enumerate(cells):
                    e = by.get((conv, c, arm))
                    if e is None:
                        continue
                    xs.append(k)
                    ys.append(e["pixels"]["plus1"]["sum_ke"])
                ax[2].plot(xs, ys, CONV_STYLE[conv], marker="s", ms=4.5,
                           lw=1.0, color=OKABE[arm],
                           mfc=(OKABE[arm] if conv == "acq_edge" else "w"))
        ax[2].axhline(0.0, color="k", lw=0.8)
        ax[2].set_xticks(range(len(cells)))
        ax[2].set_xticklabels([BASIS_LABEL[c] for c in cells])
        ax[2].set_ylabel("signed charge on the +1 pixels [ke]")
        ax[2].set_xlabel("unknown time basis")
        _ieee_axes(ax[2])
        ax[0].legend(fontsize=5.5, frameon=False)
        fig.tight_layout()
        p = figdir / "Z4_Erel_and_plus1.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- Z5: the sample ----------------------------------------------
        tab = sample["_table"]
        fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
        n = np.asarray(tab["n_records"])
        ax[0].hist(n, bins=np.arange(0.5, n.max() + 1.5, 1.0),
                   color=OKABE["coarse"], edgecolor="k", lw=0.5)
        ax[0].set_xlabel("records per pixel")
        ax[0].set_ylabel("pixels")
        ax[0].set_xticks(np.arange(1, n.max() + 1))
        px = np.asarray(tab["pixel_x"])
        py = np.asarray(tab["pixel_y"])
        tt = np.asarray(tab["first_trigger"]) * 0.05
        for xv, col, lab in ((141, OKABE["pos_a0"], "pixel_x = 141 (ionised)"),
                             (142, OKABE["pos_l1"], "pixel_x = 142 (+1)")):
            m = px == xv
            ax[1].plot(py[m], tt[m], ".", ms=3, color=col, label=lab)
        ax[1].set_xlabel("pixel_y")
        ax[1].set_ylabel(r"first trigger time [$\mu$s]")
        ax[1].legend(fontsize=6, frameon=False)
        cr = np.asarray(tab["created_ke"])
        rc_ = np.asarray(tab["recorded_ke"])
        ax[2].plot(cr, rc_, ".", ms=3, color=OKABE["coarse"])
        lim = [0, max(cr.max(), rc_.max()) * 1.05]
        ax[2].plot(lim, lim, "-", lw=0.8, color="k")
        ax[2].set_xlim(lim)
        ax[2].set_ylim(lim)
        ax[2].set_xlabel("created charge per pixel [ke]")
        ax[2].set_ylabel("recorded total per pixel [ke]")
        for a in ax:
            _ieee_axes(a)
        fig.tight_layout()
        p = figdir / "Z5_sample.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        rec = {"figures": made, "sum_y_over_truth": float(y_frac),
               "truth_total_ke": truth_total}
        print(f"[ZSFigures] wrote {len(made)} figures under {figdir}")
        self.put(store, "zs.figs", rec)
        self._emit(store, rec)
