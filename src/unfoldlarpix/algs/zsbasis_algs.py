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


def build_zs_operator(J, cell_ticks: int, windows):
    """One ``c``-tick cell operator on the given windows, UNCACHED.

    :meth:`_BasisJob.operator` caches by ``(c, convention)``, which is right
    when the windows are a function of the convention alone; a study that
    builds several row sets for one convention needs a fresh operator each
    time.
    """
    c = int(cell_ticks)
    if c == 1:
        return ZSOperator(J.K1, (J.nx, J.ny, J.nt_fine), windows, 1,
                          device=J.comp.device, dtype=J.comp.dtype)
    if J.nt_fine % c:
        raise ValueError(f"fine block {J.nt_fine} is not a multiple of the "
                         f"cell width {c}")
    return ZSOperatorUniform(J.K1, (J.nx, J.ny, J.nt_fine // c), windows, c, c,
                             device=J.comp.device, dtype=J.comp.dtype)


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

    ``per_pixel`` (default ``False``, so an archived job is unchanged) adds to
    the NPZ, for every arm, the per-pixel total ``sum_t xhat`` as an
    ``(nx, ny)`` array ``pixel_<convention>_c<c>_<label>``, together with the
    per-pixel created charge from ``effq`` (``pixel_created_ke``) and the
    per-pixel recorded charge ``sum y`` (``pixel_recorded_ke``) on the same
    grid, plus ``block_offset`` so that ``pixel_x = ix + block_offset[0]`` and
    ``pixel_y = iy + block_offset[1]``.
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
        per_pixel = bool(self.props.get("per_pixel", False))
        if per_pixel:
            created = np.zeros((J.nx, J.ny))
            okp = ((J.truth_ix >= 0) & (J.truth_ix < J.nx)
                   & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
            np.add.at(created, (J.truth_ix[okp], J.truth_iy[okp]),
                      J.truth_q[okp])
            recorded = np.zeros((J.nx, J.ny))
            hl = np.asarray(store.get("event").hits.location)
            hd = np.asarray(store.get("event").hits.data, dtype=float)[:, 3:]
            hx = hl[:, 0].astype(int) - int(J.boff[0])
            hy = hl[:, 1].astype(int) - int(J.boff[1])
            okh = (hx >= 0) & (hx < J.nx) & (hy >= 0) & (hy < J.ny)
            np.add.at(recorded, (hx[okh], hy[okh]), hd[okh].sum(axis=1))
            store_npz["pixel_created_ke"] = created.astype(np.float64)
            store_npz["pixel_recorded_ke"] = recorded.astype(np.float64)
            store_npz["block_offset"] = np.asarray(J.boff, dtype=np.float64)
            rec["per_pixel"] = {
                "pixel_created_total_ke": float(created.sum()),
                "pixel_recorded_total_ke": float(recorded.sum()),
                "created_pixel_y_min": int(
                    (np.nonzero(created.sum(axis=0))[0].min() + J.boff[1])),
                "created_pixel_y_max": int(
                    (np.nonzero(created.sum(axis=0))[0].max() + J.boff[1])),
            }

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
                    if per_pixel and out_npz:
                        store_npz[f"pixel_{conv}_c{c}_{lab}"] = \
                            x.sum(axis=2).astype(np.float64)
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


def trim_mask(pixel_y: np.ndarray, y_lo: int, y_hi: int, n: int) -> np.ndarray:
    """The pixel_y columns kept by a trim of ``n`` pixels at EACH end.

    ``pixel_y < y_lo + n`` and ``pixel_y > y_hi - n`` are excluded, on all
    ``pixel_x``.  ``y_lo`` and ``y_hi`` are the extreme ``pixel_y`` that carry
    created charge.
    """
    py = np.asarray(pixel_y)
    return (py >= int(y_lo) + int(n)) & (py <= int(y_hi) - int(n))


def trim_ratio(numer: np.ndarray, denom: np.ndarray, mask: np.ndarray) -> float:
    """``sum numer / sum denom`` over the kept columns of two ``(nx, ny)``
    per-pixel maps.  Both sides carry the SAME trim."""
    d = float(np.asarray(denom)[:, mask].sum())
    return float(np.asarray(numer)[:, mask].sum() / d) if d else float("nan")


# ---------------------------------------------------------------------------
@algorithm("ZSTrimFigure")
class ZSTrimFigure(_JsonAlg):
    """Z6: the totals as a function of how many pixels are trimmed from each
    end of the line, and the along-the-line profile.

    ``trim n`` excludes every pixel with ``pixel_y < n`` or
    ``pixel_y > y_max - n`` on ALL ``pixel_x``, with ``y_max`` the largest
    ``pixel_y`` that carries created charge.  The ratio quoted at trim ``n`` is
    ``sum xhat(n) / sum q_truth(n)``: both sides are restricted by the same
    trim, so the number answers "does the estimator recover the charge of the
    pixels it is asked about", not "does it recover the whole event".
    """

    reads = ()
    writes = ("zs.trim",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = Path(str(self.props["fig_dir"]))
        figdir.mkdir(parents=True, exist_ok=True)
        arms = json.load(open(self.props["arms_json"]))["result"]
        z = np.load(self.props["arms_npz"])
        boff = np.asarray(z["block_offset"], dtype=float)
        created = np.asarray(z["pixel_created_ke"])
        recorded = np.asarray(z["pixel_recorded_ke"])
        ny = created.shape[1]
        pixel_y = np.arange(ny) + int(boff[1])
        pixel_x = np.arange(created.shape[0]) + int(boff[0])
        occ = np.nonzero(created.sum(axis=0) > 0)[0]
        y_lo, y_hi = int(pixel_y[occ.min()]), int(pixel_y[occ.max()])
        n_max = int(self.props.get("n_max", 12))
        ns = list(range(0, n_max + 1))

        maps = {}
        for k in z.files:
            if not k.startswith("pixel_") or k in ("pixel_created_ke",
                                                   "pixel_recorded_ke"):
                continue
            body = k[len("pixel_"):]
            conv = "acq_edge" if body.startswith("acq_edge") else "acq_t0"
            rest = body[len(conv) + 1:]
            cs, lab = rest.split("_", 1)
            maps[(conv, int(cs[1:]), lab)] = np.asarray(z[k])

        masks = [trim_mask(pixel_y, y_lo, y_hi, n) for n in ns]
        curves, table = {}, {}
        rat_y = [trim_ratio(recorded, created, m) for m in masks]
        for key, arr in sorted(maps.items()):
            curves[key] = [trim_ratio(arr, created, m) for m in masks]
            vals = curves[key]
            table["%s|c%d|%s" % key] = {str(n): vals[n]
                                        for n in (0, 3, 5, 10) if n <= n_max}

        # ---- Z6 ----------------------------------------------------------
        from matplotlib.lines import Line2D
        marker_of = {1: "o", 5: "s", 30: "^"}
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
        for (conv, c, lab), vals in sorted(curves.items()):
            col = OKABE.get(lab, "0.5")
            ax[0].plot(ns, vals, CONV_STYLE[conv], marker=marker_of.get(c, "o"),
                       ms=3.8, lw=1.0, color=col,
                       mfc=(col if conv == "acq_edge" else "w"))
        ax[0].plot(ns, rat_y, "-", color="0.45", lw=1.4)
        ax[0].axhline(1.0, color="k", lw=0.9)
        ax[0].set_xlabel("pixels trimmed at each end of the line, $n$")
        ax[0].set_ylabel(r"$\Sigma\hat{x}\,/\,\Sigma q_{\rm truth}$")
        # colour = estimator, marker = basis, filled/open = convention
        handles = [Line2D([], [], color=OKABE[a], lw=1.4,
                          label=ARM_LABEL[a]) for a in ("ls", "pos_a0",
                                                        "pos_l1")]
        handles += [Line2D([], [], color="0.45", lw=1.4,
                           label=r"$\Sigma y/\Sigma q_{\rm truth}$")]
        handles += [Line2D([], [], color="0.3", lw=0, marker=marker_of[c],
                           ms=4.5, label=f"c = {c}") for c in (1, 5, 30)]
        handles += [Line2D([], [], color="0.3", lw=1.0, ls=CONV_STYLE[cv],
                           marker="o", ms=4.5,
                           mfc=("0.3" if cv == "acq_edge" else "w"),
                           label=CONV_SHORT[cv]) for cv in ("acq_edge",
                                                            "acq_t0")]
        lo_y = min(min(v) for v in curves.values())
        hi_y = max(max(max(v) for v in curves.values()), max(rat_y), 1.0)
        span = hi_y - lo_y
        ax[0].set_ylim(lo_y - 0.42 * span, hi_y + 0.04 * span)
        ax[0].legend(handles=handles, fontsize=5.0, frameon=False, ncol=3,
                     loc="lower center", handlelength=2.4,
                     columnspacing=1.0)

        prof_conv = str(self.props.get("profile_convention", "acq_edge"))
        prof_c = int(self.props.get("profile_cell_ticks", 5))
        cr_y = created.sum(axis=0)
        rc_y = recorded.sum(axis=0)
        sel = (pixel_y >= y_lo - 3) & (pixel_y <= y_hi + 3)
        ax[1].plot(pixel_y[sel], cr_y[sel], color="k", lw=1.1,
                   label="created charge")
        ax[1].plot(pixel_y[sel], rc_y[sel], color="0.55", lw=1.1,
                   label="recorded charge")
        for lab in ("ls", "pos_a0", "pos_l1"):
            arr = maps.get((prof_conv, prof_c, lab))
            if arr is None:
                continue
            ax[1].plot(pixel_y[sel], arr.sum(axis=0)[sel], lw=1.0,
                       color=OKABE[lab], label=ARM_LABEL[lab])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(cr_y > 0, arr.sum(axis=0) / cr_y, np.nan)
            ax[2].plot(pixel_y[sel], r[sel], lw=1.0, color=OKABE[lab],
                       label=ARM_LABEL[lab])
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(cr_y > 0, rc_y / cr_y, np.nan)
        ax[2].plot(pixel_y[sel], rr[sel], color="0.55", lw=1.1,
                   label="recorded charge")
        ax[2].axhline(1.0, color="k", lw=0.9)
        ax[2].set_ylim(0.6, 1.3)
        for a, ylab in ((ax[1], "charge per pixel_y [ke]"),
                        (ax[2], "ratio to the created charge")):
            a.axvline(y_lo, color=OKABE["coarse"], lw=0.9, ls="--")
            a.text(y_lo, a.get_ylim()[1], " TPC edge", fontsize=6,
                   color=OKABE["coarse"], va="top", ha="left")
            a.set_xlabel("pixel_y")
            a.set_ylabel(ylab)
        ax[1].set_title(f"c = {prof_c}, {CONV_SHORT[prof_conv]}", fontsize=8)
        ax[2].set_title(f"c = {prof_c}, {CONV_SHORT[prof_conv]}", fontsize=8)
        ax[1].legend(fontsize=6, frameon=False)
        for a in ax:
            _ieee_axes(a)
        fig.tight_layout(w_pad=1.6)
        p = figdir / "Z6_trim_and_profile.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)

        out = {
            "figure": str(p),
            "created_pixel_y_range": [y_lo, y_hi],
            "created_pixel_x_range": [int(pixel_x[created.sum(axis=1) > 0].min()),
                                      int(pixel_x[created.sum(axis=1) > 0].max())],
            "n_trim": ns,
            "recorded_over_created_vs_trim": rat_y,
            "sum_ratio_vs_trim": {"%s|c%d|%s" % k: v
                                  for k, v in sorted(curves.items())},
            "table_n_0_3_5_10": table,
            "created_total_ke": float(created.sum()),
            "recorded_total_ke": float(recorded.sum()),
        }
        # where the induced-only records sit along the line
        ind = (created == 0) & (recorded != 0)
        if ind.any():
            ix, iy = np.nonzero(ind)
            out["induced_only_pixels"] = {
                "n": int(ind.sum()),
                "pixel_x": sorted({int(v + boff[0]) for v in ix}),
                "pixel_y_min": int(iy.min() + boff[1]),
                "pixel_y_max": int(iy.max() + boff[1]),
                "recorded_total_ke": float(recorded[ind].sum()),
                "recorded_min_ke": float(recorded[ind].min()),
                "recorded_max_ke": float(recorded[ind].max()),
            }
        print(f"[ZSTrimFigure] created pixel_y {y_lo}..{y_hi}; wrote {p}")
        self.put(store, "zs.trim", out)
        self._emit(store, out)


# ---------------------------------------------------------------------------
# the lifetime ladder under zero suppression
# ---------------------------------------------------------------------------
LADDER_ESTIMATES = (
    ("sum_effq", "created charge, $\\Sigma$ effq", "#000000", "o"),
    ("sum_y", "records, $\\Sigma y$", "#666666", "s"),
    ("ls", "least squares", OKABE["ls"], "v"),
    ("pos_a0", "positivity", OKABE["pos_a0"], "^"),
    ("pos_l1", r"positivity + $\ell_1$", OKABE["pos_l1"], "D"),
)


@algorithm("ZSLadderFit")
class ZSLadderFit(_JsonAlg):
    """The electron-lifetime fit over the zero-suppressed depth ladder.

    A SOURCE algorithm (``reads = ()``): it re-reads only the per-depth JSONs
    written by :class:`ZSBasisArms` and :class:`ZSSampleFacts`, so changing the
    minimum depth never re-runs a solve.

    For every estimate ``E`` and every lifetime, ``ln E(d) = a - lambda
    t_drift(d)`` is fitted by unweighted least squares over the depths
    ``d >= d_min``, with :func:`~unfoldlarpix.algs.depthladder_algs.ls_line` —
    the same estimator and the same residual-scatter error
    ``se(lambda) = sqrt((sum r^2/(n-2)) / sum (t - tbar)^2)`` that
    ``DepthLadderFit`` uses.  ``DepthLadderFit`` itself is not called: its
    ``_values`` reads the ``variants``/``tau_cut`` schema of the min-norm
    kernel-truncation campaign, which a ZS arms record does not carry.

    ``t_drift(d) = d / v``, ``v = 0.159645`` cm/µs, the ladder generator's own
    drift velocity.

    Props
    -----
    inputs : list of ``{depth_cm, tau_ms, arms_json, sample_json}``.
    arms : list of arm labels to read from the arms record (default
        ``[ls, pos_a0, pos_l1]``).
    convention, cell_ticks : which arms record entry to read (default
        ``acq_edge``, 5).
    d_min_cm : list, default ``[4.5, 7.5, 10.5, 13.5, 16.5]``.
    velocity_cm_per_us : float, default 0.159645.
    """

    reads = ()
    writes = ("zs.ladder",)

    def execute(self, store):
        from .depthladder_algs import DRIFT_VELOCITY_CM_PER_US, ls_line
        v = float(self.props.get("velocity_cm_per_us",
                                 DRIFT_VELOCITY_CM_PER_US))
        conv = str(self.props.get("convention", "acq_edge"))
        c = int(self.props.get("cell_ticks", 5))
        arm_labels = list(self.props.get("arms", ["ls", "pos_a0", "pos_l1"]))
        d_mins = [float(x) for x in self.props.get(
            "d_min_cm", [4.5, 7.5, 10.5, 13.5, 16.5])]

        per: dict = {}
        facts: dict = {}
        for item in self.props.get("inputs", []):
            d, tau = float(item["depth_cm"]), float(item["tau_ms"])
            A = json.load(open(item["arms_json"]))["result"]
            vals = {"sum_effq": float(A["truth_total_ke"]),
                    "sum_y": float(A["data"][f"{conv}_c{c}"]["sum_y_ke"])}
            wall = {}
            for a in A["arms"]:
                if a["convention"] == conv and int(a["cell_ticks"]) == c \
                        and a["arm"] in arm_labels:
                    vals[a["arm"]] = float(a["sum_xhat_ke"])
                    wall[a["arm"]] = float(a["wall_time_s"])
            per.setdefault(tau, {})[d] = vals
            S = json.load(open(item["sample_json"]))["result"]
            bd = S["by_distance"]
            # ZSSampleFacts classifies a pixel as ionised if it carries ANY
            # created charge.  Transverse diffusion grows with drift, so past
            # ~25 cm the neighbouring pixel row carries a sub-ke tail and is
            # itself counted as ionised, which empties the +1 class.  Redo the
            # classification from the per-pixel maps with an explicit charge
            # threshold, and report both.
            extra = {}
            if item.get("arms_npz"):
                thr = float(self.props.get("ionised_min_ke", 0.5))
                z = np.load(item["arms_npz"])
                cre = np.asarray(z["pixel_created_ke"])
                rec_map = np.asarray(z["pixel_recorded_ke"])
                ion = cre > thr
                if ion.any():
                    ii = np.argwhere(ion)
                    gx, gy = np.meshgrid(np.arange(cre.shape[0]),
                                         np.arange(cre.shape[1]),
                                         indexing="ij")
                    allp = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
                    ch = np.abs(allp[:, None, :] - ii[None, :, :]
                                ).max(axis=2).min(axis=1).reshape(cre.shape)
                    p1 = (ch == 1)
                    extra = {
                        "ionised_min_ke": thr,
                        "n_ionised_pixels_thr": int(ion.sum()),
                        "created_on_ionised_ke": float(cre[ion].sum()),
                        "recorded_on_ionised_ke": float(rec_map[ion].sum()),
                        "recorded_over_created_ionised_thr":
                            float(rec_map[ion].sum() / cre[ion].sum()),
                        "n_plus1_pixels_with_records_thr":
                            int((p1 & (rec_map != 0)).sum()),
                        "plus1_recorded_ke_thr": float(rec_map[p1].sum()),
                        "created_on_plus1_ke": float(cre[p1].sum()),
                        "n_plus2_pixels_with_records_thr":
                            int(((ch == 2) & (rec_map != 0)).sum()),
                    }
            facts.setdefault(str(tau), {})[str(d)] = {
                "n_records": int(S["n_records"]),
                "n_pixels_with_records": int(S["n_pixels_with_records"]),
                "n_ionised_pixels": int(S["n_ionised_pixels"]),
                "sum_effq_ke": float(S["sum_effq_ke"]),
                "sum_recorded_ke": float(S["sum_recorded_ke"]),
                "recorded_over_created": float(S["recorded_over_created"]),
                "n_plus1_pixels_with_records":
                    int(bd.get("plus1", {}).get("n_pixels", 0)),
                "plus1_recorded_ke":
                    float(bd.get("plus1", {}).get("recorded_ke", 0.0)),
                "n_plus2_pixels_with_records":
                    int(bd.get("plus2", {}).get("n_pixels", 0)),
                "plus2_recorded_ke":
                    float(bd.get("plus2", {}).get("recorded_ke", 0.0)),
                "ionised_recorded_ke": float(bd["ionised"]["recorded_ke"]),
                "ionised_created_ke": float(bd["ionised"]["created_ke"]),
                "ionised_recorded_over_created":
                    float(bd["ionised"]["recorded_ke"]
                          / bd["ionised"]["created_ke"]),
                "n_ionised_pixels_with_records":
                    int(bd["ionised"]["n_pixels"]),
                "wall_time_s": wall,
                **extra,
            }

        rec = {"velocity_cm_per_us": v, "convention": CONV_LABEL[conv],
               "cell_ticks": c, "d_min_cm": d_mins,
               "lambda_error_definition":
                   "sqrt( (sum r^2/(n-2)) / sum (t - tbar)^2 ), unweighted "
                   "straight line ln E = a - lambda t_drift "
                   "(depthladder_algs.ls_line)",
               "sample_facts": facts, "fits": {}, "ratios": {}}
        names = [n for n, _, _, _ in LADDER_ESTIMATES]
        for tau, byd in sorted(per.items()):
            depths = np.array(sorted(byd))
            t_ms = depths / v * 1e-3
            key = f"{tau:g}ms"
            rec["fits"][key] = {}
            rec["ratios"][key] = {"depths_cm": [float(x) for x in depths]}
            print(f"[{self.name}] tau = {tau} ms, lambda_true = "
                  f"{1.0 / tau:.4f} /ms")
            for name in names:
                if not all(name in byd[d] for d in depths):
                    continue
                E = np.array([byd[d][name] for d in depths])
                if not np.all(E > 0):
                    print(f"[{self.name}]   {name}: non-positive estimate, "
                          "not fitted")
                    continue
                rec["ratios"][key][name] = [
                    float(byd[d][name] / byd[d]["sum_effq"]) for d in depths]
                rec["fits"][key][name] = {
                    "E_ke": [float(x) for x in E],
                    "depths_cm": [float(x) for x in depths],
                    "t_drift_ms": [float(x) for x in t_ms],
                    "lambda_true_per_ms": 1.0 / tau, "by_d_min": {}}
                line = f"[{self.name}]   {name:10s}"
                for dm in d_mins:
                    sel = depths >= dm - 1e-9
                    if sel.sum() < 3:
                        continue
                    f = ls_line(t_ms[sel], np.log(E[sel]))
                    f["d_min_cm"] = dm
                    f["pull"] = ((f["lambda_per_ms"] - 1.0 / tau)
                                 / max(f["lambda_err"], 1e-12))
                    rec["fits"][key][name]["by_d_min"][f"{dm:g}"] = f
                    line += (f"  {dm:g}:{f['lambda_per_ms']:7.4f}"
                             f"+-{f['lambda_err']:.4f}")
                print(line)

        # lambda(20 ms) - lambda(1 ms): -0.95 /ms for a lifetime-independent
        # estimate, since lambda_true is 0.05 and 1.0 /ms.
        keys = sorted(rec["fits"], key=lambda s: float(s[:-2]))
        if len(keys) == 2:
            lo, hi = keys[0], keys[1]
            diff = {}
            for name in names:
                if name not in rec["fits"][lo] or name not in rec["fits"][hi]:
                    continue
                per_dm = {}
                for dm in rec["fits"][lo][name]["by_d_min"]:
                    if dm not in rec["fits"][hi][name]["by_d_min"]:
                        continue
                    a = rec["fits"][hi][name]["by_d_min"][dm]
                    b = rec["fits"][lo][name]["by_d_min"][dm]
                    per_dm[dm] = {
                        "lambda_difference_per_ms":
                            a["lambda_per_ms"] - b["lambda_per_ms"],
                        "err": float(np.hypot(a["lambda_err"],
                                              b["lambda_err"])),
                        "minus_expected_-0.95":
                            a["lambda_per_ms"] - b["lambda_per_ms"] + 0.95}
                diff[name] = per_dm
            rec["lambda_difference_20ms_minus_1ms"] = {
                "expected_per_ms": -0.95,
                "note": "lambda_true is 1.0 /ms at tau = 1 ms and 0.05 /ms at "
                        "tau = 20 ms; an estimate whose depth acceptance does "
                        "not depend on the lifetime gives -0.95 exactly",
                "by_estimate": diff}
            for name, per_dm in diff.items():
                s = "  ".join(f"{k}:{e['lambda_difference_per_ms']:+7.4f}"
                              for k, e in sorted(per_dm.items(),
                                                 key=lambda kv: float(kv[0])))
                print(f"[{self.name}] lambda(20ms)-lambda(1ms) {name:10s} {s}")
        self.put(store, "zs.ladder", rec)
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("ZSLadderFigures")
class ZSLadderFigures(_JsonAlg):
    """ZL1-ZL4 from the :class:`ZSLadderFit` record."""

    reads = ()
    writes = ("zs.ladder_figs",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        R = json.load(open(self.props["ladder_json"]))["result"]
        figdir = Path(str(self.props["fig_dir"]))
        figdir.mkdir(parents=True, exist_ok=True)
        keys = sorted(R["ratios"], key=lambda s: float(s[:-2]))
        depths = np.array(R["ratios"][keys[0]]["depths_cm"])
        made = []

        # ---- ZL1 sum E / sum effq vs depth --------------------------------
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for ki, key in enumerate(keys):
            fill = ki == 0
            for name, lab, col, mk in LADDER_ESTIMATES:
                if name == "sum_effq" or name not in R["ratios"][key]:
                    continue
                ax.plot(depths, R["ratios"][key][name], mk, ls="-", color=col,
                        ms=4.0, lw=0.9, mfc=(col if fill else "none"), mew=0.9,
                        label=(lab if fill else None))
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("drift depth [cm]")
        ax.set_ylabel(r"$\Sigma E(d)\,/\,\Sigma\,\mathrm{effq}(d)$")
        ax.set_title(r"filled: $\tau$ = %s    open: $\tau$ = %s"
                     % (keys[0], keys[1] if len(keys) > 1 else "-"),
                     fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="best")
        _ieee_axes(ax)
        fig.tight_layout()
        p = figdir / "ZL1_ratio_vs_depth.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- ZL2 lambda vs d_min ------------------------------------------
        fig, axs = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.0))
        axs = np.atleast_1d(axs)
        for a, key in zip(axs, keys):
            lam_true = 1.0 / float(key[:-2])
            for name, lab, col, mk in LADDER_ESTIMATES:
                f = R["fits"].get(key, {}).get(name)
                if not f:
                    continue
                dms = sorted(f["by_d_min"], key=float)
                x = [float(k) for k in dms]
                y = [f["by_d_min"][k]["lambda_per_ms"] for k in dms]
                e = [f["by_d_min"][k]["lambda_err"] for k in dms]
                a.errorbar(x, y, yerr=e, fmt=mk, ls="-", color=col, ms=4.0,
                           lw=0.9, capsize=2.0, elinewidth=0.8, label=lab)
            a.axhline(lam_true, color="k", lw=0.8, ls="--")
            a.set_xlabel("minimum depth of the fit, $d_{\\min}$ [cm]")
            a.set_ylabel(r"$\lambda$ [ms$^{-1}$]")
            a.set_title(r"$\tau$ = %s, $\lambda_{\rm true}$ = %.3f ms$^{-1}$"
                        % (key, lam_true), fontsize=8)
            _ieee_axes(a)
        axs[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        p = figdir / "ZL2_lambda_vs_dmin.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- ZL3 ln E vs t_drift, two d_min -------------------------------
        key = str(self.props.get("profile_tau_key", keys[0]))
        dms = [str(x) for x in self.props.get("profile_d_min", [4.5, 16.5])]
        fig, axs = plt.subplots(1, len(dms), figsize=(4.4 * len(dms), 3.0),
                                sharey=True)
        axs = np.atleast_1d(axs)
        for a, dm in zip(axs, dms):
            for name, lab, col, mk in LADDER_ESTIMATES:
                f = R["fits"].get(key, {}).get(name)
                if not f or dm not in f["by_d_min"]:
                    continue
                t = np.array(f["t_drift_ms"])
                lnE = np.log(np.array(f["E_ke"]))
                a.plot(t, lnE, mk, color=col, ms=4.0, ls="none", label=lab)
                g = f["by_d_min"][dm]
                tf = np.array(g["t_drift_ms"])
                a.plot(tf, g["intercept"] - g["lambda_per_ms"] * tf, "-",
                       color=col, lw=0.9)
            a.set_xlabel(r"$t_{\rm drift}$ [ms]")
            a.set_title(r"$\tau$ = %s, $d_{\min}$ = %s cm" % (key, dm),
                        fontsize=8)
            _ieee_axes(a)
        axs[0].set_ylabel(r"$\ln E$  [$E$ in ke]")
        axs[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        p = figdir / "ZL3_lnE_vs_tdrift.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- ZL4 sample facts vs depth ------------------------------------
        F = R["sample_facts"]
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for ki, key in enumerate(keys):
            tk = str(float(key[:-2]))
            if tk not in F:
                continue
            fill = ki == 0
            dd = sorted(F[tk], key=float)
            x = [float(d) for d in dd]
            def g(d, k):
                e = F[tk][d]
                return e.get(k + "_thr", e[k])
            axs[0].plot(x, [g(d, "n_plus1_pixels_with_records") for d in dd],
                        "o-", color=OKABE["coarse"], ms=4.0, lw=0.9,
                        mfc=(OKABE["coarse"] if fill else "none"),
                        label=(r"$\tau$ = %s" % key))
            axs[1].plot(x, [g(d, "plus1_recorded_ke") for d in dd],
                        "o-", color=OKABE["coarse"], ms=4.0, lw=0.9,
                        mfc=(OKABE["coarse"] if fill else "none"),
                        label=(r"$\tau$ = %s" % key))
            axs[2].plot(x, [F[tk][d].get(
                "recorded_over_created_ionised_thr",
                F[tk][d]["ionised_recorded_over_created"]) for d in dd],
                        "o-", color=OKABE["pos_a0"], ms=4.0,
                        lw=0.9, mfc=(OKABE["pos_a0"] if fill else "none"),
                        label=(r"$\tau$ = %s" % key))
        axs[0].set_ylabel("+1 pixels with a record")
        axs[1].set_ylabel("charge recorded on the +1 pixels [ke]")
        axs[2].set_ylabel("recorded / created on the ionised pixels")
        axs[2].axhline(1.0, color="k", lw=0.8, ls="--")
        for a in axs:
            a.set_xlabel("drift depth [cm]")
            a.legend(fontsize=6, frameon=False)
            _ieee_axes(a)
        fig.tight_layout()
        p = figdir / "ZL4_sample_facts_vs_depth.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        print(f"[ZSLadderFigures] wrote {len(made)} figures under {figdir}")
        out = {"figures": made}
        self.put(store, "zs.ladder_figs", out)
        self._emit(store, out)


# ---------------------------------------------------------------------------
# threshold pseudo-measurements and censor terms
# ---------------------------------------------------------------------------
VARIANT_LABEL = {
    "A": "lumped rows only",
    "B": "split trigger (pseudo + remainder rows)",
    "C": "lumped rows + censor terms",
    "D": "split trigger + censor terms",
}
VARIANT_COLOR = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73",
                 "D": "#CC79A7"}
VARIANT_SHORT = {"A": "A\nlumped", "B": "B\nsplit trigger",
                 "C": "C\nlumped\n+ censors",
                 "D": "D\nsplit trigger\n+ censors"}


def censor_violation(term, op, q) -> dict:
    """``max over the armed bins of max(0, C - threshold)``, in ke.

    ``C`` is the term's own statistic — the running cumulative referenced at
    the CSA restart, weighted by the boundary-bin overlap — so this is the
    quantity the term penalises, read off without the penalty's beta or norm.
    Also returns how many pixels violate and the total violation.
    """
    from ..terms.base import IterCtx
    qt = q if torch.is_tensor(q) else op.to_tensor(np.ascontiguousarray(q))
    ctx = IterCtx(qt, op)
    viol, _ = term._peaks(ctx)
    v = viol.detach()
    n_armed = int(term.armed.any(dim=2).sum())
    return {"n_armed_pixels": n_armed,
            "threshold_ke": float(term.threshold),
            "max_violation_ke": float(v.max()) if v.numel() else 0.0,
            "sum_violation_ke": float(v.sum()),
            "n_violating_pixels": int((v > 0).sum())}


@algorithm("ZSCensorArms")
class ZSCensorArms(_JsonAlg):
    """Threshold pseudo-measurements and censor terms against lumped rows.

    Four operator/term variants on one basis and one event:

    ``A``  lumped rows only — ``build_latch_rows(split_threshold=None)``;
    ``B``  ``split_threshold = threshold``, ``burst_tau = resolve_burst_tau(rc,
           None)``, so each trigger that the gate calls threshold-limited
           contributes a ``pseudo`` row asserting the accumulator EQUALLED the
           threshold at the trigger and a ``remainder`` row carrying the rest
           of ``(trigger, trigger + B]``;
    ``C``  the ``A`` rows plus the two censor terms;
    ``D``  the ``B`` rows plus the same two censor terms.

    The censors are configured as ``reco_algs.build_terms`` configures them,
    with ``margin = 0`` (the sample is noiseless) and ``norm = "l2"`` on both
    so that both enter FISTA's curvature bound:

    * ``CensorRunningMax.from_hits`` — silence AFTER a pixel's last burst;
    * ``pre_trigger_censors`` — silence BEFORE each trigger, with
      ``acq_start`` equal to the variant's own first-window convention,
      ``one_tick`` from the readout config and ``close_back`` as shipped.

    ``bin_ticks = 1``: these operators sample on the fine tick, so a block bin
    IS a fine tick and every censor boundary is in fine ticks.
    """

    reads = ("event", "readout_config", "block", "block_offset", "op",
             "support", "hits_view")
    writes = ("zs.censor",)

    def execute(self, store):
        from ..model.conventions import resolve_burst_tau
        from ..solve.engine import Fista
        from ..terms.base import CoordProx
        from ..terms.censor import CensorRunningMax, pre_trigger_censors
        from ..terms.data import DataFidelity
        from .fixedgrid_algs import _SupportProx

        J = _BasisJob(self, store)
        rc = J.rc
        hv = store.get("hits_view")
        boff = store.get("block_offset")
        c = int(self.props.get("cell_ticks", 5))
        convs = list(self.props.get("conventions", ["acq_edge", "acq_t0"]))
        sigmas = [float(s) for s in self.props.get("sigma_H_us", [1.5, 2.0])]
        delta = int(self.props.get("registration_delta", -1))
        iters = int(self.props.get("iters", 1000))
        margin = float(self.props.get("censor_margin", 0.0))
        beta = float(self.props.get("censor_beta", 1.0))
        norm = str(self.props.get("censor_norm", "l2"))
        npad = int(self.props.get("censor_npad_bins", 30))
        close_back = float(self.props.get("censor_close_back", 20.0))
        post_reset = bool(self.props.get("censor_include_post_reset", True))
        variants = list(self.props.get("variants", ["A", "B", "C", "D"]))
        arms_cfg = list(self.props.get("arms", []))
        out_npz = self.props.get("out_npz")
        wave_pixels = [list(map(int, p)) for p in
                       self.props.get("waveform_pixels", [[141, 68],
                                                          [142, 68]])]
        burst_tau = resolve_burst_tau(rc, None)

        H = EvalHarness(store, store.get("op"),
                        margin_windows=int(self.props.get("margin_windows", 40)),
                        line_pixel_y_range=self.props.get(
                            "line_pixel_y_range", (5, 131)),
                        segment_pixels=int(self.props.get("segment_pixels", 7)),
                        segment_edge_exclude=int(
                            self.props.get("segment_edge_exclude", 3)))
        rec = {"truth_total_ke": J.truth_total, "cell_ticks": c,
               "sigma_H_us": sigmas, "registration_delta": delta,
               "iterations": iters,
               "burst_tau_ticks": int(burst_tau),
               "burst_tau_floor_definition":
                   "adc_hold_delay + adc_down_time + one_tick",
               "censor_settings": {
                   "margin_ke": margin, "beta": beta, "norm": norm,
                   "npad_bins_fine_ticks": npad, "bin_ticks": 1,
                   "close_back_ticks": close_back,
                   "include_post_reset": post_reset,
                   "threshold_ke": float(rc.threshold),
                   "csa_reset_time_ticks": int(rc.csa_reset_time),
                   "one_tick": int(rc.one_tick)},
               "variants": {}, "arms": []}
        store_npz: dict = {}

        for conv in convs:
            for V in variants:
                split = V in ("B", "D")
                use_censor = V in ("C", "D")
                windows, metas = zs_windows(store, conv) if not split else \
                    build_latch_rows(
                        J.ev.hits.location, J.ev.hits.data, J.B,
                        np.asarray(boff), csa_reset_time=int(rc.csa_reset_time),
                        split_threshold=float(rc.threshold),
                        acq_start=CONVENTIONS[conv], burst_tau=burst_tau)
                op = build_zs_operator(J, c, windows)
                kinds = np.array([m.kind for m in metas])
                y = op.d.cpu().numpy().astype(float)
                supp = J.support_on_basis(op, c)
                x_truth = J.truth_on_basis(op, c, delta)

                terms_extra = []
                censor_info = []
                if use_censor:
                    post = CensorRunningMax.from_hits(
                        op, hv, boff, csa_reset_time=float(rc.csa_reset_time),
                        threshold=float(rc.threshold), npad_bins=npad,
                        beta=beta, margin=margin, norm=norm, bin_ticks=1)
                    pre = pre_trigger_censors(
                        op, hv, boff, csa_reset_time=float(rc.csa_reset_time),
                        threshold=float(rc.threshold),
                        acq_start=CONVENTIONS[conv], npad_bins=npad,
                        beta=beta, margin=margin, norm=norm, bin_ticks=1,
                        one_tick=float(rc.one_tick), close_back=close_back,
                        include_post_reset=post_reset)
                    terms_extra = [post] + list(pre)
                    for i, t in enumerate(terms_extra):
                        censor_info.append({
                            "term": ("post_latch" if i == 0
                                     else f"pre_trigger_ordinal{i - 1}"),
                            "curvature": float(t.curvature()),
                            "norm": t.norm, "beta": t.beta,
                            "truth": censor_violation(t, op, x_truth)})
                data_term = DataFidelity(op)
                L_data = float(data_term.curvature())
                L_total = L_data + sum(float(t.curvature())
                                       for t in terms_extra)
                vk = f"{conv}_{V}"
                rec["variants"][vk] = {
                    "convention": conv, "convention_label": CONV_LABEL[conv],
                    "variant": V, "variant_label": VARIANT_LABEL[V],
                    "split_trigger": split, "censor_terms": use_censor,
                    "n_rows": int(op.n_data),
                    "rows_by_kind": {k: int((kinds == k).sum())
                                     for k in sorted(set(kinds))},
                    "n_sequences": int(len(metas)
                                       - (kinds == "remainder").sum()),
                    "sum_y_ke": float(y.sum()),
                    "sum_y_over_truth": float(y.sum() / J.truth_total),
                    "lipschitz_data": L_data,
                    "lipschitz_total": L_total,
                    "censor": censor_info,
                    "truth_rel_residual": J.rel_residual(op, x_truth),
                }
                if split:
                    # how many sequences the burst gate refused to split
                    n_seq = int((kinds != "remainder").sum()
                                - (kinds == "diff").sum())
                    rec["variants"][vk]["n_pseudo_rows"] = \
                        int((kinds == "pseudo").sum())
                    rec["variants"][vk]["n_lumped_rows"] = \
                        int((kinds == "lumped").sum())
                    rec["variants"][vk]["n_sequences_first_window"] = n_seq
                    rec["variants"][vk]["n_sequences_gate_suppressed"] = \
                        n_seq - int((kinds == "pseudo").sum())
                print(f"[ZSCensorArms] {vk}: rows {op.n_data} "
                      f"{rec['variants'][vk]['rows_by_kind']}, L "
                      f"{L_data:.4g} -> {L_total:.4g}")
                for ci in censor_info:
                    print(f"[ZSCensorArms]   censor {ci['term']}: "
                          f"{ci['truth']['n_armed_pixels']} armed pixels, "
                          f"curvature {ci['curvature']:.4g}, TRUTH max "
                          f"violation {ci['truth']['max_violation_ke']:.4f} ke "
                          f"on {ci['truth']['n_violating_pixels']} pixels")

                st = op.to_tensor(np.asarray(supp).astype(np.float64))
                for cfg in arms_cfg:
                    if "variants" in cfg and V not in list(cfg["variants"]):
                        continue
                    lab = str(cfg["label"])
                    alpha = float(cfg.get("alpha", 0.0))
                    pos = bool(cfg.get("positivity", True))
                    prox = (CoordProx(alpha, st) if pos else _SupportProx(st))
                    t0 = time.time()
                    q = Fista(n_iter=iters).minimize(
                        op, [data_term] + terms_extra, prox,
                        op.to_tensor(np.zeros(op.q_shape)))
                    dt = time.time() - t0
                    x = q.detach().cpu().numpy().astype(np.float64)
                    xf = J.to_fine(x, c)
                    pred = op.forward(op.to_tensor(x)).cpu().numpy()
                    res = pred.astype(float) - y
                    entry = {
                        "convention": conv, "convention_label": CONV_LABEL[conv],
                        "variant": V, "variant_label": VARIANT_LABEL[V],
                        "arm": lab, "alpha_ke_per_cell": alpha,
                        "positivity": pos, "iters": iters,
                        "wall_time_s": float(dt),
                        "lipschitz_total": L_total,
                        "sum_xhat_ke": float(x.sum()),
                        "sum_xhat_over_truth": float(x.sum() / J.truth_total),
                        "nnz": int((x > 0).sum()),
                        "rel_residual": float(np.linalg.norm(res)
                                              / np.linalg.norm(y)),
                        "pixels": ring_sums(H, xf),
                        "residual_by_row_kind": {
                            k: {"n_rows": int((kinds == k).sum()),
                                "sum_y_ke": float(y[kinds == k].sum()),
                                "sum_residual_ke": float(res[kinds == k].sum()),
                                "rms_residual_ke": float(np.sqrt(
                                    (res[kinds == k] ** 2).mean())),
                                "rel_residual": float(
                                    np.linalg.norm(res[kinds == k])
                                    / max(np.linalg.norm(y[kinds == k]),
                                          1e-30))}
                            for k in sorted(set(kinds))},
                        "censor_violation": [
                            {"term": ci["term"],
                             "solution": censor_violation(t, op, x),
                             "truth": ci["truth"]}
                            for ci, t in zip(censor_info, terms_extra)],
                    }
                    for s in sigmas:
                        m = score_rows(H, fine_xhat(H, xf, J.b0, s), s)
                        entry[f"E_rel_{s}"] = float(m["E_rel"])
                    rec["arms"].append(entry)
                    print(f"[ZSCensorArms] {vk} {lab}: sum/truth "
                          f"{entry['sum_xhat_over_truth']:.4f} +1 "
                          f"{entry['pixels']['plus1']['sum_ke']:8.2f} "
                          f"E_rel(1.5) {entry['E_rel_1.5']:.4f} res "
                          f"{entry['rel_residual']:.4f} {dt:.1f} s")
                    if out_npz:
                        for pxy in wave_pixels:
                            ip = pxy[0] - int(J.boff[0])
                            iq = pxy[1] - int(J.boff[1])
                            if 0 <= ip < J.nx and 0 <= iq < J.ny:
                                store_npz[f"wave_{conv}_{V}_{lab}_"
                                          f"{pxy[0]}_{pxy[1]}"] = \
                                    xf[ip * J.ny + iq].astype(np.float32)
                if use_censor:
                    for t in terms_extra:
                        del t
                    torch.cuda.empty_cache()
        if out_npz:
            xt_fine = np.zeros((J.nx * J.ny, J.nt_fine))
            ok = ((J.truth_ix >= 0) & (J.truth_ix < J.nx)
                  & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
            jj = J.truth_tick[ok] + delta - J.b0
            okj = (jj >= 0) & (jj < J.nt_fine)
            np.add.at(xt_fine,
                      ((J.truth_ix[ok][okj] * J.ny + J.truth_iy[ok][okj]),
                       jj[okj]), J.truth_q[ok][okj])
            for pxy in wave_pixels:
                ip = pxy[0] - int(J.boff[0])
                iq = pxy[1] - int(J.boff[1])
                if 0 <= ip < J.nx and 0 <= iq < J.ny:
                    store_npz[f"truth_{pxy[0]}_{pxy[1]}"] = \
                        xt_fine[ip * J.ny + iq].astype(np.float32)
                m = ((J.ev.hits.location[:, 0] == pxy[0])
                     & (J.ev.hits.location[:, 1] == pxy[1]))
                store_npz[f"rec_{pxy[0]}_{pxy[1]}_loc"] = \
                    np.asarray(J.ev.hits.location)[m].astype(np.int64)
                store_npz[f"rec_{pxy[0]}_{pxy[1]}_val"] = \
                    np.asarray(J.ev.hits.data, dtype=float)[m][:, 3:]
            store_npz["fine_origin_tick"] = np.array([J.b0])
            store_npz["nt_fine"] = np.array([J.nt_fine])
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **store_npz)
            print(f"[ZSCensorArms] wrote {out_npz}")
        self.put(store, "zs.censor", rec)
        self._emit(store, rec)


# ---------------------------------------------------------------------------
@algorithm("ZSCensorFigures")
class ZSCensorFigures(_JsonAlg):
    """ZC1-ZC4 from the :class:`ZSCensorArms` record and its NPZ."""

    reads = ("event", "readout_config", "block", "block_offset", "op")
    writes = ("zs.censor_figs",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        J = _BasisJob(self, store)
        R = json.load(open(self.props["censor_json"]))["result"]
        wav = np.load(self.props["censor_npz"])
        figdir = Path(str(self.props["fig_dir"]))
        figdir.mkdir(parents=True, exist_ok=True)
        truth_total = float(R["truth_total_ke"])
        cs = R["censor_settings"]
        convs = [c for c in ("acq_edge", "acq_t0")
                 if any(a["convention"] == c for a in R["arms"])]
        variants = [v for v in ("A", "B", "C", "D")
                    if any(a["variant"] == v for a in R["arms"])]
        arms = [a for a in ("ls", "pos_a0", "pos_l1")
                if any(x["arm"] == a for x in R["arms"])]
        by = {(a["convention"], a["variant"], a["arm"]): a for a in R["arms"]}
        made = []

        # ---- ZC1 ----------------------------------------------------------
        fig, axes = plt.subplots(1, len(convs), figsize=(8.0, 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        w = 0.8 / max(len(arms), 1)
        for ax, conv in zip(axes, convs):
            for i, arm in enumerate(arms):
                xs, hs = [], []
                for k, V in enumerate(variants):
                    e = by.get((conv, V, arm))
                    if e is None:
                        continue
                    xs.append(k - 0.4 + (i + 0.5) * w)
                    hs.append(e["sum_xhat_over_truth"])
                ax.bar(xs, hs, width=w * 0.92, color=OKABE[arm],
                       label=ARM_LABEL[arm], edgecolor="none")
            ax.axhline(1.0, color="k", lw=0.9)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([VARIANT_SHORT[V] for V in variants],
                               fontsize=6.5)
            ax.set_title(CONV_SHORT[conv], fontsize=8)
            ax.set_xlabel("operator / term variant")
            _ieee_axes(ax)
        axes[0].set_ylim(0.60, 1.04)
        axes[0].set_ylabel(r"$\Sigma\hat{x}\,/\,\Sigma q_{\rm truth}$")
        axes[0].legend(fontsize=6, frameon=False, loc="lower left")
        fig.tight_layout()
        p = figdir / "ZC1_sum_ratio_by_variant.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        # ---- the predicted accumulator, for the waveform panels -----------
        op1 = build_zs_operator(J, 1, zs_windows(store, convs[0])[0])
        xt = np.zeros(op1.q_shape)
        jj = J.truth_tick + int(R["registration_delta"]) - J.b0
        ok = ((jj >= 0) & (jj < op1.q_shape[2]) & (J.truth_ix >= 0)
              & (J.truth_ix < J.nx) & (J.truth_iy >= 0) & (J.truth_iy < J.ny))
        np.add.at(xt, (J.truth_ix[ok], J.truth_iy[ok], jj[ok]), J.truth_q[ok])
        acc = np.cumsum(op1.conv(op1.to_tensor(xt)).cpu().numpy(), axis=2)
        us = (J.b0 + np.arange(J.nt_fine)) * 0.05
        del op1
        torch.cuda.empty_cache()

        conv0 = str(self.props.get("waveform_convention", convs[0]))
        arm0 = str(self.props.get("waveform_arm", "pos_l1"))
        pix_a = [int(v) for v in self.props.get("pixel_a", [141, 68])]
        rl_a = wav.get(f"rec_{pix_a[0]}_{pix_a[1]}_loc")
        tr_a = wav.get(f"truth_{pix_a[0]}_{pix_a[1]}")
        t_arr = (float(rl_a[:, 3].max()) * 0.05 if rl_a is not None
                 and len(rl_a) else us[-1])
        nzt = np.nonzero(tr_a)[0]
        q_lo = (J.b0 + nzt[0]) * 0.05 - 5.0
        q_hi = (J.b0 + nzt[-1]) * 0.05 + 5.0
        npad_us = float(cs["npad_bins_fine_ticks"]) * 0.05
        cb_us = (float(cs["close_back_ticks"]) + float(cs["one_tick"])) * 0.05

        for tag, pxy in (("ZC2", pix_a),
                         ("ZC3", self.props.get("pixel_b", [142, 68]))):
            pxy = [int(v) for v in pxy]
            ip, iq = pxy[0] - int(J.boff[0]), pxy[1] - int(J.boff[1])
            if not (0 <= ip < J.nx and 0 <= iq < J.ny):
                continue
            rl = wav.get(f"rec_{pxy[0]}_{pxy[1]}_loc")
            rv = wav.get(f"rec_{pxy[0]}_{pxy[1]}_val")
            tr = wav.get(f"truth_{pxy[0]}_{pxy[1]}")
            has_rec = rl is not None and len(rl) > 0
            fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.6))
            ax[0].plot(us, acc[ip, iq], color="k", lw=1.0,
                       label=r"predicted accumulator from the truth")
            thr = float(cs["threshold_ke"])
            ax[0].axhline(thr, color="0.4", lw=0.8, ls="-.",
                          label=f"threshold {thr:.0f} ke")
            # censor armed windows, from the same boundaries the terms use
            lo_pre = (us[0] + npad_us if has_rec else us[0] + npad_us)
            if has_rec:
                hi_pre = rl[0, 2] * 0.05 - cb_us
                if hi_pre > lo_pre:
                    ax[0].axvspan(lo_pre, hi_pre, color=OKABE["pos_a0"],
                                  alpha=0.16, lw=0,
                                  label="censor armed (pre-trigger)")
                lo_post = rl[:, 4].max() * 0.05
            else:
                lo_post = us[0] + npad_us
            hi_post = us[-1] - npad_us
            if hi_post > lo_post:
                ax[0].axvspan(lo_post, hi_post, color=OKABE["coarse"],
                              alpha=0.16, lw=0,
                              label="censor armed (post-latch)")
            if has_rec:
                ax[0].plot(rl[:, 3] * 0.05, rv.ravel(), "o", ms=5,
                           color=OKABE["coarse"], label="records at their hold")
                for k in range(len(rl)):
                    ax[0].axvline(rl[k, 2] * 0.05, color=OKABE["ls"], lw=0.8,
                                  ls=":")
                # the pseudo row asserts the accumulator EQUALLED the threshold
                # at the FIRST trigger of the pixel (later ones are gated)
                ax[0].plot([rl[0, 2] * 0.05], [thr], "*", ms=11,
                           color=OKABE["ls"], mec="k", mew=0.4,
                           label="pseudo-row constraint at the trigger")
            ax[0].set_xlim(-0.5, t_arr + 10.0)
            ax[0].set_ylabel("accumulator [ke]")
            ax[0].set_xlabel(r"anode time relative to $t_0$ [$\mu$s]")
            ax[0].set_title(f"pixel ({pxy[0]}, {pxy[1]})", fontsize=9)
            ax[0].legend(fontsize=5.5, frameon=False, loc="upper left")
            if tr is not None:
                ax[1].step(_us_of(J, tr), tr, where="post", color="k", lw=1.1,
                           label="truth")
            for V in variants:
                key = f"wave_{conv0}_{V}_{arm0}_{pxy[0]}_{pxy[1]}"
                if key not in wav:
                    continue
                v = wav[key]
                ax[1].step(_us_of(J, v), v, where="post", lw=1.0,
                           color=VARIANT_COLOR[V],
                           label=f"{V}: {VARIANT_LABEL[V]}")
            ax[1].set_xlim(q_lo, q_hi)
            ax[1].set_ylabel("charge per fine tick [ke]")
            ax[1].set_xlabel(r"release time at the response plane, "
                             r"relative to $t_0$ [$\mu$s]")
            ax[1].set_title(f"{ARM_LABEL[arm0]}, {CONV_SHORT[conv0]}",
                            fontsize=8)
            ax[1].legend(fontsize=6, frameon=False)
            for a in ax:
                _ieee_axes(a)
            fig.tight_layout()
            p = figdir / f"{tag}_waveform_{pxy[0]}_{pxy[1]}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            made.append(str(p))

        # ---- ZC4 ----------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for j, key in enumerate(("E_rel_1.5", "E_rel_2.0")):
            for arm in arms:
                for conv in convs:
                    xs, ys = [], []
                    for k, V in enumerate(variants):
                        e = by.get((conv, V, arm))
                        if e is None or key not in e:
                            continue
                        xs.append(k)
                        ys.append(e[key])
                    ax[j].plot(xs, ys, CONV_STYLE[conv], marker="o", ms=4.5,
                               lw=1.0, color=OKABE[arm],
                               mfc=(OKABE[arm] if conv == "acq_edge" else "w"),
                               label=(f"{ARM_LABEL[arm]}, {CONV_SHORT[conv]}"
                                      if j == 0 else None))
            ax[j].set_ylabel(r"$E_{\rm rel}$, $\sigma_H$ = %s $\mu$s"
                             % key.split("_")[-1])
        for arm in arms:
            for conv in convs:
                xs, ys = [], []
                for k, V in enumerate(variants):
                    e = by.get((conv, V, arm))
                    if e is None:
                        continue
                    xs.append(k)
                    ys.append(e["pixels"]["plus1"]["sum_ke"])
                ax[2].plot(xs, ys, CONV_STYLE[conv], marker="s", ms=4.5,
                           lw=1.0, color=OKABE[arm],
                           mfc=(OKABE[arm] if conv == "acq_edge" else "w"))
        ax[2].axhline(0.0, color="k", lw=0.8)
        ax[2].set_ylabel("signed charge on the +1 pixels [ke]")
        for a in ax:
            a.set_xticks(range(len(variants)))
            a.set_xticklabels(variants)
            a.set_xlabel("operator / term variant")
            _ieee_axes(a)
        ax[0].legend(fontsize=5.0, frameon=False)
        fig.tight_layout()
        p = figdir / "ZC4_Erel_and_plus1.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

        print(f"[ZSCensorFigures] wrote {len(made)} figures under {figdir}")
        out = {"figures": made, "truth_total_ke": truth_total}
        self.put(store, "zs.censor_figs", out)
        self._emit(store, out)


def _us_of(J, a) -> np.ndarray:
    """Absolute time axis in microseconds for an array on the fine grid."""
    return (J.b0 + np.arange(len(a))) * 0.05
