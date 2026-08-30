"""Algorithms for the fixed-interval (threshold-free) readout probe.

A fixed-interval readout samples the CSA accumulator every
``adc_hold_delay`` fine ticks with no trigger and no reset.  Fed through
:mod:`unfoldlarpix.io.pseudo_hits` it becomes ordinary hits with
``nburst = Nsample``, so the whole reconstruction stack applies unchanged --
and because the sampling stride IS the fit bin, every latch window covers
exactly one bin at weight 1 and the window->bin allocation error vanishes.

What these algorithms exist to measure (campaign ``fixedgrid_probe``):

``FixedGridAudit``
    Proves the on-grid claim (columns per row, sampling weights) and then
    asks the question that matters: is the TRUTH a solution of ``A q = d``?
    ``sum(d) - sum(A q_truth)`` is ONE scalar out of ``n_data`` and is near
    zero by charge conservation; the per-row residual is the honest number.
    Also profiles the measurement gain ``c_v = A^T 1``.  Note ``c_v`` can be
    NEGATIVE (the kernel is bipolar across neighbour pixels), which is a
    different thing from a structurally unreachable voxel -- both counted.

``EstimatorScan``
    One matched regularisation path: same operator, same support, same
    ``q0 = 0``, only the prior changes.  The shipped prox
    (:class:`~unfoldlarpix.terms.base.CoordProx`) hard-wires positivity even
    at ``alpha = 0``; ``positivity: false`` swaps in a support-only prox so
    the positivity cost can be read off directly.

``FFTInverseScan``
    The linear inverse of the SAME system, over a filter-width scan.  Its
    ``sum q`` is pinned to ``sum d / sum(K)`` by the ``w = 0`` component for
    any data -- a conservation identity, NOT accuracy -- so the columns that
    carry information are ``sum q+`` / ``sum q-`` and the correlation.

All three write their record to ``out`` (JSON) at finalize.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..fwk.component import Algorithm, algorithm
from ..model.warm_start import deconv_fft_torch, gaussian_filter_3d_torch
from ..solve.engine import Fista
from ..solve.strategy import Ladder, SolveState
from ..terms.base import CoordProx
from ..terms.data import DataFidelity


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def fit_bin_ticks(store) -> float:
    """Fit-bin width in fine ticks: ``adc_hold_delay / time_subbin``."""
    rc = store.get("readout_config")
    S = int(store.get("time_subbin")) if "time_subbin" in store else 1
    return float(int(rc.adc_hold_delay)) / S


def grid_truth(store, op) -> np.ndarray:
    """effq summed onto the operator's own charge grid (same frame as ``d``)."""
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), dtype=float)
    B = fit_bin_ticks(store)
    el = np.asarray(ev.effq.location)
    eq = np.asarray(ev.effq.data, dtype=float)[:, -1]
    nx, ny, nt = op.q_shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (it >= 0) & (it < nt)
    qg = np.zeros(op.q_shape)
    np.add.at(qg, (ix[ok], iy[ok], it[ok]), eq[ok])
    return qg


def best_shift_r(qg: np.ndarray, q: np.ndarray, span: int = 3):
    """(r, shift) maximising the correlation over a rigid time shift.

    The fit grid sits about half a bin later than the deposit convention
    (:func:`~unfoldlarpix.model.conventions.solver_time_shift`), so on a grid
    whose phase differs from the ZS arm's the peak tips into the next index.
    Reporting the best shift keeps a declaration offset from being read as an
    anticorrelation.
    """
    out = (float("-inf"), 0)
    for s in range(-span, span + 1):
        t = np.roll(qg, s, axis=2)
        m = t > 0.01
        if m.sum() < 3 or np.std(q[m]) == 0:
            continue
        r = float(np.corrcoef(t[m], q[m])[0, 1])
        if r > out[0]:
            out = (r, s)
    return out


def loss(op, q) -> float:
    """0.5 ||A q - d||^2 on the operator's own rows."""
    qt = q if torch.is_tensor(q) else op.to_tensor(q)
    return 0.5 * float(((op.forward(qt).detach() - op.d) ** 2).sum())


def block_from_rows(op) -> np.ndarray:
    """Recover the dense block from ``d`` -- valid only one-bin-per-row.

    With a fixed-interval readout the sampling matrix is a selection, so the
    recorded charges ARE the block (bin-integrated current) and no burst
    processing is involved.
    """
    rows = op._rows.cpu().numpy()
    cols = op._cols.cpu().numpy()
    w = op._weights.cpu().numpy()
    if len(rows) != op.n_data or not np.allclose(w, 1.0):
        raise ValueError(
            f"operator is not one-bin-per-row (entries {len(rows)} for "
            f"{op.n_data} rows, weights in [{w.min():.4g}, {w.max():.4g}]) -- "
            "block_from_rows only applies to a fixed-interval readout")
    blk = np.zeros(int(np.prod(op.block_shape)))
    blk[cols] = op.d.cpu().numpy()[rows]
    return blk.reshape(op.block_shape)


class _JsonRecorder(Algorithm):
    """Accumulate per-event records, dump to ``out`` at finalize."""

    def initialize(self, services):
        super().initialize(services)
        self._records: list[dict] = []
        self.out_path = self.props.get("out")

    def _emit(self, store, rec):
        self.put(store, self.writes[0], rec)
        self._records.append(rec)

    def finalize(self):
        if not self._records:
            return {}
        if self.out_path:
            with open(self.out_path, "w") as fh:
                json.dump(self._records if len(self._records) > 1
                          else self._records[0], fh, indent=1)
            print(f"[{self.name}] wrote {self.out_path}")
        return (self._records[0] if len(self._records) == 1
                else {"events": self._records})


# ---------------------------------------------------------------------------
@algorithm("FixedGridAudit")
class FixedGridAudit(_JsonRecorder):
    """On-grid check, operator closure against truth, measurement-gain map.

    Props
    -----
    out : str, optional
        JSON path for the record.
    gain_stride : int
        Stride for the printed per-time-bin gain profile (default 10).
    """

    reads = ("event", "readout_config", "op", "block_offset")
    writes = ("fixedgrid.audit",)

    def execute(self, store):
        op = store.get("op")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        B = fit_bin_ticks(store)
        d = op.d.detach()

        rows = op._rows.cpu().numpy()
        w = op._weights.cpu().numpy()
        per_row = np.bincount(rows, minlength=op.n_data)

        qg = grid_truth(store, op)
        Aqt = op.forward(op.to_tensor(qg)).detach()
        resid = (d - Aqt).cpu().numpy()
        row_scale = float(np.sqrt((d.cpu().numpy() ** 2).mean()))
        row_rms = float(np.sqrt((resid ** 2).mean()))

        # measurement gain c_v = A^T 1; bipolar kernel -> c_v may be negative,
        # which is NOT the same as the voxel being unobservable.
        c = op.measurement_gain().cpu().numpy()
        cmax = float(np.abs(c).max())
        reach = op.sample_adjoint(torch.ones(op.n_data, dtype=op.dtype,
                                             device=op.device)).detach()
        prep = self.services["detector"].prepared(int(round(B)))
        kabs = torch.as_tensor(np.abs(np.asarray(prep.integrated_response)),
                               dtype=op.dtype, device=op.device)
        sh = op.fft_shape
        R = torch.fft.irfftn(
            torch.fft.rfftn(reach, s=sh, dim=(0, 1, 2))
            * torch.conj(torch.fft.rfftn(kabs, s=sh, dim=(0, 1, 2))),
            s=sh, dim=(0, 1, 2))
        R = torch.roll(torch.roll(R, op.cx, dims=0), op.cy, dims=1)
        R = R[:op.q_shape[0], :op.q_shape[1], :op.q_shape[2]].cpu().numpy()
        unreachable = R <= 1e-9 * float(R.max())
        ct = c.max(axis=(0, 1))
        stride = int(self.props.get("gain_stride", 10))

        rec = {
            "rows": int(op.n_data),
            "block_shape": [int(s) for s in op.block_shape],
            "q_shape": [int(s) for s in op.q_shape],
            "kernel_bins": int(op.block_shape[2] - op.q_shape[2]),
            "fit_bin_ticks": B,
            "block_offset": [float(v) for v in boff],
            "block_offset_mod_bin": float(boff[2] % B),
            # the on-grid claim
            "cols_per_row_min": int(per_row.min()),
            "cols_per_row_max": int(per_row.max()),
            "cols_per_row_mean": float(per_row.mean()),
            "weight_min": float(w.min()), "weight_max": float(w.max()),
            "frac_weight_one": float(np.mean(np.isclose(w, 1.0))),
            "on_fit_grid": bool(per_row.max() == 1 and np.allclose(w, 1.0)),
            # is the truth a solution?
            "truth_on_grid": float(qg.sum()),
            "sum_d": float(d.sum()), "sum_A_qtruth": float(Aqt.sum()),
            "closure_sum": float(resid.sum()),
            "row_resid_mean": float(resid.mean()),
            "row_resid_rms": row_rms,
            "row_scale_rms": row_scale,
            "row_resid_rel_pct": 100.0 * row_rms / max(row_scale, 1e-12),
            "L_qtruth": 0.5 * float((resid ** 2).sum()),
            # what the measurement can see
            "gain_max": cmax,
            "n_gain_pos": int((c > 1e-6 * cmax).sum()),
            "n_gain_zero": int((np.abs(c) <= 1e-6 * cmax).sum()),
            "n_gain_neg": int((c < -1e-6 * cmax).sum()),
            "n_unreachable": int(unreachable.sum()),
            "gain_profile_stride": stride,
            "gain_profile": [float(v / max(cmax, 1e-12))
                             for v in ct[::stride]],
        }
        print(f"[{self.name}] on_fit_grid={rec['on_fit_grid']} "
              f"cols/row {rec['cols_per_row_mean']:.4f} "
              f"w=1 {rec['frac_weight_one']:.4f} | closure "
              f"{rec['closure_sum']:+.3f} ke, row resid "
              f"{row_rms:.4f} ke = {rec['row_resid_rel_pct']:.1f}% of scale | "
              f"gain +/0/- {rec['n_gain_pos']}/{rec['n_gain_zero']}/"
              f"{rec['n_gain_neg']}, unreachable {rec['n_unreachable']}")
        self._emit(store, rec)


# ---------------------------------------------------------------------------
class _SupportProx:
    """Support mask only -- no positivity, no l1."""

    def __init__(self, support):
        self.support = support
        self.alpha = 0.0

    def __call__(self, v, step):
        return v * self.support


@algorithm("EstimatorScan")
class EstimatorScan(_JsonRecorder):
    """Matched regularisation path: only the prior changes between arms.

    Props
    -----
    arms : list of dict
        Each ``{label, alpha, positivity, gain_cut, iters}``.  ``alpha`` may
        be a number (single prox) or a list (soft-seeded ladder).
        ``positivity: false`` replaces the shipped prox with a support-only
        one, which is the only way to separate the positivity bias from the
        l1 shrinkage.  ``gain_cut`` (fraction of max ``c_v``) trims the
        support to what the measurement can see; the invisible region
        otherwise dominates every unregularised arm.
    seed_cut, soft_len : float
        Ladder parameters (defaults 0.5 / 2.0, the shipped values).
    out : str, optional
    """

    reads = ("op", "support", "event", "readout_config", "block_offset")
    writes = ("fixedgrid.estimators",)

    def execute(self, store):
        op = store.get("op")
        base = np.asarray(store.get("support"))
        qg = grid_truth(store, op)
        T = float(qg.sum())
        c = op.measurement_gain().cpu().numpy()
        cmax = float(np.abs(c).max())
        seed_cut = float(self.props.get("seed_cut", 0.5))
        soft_len = float(self.props.get("soft_len", 2.0))

        arms = []
        for spec in self.props.get("arms", []):
            alpha = spec.get("alpha", 0.0)
            pos = bool(spec.get("positivity", True))
            gcut = spec.get("gain_cut")
            nit = int(spec.get("iters", 6000))
            supp = base if gcut is None else (base & (c > float(gcut) * cmax))
            st = op.to_tensor(supp.astype(np.float64))
            q0 = SolveState(q=op.to_tensor(np.zeros(op.q_shape)))
            if isinstance(alpha, (list, tuple)):
                if not pos:
                    raise ValueError("ladder arms are positivity-only "
                                     "(the ladder soft seed assumes q >= 0)")
                lad = Ladder(alphas=list(alpha), seed_cut=seed_cut,
                             soft_len=soft_len, n_iter=nit)
                q = lad.run(Fista(n_iter=nit), op, [DataFidelity(op)],
                            st, q0).q
            else:
                prox = (CoordProx(float(alpha), st) if pos
                        else _SupportProx(st))
                q = Fista(n_iter=nit).minimize(op, [DataFidelity(op)],
                                               prox, q0.q)
            q = q.detach().cpu().numpy().astype(np.float64)
            r, shift = best_shift_r(qg, q)
            rec = {
                "label": spec.get("label", f"alpha={alpha}"),
                "alpha": alpha, "positivity": pos, "gain_cut": gcut,
                "iters": nit,
                "support": int(supp.sum()),
                "rows_over_support": float(op.n_data / max(supp.sum(), 1)),
                "sum_q": float(q.sum()), "ratio_q": float(q.sum() / T),
                "sum_q_pos": float(q[q > 0].sum()),
                "sum_q_neg": float(q[q < 0].sum()),
                "pos_over_truth": float(q[q > 0].sum() / T),
                "nnz": int((np.abs(q) > 0.01).sum()),
                "max_abs_q": float(np.abs(q).max()),
                "max_q": float(q.max()), "min_q": float(q.min()),
                "n_pos": int((q > 0).sum()), "n_neg": int((q < 0).sum()),
                "L": loss(op, q), "r_best": r, "t_shift": shift,
            }
            print(f"[{self.name}] {rec['label']:28s} sum_q {rec['sum_q']:10.1f} "
                  f"({rec['ratio_q']:.4f}x)  q+ {rec['sum_q_pos']:9.1f}  "
                  f"q- {rec['sum_q_neg']:9.1f}  nnz {rec['nnz']:6d}  "
                  f"L {rec['L']:.4g}  r {r:+.3f}@{shift:+d}")
            arms.append(rec)
            del q
            torch.cuda.empty_cache()
        self._emit(store, {"truth_on_grid": T, "rows": int(op.n_data),
                           "L_qtruth": loss(op, qg), "arms": arms})


# ---------------------------------------------------------------------------
@algorithm("FFTInverseScan")
class FFTInverseScan(_JsonRecorder):
    """Linear (FFT) inverse of the same system over a filter-width scan.

    ``sum q`` is pinned to ``sum d / sum(K)`` by the ``w = 0`` component for
    ANY data, so it is a conservation identity and must not be quoted as
    accuracy; ``sum q+`` / ``sum q-`` and the correlation are the columns
    that carry information.

    Props
    -----
    sigmas_time : list
        Frequency-domain widths; ``null`` entry = unfiltered.  The
        time-domain width is ``1/(2 pi sigma_t)`` fine ticks and is reported.
    sigma_pixel : float
        Spatial filter width (default 0.2, the shipped value).
    out : str, optional
    """

    reads = ("event", "readout_config", "op", "block_offset")
    writes = ("fixedgrid.fft",)

    def execute(self, store):
        op = store.get("op")
        B = fit_bin_ticks(store)
        qg = grid_truth(store, op)
        T = float(qg.sum())
        blk = op.to_tensor(block_from_rows(op))
        bs = tuple(blk.shape)
        prep = self.services["detector"].prepared(int(round(B)))
        kern = torch.as_tensor(prep.integrated_response, dtype=op.dtype,
                               device=op.device)
        sig_p = float(self.props.get("sigma_pixel", 0.2))
        nt = op.q_shape[2]
        out = []
        for st_ in self.props.get("sigmas_time", [None, 0.005]):
            filt = None
            if st_ is not None:
                filt = gaussian_filter_3d_torch(
                    (bs[0] + kern.shape[0] - 1, bs[1] + kern.shape[1] - 1,
                     bs[2]), dt=(1, 1, B), sigma=(sig_p, sig_p, float(st_)),
                    device=op.device, dtype=op.dtype)
            q = deconv_fft_torch(blk, kern, filt).detach().cpu().numpy()
            q = q.astype(np.float64)[:, :, :nt]
            r, shift = best_shift_r(qg, q)
            pos, neg = float(q[q > 0].sum()), float(q[q < 0].sum())
            rec = {
                "sigma_time": st_,
                "time_width_ticks": (None if st_ is None
                                     else float(1.0 / (2 * np.pi * st_))),
                "sigma_pixel": sig_p,
                "sum_q": float(q.sum()), "ratio_q": float(q.sum() / T),
                "sum_q_pos": pos, "sum_q_neg": neg,
                "neg_over_pos": float(abs(neg) / max(pos, 1e-12)),
                "pos_over_truth": float(pos / T),
                "max_abs_q": float(np.abs(q).max()),
                "max_q": float(q.max()), "min_q": float(q.min()),
                "n_pos": int((q > 0).sum()), "n_neg": int((q < 0).sum()),
                "L": loss(op, q), "r_best": r, "t_shift": shift,
            }
            lab = "none" if st_ is None else f"{st_:g}"
            print(f"[{self.name}] sigma_t {lab:>7s} sum_q {rec['sum_q']:9.1f} "
                  f"({rec['ratio_q']:.4f}x)  q+ {pos:9.1f}  q- {neg:10.1f}  "
                  f"|q-|/q+ {rec['neg_over_pos']:.3f}  q+/truth "
                  f"{rec['pos_over_truth']:.3f}  max q {rec['max_q']:.2f}  "
                  f"min q {rec['min_q']:.2f}  L {rec['L']:.3g}  "
                  f"r {r:+.3f}@{shift:+d}")
            out.append(rec)
        self._emit(store, {"truth_on_grid": T, "sum_d": float(op.d.sum()),
                           "kernel_dc_gain": float(kern.sum()),
                           "filters": out})
