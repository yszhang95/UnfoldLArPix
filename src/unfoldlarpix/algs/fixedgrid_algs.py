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

from ..eval.universal import metrics_from_blocks, universal_rebin
from ..fwk.component import Algorithm, algorithm
from ..model.warm_start import deconv_fft_torch, gaussian_filter_3d_torch
from ..solve.engine import Fista
from ..solve.strategy import Ladder, SolveState
from ..terms.base import CoordProx
from ..smear_truth import gaus_smear_true_3d
from ..terms.data import DataFidelity


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def fit_bin_ticks(store) -> float:
    """Fit-bin width in fine ticks: ``adc_hold_delay / time_subbin``."""
    rc = store.get("readout_config")
    S = int(store.get("time_subbin")) if "time_subbin" in store else 1
    return float(int(rc.adc_hold_delay)) / S


def grid_truth(store, op, mode: str = "round") -> np.ndarray:
    """effq summed onto the operator's own charge grid (same frame as ``d``).

    FOR OPERATOR INPUT ONLY -- this is the true charge to push through ``A``
    (``A q_truth`` vs ``d``), where an unsmeared truth is exactly right.
    NEVER use it to score a reconstruction: the analysis filter smears the
    reco, and comparing a smeared reco against an unsmeared truth is the
    one-sided smearing that fakes the slope.  Use :func:`score_universal`.

    ``mode="round"`` is the BIN-CENTRE deposit, the adopted eval protocol
    (decided 2026-08-16 on the criterion that slope must be unbiased); the
    charge goes to the bin whose centre is nearest.  ``mode="floor"`` is the
    older nearest-lower-edge assignment kept only to reproduce archived
    numbers -- it moves a deposit sitting late in a bin a whole bin early,
    which on a 2-bin-wide isochronous feature shows up as an apparent
    reco-late offset (measured: L(q_truth) 1.16e4 with floor vs 1243 with
    round on isoline d16p5, i.e. a 9.3x difference in what looks like
    "operator error").
    """
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), dtype=float)
    B = fit_bin_ticks(store)
    el = np.asarray(ev.effq.location)
    eq = np.asarray(ev.effq.data, dtype=float)[:, -1]
    nx, ny, nt = op.q_shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    f = (el[:, 2] - boff[2]) / B
    it = (np.rint(f) if mode == "round" else np.floor(f)).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (it >= 0) & (it < nt)
    qg = np.zeros(op.q_shape)
    np.add.at(qg, (ix[ok], iy[ok], it[ok]), eq[ok])
    return qg


def smeared_truth(store, sigma_pixel: float = 0.5,
                  sigma_time: float = 0.005):
    """The event's truth smeared with the analysis filter (cached per event).

    Widths are the ADOPTED protocol (sigma_pxl 0.5, sigma_time 0.005), not
    the 0.2 that older production NPZs embed -- reusing the embedded 0.2
    truth against a 0.5 reco inflates the slope to 1.7-2.6.
    """
    key = ("_smeared", sigma_pixel, sigma_time)
    cache = getattr(store, "_fixedgrid_cache", None)
    if cache is None:
        cache = {}
        try:
            store._fixedgrid_cache = cache
        except Exception:
            pass
    if key in cache:
        return cache[key]
    ev = store.get("event")
    off, sm = gaus_smear_true_3d(np.asarray(ev.effq.location),
                                 np.asarray(ev.effq.data, dtype=float),
                                 width=np.array([sigma_pixel, sigma_pixel,
                                                 sigma_time]))
    cache[key] = (np.asarray(off), np.asarray(sm))
    return cache[key]


def score_universal(store, op, q: np.ndarray, sigma_pixel: float = 0.5,
                    sigma_time: float = 0.005,
                    corr_threshold: float = 0.5) -> dict:
    """Score a reconstruction against truth with BOTH SIDES SMEARED.

    The adopted eval protocol: universal grid (edges at global multiples of
    B), gaussian deposit of the sharp charge, no fitted sub-bin offsets,
    sigma_pxl 0.5 / sigma_time 0.005, corr_threshold 0.5.  Delegates the
    binning to :func:`~unfoldlarpix.eval.universal.universal_rebin` and the
    scalars to :func:`~unfoldlarpix.eval.universal.metrics_from_blocks`, so
    this cannot drift from the production numbers.

    Adds ``transport``: where the charge sits relative to the truth's own
    voxels on that same grid -- the test for a prior that concentrates a
    diffuse halo onto a few voxels rather than reconstructing it.
    """
    import tempfile
    from pathlib import Path
    boff = np.asarray(store.get("block_offset"), dtype=float)
    rc = store.get("readout_config")
    S = int(store.get("time_subbin")) if "time_subbin" in store else 1
    off, sm = smeared_truth(store, sigma_pixel, sigma_time)
    qf = np.asarray(q, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "arm.npz"
        np.savez(f, deconv_q=qf, deconv_q_sharp=qf,
                 boffset=boff, boffset_raw=boff,
                 adc_hold_delay=np.array(int(rc.adc_hold_delay) // S),
                 time_convention=np.array("release_point"),
                 smeared_true=sm, smear_offset=off)
        tru, reco = universal_rebin(f, deposit_shape="gaussian",
                                    sigma_time=sigma_time,
                                    sigma_pxl=sigma_pixel)
    out = dict(metrics_from_blocks(tru, reco, corr_threshold=corr_threshold))
    out["transport"] = transport_profile(tru, reco, cut=corr_threshold)
    return out


def _grow(mask: np.ndarray) -> np.ndarray:
    """One-voxel dilation with OPEN boundaries (np.roll would wrap)."""
    g = mask.copy()
    g[1:] |= mask[:-1]; g[:-1] |= mask[1:]
    g[:, 1:] |= mask[:, :-1]; g[:, :-1] |= mask[:, 1:]
    g[:, :, 1:] |= mask[:, :, :-1]; g[:, :, :-1] |= mask[:, :, 1:]
    return g


def transport_profile(tru: np.ndarray, reco: np.ndarray, cut: float = 0.5,
                      n_rings: int = 3) -> dict:
    """Where an estimator puts its charge relative to the TRUTH's voxels.

    Both blocks must already be on the same grid and smeared the same way
    (see :func:`score_universal`).  A sparsifying prior shows MORE charge on
    the truth voxels and less in the surrounding shells than a filtered
    inverse at the same total -- that is charge transported, not recovered.
    Measures concentration, not correctness: the truth mask is truth-derived,
    so read it beside the on-core excess, never alone.
    """
    core = tru > cut
    out = {"truth_total": float(tru.sum()), "q_on_truth": float(reco[core].sum()),
           "truth_on_core": float(tru[core].sum()), "n_core": int(core.sum())}
    seen = core.copy()
    for r in range(1, n_rings + 1):
        grown = _grow(seen)
        ring = grown & ~seen
        out[f"q_ring{r}"] = float(reco[ring].sum())
        seen = grown
    out["q_outside"] = float(reco[~seen].sum())
    out["nnz_pos"] = int((reco > cut).sum())
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
            sc = score_universal(store, op, q); tp = sc.pop("transport")
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
                "L": loss(op, q), "universal": sc, "transport": tp,
            }
            print(f"[{self.name}] {rec['label']:28s} sum_q {rec['sum_q']:10.1f} "
                  f"({rec['ratio_q']:.4f}x)  q+ {rec['sum_q_pos']:9.1f}  "
                  f"q- {rec['sum_q_neg']:9.1f}  nnz {rec['nnz']:6d}  "
                  f"L {rec['L']:.4g} | U r {sc['pearson_r']:+.4f} slope "
                  f"{sc['slope']:+.4f} int% {sc['integral_pct']:+.2f} "
                  f"ghostQ {sc['ghost_charge']:8.1f} killed {sc['true_killed']:7.1f}"
                  f" | core {tp['q_on_truth']:8.1f}/{tp['truth_on_core']:.1f} "
                  f"ring1 {tp['q_ring1']:7.1f} out {tp['q_outside']:7.1f}")
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
            sc = score_universal(store, op, q); tp = sc.pop("transport")
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
                "L": loss(op, q), "universal": sc, "transport": tp,
            }
            lab = "none" if st_ is None else f"{st_:g}"
            print(f"[{self.name}] sigma_t {lab:>7s} sum_q {rec['sum_q']:9.1f} "
                  f"({rec['ratio_q']:.4f}x)  q+ {pos:9.1f}  q- {neg:10.1f}  "
                  f"|q-|/q+ {rec['neg_over_pos']:.3f}  q+/truth "
                  f"{rec['pos_over_truth']:.3f}  max q {rec['max_q']:.2f}  "
                  f"min q {rec['min_q']:.2f}  L {rec['L']:.3g}  "
                  f"U r {sc['pearson_r']:+.4f} slope {sc['slope']:+.4f} "
                  f"int% {sc['integral_pct']:+.2f} | nnz+ {tp['nnz_pos']:6d} "
                  f"core {tp['q_on_truth']:8.1f}/{tp['truth_on_core']:.1f} "
                  f"ring1 {tp['q_ring1']:7.1f} out {tp['q_outside']:7.1f}")
            out.append(rec)
        self._emit(store, {"truth_on_grid": T, "sum_d": float(op.d.sum()),
                           "kernel_dc_gain": float(kern.sum()),
                           "filters": out})
