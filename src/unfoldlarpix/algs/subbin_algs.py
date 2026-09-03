"""Uniform-within-bin forward model: the equivalence check.

Two routes implement the same model (see
:mod:`unfoldlarpix.model.subbin_operator`):

``fine``    ``BuildMeasurement fwd_subbin: S`` — expand each coarse unknown
            flat over S sub-bins, convolve at ``B/S``, sample the FINE
            waveform with the ordinary overlap weights.
``kernel``  detector service ``within_bin: uniform, subbin: S`` — the same
            model reduced analytically to one coarse convolution with a
            Bartlett-combed kernel, S times cheaper.

They are identical iff every window edge lands on a fit-bin boundary, which
is what a fixed-interval readout guarantees and what the audit's
``on_fit_grid`` column reports.  This algorithm builds whichever route the
job did NOT build and compares them where it matters: the predicted rows
``A q_truth``, and the measurement gain ``c_v = A^T 1``.  A campaign that
quotes the cheap route needs this number in the record.
"""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import (build_latch_rows, windows_to_sampling)
from ..deconv_workflow import (integrate_kernel_over_time,
                               uniform_within_bin_kernel)
from ..fwk.component import algorithm
from ..model.operator import ZSOperator
from ..model.subbin_operator import ZSOperatorUniform
from .fixedgrid_algs import _JsonRecorder, grid_truth


@algorithm("UniformForwardCheck")
class UniformForwardCheck(_JsonRecorder):
    """Cross-check the two implementations of the uniform-in-bin operator.

    Props
    -----
    subbin : int
        ``S``.  Must match whatever the job configured.
    acq_start, split_trigger : as ``BuildMeasurement``
        Needed to rebuild the windows for the alternative operator; pass the
        SAME values the job's ``BuildMeasurement`` used.
    out : str, optional
    """

    reads = ("event", "readout_config", "op", "block", "block_offset")
    writes = ("subbin.check",)

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        op = store.get("op")
        block = store.get("block")
        boff = np.asarray(store.get("block_offset"))
        comp = self.services["compute"]
        det = self.services["detector"]
        S = int(self.props.get("subbin", getattr(op, "subbin", 1)))
        B = int(rc.adc_hold_delay)

        acq = self.props.get("acq_start")
        if acq == "event":
            acq = getattr(ev, "acq_start", None)
        elif acq is not None:
            acq = float(acq)
        windows, _ = build_latch_rows(
            ev.hits.location, ev.hits.data, B, boff,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=(float(rc.threshold)
                             if bool(self.props.get("split_trigger", False))
                             else None),
            acq_start=acq, burst_tau=None)
        nx, ny, nt = block.shape

        have_fine = isinstance(op, ZSOperatorUniform)
        if have_fine:
            kern = uniform_within_bin_kernel(
                det.prepared_raw(B).full_response, B, S,
                start_tick=det.start_tick)
            other = ZSOperator(kern, (nx, ny, nt), windows, B,
                               device=comp.device, dtype=comp.dtype)
            labels = ("fine", "kernel")
        else:
            fine = det.prepared_raw(B // S)
            other = ZSOperatorUniform(fine.integrated_response, (nx, ny, nt),
                                      windows, B, S, device=comp.device,
                                      dtype=comp.dtype)
            labels = ("kernel", "fine")

        if tuple(op.q_shape) != tuple(other.q_shape):
            raise RuntimeError(f"q_shape {op.q_shape} vs {other.q_shape}")
        if int(op.n_data) != int(other.n_data):
            raise RuntimeError(f"n_data {op.n_data} vs {other.n_data}")

        qg = grid_truth(store, op)
        a = op.forward(op.to_tensor(qg)).detach().cpu().numpy()
        b = other.forward(other.to_tensor(qg)).detach().cpu().numpy()
        ca = op.measurement_gain().detach().cpu().numpy()
        cb = other.measurement_gain().detach().cpu().numpy()
        scale = float(np.sqrt((a ** 2).mean()))

        rec = {
            "subbin": S,
            "route_in_job": labels[0], "route_rebuilt": labels[1],
            "q_shape": [int(v) for v in op.q_shape],
            "rows": int(op.n_data),
            "row_scale_rms": scale,
            "row_max_abs_diff": float(np.abs(a - b).max()),
            "row_rms_diff": float(np.sqrt(((a - b) ** 2).mean())),
            "row_rel_pct": float(100 * np.sqrt(((a - b) ** 2).mean())
                                 / max(scale, 1e-12)),
            "sum_A_qtruth": [float(a.sum()), float(b.sum())],
            "gain_max_abs_diff": float(np.abs(ca - cb).max()),
            "gain_scale": float(np.abs(ca).max()),
        }
        print(f"[{self.name}] S={S}  {labels[0]} vs {labels[1]}:  rows "
              f"max|d| {rec['row_max_abs_diff']:.3e} ke, rms "
              f"{rec['row_rms_diff']:.3e} = {rec['row_rel_pct']:.2e}% of "
              f"scale {scale:.4f} | sum A q_truth "
              f"{rec['sum_A_qtruth'][0]:.4f} vs {rec['sum_A_qtruth'][1]:.4f} "
              f"| max|dc_v| {rec['gain_max_abs_diff']:.3e} of "
              f"{rec['gain_scale']:.4f}")
        del other
        torch.cuda.empty_cache()
        self._emit(store, rec)


@algorithm("UniformKernelAudit")
class UniformKernelAudit(_JsonRecorder):
    """Prove the uniform-within-bin kernel is what it claims to be.

    A source algorithm (``reads = ()``): it needs no event, only the response
    file, so it runs in seconds and its JSON is the record that the operator
    used here is the model that was intended.  Four checks per ``subbin``:

    ``brute_force``    ``Keff[j]`` against the DEFINITION evaluated as a
        literal double sum, ``(1/S) sum_{u,s in [0,S)} Kf[j*S + u - s]`` --
        no algebra shared with the implementation.
    ``charge``         ``sum Keff == sum Kf == sum K``; the comb must not
        create or destroy charge.
    ``identity``       ``subbin = 1`` must return
        :func:`integrate_kernel_over_time` bit for bit.
    ``moments``        the kernel's charge-weighted mean time minus the delta
        form's, which is the RELEASE OFFSET the rest of the chain assumes is
        ``B(S-1)/(2S)``.  Reported, not asserted -- the evaluation-offset
        scan measures what the data prefers, and the two need not agree.

    Props: ``subbins`` (list, default 1..30), ``bin_ticks`` (default 30),
    ``out``.
    """

    reads = ()
    writes = ("subbin.kernel_audit",)

    def execute(self, store):
        det = self.services["detector"]
        B = int(self.props.get("bin_ticks", 30))
        subbins = [int(v) for v in self.props.get(
            "subbins", [1, 2, 3, 5, 6, 10, 15, 30])]
        full = np.asarray(det.prepared_raw(B).full_response, dtype=np.float64)
        Kc = integrate_kernel_over_time(full, B, start_tick=det.start_tick)
        tc = np.arange(Kc.shape[-1], dtype=np.float64) * B
        mean_delta = float((Kc * tc).sum() / Kc.sum())

        rows = []
        for S in subbins:
            Kf = integrate_kernel_over_time(full, B // S,
                                            start_tick=det.start_tick)
            Ke = uniform_within_bin_kernel(full, B, S,
                                           start_tick=det.start_tick)
            # the definition, written out
            ktc, ktf = Ke.shape[-1], Kf.shape[-1]
            bf = np.zeros_like(Ke)
            for j in range(ktc):
                acc = np.zeros(Ke.shape[:2])
                for u in range(S):
                    for v in range(S):
                        n = j * S + u - v
                        if 0 <= n < ktf:
                            acc += Kf[:, :, n]
                bf[:, :, j] = acc / S
            te = np.arange(ktc, dtype=np.float64) * B
            mean_uni = float((Ke * te).sum() / Ke.sum())
            rec = {
                "subbin": S, "sub_bin_ticks": B // S,
                "kernel_bins": int(ktc), "fine_kernel_bins": int(ktf),
                "max_abs_diff_vs_definition": float(np.abs(Ke - bf).max()),
                "sum_Keff": float(Ke.sum()), "sum_Kfine": float(Kf.sum()),
                "sum_Kdelta": float(Kc.sum()),
                "charge_conserved": bool(
                    abs(Ke.sum() - Kc.sum()) < 1e-12 * abs(Kc.sum())),
                "identity_at_S1": (None if S != 1 else
                                   bool(np.array_equal(Ke, Kc))),
                "mean_time_ticks": mean_uni,
                "release_offset_measured_ticks": mean_uni - mean_delta,
                "release_offset_assumed_ticks": (0.0 if S == 1 else
                                                 B * (S - 1) / (2.0 * S)),
            }
            print(f"[{self.name}] S={S:3d} sub-bin {B // S:2d} tk  "
                  f"kt {ktc:3d}  max|Keff-definition| "
                  f"{rec['max_abs_diff_vs_definition']:.3e}  sum "
                  f"{rec['sum_Keff']:.9f} vs {rec['sum_Kdelta']:.9f}  "
                  f"release offset measured "
                  f"{rec['release_offset_measured_ticks']:+7.3f} vs assumed "
                  f"{rec['release_offset_assumed_ticks']:+7.3f} ticks")
            rows.append(rec)
        self._emit(store, {"bin_ticks": B, "mean_time_delta_ticks": mean_delta,
                           "subbins": rows})


@algorithm("WindowGeometryAudit")
class WindowGeometryAudit(_JsonRecorder):
    """Are the readout windows compatible with the block the warm start built?

    Runs BEFORE ``BuildMeasurement`` and needs no operator, which is the
    point: ``BuildMeasurement`` raises on exactly the configurations this is
    meant to diagnose.  It rebuilds the windows with the same arguments and
    reports, per event:

    * the block geometry and ``block_offset``;
    * how many windows fail ``t_hi > max(t_lo, 0)`` -- i.e. the asserted
      ``acq_start`` edge lands AFTER the pixel's own first latch -- and, the
      column that decides whether it matters, **how much charge they carry**;
    * the columns-per-row distribution from :func:`windows_to_sampling`.
      ``max == 1`` is the on-fit-grid condition that ``block_from_rows``
      (and hence every FFT arm) requires.

    ``acq_start`` is an ABSOLUTE tick while ``block_offset`` moves with the
    drift depth, so both failure modes are depth-dependent and neither is a
    property of the charge model -- run this with and without
    ``within_bin: uniform`` and the numbers are identical.

    Props: ``acq_start``, ``split_trigger``, ``burst_tau``, ``out``.
    """

    reads = ("event", "readout_config", "block", "block_offset")
    writes = ("subbin.window_geometry",)

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        block = store.get("block")
        boff = np.asarray(store.get("block_offset"))
        B = int(rc.adc_hold_delay)
        nx, ny, nt = (int(v) for v in block.shape)

        acq = self.props.get("acq_start")
        if acq == "event":
            acq = getattr(ev, "acq_start", None)
        elif acq is not None:
            acq = float(acq)
        split = bool(self.props.get("split_trigger", False))
        windows, metas = build_latch_rows(
            ev.hits.location, ev.hits.data, B, boff,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=float(rc.threshold) if split else None,
            acq_start=acq, burst_tau=self.props.get("burst_tau"))

        t_lo = np.array([w.t_lo for w in windows], dtype=np.float64)
        t_hi = np.array([w.t_hi for w in windows], dtype=np.float64)
        px = np.array([w.px for w in windows])
        py = np.array([w.py for w in windows])
        val = np.array([w.value for w in windows], dtype=np.float64)
        in_pix = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        kept = in_pix & (t_hi > np.maximum(t_lo, 0.0))
        dropped = in_pix & ~kept

        rows, cols, wts = windows_to_sampling(windows, (nx, ny, nt), B)
        per_row = np.bincount(rows, minlength=len(windows))
        hist = {str(int(k)): int(v) for k, v in
                zip(*np.unique(per_row, return_counts=True))}

        rec = {
            "acq_start": (None if acq is None else float(acq)),
            "block_shape": [nx, ny, nt], "bin_ticks": B,
            "block_offset": [float(v) for v in boff],
            "n_windows": int(len(windows)),
            "n_outside_pixels": int((~in_pix).sum()),
            "n_dropped_edge": int(dropped.sum()),
            "charge_total_ke": float(val.sum()),
            "charge_in_dropped_ke": float(val[dropped].sum()),
            "charge_in_dropped_frac": float(
                abs(val[dropped].sum()) / max(abs(val.sum()), 1e-12)),
            "dropped_t_lo_range": ([float(t_lo[dropped].min()),
                                    float(t_lo[dropped].max())]
                                   if dropped.any() else None),
            "dropped_t_hi_range": ([float(t_hi[dropped].min()),
                                    float(t_hi[dropped].max())]
                                   if dropped.any() else None),
            "dropped_kinds": ({} if not dropped.any() else {
                k: int(sum(1 for i in np.nonzero(dropped)[0]
                           if metas[i].kind == k))
                for k in sorted({metas[i].kind
                                 for i in np.nonzero(dropped)[0]})}),
            "sampling_entries": int(len(rows)),
            "cols_per_row_min": int(per_row.min()),
            "cols_per_row_max": int(per_row.max()),
            "cols_per_row_hist": hist,
            "weight_min": float(wts.min()), "weight_max": float(wts.max()),
            "on_fit_grid": bool(per_row.max() == 1
                                and np.allclose(wts, 1.0)),
            "buildmeasurement_would_raise": bool(dropped.any()),
            "block_from_rows_would_raise": bool(
                len(rows) != len(windows) or not np.allclose(wts, 1.0)),
        }
        print(f"[{self.name}] offset {rec['block_offset']}  windows "
              f"{rec['n_windows']}  dropped {rec['n_dropped_edge']} carrying "
              f"{rec['charge_in_dropped_ke']:.4f} ke of "
              f"{rec['charge_total_ke']:.3f} | entries "
              f"{rec['sampling_entries']}  cols/row "
              f"{rec['cols_per_row_min']}..{rec['cols_per_row_max']}  "
              f"on_fit_grid={rec['on_fit_grid']} | would raise: "
              f"BuildMeasurement={rec['buildmeasurement_would_raise']} "
              f"block_from_rows={rec['block_from_rows_would_raise']}")
        self._emit(store, rec)
