"""Reconstruction algorithms: warm start, measurement, support, solve."""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import (build_latch_rows, centroid_bin_offsets,
                                  gaussian_post_smooth)
from ..fwk.component import Algorithm, algorithm
from ..model.conventions import resolve_burst_tau
from ..model.noise import row_weights
from ..model.operator import ZSOperator
from ..model.warm_start import fft_warm_start
from ..solve.engine import Fista
from ..solve.strategy import FinalRefit, Ladder, SolveState
from ..terms.censor import CensorRunningMax
from ..terms.data import DataFidelity


@algorithm("FFTWarmStart")
class FFTWarmStart(Algorithm):
    """Compensated-block FFT deconvolution (GPU) — warm-start provider.

    Also publishes the block geometry (offset/shape), which downstream
    measurement building shares.
    """

    reads = ("event", "readout_config")
    writes = ("warm.deconv_q", "block", "block_offset")

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        comp = self.services["compute"]
        prepared = self.services["detector"].prepared(rc.adc_hold_delay)
        tau = self.props.get("tau")   # None -> physical floor (resolve_burst_tau)
        ws = fft_warm_start(
            ev.hits, rc, prepared,
            sigma_time=float(self.props.get("sigma_time", 0.005)),
            sigma_pixel=float(self.props.get("sigma_pixel", 0.2)),
            pad_pixels=int(self.props.get("pad_pixels", 0)),
            tau=None if tau is None else int(tau),
            device=comp.device, dtype=comp.dtype)
        self.put(store, "warm.deconv_q", ws.deconv_q)
        self.put(store, "block", ws.block)
        self.put(store, "block_offset", ws.block_offset)


@algorithm("BuildMeasurement")
class BuildMeasurement(Algorithm):
    """Immutable event measurement: latch windows and the operator A(d)."""

    reads = ("event", "readout_config", "block", "block_offset")
    writes = ("op", "time_subbin")

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        block = store.get("block")
        block_offset = np.asarray(store.get("block_offset"))
        comp = self.services["compute"]
        S = int(self.props.get("time_subbin", 1))
        B = int(rc.adc_hold_delay)
        if S < 1 or B % S != 0:
            raise ValueError(f"time_subbin {S} must be >=1 and divide "
                             f"adc_hold_delay {B}")
        # finer operator bin = B/S; the physical latch windows are unchanged.
        prepared = self.services["detector"].prepared(B // S)
        thr = float(rc.threshold)
        split = bool(self.props.get("split_trigger", True))
        # acq_start: lower edge of each channel's first window.
        #   None (default) -> legacy -inf;
        #   "event"        -> the event's canonical acq_start (scalar or
        #                     channel-wise callable; the LOADER translates
        #                     any file-format specifics into it);
        #   number         -> absolute ticks (diagnostics only).
        acq = self.props.get("acq_start")
        if acq == "event":
            acq = getattr(ev, "acq_start", None)
            if acq is None:
                raise ValueError("acq_start: 'event' but the event carries "
                                 "no canonical acq_start")
        elif acq is not None:
            acq = float(acq)
        # burst_tau: gate on the split_trigger pseudo-measurement.
        #   absent -> legacy (every trigger treated as threshold-limited);
        #   "auto" -> the physical floor derived from the readout config;
        #   number -> ticks, clamped to [floor, cap] with a warning.
        # Resolution is a CONVENTION (pure function of the readout config);
        # build_latch_windows receives only the resolved integer.
        tau_prop = self.props.get("burst_tau")
        if tau_prop is None:
            burst_tau = None
        elif tau_prop == "auto":
            burst_tau = resolve_burst_tau(rc, None)
        else:
            burst_tau = resolve_burst_tau(rc, int(tau_prop))
        windows, metas = build_latch_rows(
            ev.hits.location, ev.hits.data, B, block_offset,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=thr if split else None,
            acq_start=acq, burst_tau=burst_tau)
        # row_weights: diagonal data-fidelity weighting from the readout
        # noise model (model.noise; scales travel with the data file).
        #   absent -> legacy unweighted;
        #   "split" -> only the trigger-split rows re-weighted;
        #   "diag"  -> every row weighted by ref_var / var.
        # The reference is the burst-diff variance, so ordinary rows keep
        # weight 1 and the l1/censor balance is unchanged by construction.
        rw_prop = self.props.get("row_weights")
        weights = (None if rw_prop is None
                   else row_weights(metas, rc, mode=str(rw_prop)))
        nx, ny, nt = block.shape
        op = ZSOperator(prepared.integrated_response, (nx, ny, nt * S), windows,
                        B // S, device=comp.device, dtype=comp.dtype,
                        row_weights=weights)
        self.put(store, "op", op)
        self.put(store, "time_subbin", S)


@algorithm("BuildSupport")
class BuildSupport(Algorithm):
    """ROI support on the fit grid.

    source=warm: threshold the (legacy-compatible: re-smoothed) warm
    start.  ``smooth_first`` defaults to True for parity with the
    adopted pipeline; FINDINGS item 18(b) documents that the direct
    threshold is the clean equivalent.
    """

    reads = ("warm.deconv_q", "op", "readout_config", "time_subbin")
    writes = ("support",)

    def execute(self, store):
        rc = store.get("readout_config")
        op = store.get("op")
        S = int(store.get("time_subbin") or 1)
        dq = np.clip(store.get("warm.deconv_q"), 0.0, None)
        eps = float(self.props.get("eps", 0.3))
        dilate = int(self.props.get("dilate", 1))
        if bool(self.props.get("smooth_first", True)):
            dq = gaussian_post_smooth(
                dq, rc.adc_hold_delay,
                float(self.props.get("sigma_time", 0.005)),
                float(self.props.get("sigma_pixel", 0.2)))
        support = dq > eps
        for _ in range(dilate):
            grown = support.copy()
            for ax in range(3):
                for shift in (-1, 1):
                    grown |= np.roll(support, shift, axis=ax)
            support = grown
        if S > 1:                       # lift the B-grid support to the B/S grid
            support = np.repeat(support, S, axis=2)
        support = support[:, :, : op.q_shape[2]]
        print(f"[BuildSupport] {support.mean() * 100:.2f}% of q voxels"
              f"{f' (time_subbin={S})' if S > 1 else ''}")
        self.put(store, "support", support)


@algorithm("Solve")
class Solve(Algorithm):
    """Constrained solve: terms + engine + strategies from config."""

    reads = ("op", "support", "warm.deconv_q",
             "hits_view", "block_offset", "readout_config", "time_subbin")
    writes = ("solve.q", "solve.state")

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        thr = float(rc.threshold)
        S = int(store.get("time_subbin") or 1)
        bin_ticks = int(rc.adc_hold_delay) // S

        terms = [DataFidelity(op)]
        for tcfg in self.props.get("terms", []):
            kind = tcfg["type"]
            if kind == "censor":
                terms.append(CensorRunningMax.from_hits(
                    op, store.get("hits_view"), store.get("block_offset"),
                    csa_reset_time=float(rc.csa_reset_time or 0),
                    threshold=thr,
                    npad_bins=int(tcfg.get("npad_bins", 50)) * S,
                    beta=float(tcfg.get("beta", 1.0)),
                    margin=float(tcfg.get("margin", 3.0)),
                    norm=tcfg.get("norm", "l2"),
                    bin_ticks=bin_ticks))
            else:
                raise ValueError(f"unknown term type: {kind}")

        support_np = store.get("support")
        support = op.to_tensor(support_np.astype(np.float64))
        q0_np = np.clip(store.get("warm.deconv_q"), 0.0, None)
        if S > 1:                       # lift the B-grid warm seed to B/S (conserve)
            q0_np = np.repeat(q0_np, S, axis=2) / S
        q0 = op.to_tensor(q0_np[:, :, : op.q_shape[2]])

        engine = Fista(n_iter=int(self.props.get("engine", {})
                                  .get("iters", 150)))
        scfg = dict(self.props.get("strategy", {}))
        stype = scfg.pop("type", "ladder")
        if stype != "ladder":
            raise ValueError(f"unknown strategy: {stype}")
        # gain_alpha: scale the l1 weights by the per-voxel measurement
        # gain c_v = A^T 1 (an operator/geometry quantity -- blind to the
        # data and to truth), normalised to its median over the support and
        # clipped.  Equalises the coordinate-activation condition so
        # weak-coverage charge is no longer preferentially shrunk.
        ga = scfg.pop("gain_alpha", None)
        if ga:
            ga = ga if isinstance(ga, dict) else {}
            c = op.measurement_gain()
            on = c[support > 0]
            ref = float(on.median()) if on.numel() else 1.0
            scale = torch.clamp(c / max(ref, 1e-12),
                                float(ga.get("floor", 0.05)),
                                float(ga.get("cap", 2.0)))
            scfg["alpha_scale"] = scale
        ladder = Ladder(n_iter=engine.n_iter, **scfg)
        state = ladder.run(engine, op, terms, support, SolveState(q=q0))
        if "refit" in self.props:
            rcfg = self.props["refit"]
            state = FinalRefit(eps=float(rcfg.get("eps", 0.5)),
                               alpha=float(rcfg.get("alpha", 0.0)),
                               n_iter=engine.n_iter).run(
                engine, op, terms, support, state)
        for rec in state.history:
            print(f"[Solve] {rec.label}: alpha={rec.alpha} "
                  f"q_sum={rec.q_sum:.1f} nnz={rec.nnz}")
        q = state.q.cpu().numpy().astype(np.float64)
        if S > 1:                       # fit at B/S, report at B (sum sub-bins)
            nx, ny, qt = q.shape
            q = q[:, :, : (qt // S) * S].reshape(nx, ny, qt // S, S).sum(axis=3)
        self.put(store, "solve.q", q)
        self.put(store, "solve.state", state)


@algorithm("CentroidPositions")
class CentroidPositions(Algorithm):
    """Sub-bin positions = local reco centroid (truth-free; FINDINGS 16)."""

    reads = ("solve.q",)
    writes = ("offsets.u",)

    def execute(self, store):
        u = centroid_bin_offsets(
            store.get("solve.q"),
            window_bins=int(self.props.get("window", 1)))
        self.put(store, "offsets.u", u)
