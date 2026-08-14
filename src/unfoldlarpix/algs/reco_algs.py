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
from ..terms.base import IterCtx
from ..terms.censor import CensorRunningMax, pre_trigger_censors
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

    source=warm (default): threshold the (legacy-compatible: re-smoothed)
    warm start.  ``smooth_first`` defaults to True for parity with the
    adopted pipeline; FINDINGS item 18(b) documents that the direct
    threshold is the clean equivalent.  CAVEAT: the threshold ``eps`` is
    an absolute charge, so the support SHRINKS as the signal attenuates
    (measured: coverage 1.94% -> 1.40% over a 30-cm drift at 1 ms
    lifetime) -- an amplitude coupling that a lifetime analysis sees as
    fake decay.

    source=hits: build the support from the DATA -- every pixel within
    ``hits_dilate`` (Chebyshev) of a fired pixel, over each hit's
    [trigger - t_pad, last latch + t_pad] time-bin extent.  Whether a
    pixel fired is a binary fact, so this support is amplitude-blind by
    construction.
    """

    reads = ("warm.deconv_q", "op", "readout_config", "time_subbin",
             "hits_view", "block_offset")
    writes = ("support",)

    def _from_hits(self, store, op, S):
        rc = store.get("readout_config")
        hv = store.get("hits_view")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        B = float(rc.adc_hold_delay) / S
        N = int(self.props.get("hits_dilate", 2))
        tpad = int(self.props.get("t_pad", 2))
        nx, ny, qt = op.q_shape
        supp = np.zeros((nx, ny, qt), dtype=bool)
        px = (hv.pixel_x - int(boff[0])).astype(int)
        py = (hv.pixel_y - int(boff[1])).astype(int)
        # hits live at the ANODE-ARRIVAL time; the charge grid lives at the
        # response reference plane, one kernel length earlier.  The kernel
        # length in bins is exactly the block/q time-extent difference.
        kshift = op.block_shape[2] - qt
        trig = (hv.trigger - boff[2]) / B - kshift
        last = (hv.last_latch - boff[2]) / B - kshift
        for i in range(len(px)):
            if not (-N <= px[i] < nx + N and -N <= py[i] < ny + N):
                continue
            b0 = max(int(np.floor(trig[i])) - tpad, 0)
            b1 = min(int(np.ceil(last[i])) + tpad + 1, qt)
            if b1 <= b0:
                continue
            supp[max(px[i]-N, 0):px[i]+N+1,
                 max(py[i]-N, 0):py[i]+N+1, b0:b1] = True
        return supp

    def execute(self, store):
        rc = store.get("readout_config")
        op = store.get("op")
        S = int(store.get("time_subbin") or 1)
        if self.props.get("source", "warm") == "hits":
            support = self._from_hits(store, op, S)
            print(f"[BuildSupport] {support.mean() * 100:.2f}% of q voxels "
                  f"(source=hits)")
            self.put(store, "support", support)
            return
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


def post_reset_wanted(tcfg: dict) -> bool:
    """Resolve the censor_pre interval scope from a term config.

    ``include_post_reset`` is the key.  ``max_slot`` is a LEGACY alias kept
    read-only: the archived censor_pre* campaigns wrote it (0 = the
    pre-trigger interval alone, None = every interval), and dropping it would
    make their embedded job_config unreplayable.
    """
    if "include_post_reset" in tcfg:
        return bool(tcfg["include_post_reset"])
    if "max_slot" in tcfg:
        return tcfg["max_slot"] is None
    return False


@algorithm("Solve")
class Solve(Algorithm):
    """Constrained solve: terms + engine + strategies from config."""

    reads = ("op", "support", "warm.deconv_q", "event",
             "hits_view", "block_offset", "readout_config", "time_subbin")
    writes = ("solve.q", "solve.state", "solve.loss", "solve.trace")

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
            elif kind == "censor_pre":
                # silence BEFORE a trigger: the pre-trigger interval (a
                # pixel's first) plus, with include_post_reset, the later
                # post-reset ones.  The pre-trigger reference is the
                # acquisition edge, so it must match BuildMeasurement's
                # acq_start convention: pass acq_start: event to take it from
                # the event, as the operator does.
                acq = tcfg.get("acq_start")
                if acq == "event":
                    acq = getattr(store.get("event"), "acq_start", None)
                    if acq is None:
                        raise ValueError("censor_pre acq_start: 'event' but "
                                         "the event carries no acq_start")
                elif acq is not None:
                    acq = float(acq)
                pre = pre_trigger_censors(
                    op, store.get("hits_view"), store.get("block_offset"),
                    csa_reset_time=float(rc.csa_reset_time or 0),
                    threshold=thr, acq_start=acq,
                    npad_bins=int(tcfg.get("npad_bins", 50)) * S,
                    beta=float(tcfg.get("beta", 1.0)),
                    margin=float(tcfg.get("margin", 3.0)),
                    norm=tcfg.get("norm", "l1"),
                    bin_ticks=bin_ticks,
                    one_tick=float(rc.one_tick or 1),
                    close_back=float(tcfg.get("close_back", 20.0)),
                    include_post_reset=post_reset_wanted(tcfg))
                print(f"[Solve] censor_pre: {len(pre)} interval kind(s), "
                      f"{sum(int(t.armed.any(dim=2).sum()) for t in pre)} "
                      f"constrained pixel-intervals")
                terms.extend(pre)
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
        tr = int(self.props.get("trace_every", 0))
        ladder = Ladder(n_iter=engine.n_iter, trace_every=tr, **scfg)
        state = ladder.run(engine, op, terms, support, SolveState(q=q0))
        if "refit" in self.props:
            rcfg = self.props["refit"]
            state = FinalRefit(eps=float(rcfg.get("eps", 0.5)),
                               alpha=float(rcfg.get("alpha", 0.0)),
                               n_iter=engine.n_iter,
                               trace_every=tr).run(
                engine, op, terms, support, state)
        for rec in state.history:
            print(f"[Solve] {rec.label}: alpha={rec.alpha} "
                  f"q_sum={rec.q_sum:.1f} nnz={rec.nnz}")
        # loss ledger: every objective component evaluated at the final
        # solution, with the weights actually used (censor includes beta).
        # The l1 LOSS at the solution is alpha-field dependent; what is
        # recorded is the norm (sum q) plus the configured weights, which
        # together with the stored job_config reproduce the objective.
        ctx = IterCtx(state.q, op)
        loss = {type(t).__name__: float(t.value(ctx)) for t in terms}
        loss["l1_sum_q"] = float(state.q.sum())
        if "refit" in self.props:
            loss["refit_alpha"] = float(
                self.props["refit"].get("alpha", 0.0))
        self.put(store, "solve.loss", loss)
        self.put(store, "solve.trace", state.trace)
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
