"""Reconstruction algorithms: warm start, measurement, support, solve."""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import (build_latch_windows, centroid_bin_offsets,
                                  gaussian_post_smooth)
from ..fwk.component import Algorithm, algorithm
from ..model.operator import ZSOperator
from ..model.warm_start import fft_warm_start
from ..solve.engine import Fista
from ..solve.strategy import FinalRefit, Ladder, SolveState
from ..terms.censor import CensorRunningMax
from ..terms.data import DataFidelity
from ..terms.quiet import QuietHinge


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
        ws = fft_warm_start(
            ev.hits, rc, prepared,
            sigma_time=float(self.props.get("sigma_time", 0.005)),
            sigma_pixel=float(self.props.get("sigma_pixel", 0.2)),
            pad_pixels=int(self.props.get("pad_pixels", 0)),
            device=comp.device, dtype=comp.dtype)
        self.put(store, "warm.deconv_q", ws.deconv_q)
        self.put(store, "block", ws.block)
        self.put(store, "block_offset", ws.block_offset)


@algorithm("BuildMeasurement")
class BuildMeasurement(Algorithm):
    """Immutable event measurement: windows, operator A(d), quiet mask."""

    reads = ("event", "readout_config", "block", "block_offset")
    writes = ("op", "quiet_mask")

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        block = store.get("block")
        block_offset = np.asarray(store.get("block_offset"))
        comp = self.services["compute"]
        prepared = self.services["detector"].prepared(rc.adc_hold_delay)
        thr = float(rc.threshold)
        split = bool(self.props.get("split_trigger", True))
        windows = build_latch_windows(
            ev.hits.location, ev.hits.data, rc.adc_hold_delay, block_offset,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=thr if split else None)
        op = ZSOperator(prepared.integrated_response, block.shape, windows,
                        rc.adc_hold_delay, device=comp.device,
                        dtype=comp.dtype)
        quiet = np.ones(block.shape, dtype=bool)
        for row in ev.hits.location:
            px = int(row[0] - block_offset[0])
            py = int(row[1] - block_offset[1])
            if 0 <= px < block.shape[0] and 0 <= py < block.shape[1]:
                quiet[px, py, :] = False
        self.put(store, "op", op)
        self.put(store, "quiet_mask", quiet)


@algorithm("BuildSupport")
class BuildSupport(Algorithm):
    """ROI support on the fit grid.

    source=warm: threshold the (legacy-compatible: re-smoothed) warm
    start.  ``smooth_first`` defaults to True for parity with the
    adopted pipeline; FINDINGS item 18(b) documents that the direct
    threshold is the clean equivalent.
    """

    reads = ("warm.deconv_q", "op", "readout_config")
    writes = ("support",)

    def execute(self, store):
        rc = store.get("readout_config")
        op = store.get("op")
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
        support = support[:, :, : op.q_shape[2]]
        print(f"[BuildSupport] {support.mean() * 100:.2f}% of q voxels")
        self.put(store, "support", support)


@algorithm("Solve")
class Solve(Algorithm):
    """Constrained solve: terms + engine + strategies from config."""

    reads = ("op", "quiet_mask", "support", "warm.deconv_q",
             "hits_view", "block_offset", "readout_config")
    writes = ("solve.q", "solve.state")

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        thr = float(rc.threshold)

        terms = [DataFidelity(op)]
        for tcfg in self.props.get("terms", []):
            kind = tcfg["type"]
            if kind == "quiet":
                terms.append(QuietHinge(op, store.get("quiet_mask"), thr,
                                        beta=float(tcfg.get("beta", 1.0))))
            elif kind == "censor":
                terms.append(CensorRunningMax.from_hits(
                    op, store.get("hits_view"), store.get("block_offset"),
                    csa_reset_time=float(rc.csa_reset_time or 0),
                    threshold=thr,
                    npad_bins=int(tcfg.get("npad_bins", 50)),
                    beta=float(tcfg.get("beta", 1.0)),
                    margin=float(tcfg.get("margin", 3.0)),
                    norm=tcfg.get("norm", "l2")))
            else:
                raise ValueError(f"unknown term type: {kind}")

        support_np = store.get("support")
        support = op.to_tensor(support_np.astype(np.float64))
        q0 = op.to_tensor(np.clip(
            store.get("warm.deconv_q")[:, :, : op.q_shape[2]], 0.0, None))

        engine = Fista(n_iter=int(self.props.get("engine", {})
                                  .get("iters", 150)))
        scfg = dict(self.props.get("strategy", {}))
        stype = scfg.pop("type", "ladder")
        if stype != "ladder":
            raise ValueError(f"unknown strategy: {stype}")
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
        self.put(store, "solve.q",
                 state.q.cpu().numpy().astype(np.float64))
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
