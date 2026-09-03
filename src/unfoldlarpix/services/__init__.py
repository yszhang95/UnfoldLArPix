"""Framework services (long-lived, shared infrastructure)."""
from __future__ import annotations

import numpy as np
import torch

from ..fwk.component import Service, service


@service("compute")
class ComputeService(Service):
    """Device/dtype policy for the whole job."""

    def initialize(self) -> None:
        self.device = torch.device(self.props.get("device", "cuda"))
        self.dtype = {"float32": torch.float32,
                      "float64": torch.float64}[
                          self.props.get("dtype", "float32")]


@service("detector")
class DetectorService(Service):
    """Field response and its prepared (bin-integrated) form.

    ``prepared(B)`` caches per adc_hold_delay — run-level constant data.

    ``within_bin`` selects the operator's WITHIN-BIN CHARGE MODEL, which is
    the thing the bin-integrated kernel actually encodes:

    ``delta`` (default, shipped)
        all of a fit bin's charge released as a delta at the bin's LOWER
        edge ``t_b = b*B``.
    ``uniform``
        the bin's charge spread uniformly across the bin, realised as
        ``subbin`` equal sub-deposits one fine sub-bin apart, convolved with
        the response integrated at ``B/subbin`` and re-summed onto the fit
        grid (:func:`~unfoldlarpix.deconv_workflow.uniform_within_bin_kernel`).
        Exact at ``subbin = B``.  The mean release instant is then the bin
        CENTRE, which is the convention ``grid_truth(mode="round")`` and
        ``universal_rebin``'s release-point deposit already assume for the
        truth -- so this is the setting that makes the operator agree with
        the adopted truth convention instead of sitting half a bin early.

    The switch is on the SERVICE, not on one algorithm, so every consumer of
    ``prepared()`` -- the operator, the FFT inverse, the warm start, the
    audit's reachability map -- sees the same charge model by construction.
    ``prepared_raw()`` bypasses it and always returns the delta form; it is
    what :class:`~unfoldlarpix.model.subbin_operator.ZSOperatorUniform`
    needs, because that operator does the sub-bin expansion itself.
    """

    def initialize(self) -> None:
        self.response_path = self.props["response"]
        # response_start_tick: offset of the kernel's bin-integration windows
        # [fine ticks]; 0 is the shipped convention (alignment probe only).
        self.start_tick = int(self.props.get("response_start_tick", 0))
        self.within_bin = str(self.props.get("within_bin", "delta"))
        self.subbin = int(self.props.get("subbin", 1))
        if self.within_bin not in ("delta", "uniform"):
            raise ValueError(f"within_bin {self.within_bin!r} "
                             "(want 'delta' or 'uniform')")
        if self.within_bin == "uniform" and self.subbin < 1:
            raise ValueError("subbin must be >= 1")
        self._prepared: dict[int, object] = {}
        self._raw: dict[int, object] = {}

    def prepared_raw(self, adc_hold_delay: int):
        """The shipped delta-at-bin-start kernel, whatever ``within_bin`` says."""
        adc_hold_delay = int(adc_hold_delay)
        if adc_hold_delay not in self._raw:
            from ..deconv_workflow import prepare_field_response
            self._raw[adc_hold_delay] = prepare_field_response(
                self.response_path, adc_hold_delay, normalized=False,
                start_tick=self.start_tick)
        return self._raw[adc_hold_delay]

    def prepared(self, adc_hold_delay: int):
        adc_hold_delay = int(adc_hold_delay)
        if adc_hold_delay not in self._prepared:
            base = self.prepared_raw(adc_hold_delay)
            if self.within_bin == "uniform" and self.subbin > 1:
                import dataclasses

                from ..deconv_workflow import uniform_within_bin_kernel
                k = uniform_within_bin_kernel(
                    base.full_response, adc_hold_delay, self.subbin,
                    start_tick=self.start_tick)
                was = tuple(base.integrated_response.shape)
                base = dataclasses.replace(base, integrated_response=k)
                print(f"[detector] within_bin=uniform subbin={self.subbin} "
                      f"@ B={adc_hold_delay}: kernel {was} -> {tuple(k.shape)}"
                      f", sum {k.sum():.6f}")
            self._prepared[adc_hold_delay] = base
        return self._prepared[adc_hold_delay]


@service("rng")
class RngService(Service):
    """Seeded randomness for reproducibility."""

    def initialize(self) -> None:
        self.seed = int(self.props.get("seed", 0))
        self.numpy = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
