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
    """

    def initialize(self) -> None:
        self.response_path = self.props["response"]
        # response_start_tick: offset of the kernel's bin-integration windows
        # [fine ticks]; 0 is the shipped convention (alignment probe only).
        self.start_tick = int(self.props.get("response_start_tick", 0))
        self._prepared: dict[int, object] = {}

    def prepared(self, adc_hold_delay: int):
        if adc_hold_delay not in self._prepared:
            from ..deconv_workflow import prepare_field_response
            self._prepared[adc_hold_delay] = prepare_field_response(
                self.response_path, adc_hold_delay, normalized=False,
                start_tick=self.start_tick)
        return self._prepared[adc_hold_delay]


@service("rng")
class RngService(Service):
    """Seeded randomness for reproducibility."""

    def initialize(self) -> None:
        self.seed = int(self.props.get("seed", 0))
        self.numpy = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
