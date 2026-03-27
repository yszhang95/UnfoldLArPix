"""UnfoldLArPix - Signal processing package for LArPix data unfolding."""

__version__ = "0.1.0"
__author__ = "Yousen Zhang"

from .data_containers import (
    Current,
    EffectiveCharge,
    EventData,
    Geometry,
    Hits,
    ReadoutConfig,
)
from .data_loader import DataLoader
from .field_response import FieldResponseProcessor
from .burst_processor import (
    BurstSequence,
    BurstSequenceProcessor,
    MergedSequence,
)
from .burst_processor_v2 import BurstSequenceProcessorV2
from .deconv_workflow import (
    EventDeconvolutionResult,
    PreparedFieldResponse,
    build_event_output_payload,
    create_burst_processor,
    integrate_kernel_over_time,
    prepare_field_response,
    process_event_deconvolution,
    shift_time_offset,
)

__all__ = [
    "DataLoader",
    "EffectiveCharge",
    "Current",
    "Hits",
    "EventData",
    "Geometry",
    "ReadoutConfig",
    "FieldResponseProcessor",
    "BurstSequence",
    "BurstSequenceProcessor",
    "BurstSequenceProcessorV2",
    "MergedSequence",
    "PreparedFieldResponse",
    "EventDeconvolutionResult",
    "integrate_kernel_over_time",
    "prepare_field_response",
    "create_burst_processor",
    "process_event_deconvolution",
    "build_event_output_payload",
    "shift_time_offset",
]
