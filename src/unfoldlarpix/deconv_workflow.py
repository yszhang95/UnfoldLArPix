"""Shared deconvolution workflow helpers used by the example scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .burst_processor import BurstSequenceProcessor, merged_sequences_to_block
from .burst_processor_v2 import BurstSequenceProcessorV2
from .data_containers import EventData, Geometry, Hits, ReadoutConfig
from .deconv import deconv_fft, gaussian_filter_3d
from .field_response import FieldResponseProcessor
from .smear_truth import gaus_smear_true_3d


@dataclass(frozen=True)
class PreparedFieldResponse:
    """Field response products shared across event processing."""

    processor: FieldResponseProcessor
    full_response: np.ndarray
    integrated_response: np.ndarray
    center_response: np.ndarray
    metadata: dict[str, Any]

    @property
    def drift_length(self) -> float | None:
        """Return the field-response drift length when available."""
        if "drift_length" not in self.metadata:
            return None
        return float(np.squeeze(self.metadata["drift_length"]))


@dataclass(frozen=True)
class EventDeconvolutionResult:
    """Outputs produced by a single event deconvolution pass."""

    compensated_charge: float
    hwf_block: np.ndarray
    hwf_block_offset: np.ndarray
    deconv_q: np.ndarray
    local_offset: tuple[int, ...]
    smeared_true: np.ndarray
    smear_offset: np.ndarray


BurstProcessorClass = type[BurstSequenceProcessor] | type[BurstSequenceProcessorV2]


def integrate_kernel_over_time(kernel: np.ndarray, ticks_per_bin: int) -> np.ndarray:
    """Integrate a time-domain kernel into coarser bins."""
    if ticks_per_bin <= 0:
        raise ValueError("ticks_per_bin must be positive.")

    kernel = np.asarray(kernel)
    n_ticks = kernel.shape[-1] // ticks_per_bin * ticks_per_bin
    if np.any(np.abs(kernel[..., n_ticks:]) > 1e-6):
        raise ValueError(
            "Kernel tail must be zero-padded before time integration."
        )

    reshaped = kernel[..., :n_ticks].reshape(
        *kernel.shape[:-1], n_ticks // ticks_per_bin, ticks_per_bin
    )
    return reshaped.sum(axis=-1)


def prepare_field_response(
    field_response_path: str | Path,
    adc_hold_delay: int,
    *,
    normalized: bool = False,
) -> PreparedFieldResponse:
    """Load, process, and integrate the field response for deconvolution."""
    processor = FieldResponseProcessor(field_response_path, normalized=normalized)
    full_response = processor.process_response()
    center_response = full_response[
        full_response.shape[0] // 2, full_response.shape[1] // 2, :
    ].copy()
    integrated_response = integrate_kernel_over_time(full_response, adc_hold_delay)
    return PreparedFieldResponse(
        processor=processor,
        full_response=full_response,
        integrated_response=integrated_response,
        center_response=center_response,
        metadata=processor.get_metadata(),
    )


def create_burst_processor(
    readout_config: ReadoutConfig,
    center_response: np.ndarray,
    *,
    processor_cls: BurstProcessorClass = BurstSequenceProcessor,
    tau: float | None = None,
):
    """Create a burst processor configured from the readout and field response."""
    tau_value = readout_config.adc_hold_delay if tau is None else tau
    return processor_cls(
        readout_config.adc_hold_delay,
        tau=tau_value,
        deadtime=readout_config.csa_reset_time,
        template=np.cumsum(center_response),
        threshold=readout_config.threshold,
    )


def hits_to_merged_block(
    hits: Hits,
    readout_config: ReadoutConfig,
    center_response: np.ndarray,
    *,
    processor_cls: BurstProcessorClass = BurstSequenceProcessor,
    tau: float | None = None,
    npadbin: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert hit bursts into a dense 3D block for deconvolution."""
    burst_processor = create_burst_processor(
        readout_config,
        center_response,
        processor_cls=processor_cls,
        tau=tau,
    )
    merged_sequences = burst_processor.process_hits(hits)
    compensated_charge = float(
        sum(np.sum(sequence.charges) for sequence in merged_sequences.values())
    )
    block_offset, block_data = merged_sequences_to_block(
        merged_sequences,
        readout_config.adc_hold_delay,
        npadbin=npadbin,
    )
    return block_offset, block_data, compensated_charge


def build_gaussian_deconv_kernel(
    block_shape: tuple[int, int, int],
    response_shape: tuple[int, int, int],
    adc_hold_delay: int,
    sigma_time: float,
    sigma_pixel: float,
) -> np.ndarray:
    """Build the 3D Gaussian filter used after FFT deconvolution."""
    return gaussian_filter_3d(
        (
            block_shape[0] + response_shape[0] - 1,
            block_shape[1] + response_shape[1] - 1,
            block_shape[2],
        ),
        dt=(1, 1, adc_hold_delay),
        sigma=(sigma_pixel, sigma_pixel, sigma_time),
    )


def smear_effective_charge(
    event: EventData,
    *,
    sigma_time: float,
    sigma_pixel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same 3D Gaussian smearing to effective charge truth."""
    if event.effq is None:
        raise ValueError("Event does not contain effective charge data.")
    return gaus_smear_true_3d(
        event.effq.location,
        event.effq.data,
        width=np.array([sigma_pixel, sigma_pixel, sigma_time], dtype=float),
    )


def process_event_deconvolution(
    event: EventData,
    readout_config: ReadoutConfig,
    prepared_response: PreparedFieldResponse,
    *,
    sigma_time: float,
    sigma_pixel: float,
    processor_cls: BurstProcessorClass = BurstSequenceProcessor,
    tau: float | None = None,
    npadbin: int = 50,
    require_zero_local_offset: bool = False,
) -> EventDeconvolutionResult:
    """Run the common hit-block deconvolution workflow for one event."""
    if event.hits is None:
        raise ValueError("Event does not contain hit data.")

    block_offset, block_data, compensated_charge = hits_to_merged_block(
        event.hits,
        readout_config,
        prepared_response.center_response,
        processor_cls=processor_cls,
        tau=tau,
        npadbin=npadbin,
    )
    gaussian_kernel = build_gaussian_deconv_kernel(
        tuple(block_data.shape),
        tuple(prepared_response.integrated_response.shape),
        readout_config.adc_hold_delay,
        sigma_time,
        sigma_pixel,
    )
    deconv_q, local_offset = deconv_fft(
        block_data,
        prepared_response.integrated_response,
        gaussian_kernel,
    )
    local_offset = tuple(int(offset) for offset in local_offset)
    if require_zero_local_offset and any(offset != 0 for offset in local_offset):
        raise ValueError(f"Expected zero local offset, got {local_offset}")

    smear_offset, smeared_true = smear_effective_charge(
        event,
        sigma_time=sigma_time,
        sigma_pixel=sigma_pixel,
    )
    return EventDeconvolutionResult(
        compensated_charge=compensated_charge,
        hwf_block=block_data,
        hwf_block_offset=np.array(block_offset, copy=True),
        deconv_q=deconv_q,
        local_offset=local_offset,
        smeared_true=smeared_true,
        smear_offset=np.array(smear_offset, copy=True),
    )


def shift_time_offset(offset: np.ndarray, delta_t: int | float) -> np.ndarray:
    """Return a copy of an offset array with its time component shifted."""
    shifted = np.array(offset, copy=True)
    shifted[-1] += delta_t
    return shifted


def build_event_output_payload(
    event: EventData,
    geometry: Geometry,
    readout_config: ReadoutConfig,
    result: EventDeconvolutionResult,
    *,
    drift_length: float | None,
    boffset_time_shift: int | float = 0,
    include_hwf_block: bool = False,
) -> dict[str, Any]:
    """Assemble a standard `.npz` payload for deconvolution outputs."""
    if event.hits is None or event.effq is None:
        raise ValueError("Event output requires both hit and effective charge data.")

    block_offset = shift_time_offset(result.hwf_block_offset, boffset_time_shift)
    payload: dict[str, Any] = {
        "deconv_q": result.deconv_q,
        "boffset": block_offset,
        "smeared_true": result.smeared_true,
        "smear_offset": result.smear_offset,
        "effq_location": event.effq.location,
        "effq_data": event.effq.data,
        "hits_location": event.hits.location,
        "hits_data": event.hits.data,
        "adc_hold_delay": readout_config.adc_hold_delay,
        "csa_reset_time": readout_config.csa_reset_time,
        "adc_downsample_factor": readout_config.adc_hold_delay,
        "anode_position": geometry.anode_position,
        "drift_direction": geometry.drift_direction,
        "global_tref": event.global_tref,
        "tpc_lower": geometry.lower,
        "drtoa": drift_length,
    }
    if include_hwf_block:
        payload["hwf_block"] = result.hwf_block
        payload["hwf_block_offset"] = block_offset
    return payload
