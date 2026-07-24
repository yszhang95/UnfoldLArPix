"""Warm-start providers — decoupled from measurement building and solve.

The solver takes ``q0`` (and a support) as INPUTS; where they come from
is a configuration choice.  Providers here:

- :func:`fft_warm_start` — the legacy compensated-FFT deconvolution,
  with the FFT part on torch (GPU): template compensation builds the
  dense block on CPU (object logic, cheap), the deconvolution runs as
  torch rFFTs.  Single pass — the iterative recompensation was retired:
  it mattered when the pipeline WAS the estimator; for a warm start the
  seed-source study showed support/warm-start choices are not binding.
- cold start: pass ``q0=None`` to the solver (no provider needed).

Truth smearing does NOT belong here (evaluation concern).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..deconv_workflow import hits_to_merged_block
from .conventions import DEPOSIT_MODE, DEPOSIT_PHASE, burst_tau_min


def resolve_burst_tau(readout_config, tau: int | None = None) -> int:
    """Resolve the tunable burst-merge gap ``tau`` against its physical bounds.

    Bounds (both set by the readout, ticks):
    - FLOOR = adc_hold_delay + adc_down_time + one_tick (:func:`burst_tau_min`).
      Below it, immediate re-triggers are misrouted to template compensation
      and lose ~threshold of charge each.
    - CAP = 2 * adc_hold_delay.  Above it, dead-time compensation is applied
      across a gap longer than one burst window plus one dead window — a
      genuine multi-bin silence.  Dead-time merge does not create the
      intermediate gap bins that template compensation does, so the merged
      sequence comes out with a different length and its start displaced by
      ~one bin (measured: len 130->128, start +1*adc_hold_delay for a
      gap-82 pixel).  Gaps beyond the cap are real silences and belong to
      template compensation.

    ``tau=None`` -> FLOOR.  A value in [FLOOR, CAP] is used as given (tune
    upward for broader responses).  Out-of-range values warn and clamp.  If
    FLOOR > CAP (adc_down_time too large relative to adc_hold_delay) the two
    constraints cannot both hold; warn and keep the FLOOR (charge
    conservation over timing).
    """
    tau_min = burst_tau_min(readout_config)
    tau_max = 2 * int(readout_config.adc_hold_delay)
    if tau_min > tau_max:
        warnings.warn(
            f"burst tau floor {tau_min} exceeds cap {tau_max} "
            f"(=2*adc_hold_delay): adc_down_time={readout_config.adc_down_time} "
            f"is too large relative to adc_hold_delay="
            f"{readout_config.adc_hold_delay} for a well-posed burst-merge "
            f"window (no tau conserves charge AND avoids the >2B shift). "
            f"Keeping the floor {tau_min}.", RuntimeWarning, stacklevel=2)
        return tau_min
    if tau is None:
        return tau_min
    tau = int(tau)
    if tau < tau_min:
        warnings.warn(
            f"burst tau={tau} is below the physical floor {tau_min} "
            f"(adc_hold_delay+adc_down_time+one_tick): immediate re-triggers "
            f"would be routed to template compensation and lose ~threshold of "
            f"charge each. Clamping to {tau_min}.",
            RuntimeWarning, stacklevel=2)
        return tau_min
    if tau > tau_max:
        warnings.warn(
            f"burst tau={tau} exceeds the cap {tau_max} (=2*adc_hold_delay): "
            f"gaps >2*adc_hold_delay are genuine silences; dead-time merge "
            f"across them changes the merged-sequence length and shifts the "
            f"following sequence by ~one bin. Clamping to {tau_max}.",
            RuntimeWarning, stacklevel=2)
        return tau_max
    return tau


def gaussian_filter_3d_torch(shape, dt, sigma, device, dtype) -> torch.Tensor:
    """Frequency-domain 3D Gaussian (rfft along the last axis)."""
    freqs = torch.fft.rfftfreq(shape[-1], d=dt[-1], device=device,
                               dtype=torch.float64)
    g = torch.exp(-0.5 * freqs**2 / sigma[-1] ** 2)
    for i in range(len(shape) - 1):
        fi = torch.fft.fftfreq(shape[i], d=dt[i], device=device,
                               dtype=torch.float64)
        gi = torch.exp(-0.5 * fi**2 / sigma[i] ** 2)
        g = gi[None, :] * g[..., None]
    g = torch.movedim(g, 0, -1)
    return g.to(dtype=torch.complex64 if dtype == torch.float32
                else torch.complex128)


def deconv_fft_torch(measurement: torch.Tensor, kernel: torch.Tensor,
                     filter_fft: torch.Tensor | None = None) -> torch.Tensor:
    """Torch port of ``deconv.deconv_fft`` (same shapes, rolls, trims)."""
    shape = list(measurement.shape)
    shape[0] += kernel.shape[0] - 1
    shape[1] += kernel.shape[1] - 1
    m_fft = torch.fft.rfftn(measurement, s=shape)
    k_fft = torch.fft.rfftn(kernel, s=shape)
    eps = 1e-10
    k_fft = torch.where(torch.abs(k_fft) < eps,
                        torch.full_like(k_fft, eps), k_fft)
    s_fft = m_fft / k_fft
    if filter_fft is not None:
        s_fft = s_fft * filter_fft
    sig = torch.fft.irfftn(s_fft, s=shape)
    sig = torch.roll(sig, (kernel.shape[0] - 1) // 2, dims=0)
    sig = torch.roll(sig, (kernel.shape[1] - 1) // 2, dims=1)
    expected = [shape[i] - kernel.shape[i] + 1 for i in range(3)]
    return sig[: expected[0], : expected[1], : expected[2]]


@dataclass(frozen=True)
class WarmStartResult:
    deconv_q: np.ndarray          # smoothed linear estimate on the block grid
    block: np.ndarray             # compensated dense block (hwf)
    block_offset: np.ndarray      # (3,) raw block lower corner


def fft_warm_start(
    hits,
    readout_config,
    prepared_response,
    *,
    sigma_time: float,
    sigma_pixel: float,
    pad_pixels: int = 0,
    npadbin: int = 50,
    tau: int | None = None,
    processor_cls=None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> WarmStartResult:
    """Compensated block (CPU) + Gaussian-regularized FFT deconv (torch).

    ``tau`` is the burst-merge gap [ticks]; ``None`` uses the physical floor
    ``adc_hold_delay + adc_down_time + one_tick`` (see
    :func:`resolve_burst_tau`).  Raise it above the floor (up to
    ``2*adc_hold_delay``) to merge more-separated re-triggers for a broader
    field response.
    """
    from ..burst_processor_v3 import BurstSequenceProcessorV3

    processor_cls = processor_cls or BurstSequenceProcessorV3
    response_indu = (prepared_response.response_indu
                     if processor_cls is BurstSequenceProcessorV3 else None)
    tau = resolve_burst_tau(readout_config, tau)
    block_offset, block_data, _comp, _anchors = hits_to_merged_block(
        hits, readout_config, prepared_response.selected_response,
        processor_cls=processor_cls,
        tau=tau,
        template_search_mode=prepared_response.template_search_mode,
        npadbin=npadbin, response_indu=response_indu,
        deposit_mode=DEPOSIT_MODE, deposit_phase=DEPOSIT_PHASE,
        pad_pixels=pad_pixels,
    )
    dev = torch.device(device)
    block_t = torch.as_tensor(block_data, dtype=dtype, device=dev)
    kernel_t = torch.as_tensor(prepared_response.integrated_response,
                               dtype=dtype, device=dev)
    # same shape convention as build_gaussian_deconv_kernel: spatial axes
    # padded to the linear-deconvolution size, time axis unpadded
    filt_shape = (
        block_data.shape[0] + kernel_t.shape[0] - 1,
        block_data.shape[1] + kernel_t.shape[1] - 1,
        block_data.shape[2],
    )
    filt = gaussian_filter_3d_torch(
        filt_shape,
        dt=(1, 1, readout_config.adc_hold_delay),
        sigma=(sigma_pixel, sigma_pixel, sigma_time),
        device=dev, dtype=dtype)
    q = deconv_fft_torch(block_t, kernel_t, filt)
    return WarmStartResult(
        deconv_q=q.cpu().numpy().astype(np.float64),
        block=block_data,
        block_offset=np.asarray(block_offset),
    )
