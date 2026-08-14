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
from .conventions import (DEPOSIT_MODE, DEPOSIT_PHASE, burst_tau_min,
                          resolve_burst_tau)

__all__ = ["resolve_burst_tau", "burst_tau_min", "fft_warm_start",
           "gaussian_filter_3d_torch"]


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
    align_origin: bool = False,
    align_phase: float = 0.0,
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
        pad_pixels=pad_pixels, align_origin=align_origin,
        align_phase=align_phase,
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
