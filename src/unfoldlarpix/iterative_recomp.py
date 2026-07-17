"""Iterative self-consistent template recompensation.

Zero-suppressed readout only measures burst integrals at known latch times;
the pre-trigger gaps are filled with an *assumed* field-response template
before deconvolution.  This module replaces those assumed segments with a
self-consistent prediction:

1. deconvolve the compensated block (first pass),
2. forward-model the deconvolved charge through the same integrated
   response to predict what the dense measurement should look like,
3. overwrite ONLY the template-injected (unmeasured) bins with the
   prediction — rescaled per contiguous segment so each gap keeps the
   integral assigned by the compensation (charge conservation with the
   recorded cumulative is preserved exactly),
4. re-deconvolve.  Optionally iterate.

The recorded samples are never modified: they are data.  The template
segments are model; the forward prediction replaces the fixed template
shape with an event-adaptive one that automatically carries diffusion,
track angle, and neighbor-coupling information through the 2D response.

IMPORTANT — the loop needs a nonlinearity to gain information.  For a
purely linear deconvolution the forward model reproduces the input block
identically (``pred_fft = (M/K)·F·K = M·F``), template errors included,
and the refinement is a fixed point of any block.  The physics prior
that breaks the identity is POSITIVITY of the ionization charge: the
deconvolved charge is clipped at zero before forward modeling, so
unphysical ringing (the signature of a wrongly-shaped template segment)
is removed and the prediction disagrees with the block exactly where the
template shape is inconsistent with the recorded samples around it.
"""

from __future__ import annotations

import numpy as np
from numpy import fft

from .deconv import deconv_fft


def forward_model_block(
    deconv_q: np.ndarray,
    kernel: np.ndarray,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    """Convolve a deconvolved-charge block back into measurement space.

    Exact adjoint of the alignment conventions in :func:`deconv_fft`:
    spatial axes are center-aligned (kernel center = zero displacement,
    inverted by rolling back ``(k-1)//2``), the time axis is causal.

    Args:
        deconv_q: Deconvolved charge, shape ``(nx, ny, nt - kt + 1)`` as
            returned by ``deconv_fft`` for a measurement of shape
            ``(nx, ny, nt)``.
        kernel: Integrated response, shape ``(kx, ky, kt)`` with odd
            ``kx``/``ky``.
        block_shape: Shape ``(nx, ny, nt)`` of the measurement block to
            predict.

    Returns:
        Predicted measurement block of shape ``block_shape``.
    """
    fft_shape = (
        block_shape[0] + kernel.shape[0] - 1,
        block_shape[1] + kernel.shape[1] - 1,
        block_shape[2],
    )
    axes = (0, 1, 2)
    pred = fft.irfftn(
        fft.rfftn(deconv_q, s=fft_shape, axes=axes)
        * fft.rfftn(kernel, s=fft_shape, axes=axes),
        s=fft_shape,
        axes=axes,
    )
    pred = np.roll(pred, -((kernel.shape[0] - 1) // 2), axis=0)
    pred = np.roll(pred, -((kernel.shape[1] - 1) // 2), axis=1)
    return pred[: block_shape[0], : block_shape[1], : block_shape[2]]


def measured_bin_mask(
    hits_location: np.ndarray,
    nburst: int,
    adc_hold_delay: int,
    block_offset: np.ndarray,
    block_shape: tuple[int, int, int],
    deposit_mode: str = "floor",
    deposit_phase: float = 0.0,
) -> np.ndarray:
    """Mark block bins that contain recorded (measured) burst charge.

    Every latch time of every burst sequence marks the bin(s) its charge
    was deposited into — one bin in ``floor`` mode, the two split bins in
    ``linear`` mode.  All other bins are model-filled (template segments,
    bootstrap segments, or empty padding).

    Args:
        hits_location: ``(n, >=3)`` array; columns are pixel_x, pixel_y,
            trigger_time_idx (fine ticks).
        nburst: Number of burst charges per sequence.
        adc_hold_delay: Burst window length in fine ticks.
        block_offset: ``(3,)`` block lower corner (pixel_x, pixel_y, time).
        block_shape: Dense block shape ``(nx, ny, nt)``.
        deposit_mode / deposit_phase: Must match the values used by
            ``merged_sequences_to_block``.
    """
    mask = np.zeros(block_shape, dtype=bool)
    bin_size = float(adc_hold_delay)
    for row in np.asarray(hits_location):
        px = int(row[0] - block_offset[0])
        py = int(row[1] - block_offset[1])
        if not (0 <= px < block_shape[0] and 0 <= py < block_shape[1]):
            continue
        trigger = float(row[2])
        latches = trigger + bin_size * np.arange(1, nburst + 1)
        fpos = (latches - float(block_offset[2])) / bin_size
        if deposit_mode == "linear":
            fpos = fpos + deposit_phase
            i0 = np.floor(fpos).astype(int)
            inds = np.concatenate([i0, i0 + 1])
        else:
            inds = np.floor(fpos).astype(int)
        inds = inds[(inds >= 0) & (inds < block_shape[2])]
        mask[px, py, inds] = True
    return mask


def refine_unmeasured_segments(
    block: np.ndarray,
    pred: np.ndarray,
    measured_mask: np.ndarray,
    *,
    min_segment_charge: float = 1e-9,
    clip_negative: bool = True,
) -> np.ndarray:
    """Replace model-filled segments of ``block`` with the forward prediction.

    Works per pixel on contiguous runs of unmeasured bins that carry
    nonzero compensated charge (i.e. injected template/bootstrap
    segments).  Each run is replaced by the (optionally non-negative part
    of the) prediction rescaled to the run's original integral, so the
    total compensated charge — anchored to the recorded cumulative — is
    conserved exactly.  Runs with no prediction support are left alone.
    """
    out = block.copy()
    nx, ny, nt = block.shape
    for x in range(nx):
        for y in range(ny):
            trace = block[x, y]
            meas = measured_mask[x, y]
            model_bins = (~meas) & (trace != 0.0)
            if not model_bins.any():
                continue
            # contiguous runs of model-filled bins
            padded = np.concatenate([[False], model_bins, [False]])
            edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
            for start, stop in zip(edges[::2], edges[1::2]):
                seg_sum = float(trace[start:stop].sum())
                if abs(seg_sum) < min_segment_charge:
                    continue
                p = pred[x, y, start:stop]
                if clip_negative:
                    p = np.clip(p, 0.0, None)
                p_sum = float(p.sum())
                if p_sum <= min_segment_charge:
                    continue
                out[x, y, start:stop] = p * (seg_sum / p_sum)
    return out


def iterative_recompensation(
    block: np.ndarray,
    kernel: np.ndarray,
    filter_fft: np.ndarray | None,
    measured_mask: np.ndarray,
    n_iter: int = 1,
    positivity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the deconvolve → clip → forward-model → refine loop.

    ``positivity=True`` (default) clips the deconvolved charge at zero
    before forward modeling — the nonlinearity that makes the iteration
    informative (see module docstring).  Each refinement always starts
    from the ORIGINAL block, so measured bins are never touched and the
    model bins are re-derived from the newest prediction each pass.

    Returns ``(deconv_q, refined_block)`` after ``n_iter`` refinement
    passes (``n_iter = 0`` reduces to a single plain deconvolution).
    """
    current = block
    deconv_q, _ = deconv_fft(current, kernel, filter_fft)
    for _ in range(int(n_iter)):
        source = np.clip(deconv_q, 0.0, None) if positivity else deconv_q
        pred = forward_model_block(source, kernel, current.shape)
        current = refine_unmeasured_segments(
            block, pred, measured_mask, clip_negative=positivity
        )
        deconv_q, _ = deconv_fft(current, kernel, filter_fft)
    return deconv_q, current
