"""Region-of-interest identification on a Wiener-deconvolved block.

The ROI is found on a Wiener-inspired deconvolution (sharp time localization)
and then applied as a mask to the Gaussian-deconvolved charge map used for the
final output. See ``wiener_filter.py`` for the filter form.
"""

from __future__ import annotations

import numpy as np


def estimate_quiet_pixel_noise(
    deconv_q_wiener: np.ndarray,
    block_offset: np.ndarray,
    hit_pixel_xy: np.ndarray,
    *,
    min_quiet_pixels: int = 8,
) -> float:
    """Estimate noise RMS from pixels with no hits in this event.

    Args:
        deconv_q_wiener: 3D block ``(nx, ny, nt)`` from the Wiener deconvolution.
        block_offset: Block origin ``(x0, y0, t0)`` in global indices (only the
            spatial components are used).
        hit_pixel_xy: ``(N, 2)`` array of global ``(x, y)`` indices for pixels
            with hits in this event.
        min_quiet_pixels: Minimum number of quiet pixels required to estimate
            noise; below this, raises ``ValueError`` so the caller can supply a
            fallback rather than silently producing a wrong threshold.

    Returns:
        Standard deviation across all bins of the quiet-pixel slab.
    """
    if deconv_q_wiener.ndim != 3:
        raise ValueError(
            f"deconv_q_wiener must be 3D, got shape {deconv_q_wiener.shape}."
        )
    nx, ny, _ = deconv_q_wiener.shape
    x0, y0 = int(block_offset[0]), int(block_offset[1])

    hit_local = np.asarray(hit_pixel_xy, dtype=int) - np.array([x0, y0], dtype=int)
    in_block = (
        (hit_local[:, 0] >= 0)
        & (hit_local[:, 0] < nx)
        & (hit_local[:, 1] >= 0)
        & (hit_local[:, 1] < ny)
    )
    hit_local = hit_local[in_block]

    busy_mask = np.zeros((nx, ny), dtype=bool)
    busy_mask[hit_local[:, 0], hit_local[:, 1]] = True
    quiet_mask = ~busy_mask

    n_quiet = int(quiet_mask.sum())
    if n_quiet < min_quiet_pixels:
        raise ValueError(
            f"Only {n_quiet} quiet pixels available "
            f"(< min_quiet_pixels={min_quiet_pixels}); cannot estimate noise."
        )

    quiet_slab = deconv_q_wiener[quiet_mask, :]
    return float(np.std(quiet_slab))


def find_roi_mask(
    deconv_q_wiener: np.ndarray,
    noise_rms: float,
    *,
    threshold_sigma: float = 5.0,
    merge_gap: int = 2,
    expand: int = 2,
) -> np.ndarray:
    """Build a per-pixel ROI mask along the time axis.

    For each pixel, samples above ``threshold_sigma * noise_rms`` are marked as
    ROI; runs separated by ``<= merge_gap`` zero bins are merged; each merged
    run is expanded by ``expand`` bins on each side.

    Args:
        deconv_q_wiener: 3D block ``(nx, ny, nt)``.
        noise_rms: Noise RMS as estimated by :func:`estimate_quiet_pixel_noise`.
        threshold_sigma: Threshold in units of ``noise_rms`` (paper §3.2.3
            uses 5x for collection planes).
        merge_gap: Maximum number of below-threshold time bins separating two
            ROI segments that should be merged.
        expand: Number of bins to add on each side of each merged ROI.

    Returns:
        Boolean array of the same shape as ``deconv_q_wiener``.
    """
    if noise_rms <= 0:
        raise ValueError(f"noise_rms must be positive, got {noise_rms}.")
    if merge_gap < 0 or expand < 0:
        raise ValueError("merge_gap and expand must be non-negative.")

    threshold = threshold_sigma * noise_rms
    above = deconv_q_wiener > threshold

    if merge_gap > 0:
        mask = _close_gaps_along_last_axis(above, merge_gap)
    else:
        mask = above.copy()

    if expand > 0:
        mask = _expand_along_last_axis(mask, expand)

    return mask


def apply_roi_mask(deconv_q: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Zero samples of ``deconv_q`` outside the ROI mask."""
    if deconv_q.shape != roi_mask.shape:
        raise ValueError(
            f"Shape mismatch: deconv_q {deconv_q.shape} vs roi_mask {roi_mask.shape}"
        )
    return np.where(roi_mask, deconv_q, 0.0)


def _close_gaps_along_last_axis(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill False runs of length <= ``gap`` between True runs along the last axis.

    Implemented as a binary closing along the last axis with a structuring
    element of length ``2 * radius + 1`` where ``radius = (gap + 1) // 2``.
    A radius of ``r`` closes False runs of length up to ``2 * r``; thus the
    chosen ``radius`` closes runs up to ``gap`` for any ``gap >= 1``.
    """
    if gap <= 0:
        return mask.copy()
    radius = (gap + 1) // 2
    dilated = _dilate_along_last_axis(mask, radius)
    return ~_dilate_along_last_axis(~dilated, radius)


def _dilate_along_last_axis(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation along the last axis by ``radius`` bins on each side."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[..., shift:] |= mask[..., :-shift]
        out[..., :-shift] |= mask[..., shift:]
    return out


def _expand_along_last_axis(mask: np.ndarray, expand: int) -> np.ndarray:
    return _dilate_along_last_axis(mask, expand)
