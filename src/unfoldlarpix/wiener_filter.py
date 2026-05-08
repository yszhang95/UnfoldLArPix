"""Wiener-inspired regularization filter for ROI-finding deconvolution.

Implements the parametric form from arxiv 1802.08709 §3.1.1 (Eqs. 3.9-3.10):

    F(f) = exp(-0.5 * (f / f_c) ** b)   for f > 0
    F(f) = 0                              for f = 0

The DC component is suppressed so the time-domain smearing function integrates
to one and constant baselines do not propagate into the deconvolved waveform.
The spatial axes use a Gaussian (paper Eq. 3.13).

Tuning recipe: start with ``b = 2`` and ``omega_c ~ 1 / (3 * adc_hold_delay)``
(in the same units as ``np.fft.rfftfreq(n, d=adc_hold_delay)``); raise
``omega_c`` for sharper time localization, lower it for more noise suppression.
"""

import numpy as np
from numpy import fft


def wiener_inspired_filter_3d(
    s: tuple[int, int, int],
    dt: tuple[float, float, float],
    sigma_pixel: tuple[float, float],
    omega_c: float,
    b: float = 2.0,
) -> np.ndarray:
    """Build the 3D Wiener-inspired filter consumed by ``deconv_fft``.

    Args:
        s: FFT shape ``(nx, ny, nt)`` matching the measurement passed to
            ``deconv_fft`` (i.e. block + response - 1 along spatial axes).
        dt: Sample spacing along ``(x, y, t)``. Use ``(1, 1, adc_hold_delay)``
            to match :func:`gaussian_filter_3d`.
        sigma_pixel: Spatial Gaussian widths ``(sigma_x, sigma_y)`` in pixel
            units (as used by :func:`gaussian_filter_3d`).
        omega_c: Time-axis cutoff frequency in the same units as
            ``np.fft.rfftfreq(s[-1], d=dt[-1])``.
        b: Rolloff exponent. ``b = 2`` recovers a Gaussian time response;
            larger values give a sharper edge (paper uses values near 2).

    Returns:
        Real array of shape ``(s[0], s[1], s[2] // 2 + 1)`` -- compatible with
        the FFT output of ``rfftn`` on a real array of shape ``s``.
    """
    if omega_c <= 0:
        raise ValueError("omega_c must be positive.")
    if b <= 0:
        raise ValueError("b must be positive.")

    freqs_t = fft.rfftfreq(s[-1], d=dt[-1])
    time_filter = np.exp(-0.5 * (freqs_t / omega_c) ** b)
    time_filter[freqs_t == 0] = 0.0

    freqs_x = fft.fftfreq(s[0], d=dt[0])
    freqs_y = fft.fftfreq(s[1], d=dt[1])
    gx = np.exp(-0.5 * freqs_x**2 / sigma_pixel[0] ** 2)
    gy = np.exp(-0.5 * freqs_y**2 / sigma_pixel[1] ** 2)

    return gx[:, None, None] * gy[None, :, None] * time_filter[None, None, :]
