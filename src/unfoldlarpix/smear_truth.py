#!/usr/bin/env python

import numpy as np

def gaus_smear_true(ticks: np.ndarray, true_charge: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Smear true charge with kernel to get smeared charge."""
    if len(ticks.shape) != 2:
        raise ValueError("ticks should be 3D array")
    # get a minimum shape of charge block
    loc_min = [np.min(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_max = [np.max(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_min = np.array(loc_min)
    loc_max = np.array(loc_max)
    shape = [loc_max[i] - loc_min[i] + 1 for i in range(ticks.shape[1])]
    data = np.zeros(shape, dtype=true_charge.dtype)
    # fill data with true charge
    for i in range(ticks.shape[0]):
        data[tuple(ticks[i] - loc_min)] += true_charge[i, -1]
    n = data.shape[-1]
    n_single_side = int((8*1/2/np.pi/width) // n + 1)
    ktimes = n_single_side * 2 + 1
    m = int(ktimes * n)
    smeared = np.zeros((m,))
    oshape = list(data.shape)
    oshape[-1] = m
    smeared = np.fft.ifftn(np.fft.fftn(data, s=oshape) *
                           np.exp(-np.fft.fftfreq(n=m, d=1)**2/width**2/2)[None, None, :], s=oshape).real
    smeared = np.roll(smeared, n_single_side*n)
    offset = loc_min.copy()
    offset[-1] = offset[-1] - n_single_side*n
    return offset, smeared


def gaus_smear_true_3d(ticks: np.ndarray, true_charge: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smear true charge with kernel to get smeared charge.

    The convolution is done by FFT, which is CIRCULAR, so every axis has to be
    padded past the kernel's reach or charge leaving one edge re-enters at the
    other.  The time axis was always padded (``n_single_side`` below).  The
    PIXEL axes were not, and the array was opened at exactly the charge's
    extent -- so a track narrow in pixels wrapped into itself.

    Measured on ``mu_a50`` (effq spans three pixel columns, 138-140,
    sigma_pixel = 0.5 -> 0.318 px real space, kernel taps to +-4 px):
    \SI{83.5}{\ke} that belongs in column 141 appeared in column 138 instead,
    inflating it from 87.0 to \SI{167.8}{\ke} (+93%) and leaving column 141
    empty.  Charge is conserved -- circular convolution conserves the sum --
    so integral-based metrics could not see it, while ``true_killed`` (column
    138's truth twice too large) and ``ghost`` (column 141's reco with no
    truth to support it) were both wrong by tens of ke.

    Padding is ``ceil(8 * sigma_realspace)`` cells per side, floored at 8.  The
    kernel is an aliased Gaussian whose taps decay only as ~1/d^2, so no pad is
    exact; measured max deviation per unit charge is 2.5e-3 at pad 4,
    7.7e-4 at 8, 2.1e-4 at 16.  Eight is where the residual falls below the
    other errors in the protocol at a 1.4x memory cost.
    """
    if len(ticks.shape) != 2:
        raise ValueError("ticks should be 3D array")
    # get a minimum shape of charge block
    loc_min = [np.min(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_max = [np.max(ticks[:, i]) for i in range(ticks.shape[1])]
    loc_min = np.array(loc_min)
    loc_max = np.array(loc_max)
    shape = [loc_max[i] - loc_min[i] + 1 for i in range(ticks.shape[1])]
    # pad every axis but the last (the last is padded by n_single_side below)
    pad = [max(int(np.ceil(8.0 / (2.0 * np.pi * width[i]))), 8)
           for i in range(ticks.shape[1] - 1)] + [0]
    shape = [shape[i] + 2 * pad[i] for i in range(ticks.shape[1])]
    loc_min = loc_min - np.array(pad, dtype=loc_min.dtype)
    data = np.zeros(shape, dtype=true_charge.dtype)
    # fill data with true charge
    for i in range(ticks.shape[0]):
        data[tuple(ticks[i] - loc_min)] += true_charge[i, -1]
    n = data.shape[-1]
    n_single_side = int((8*1/2/np.pi/width[-1]) // n + 1)
    ktimes = n_single_side * 2 + 1
    m = int(ktimes * n)
    smeared = np.zeros((m,))
    oshape = list(data.shape)
    oshape[-1] = m

    freqs = np.fft.rfftfreq(oshape[-1], d=1)
    gaussian = np.exp(-0.5 * freqs**2/width[-1]**2)
    for i in range(len(oshape[:-1])):
        freqs_i = np.fft.fftfreq(oshape[i], d=1)
        gaussian_i = np.exp(-0.5 * freqs_i**2/width[i]**2)
        gaussian = gaussian_i[None, :] * gaussian[..., None]
    gaussian = np.moveaxis(gaussian, 0, -1)

    smeared = np.fft.irfftn(np.fft.rfftn(data, s=oshape) *
                           gaussian, s=oshape)
    smeared = np.roll(smeared, n_single_side*n, axis=-1)
    offset = loc_min.copy()
    offset[-1] = offset[-1] - n_single_side*n
    return offset, smeared
