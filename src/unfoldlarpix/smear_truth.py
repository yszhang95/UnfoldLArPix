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
    print("loc_min:", loc_min, "loc_max:", loc_max)
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
