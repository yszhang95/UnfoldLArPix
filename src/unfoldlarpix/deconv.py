"""
deconvolution using FFT
"""

import numpy as np
from numpy import fft

def deconv_fft(measurement: np.ndarray, kernel: np.ndarray,
               filter_fft: np.ndarray | None = None) -> np.ndarray:
    """Deconvolve measurement with kernel using FFT.
    This function is generic for N-d real input using rfftn and irfftn.
    Assume measurement is len(m) = len(s) + len(k) - 1 in time.
    Args:
        measurement: Input measurement array.
        kernel: Deconvolution kernel array.

    Returns:
        Deconvolved signal array.
    """
    if filter_fft is not None:
        raise NotImplementedError("filter_fft is not implemented yet.")
    # Determine the shape for FFT
    shape = np.array(measurement.shape)  # Copy to avoid modifying input
    shape[0] = measurement.shape[0] + (kernel.shape[0] - 1)  # spatial dimension
    shape[1] = measurement.shape[1] + (kernel.shape[1] - 1)  # spatial dimension

    # Compute the FFT of the measurement
    measurement_fft = fft.rfftn(measurement, s=shape)

    # Compute the FFT of the kernel, zero-padded to match measurement shape
    kernel_fft = fft.rfftn(kernel, s=shape)

    # Avoid division by zero by adding a small epsilon where kernel_fft is zero
    epsilon = 1e-10
    kernel_fft = np.where(np.abs(kernel_fft) < epsilon, epsilon, kernel_fft)

    # Perform deconvolution in the frequency domain
    signal_fft = measurement_fft / kernel_fft

    # Compute the inverse FFT to get back to the time/spatial domain
    signal = fft.irfftn(signal_fft, s=shape)

    # Trim the signal to the expected length (len(measurement) - len(kernel) + 1)
    signal = np.roll(signal, (kernel.shape[0] - 1) // 2, axis=0)
    signal = np.roll(signal, (kernel.shape[1] - 1) // 2, axis=1)

    expected_length = list(int(shape[i]) - kernel.shape[i] + 1  for i in range(len(shape)))

    slices = tuple(slice(0, expected_length[i]) for i in range(len(shape)))
    signal = signal[slices]
    loc_offset = (0, 0, 0)

    return signal, loc_offset
