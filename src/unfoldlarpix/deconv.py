"""
deconvolution using FFT
"""

import numpy as np
from numpy import fft

def gaussian_filter(n, dt, sigma):
    """Generate a Gaussian filter in the frequency domain.
    Args:
        n: Length of the filter.
        dt: Time step size.
        sigma: Standard deviation of the Gaussian in time domain.
    Returns:
        Gaussian filter in the frequency domain.
    """
    freqs = fft.rfftfreq(n, d=dt)
    gaussian = np.exp(-0.5 * freqs**2/sigma**2)
    return gaussian

def gaussian_filter_3d(s, dt, sigma):
    freqs = fft.rfftfreq(s[-1], d=dt[-1])
    gaussian = np.exp(-0.5 * freqs**2/sigma[-1]**2)
    for i in range(len(s[:-1])):
        freqs_i = fft.fftfreq(s[i], d=dt[i])
        gaussian_i = np.exp(-0.5 * freqs_i**2/sigma[i]**2)
        gaussian = gaussian_i[None, :] * gaussian[..., None]
    gaussian = np.moveaxis(gaussian, 0, -1)
    return gaussian

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
    # if filter_fft is not None:
    #     raise NotImplementedError("filter_fft is not implemented yet.")
    # if isinstance(filter_fft, np.ndarray):
    #     if len(filter_fft.shape) != 1 or filter_fft.shape[-1] != measurement.shape[-1]//2+1:
    #         print('filter_fft shape:', filter_fft.shape, 'measurement shape:', measurement.shape)
    #         raise ValueError(f"filter_fft shape is assumed to be 1D in time, got {filter_fft.shape}")
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

    print(measurement_fft.shape, kernel_fft.shape, filter_fft.shape if filter_fft is not None else None)

    # Perform deconvolution in the frequency domain
    signal_fft = measurement_fft / kernel_fft
    if filter_fft is not None:
        signal_fft *= filter_fft

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
