#!/usr/bin/env python3
"""
2D deconvolution toy demo with asymmetric kernel and multiple scenarios.

Creates a simple delta signal, convolves with an asymmetric 2D kernel
(Gaussian in space, rise-then-decay in time), and demonstrates deconvolution
with 2D Gaussian regularization filter under 4 scenarios:
1. Ideal (no noise, no misalignment)
2. Noisy (Gaussian noise in time domain)
3. Misaligned (spatial shift in x dimension)
4. Noisy + Misaligned
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


# ============================================================================
# Kernel Creation
# ============================================================================

def create_spatial_gaussian(ksize, sigma):
    """Create 1D Gaussian kernel.

    Args:
        ksize: kernel size (odd)
        sigma: standard deviation

    Returns:
        1D array of shape (ksize,)
    """
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-0.5 * x**2 / sigma**2)
    return kernel / kernel.sum()


def create_temporal_asymmetric(ksize, rise_idx, rise_sigma, decay_tau):
    """Create asymmetric temporal kernel with rise then decay.

    Args:
        ksize: kernel size
        rise_idx: index where rise peaks (e.g., 5)
        rise_sigma: width of rise (Gaussian)
        decay_tau: decay time constant (exponential)

    Returns:
        1D array of shape (ksize,)
    """
    t = np.arange(ksize)

    # Gaussian rise
    rise = np.exp(-0.5 * (t - rise_idx)**2 / rise_sigma**2)
    rise[t < rise_idx] = rise[t < rise_idx]  # keep only rising side for asymmetry

    # Exponential decay after peak
    decay = np.zeros(ksize)
    decay_start = rise_idx
    decay[decay_start:] = np.exp(-(t[decay_start:] - decay_start) / decay_tau)

    # Combine: use rise where it's larger, then decay
    kernel = np.where(t <= rise_idx, rise, decay)

    return kernel / kernel.sum()


def create_2d_kernel(kx_size=11, kt_size=20, spatial_sigma=1.5,
                     temporal_rise_idx=5, temporal_rise_sigma=2.0,
                     temporal_decay_tau=6.0):
    """Create 2D asymmetric convolution kernel.

    Args:
        kx_size: spatial kernel size
        kt_size: temporal kernel size
        spatial_sigma: Gaussian width in spatial dimension
        temporal_rise_idx: index of rise peak in time
        temporal_rise_sigma: Gaussian width of rise
        temporal_decay_tau: exponential decay time constant

    Returns:
        2D array of shape (kx_size, kt_size), normalized to sum to 1
    """
    kx = create_spatial_gaussian(kx_size, spatial_sigma)
    kt = create_temporal_asymmetric(kt_size, temporal_rise_idx,
                                    temporal_rise_sigma, temporal_decay_tau)

    kernel_2d = kx[:, np.newaxis] * kt[np.newaxis, :]
    return kernel_2d / kernel_2d.sum()


# ============================================================================
# Forward Convolution (FFT-based)
# ============================================================================

def convolve_2d_fft(signal, kernel):
    """Convolve signal with kernel using FFT.

    Args:
        signal: 2D array (nx, nt)
        kernel: 2D array (kx, kt)

    Returns:
        2D array of shape (nx, nt) - same size as signal
    """
    nx, nt = signal.shape
    kx, kt = kernel.shape

    # Pad to avoid circular convolution: output size = input + kernel - 1
    pad_nx = nx + kx - 1
    pad_nt = nt + kt - 1

    # FFT
    sig_fft = np.fft.rfft2(signal, s=(pad_nx, pad_nt))
    ker_fft = np.fft.rfft2(kernel, s=(pad_nx, pad_nt))

    # Convolve in frequency domain
    conv_fft = sig_fft * ker_fft

    # IFFT
    conv = np.fft.irfft2(conv_fft, s=(pad_nx, pad_nt))

    # Crop back to original signal size (centered)
    # The convolution output starts at (0, 0), we want to keep the
    # physically meaningful part
    start_x = (kx - 1) // 2
    start_t = (kt - 1) // 2
    measurement = conv[start_x : start_x + nx, start_t : start_t + nt]

    return measurement


# ============================================================================
# Deconvolution with 2D Gaussian Filter
# ============================================================================

def gaussian_filter_2d(shape, dt_x, dt_t, sigma_x, sigma_t):
    """Create 2D Gaussian regularization filter in frequency domain.

    Args:
        shape: (nx_fft, nt_fft) - FFT shape
        dt_x: spatial grid spacing
        dt_t: temporal grid spacing
        sigma_x: Gaussian sigma in spatial domain
        sigma_t: Gaussian sigma in temporal domain

    Returns:
        2D array matching rfft2 output shape: (nx_fft, nt_fft // 2 + 1)
    """
    nx_fft, nt_fft = shape

    # Spatial frequencies (full spectrum for x)
    freqs_x = np.fft.fftfreq(nx_fft, d=dt_x)
    # Temporal frequencies (half spectrum for t, since we use rfft)
    freqs_t = np.fft.rfftfreq(nt_fft, d=dt_t)

    # 2D Gaussian
    gaussian_x = np.exp(-0.5 * freqs_x[:, np.newaxis]**2 / sigma_x**2)
    gaussian_t = np.exp(-0.5 * freqs_t[np.newaxis, :]**2 / sigma_t**2)
    print(gaussian_t, gaussian_x)

    gaussian_2d = gaussian_x * gaussian_t
    return gaussian_2d


def deconvolve_2d_fft(measurement, kernel, filter_2d=None):
    """Deconvolve measurement using FFT with optional Gaussian filter.

    Args:
        measurement: 2D array (nx, nt)
        kernel: 2D array (kx, kt)
        filter_2d: optional 2D filter in frequency domain

    Returns:
        2D array of shape (nx, nt) - deconvolved signal
    """
    nx, nt = measurement.shape
    kx, kt = kernel.shape

    # Padding for FFT
    pad_nx = nx + kx - 1
    pad_nt = nt + kt - 1

    # FFT
    meas_fft = np.fft.rfft2(measurement, s=(pad_nx, pad_nt))
    ker_fft = np.fft.rfft2(kernel, s=(pad_nx, pad_nt))

    # Avoid division by zero
    epsilon = 1e-10
    ker_fft = np.where(np.abs(ker_fft) < epsilon, epsilon, ker_fft)

    # Deconvolve
    signal_fft = meas_fft / ker_fft

    # Apply Gaussian filter if provided
    if filter_2d is not None:
        signal_fft *= filter_2d

    # IFFT
    signal = np.fft.irfft2(signal_fft, s=(pad_nx, pad_nt))

    # Crop back to original size
    start_x = (kx - 1) // 2
    start_t = (kt - 1) // 2
    signal_cropped = signal[start_x : start_x + nx, start_t : start_t + nt]

    return signal_cropped


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run 2D deconvolution demo."""

    # Output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    nx, nt = 32, 128
    delta_x, delta_t = 16, 32

    # Kernel parameters
    kernel = create_2d_kernel(kx_size=11, kt_size=20, spatial_sigma=1.5,
                              temporal_rise_idx=5, temporal_rise_sigma=2.0,
                              temporal_decay_tau=6.0)

    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel sum: {kernel.sum():.4f}")

    # Create ideal delta signal
    signal_ideal = np.zeros((nx, nt))
    signal_ideal[delta_x, delta_t] = 1.0

    # Forward convolution to get clean measurement
    measurement_ideal = convolve_2d_fft(signal_ideal, kernel)

    # Case 1: Add noise to measurement (Gaussian noise in time domain)
    noise_std = 0.05 * np.max(np.abs(measurement_ideal))
    noise_std = 0
    print(noise_std, np.max(np.abs(measurement_ideal)))
    noise = np.random.normal(0, noise_std, measurement_ideal.shape)
    measurement_noisy = measurement_ideal + noise

    # Case 2: Misalignment (spatial shift)
    misalign_shift = -3  # pixels
    measurement_misaligned = measurement_ideal.copy()
    measurement_misaligned[delta_x, :] = np.roll(measurement_ideal[delta_x, :], misalign_shift, axis=-1)
    measurement_misaligned[delta_x, misalign_shift:] = 0

    # Case 1+2: Both noise and misalignment
    measurement_both = measurement_noisy.copy()
    measurement_both[delta_x, :] = np.roll(measurement_noisy[delta_x, :], misalign_shift, axis=-1)
    measurement_both[delta_x, misalign_shift:] = 0

    # Deconvolution
    # Ideal: no filter
    deconv_ideal = deconvolve_2d_fft(measurement_ideal, kernel, filter_2d=None)

    # With Gaussian filter for noisy cases
    # Choose filter parameters based on problem scale
    sigma_x_filter = 20  # smoothing in spatial dimension
    sigma_t_filter = 20  # smoothing in time dimension
    pad_nx = nx + kernel.shape[0] - 1
    pad_nt = nt + kernel.shape[1] - 1
    filter_2d = gaussian_filter_2d((pad_nx, pad_nt), dt_x=1.0, dt_t=1.0,
                                   sigma_x=sigma_x_filter, sigma_t=sigma_t_filter)

    deconv_noisy = deconvolve_2d_fft(measurement_noisy, kernel, filter_2d=filter_2d)
    deconv_misaligned = deconvolve_2d_fft(measurement_misaligned, kernel,
                                          filter_2d=filter_2d)
    deconv_both = deconvolve_2d_fft(measurement_both, kernel, filter_2d=filter_2d)

    # Print statistics
    print("\n--- Signal and Measurement Statistics ---")
    print(f"Signal ideal sum: {signal_ideal.sum():.4f}")
    print(f"Measurement ideal sum: {measurement_ideal.sum():.4f}")
    print(f"Measurement noisy sum: {measurement_noisy.sum():.4f}")
    print(f"Measurement misaligned sum: {measurement_misaligned.sum():.4f}")
    print(f"Measurement both sum: {measurement_both.sum():.4f}")
    print(f"\nDeconv ideal sum: {deconv_ideal.sum():.4f}")
    print(f"Deconv noisy sum: {deconv_noisy.sum():.4f}")
    print(f"Deconv misaligned sum: {deconv_misaligned.sum():.4f}")
    print(f"Deconv both sum: {deconv_both.sum():.4f}")

    # ========================================================================
    # Plotting
    # ========================================================================

    # Plot 1: Detailed kernel visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(kernel, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_xlabel('Time (bin)')
    axes[0].set_ylabel('Space (pixel)')
    axes[0].set_title('2D Asymmetric Kernel')
    plt.colorbar(im0, ax=axes[0])

    # Kernel profiles
    axes[1].plot(kernel[kernel.shape[0]//2, :], 'b-o', label='Temporal (at center)')
    axes[1].plot(kernel[:, kernel.shape[1]//2], 'r-s', label='Spatial (at center)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Kernel Profiles (Asymmetric!)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'kernel_detail.png', dpi=150)
    plt.close()

    # Plot 2: Main 4×4 grid showing signal, measurements, and deconvolutions
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    vmin_meas = min(measurement_ideal.min(), measurement_noisy.min(),
                    measurement_misaligned.min(), measurement_both.min())
    vmax_meas = max(measurement_ideal.max(), measurement_noisy.max(),
                    measurement_misaligned.max(), measurement_both.max())

    vmin_deconv = min(deconv_ideal.min(), deconv_noisy.min(),
                      deconv_misaligned.min(), deconv_both.min())
    vmax_deconv = max(deconv_ideal.max(), deconv_noisy.max(),
                      deconv_misaligned.max(), deconv_both.max())

    # Row 0: Signal, Kernel, Ideal measurement, Ideal deconvolution
    im = axes[0, 0].imshow(signal_ideal, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title('True Signal\n(delta at (16, 16))')
    axes[0, 0].plot(delta_t, delta_x, 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(kernel, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 1].set_title('Asymmetric Kernel')
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].imshow(measurement_ideal, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_meas, vmax=vmax_meas)
    axes[0, 2].set_title('Ideal Measurement\n(no noise, no misalignment)')
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[0, 3].imshow(deconv_ideal, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_deconv, vmax=vmax_deconv)
    axes[0, 3].set_title('Deconv Ideal\n(no filter applied)')
    axes[0, 3].plot(delta_t, delta_x, 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im, ax=axes[0, 3])

    # Row 1: Noisy measurement and deconvolution
    im = axes[1, 0].imshow(measurement_noisy, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_meas, vmax=vmax_meas)
    axes[1, 0].set_title('Noisy Measurement\n(time-domain noise)')
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(deconv_noisy, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_deconv, vmax=vmax_deconv)
    axes[1, 1].set_title('Deconv Noisy\n(+ 2D Gaussian filter)')
    axes[1, 1].plot(delta_t, delta_x, 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im, ax=axes[1, 1])

    # Row 2: Misaligned measurement and deconvolution
    im = axes[2, 0].imshow(measurement_misaligned, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_meas, vmax=vmax_meas)
    axes[2, 0].set_title('Misaligned Measurement\n(Time at charge center shift +3 ticks)')
    plt.colorbar(im, ax=axes[2, 0])

    im = axes[2, 1].imshow(deconv_misaligned, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_deconv, vmax=vmax_deconv)
    axes[2, 1].set_title('Deconv Misaligned\n(+ 2D Gaussian filter)')
    axes[2, 1].plot(delta_t - misalign_shift, delta_x, 'r+', markersize=15,
                    markeredgewidth=2, label='True pos (shifted)')
    plt.colorbar(im, ax=axes[2, 1])

    # Row 3: Both noise and misalignment
    im = axes[3, 0].imshow(measurement_both, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_meas, vmax=vmax_meas)
    axes[3, 0].set_title('Both: Noisy + Misaligned\nMeasurement')
    plt.colorbar(im, ax=axes[3, 0])

    im = axes[3, 1].imshow(deconv_both, aspect='auto', cmap='viridis',
                           origin='lower', vmin=vmin_deconv, vmax=vmax_deconv)
    axes[3, 1].set_title('Deconv Both\n(+ 2D Gaussian filter)')
    axes[3, 1].plot(delta_t, delta_x - misalign_shift, 'r+', markersize=15,
                    markeredgewidth=2, label='True pos (shifted)')
    plt.colorbar(im, ax=axes[3, 1])

    # Columns 2-3: 1D profiles (spatial and temporal)
    # Spatial profile at t=delta_t
    axes[0, 2].plot(signal_ideal[:, delta_t], 'o-', label='Signal', linewidth=2)
    axes[1, 2].plot(measurement_noisy[:, delta_t], 'o-', label='Noisy meas',
                    linewidth=2)
    axes[2, 2].plot(measurement_misaligned[:, delta_t], 'o-', label='Misaligned meas',
                    linewidth=2)
    axes[3, 2].plot(measurement_both[:, delta_t], 'o-', label='Both meas', linewidth=2)

    for i in range(4):
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend(loc='upper right')
    axes[0, 2].set_title('Spatial Profile at t={}'.format(delta_t))

    # Temporal profile at x=delta_x (accounting for misalignment)
    axes[0, 3].plot(signal_ideal[delta_x, :], 'o-', label='Signal', linewidth=2)
    axes[1, 3].plot(deconv_noisy[delta_x, :], 'o-', label='Deconv noisy',
                    linewidth=2)
    axes[2, 3].plot(deconv_misaligned[delta_x, :], 'o-',
                    label='Deconv misaligned', linewidth=2)
    axes[2, 3].set_xlabel('Time')
    axes[3, 3].plot(deconv_both[delta_x, :], 'o-', label='Deconv both',
                    linewidth=2)
    axes[3, 3].set_xlabel('Time')

    for i in range(4):
        axes[i, 3].set_xlabel('Time (bin)')
        axes[i, 3].set_ylabel('Amplitude')
        axes[i, 3].grid(True, alpha=0.3)
        axes[i, 3].legend(loc='upper right')
    axes[0, 3].set_title('Temporal Profile at x={}'.format(delta_x))

    # Hide unused subplots
    axes[0, 0].set_xlabel('Time (bin)')
    axes[0, 0].set_ylabel('Space (pixel)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Space (pixel)')
    for ax in axes[3, :]:
        ax.set_xlabel('Time (bin)')

    plt.tight_layout()
    plt.savefig(output_dir / 'deconv2d_comparison.png', dpi=150)
    plt.close()

    print("\n✓ Plots saved:")
    print(f"  - {output_dir / 'kernel_detail.png'}")
    print(f"  - {output_dir / 'deconv2d_comparison.png'}")


if __name__ == '__main__':
    main()
