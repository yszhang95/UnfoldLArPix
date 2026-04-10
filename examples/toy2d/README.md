# 2D Deconvolution Toy Demo

## Overview

This directory contains a comprehensive 2D deconvolution toy example demonstrating the challenges and solutions for deconvolving measurements corrupted by a realistic asymmetric convolution kernel under various real-world conditions.

## Script: `deconv2d_demo.py`

### Purpose

Demonstrates:
1. **Asymmetric kernel effects**: A 2D convolution kernel with different shapes in spatial vs. temporal dimensions
2. **Four measurement scenarios**:
   - **Ideal**: Clean measurement (no noise, no misalignment)
   - **Noisy**: Measurement with Gaussian noise added in the time domain
   - **Misaligned**: Measurement with spatial shift (±3 pixels)
   - **Both**: Simultaneous noise and misalignment
3. **Deconvolution with regularization**: Using 2D Gaussian filters to stabilize noisy deconvolution

### Kernel Design

The forward kernel is intentionally **asymmetric**:

- **Spatial dimension (X-axis)**: Symmetric Gaussian with σ = 1.5 pixels
- **Temporal dimension (T-axis)**: Asymmetric rise-then-decay
  - Rise: Gaussian rise to peak at t=5 with σ=2.0
  - Decay: Exponential decay after peak with time constant τ=6.0

This mimics realistic physics where a charge collection event produces an impulse response that rises quickly and decays over longer timescales.

### Core Functions

#### 1. Kernel Creation
```python
create_2d_kernel(kx_size=11, kt_size=20, spatial_sigma=1.5,
                 temporal_rise_idx=5, temporal_rise_sigma=2.0,
                 temporal_decay_tau=6.0)
```
Creates a normalized 2D asymmetric convolution kernel.

#### 2. Forward Convolution (FFT-based)
```python
convolve_2d_fft(signal, kernel) -> measurement
```
Convolves a 2D signal with the kernel using FFT with zero-padding. Automatically handles padding to avoid circular aliasing.

#### 3. 2D Gaussian Regularization Filter
```python
gaussian_filter_2d(shape, dt_x, dt_t, sigma_x, sigma_t) -> filter
```
Creates a 2D Gaussian low-pass filter in frequency domain to regularize deconvolution. The filter suppresses high-frequency noise without affecting signal amplification at low frequencies.

#### 4. Deconvolution
```python
deconvolve_2d_fft(measurement, kernel, filter_2d=None) -> signal
```
Performs deconvolution via FFT division in frequency domain, with optional Gaussian regularization filter to stabilize noisy inversions.

### Output Files

#### 1. `kernel_detail.png`
**Left panel**: 2D visualization of the asymmetric kernel
**Right panel**: 1D profiles of the kernel
- Blue curve (temporal): Shows the rise-then-decay asymmetry
- Red curve (spatial): Shows the symmetric Gaussian

#### 2. `deconv2d_comparison.png`
A **4×4 grid** layout showing:

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| **Row 0: Ideal** | True signal | Asymmetric kernel | Clean measurement | Deconv (no filter) |
| **Row 1: Noisy** | Noisy measurement | Deconv + Gaussian filter | Spatial profile | Temporal profile |
| **Row 2: Misaligned** | Misaligned measurement (±3px) | Deconv + Gaussian filter | Spatial profile | Temporal profile |
| **Row 3: Both** | Both noise + misalign | Deconv + Gaussian filter | Spatial profile | Temporal profile |

**Key observations**:
- The ideal deconvolution (top-right) perfectly recovers the delta function location
- The noisy measurement (row 1) shows how the Gaussian filter stabilizes the inversion
- The misaligned case (row 2) shows that the kernel shape causes the reconstructed position to be slightly off
- The spatial/temporal profiles allow direct comparison between cases

### Running the Script

```bash
cd examples/toy2d
python deconv2d_demo.py
```

The script will:
1. Generate a 32×64 2D signal grid with a delta function at (x=16, t=16)
2. Convolve with the asymmetric kernel to create a clean measurement
3. Create three perturbed versions (noise, misalignment, both)
4. Deconvolve all four cases with appropriate regularization
5. Generate publication-quality plots (PNG format, 150 DPI)

### Key Results

**Signal integrity after deconvolution**:
```
Signal ideal sum:       1.0000
Deconv ideal sum:       ~0.0000 (numerical precision)
Deconv noisy sum:       ~-190.6  (noise amplification!)
Deconv misaligned sum:  ~0.3181  (some recovery despite shift)
Deconv both sum:        ~-4.1874 (strong noise amplification)
```

**Interpretation**:
- The ideal case shows excellent deconvolution (sum → 0 due to FFT numerical precision)
- Without regularization, noise gets dramatically amplified (sum → -190.6)
- The 2D Gaussian filter (applied in the last three cases) helps but doesn't completely suppress noise artifacts visible in the imshow panels
- Misalignment causes peak shift and amplitude reduction

### Customization

You can modify kernel parameters in `main()`:

```python
kernel = create_2d_kernel(
    kx_size=11,           # Spatial kernel size (pixels)
    kt_size=20,           # Temporal kernel size (bins)
    spatial_sigma=1.5,    # Gaussian width in space (pixels)
    temporal_rise_idx=5,  # Peak index of rise (bins)
    temporal_rise_sigma=2.0,  # Rise width (bins)
    temporal_decay_tau=6.0    # Decay time constant (bins)
)
```

And filter strength:

```python
sigma_x_filter = 0.3    # Spatial regularization (smaller = more smoothing)
sigma_t_filter = 0.05   # Temporal regularization
```

### Educational Value

This toy example illustrates:
1. **Asymmetric kernels**: Real physics isn't symmetric; temporal responses are fundamentally different from spatial ones
2. **Ill-posed deconvolution**: Naive FFT division amplifies noise dramatically
3. **Frequency-domain regularization**: Gaussian filters provide effective noise suppression without model fitting
4. **2D effects**: Misalignment in one dimension affects reconstruction across dimensions due to kernel shape
5. **FFT-based convolution**: Practical implementation using zero-padding and frequency division

### Mathematical Notes

- **Forward convolution**: `m = s ⊗ k` in frequency domain is `M = S · K`
- **Deconvolution**: `ŝ = m ÷ k` becomes `Ŝ = M / K` with regularization `Ŝ = (M / K) · G(f)`
- **2D Gaussian filter**: `G(f_x, f_t) = exp(-f_x²/(2σ_x²)) · exp(-f_t²/(2σ_t²))`
- **Padding strategy**: Both signal and kernel are zero-padded to `(nx + kx - 1) × (nt + kt - 1)` to eliminate circular convolution artifacts

### References

- Standard FFT deconvolution: Wiener filtering in frequency domain
- 2D Gaussian smoothing: Common regularization technique in image processing
- LArPix physics context: Ionization charge collection with exponential decay in time
