# Algorithm Overview: UnfoldLArPix 3D Deconvolution

**Date:** 2026-04-13  
**Source revision:** `main` branch (commit d8ea6df)  
**Files covered:** `src/unfoldlarpix/` — all source modules

---

## 1. Problem Statement

The LArPix pixelated TPC readout measures charge at discrete pixels on an anode plane. Each pixel records a set of ADC-sampled "bursts" — the integrated charge accumulated between threshold crossings. The goal of this package is to recover the true 3D ionization charge density `x(px, py, t)` from the measured hit-burst data `y(px, py, t)`, given a known detector response `R`.

The forward model is:

```
y = R ∗ x + n
```

where `∗` denotes 3D convolution, `R` is the combined field-response and electronics-integration kernel, and `n` is additive noise.

---

## 2. The Detector Response Kernel R

### 2.1 Field Response

The field response describes how charge drifting toward pixel `(px, py)` from a given (x, y, z) position in the active volume induces a current signal. It is loaded from an NPZ file in `field_response.py`.

Raw data format: `response` array of shape `(r_npath, r_npath, Nt_fine)`, representing one positive-positive quadrant of the transverse response for `npath × npath` sub-pixel impact positions.

Processing pipeline (`FieldResponseProcessor.process_response`, `field_response.py:154`):

1. **Normalization** (`field_response.py:187-196`): if not already normalized, multiply by `time_tick` so the center-pixel response integrates to 1.0:
   ```
   sum(response[0, 0, :]) * time_tick == 1.0
   ```
   The kernel represents dQ/dt; multiplying by dt gives dQ per fine tick.

2. **Quadrant copy** (`_quadrant_copy`, `field_response.py:38-85`): The positive-positive quadrant is mirrored across both spatial axes to produce the full `(2*r_npath, 2*r_npath, Nt_fine)` plane. Origin pixel sits at index `(r_npath, r_npath)`.

3. **Per-path flip** (`_flip_kernel_for_convolution`, `field_response.py:121-152`): The response encodes a cross-correlation kernel (detector response at pixel (i,j) for source at path offset (di,dj)). Flipping the pixel indices converts it to a convolution kernel.

4. **Sub-pixel average** (`_downsample_by_averaging`, `field_response.py:87-119`): Average over the `npath × npath` sub-pixel positions within each macro-pixel. Output shape: `(2r, 2r, Nt_fine)` where `r = r_npath/npath` (number of pixels in one quadrant).

   > **Note:** The resulting shape is `(2r, 2r, Nt_fine)` — an **even** spatial extent. The "center" pixel used downstream is at `[r, r]` (`deconv_workflow.py:84`), which is one pixel toward positive-positive of the geometric center. See bug report §3.2.

### 2.2 Electronics Integration

The ADC integrates charge for `adc_hold_delay` fine ticks per output coarse sample. This is modeled by summing the fine-tick field response in non-overlapping `adc_hold_delay`-tick windows (`integrate_kernel_over_time`, `deconv_workflow.py:56-71`):

```python
reshaped = kernel[..., :n_ticks].reshape(*kernel.shape[:-1],
                                          n_ticks // ticks_per_bin,
                                          ticks_per_bin)
integrated = reshaped.sum(axis=-1)   # shape: (2r, 2r, Nt_coarse)
```

The integrated response `R_int` has shape `(2r, 2r, Nt_coarse)` where `Nt_coarse = Nt_fine / adc_hold_delay`. This is the kernel used in deconvolution.

### 2.3 Center Response (Template)

```python
center_response = full_response[shape//2, shape//2, :]   # fine ticks
```
(`deconv_workflow.py:83-85`)

This 1D cumulative template (after `np.cumsum`) is used by the burst processor to model the rising edge of a new hit during threshold-gap intervals. It is **not** used in the frequency-domain deconvolution itself.

---

## 3. From Hit Bursts to a Dense 3D Block

### 3.1 Hit Data Format

`Hits.data` shape: `(N_pix_hit, 3 + nburst)` — columns `[x, y, z, q1, …, q_nburst]` where `q_k` is the **cumulative** integrated charge through burst `k`.  
`Hits.location` shape: `(N_pix_hit, 5)` — columns `[pixel_x, pixel_y, trigger_time_idx, last_adc_latch, next_integration_start]`.

### 3.2 Burst Sequence Extraction

Each row in `Hits` corresponds to one "trigger event" on one pixel. The cumulative charges are differentiated to get per-burst increments:

```python
charges = [raw[0]] + np.diff(raw).tolist()   # burst_processor.py:155 (v1)
```

Time coordinates:
```
t_first = trigger_time_idx + adc_hold_delay     # end of first integration window
t_last  = trigger_time_idx + adc_hold_delay * nburst
```

Each pixel collects a list of `BurstSequence` objects, sorted by `trigger_time_idx`.

### 3.3 Sequence Merging (Burst Processor)

The core challenge: between trigger events the signal may have dipped below threshold, been absent entirely, or just experienced hardware dead-time. The burst processor interpolates the charge during these gaps.

**V1 (`BurstSequenceProcessor`, `burst_processor.py`):**

For each consecutive pair of sequences per pixel:

- `0 < gap ≤ tau` (signal dipped briefly): **dead-time slope compensation** — extrapolates a linear ramp bridging the gap using `slope = charges[0] / (gap - deadtime)`, adds `slope * deadtime` as compensated charge at the gap boundary.
- `gap > tau` (long absence): **template compensation** — inserts a reversed, downsampled slice of the cumulative `center_response` template to model the sub-threshold rising edge before the next trigger fires.

Before the very first sequence, a bootstrap template compensation is always applied (the signal must have been rising to reach threshold).

**V2 (`BurstSequenceProcessorV2`, `burst_processor_v2.py`):**

Key differences from V1:

1. **No dead-time slope compensation** — replaced by first-interval scaling: for `gap < tau`, the first charge of the next sequence is rescaled by `adc_hold_delay / (gap - deadtime)` to account for the shortened first integration window.

2. **Collision-stopping template filter** — during template compensation, candidate template times are filtered to the open interval `(last_time, next_seq.t_first)`, preventing template points from colliding with the next burst.

3. **Fractional-shift phase alignment (Phase 2)** — each pixel's trigger has a sub-sample jitter `delta_T = trigger_time_idx % adc_hold_delay`. After all merging, V2 applies a frequency-domain fractional shift `exp(-i 2π k delta_T / (M * adc_hold_delay))` to each contiguous block of charges sharing the same `delta_T`, then snaps their times to the common `n * adc_hold_delay` grid (`burst_processor_v2.py:503-514`).

### 3.4 Dense Block Assembly

After merging all sequences per pixel, `merged_sequences_to_block` (`burst_processor.py:523-566`) scatters the per-pixel 1D charge-vs-time arrays into a dense 3D array:

- Shape: `(Nx, Ny, Nt)` where:
  - `Nx = pmax_x - pmin_x + 1` (pixel bounding box)
  - `Ny = pmax_y - pmin_y + 1`
  - `Nt = ceil((tmax - tmin + 2*pad_length) / bin_size) + 1`
  - `pad_length = npadbin * bin_size`, default `npadbin=50`
- Offset: `block_offset = [pmin_x, pmin_y, tmin - pad_length]`
- `bin_size = adc_hold_delay` — times are snapped to coarse bins

---

## 4. Deconvolution

### 4.1 Frequency-Domain Solver

The deconvolution is performed by `deconv_fft` (`deconv.py:31-82`). It is a **one-pass pseudo-Wiener filter** — not iterative. Given:
- `measurement` = dense block `y` of shape `(Nx, Ny, Nt)`
- `kernel` = integrated response `R_int` of shape `(Kx, Ky, Kt)` — compact support
- `filter_fft` = 3D Gaussian low-pass filter (pre-built, in frequency domain)

The steps:

```
1. fft_shape = (Nx + Kx - 1, Ny + Ky - 1, Nt)   # only spatial dims padded
2. Y(k) = rfftn(measurement, s=fft_shape)           # deconv.py:55
3. R(k) = rfftn(kernel,      s=fft_shape)           # deconv.py:58
4. R_clamped(k) = |R(k)| < ε ? ε : R(k),  ε=1e-10  # deconv.py:62
5. X̂(k) = Y(k) / R_clamped(k)                      # deconv.py:65
6. X̂(k) *= G(k)                  # Gaussian low-pass  deconv.py:67
7. x̂    = irfftn(X̂(k), s=fft_shape)                 # deconv.py:70
8. x̂    = roll(x̂, (Kx-1)//2, axis=0)               # recenter axis 0  deconv.py:73
9. x̂    = roll(x̂, (Ky-1)//2, axis=1)               # recenter axis 1  deconv.py:74
10. x̂   = x̂[0:Nx, 0:Ny, 0:Nt]                      # crop to measurement shape
```

**Important detail:** The time axis (axis 2) is **not FFT-padded** and **not rolled**. This is a silent assumption that (a) the caller has pre-padded `measurement` by at least `Kt - 1` ticks in time via `npadbin`, and (b) the integrated response kernel is **causal** (time origin at index 0). If either assumption is violated, the time-axis deconvolution wraps around circularly and/or the output is time-shifted. See bug report §2.1.

### 4.2 Regularization: 3D Gaussian Low-Pass Filter

The filter `G(k)` is a separable 3D Gaussian in the frequency domain:

```
G(kx, ky, kt) = exp(-0.5 * kx²/σ_pix²) · exp(-0.5 * ky²/σ_pix²) · exp(-0.5 * kt²/σ_t²)
```

Built by `gaussian_filter_3d` (`deconv.py:21-29`). The rfft half-spectrum is used for the time axis; full spectra for the spatial axes. Frequencies are computed as:

```python
freqs_t  = rfftfreq(Nt,         d=adc_hold_delay)   # units: 1/tick
freqs_xy = fftfreq(Nx_fft or Ny_fft, d=1)          # units: 1/pixel
```

Parameters `sigma_time` and `sigma_pixel` are passed as frequency-domain standard deviations in units of `1/tick` and `1/pixel` respectively (note: the docstring says "sigma in time domain" but the math uses `exp(-f²/(2σ²))` directly in the frequency domain — so σ is a frequency-domain width, not a time-domain width). See bug report §3.4.

Both parameters are supplied at call sites in `deconv_workflow.py:289-295` and re-used for ground-truth smearing to keep the comparison matched.

### 4.3 Ground-Truth Smearing

The same `(sigma_pixel, sigma_pixel, sigma_time)` Gaussian is convolved with the effective-charge truth to produce a matched-resolution reference:

```python
smear_offset, smeared_true = gaus_smear_true_3d(
    event.effq.location, event.effq.data,
    width=np.array([sigma_pixel, sigma_pixel, sigma_time], dtype=float),
)
```
(`deconv_workflow.py:305-309`, `smear_truth.py:34-69`)

The truth is zero-padded to `ktimes = 2*n_single_side + 1` times its time extent for the circular FFT convolution, then rolled back. The rolling ensures the smeared peak aligns with the original time coordinate.

---

## 5. Full Call Graph (per event)

```
process_event_deconvolution (deconv_workflow.py:265)
  │
  ├─ hits_to_merged_block (deconv_workflow.py:114)
  │    ├─ create_burst_processor → BurstSequenceProcessor[V2] (deconv_workflow.py:96)
  │    ├─ burst_processor.process_hits (burst_processor*.py:502/525)
  │    │    ├─ extract_sequences_from_hits   [loop over hits]
  │    │    └─ process_pixel_sequences       [loop over pixels]
  │    │         ├─ _template_compensation   [bootstrap]
  │    │         ├─ [loop over sequences:]
  │    │         │    ├─ _append_shifted     [gap < tau, V2 only]
  │    │         │    └─ _template_compensation  [gap >= tau]
  │    │         └─ Phase 2: _fractional_shift per delta_T block  [V2 only]
  │    └─ merged_sequences_to_block → (block_offset, block_data)   [scatter]
  │
  ├─ build_gaussian_deconv_kernel (deconv_workflow.py:230)
  │    └─ gaussian_filter_3d (deconv.py:21) → filter_fft array
  │
  ├─ deconv_fft (deconv.py:31)
  │    ├─ rfftn(measurement)     [deconv.py:55]
  │    ├─ rfftn(kernel)          [deconv.py:58]
  │    ├─ ε-clamp kernel FFT     [deconv.py:62]
  │    ├─ / kernel FFT           [deconv.py:65]
  │    ├─ × filter_fft           [deconv.py:67]
  │    ├─ irfftn                 [deconv.py:70]
  │    ├─ np.roll × 2 (axes 0,1) [deconv.py:73-74]
  │    └─ crop → (deconv_q, local_offset)
  │
  └─ smear_effective_charge (deconv_workflow.py:249)
       └─ gaus_smear_true_3d (smear_truth.py:34)
            ├─ scatter effq into dense grid
            ├─ rfftn
            ├─ × gaussian filter
            └─ irfftn + roll → (smear_offset, smeared_true)
```

---

## 6. Tensor Shape Table

| Stage | Array | Shape | Notes |
|-------|-------|-------|-------|
| Input hits | `Hits.data` | `(N_pix, 3+nburst)` | Cumulative charges |
| | `Hits.location` | `(N_pix, 5)` | `[px, py, trig, latch, next]` |
| Field response (raw quadrant) | `response` | `(r·npath, r·npath, Nt_fine)` | One quadrant |
| After quadrant copy | `expanded_response` | `(2r·npath, 2r·npath, Nt_fine)` | Full plane |
| After downsample | `full_response` | `(2r, 2r, Nt_fine)` | Even spatial dims |
| After time integration | `integrated_response` | `(2r, 2r, Nt_coarse)` | `Nt_coarse = Nt_fine/adc_hold_delay` |
| Dense merged block | `block_data` | `(Nx, Ny, Nt)` | Event bounding box + pad |
| FFT working arrays | `measurement_fft`, `kernel_fft`, `signal_fft` | `(Nx+2r-1, Ny+2r-1, Nt/2+1)` | Complex128 |
| 3D Gaussian filter | `filter_fft` | `(Nx+2r-1, Ny+2r-1, Nt/2+1)` | Real → broadcast |
| Deconvolved output | `deconv_q` | `(Nx, Ny, Nt)` | Same as `block_data` |
| Truth scatter | `data` in smear_truth | `(Nx_eff, Ny_eff, Nt_eff)` | Effq bounding box |
| Truth smeared | `smeared_true` | `(Nx_eff, Ny_eff, ktimes·Nt_eff)` | Zero-padded for FFT |

**Typical numeric values** (from project analysis notes):
- `Nt` ≈ 11,970 (nburst16) or 16,500 (nburst256) coarse bins
- Pixel bounding box: tens to hundreds of pixels per axis depending on event topology

---

## 7. V1 vs. V2 Burst Processor Summary

| Feature | V1 (`burst_processor.py`) | V2 (`burst_processor_v2.py`) |
|---------|--------------------------|------------------------------|
| Gap < tau | Dead-time slope compensation (linear ramp) | First-interval scaling only (no template) |
| Gap ≥ tau | Template insertion | Template insertion with collision filter |
| Template filter | None (`valid_mask = times > last_time`) | Strict: `(last_time, next_seq.t_first)` |
| Sub-sample jitter | Not corrected | Phase-2 fractional FFT shift |
| First-charge scaling | None | `charges[0] *= adc_hold_delay / active_time_first` |
| Output times | May be non-uniform across gaps | Snapped to `n * adc_hold_delay` grid |

---

## 8. Regularization Parameters

| Parameter | Symbol | Where used | Units |
|-----------|--------|------------|-------|
| `sigma_time` | σ_t | Gaussian filter time axis | 1/tick (frequency domain) |
| `sigma_pixel` | σ_pix | Gaussian filter spatial axes (both x and y) | 1/pixel (frequency domain) |
| `epsilon` | ε=1e-10 | Kernel FFT magnitude clamp | Absolute (not scaled) |

Both σ parameters are applied identically to the deconvolution filter and the ground-truth smearing kernel, so comparisons between `deconv_q` and `smeared_true` are always made in the same resolution space.

Larger σ → more aggressive low-pass → smoother output, lower spatial/temporal resolution.  
Smaller σ → less filtering → sharper output, but more noise amplification from `1/R` at high frequencies.
