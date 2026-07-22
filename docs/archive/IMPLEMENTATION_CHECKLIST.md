# Wiener-Like Filter Implementation Checklist

## Quick Reference for Building Your Filter

---

## Phase 1: Collection Plane (Simpler Baseline)

### [ ] Step 1: Load Response Functions
```python
# Pseudo-code
field_response = load_garfield_responses()  # Shape: (n_wires, n_positions, n_time)
electronics_response = calculate_rc_impulse(gain=14e-3, peaking_time=2e-6)
total_response = convolve(field_response, electronics_response)
```

**What you need:**
- Garfield simulated field responses (position-dependent)
- Electronics impulse response (RC filter parameters)
- Pre-calculated convolution

---

### [ ] Step 2: Position-Average Response
```python
# Equation 3.8: Average over 11 transverse positions
positions = [0, 0.3, 0.6, 0.9, 1.2, 1.5] # mm within wire pitch
weights = [0.5, 1, 1, 1, 1, 0.5]  # Boundary weighting

R_averaged = zeros_like(R[0])
for i, (pos, w) in enumerate(zip(positions, weights)):
    R_averaged += w * R[pos] / sum(weights)
```

**Output:**
- R̄(t): Averaged response function in time domain
- R̄_fft(ω): FFT of averaged response

---

### [ ] Step 3: 2D Deconvolution (Frequency Domain)
```python
# Equation 3.7: Toeplitz matrix inversion
def deconvolve_2d(signals, response_matrix):
    # signals shape: (n_wires, n_samples)
    # response_matrix: Toeplitz convolution matrix
    
    # FFT along time axis
    M_fft = fft(signals, axis=1)
    
    # For each frequency bin, invert the matrix
    S_fft = zeros_like(M_fft)
    for freq in range(n_samples):
        S_fft[:, freq] = linalg.solve(response_matrix[freq], M_fft[:, freq])
    
    # IFFT back to time domain
    S = ifft(S_fft, axis=1).real
    return S
```

**Key points:**
- Toeplitz matrix (symmetric, constant diagonals)
- Solves: M = R·S → S = R⁻¹·M
- Can use FFT-based convolution for efficiency

---

### [ ] Step 4: Test with Gaussian Filter First
```python
# Equation 3.12: Simple Gaussian in frequency domain
def gaussian_filter(signal, sigma_omega):
    S_fft = fft(signal)
    freqs = fftfreq(len(signal), dt)
    
    # F(ω) = exp(-0.5 * (ω/σ_ω)²)
    F = exp(-0.5 * (freqs / sigma_omega)**2)
    
    S_filtered_fft = S_fft * F
    S_filtered = ifft(S_filtered_fft).real
    return S_filtered

# Tune sigma_omega for your collection plane
sigma_omega_collection = tune_for_best_charge_resolution()
```

**Validation:**
- Compare deconvolved charge vs true charge
- Tune σ_ω to minimize bias and resolution

---

### [ ] Step 5: Implement Wiener-Inspired Filter
```python
# Equations 3.9-3.10: Modified Wiener with zero at DC
def wiener_inspired_filter(signal, omega_c, b, c):
    """
    F(ω) = c·exp(-1/2·(ω/ω_c)^b)  for ω > 0
    F(ω) = 0                        for ω = 0
    """
    S_fft = fft(signal)
    freqs = fftfreq(len(signal), dt)
    
    # Zero at DC
    F = zeros_like(freqs)
    mask = (freqs > 0)
    F[mask] = c * exp(-0.5 * (freqs[mask] / omega_c)**b)
    
    S_filtered_fft = S_fft * F
    S_filtered = ifft(S_filtered_fft).real
    return S_filtered

# Fit parameters ω_c, b, c from simulation
omega_c_fit, b_fit, c_fit = fit_to_ideal_wiener(...)
```

**Fitting procedure:**
- Generate ideal Wiener results from simulation
- Fit exponential form to frequency response
- Verify time-domain smearing is local

---

### [ ] Step 6: ROI Finding & Charge Extraction
```python
def find_rois_collection(deconvolved_charge):
    # Calculate RMS from quiet regions
    quiet_mask = is_quiet(deconvolved_charge)
    rms = sqrt(mean(deconvolved_charge[quiet_mask]**2))
    
    # Threshold: 5 × RMS
    threshold = 5 * rms
    signal_mask = (deconvolved_charge > threshold)
    
    # Merge adjacent hits
    rois = merge_adjacent_hits(signal_mask)
    
    return rois

def extract_charge(deconvolved, rois):
    charge_spectrum = []
    for roi in rois:
        # Sum charge in ROI window
        charge = sum(deconvolved[roi.start:roi.end])
        charge_spectrum.append(charge)
    return charge_spectrum
```

**Output:**
- List of integrated charge values
- One value per signal/electron bunch

---

## Phase 2: Induction Planes (More Complex)

### [ ] Step 7: Same HF-cut Wiener on Induction
```python
# Apply same Wiener-inspired filter
# May adjust parameters slightly (more noise on induction)
omega_c_induction = 0.9 * omega_c_collection  # Slightly more aggressive
S_induction_filtered = wiener_inspired_filter(
    S_induction_deconv, 
    omega_c_induction, b_fit, c_fit
)
```

---

### [ ] Step 8: Wire Dimension Gaussian Smoothing
```python
# Equation 3.13: Gaussian in wire direction
def wire_dimension_filter(charge_spectrum, sigma_w):
    """
    F(ω_w) = exp(-0.5 * (ω_w / ω_wc)²)
    Applied in wire number domain
    """
    # FFT over wire dimension
    spectrum_fft = fft(charge_spectrum)
    n_wires = len(charge_spectrum)
    freqs_wire = fftfreq(n_wires)
    
    # Gaussian in wire domain
    F_wire = exp(-0.5 * (freqs_wire / sigma_w)**2)
    
    spectrum_filtered = ifft(spectrum_fft * F_wire).real
    return spectrum_filtered

# Induction planes: larger σ_w (more smoothing)
sigma_w_induction = tune_for_bipolar_cancellation()
```

---

### [ ] Step 9: Low-Frequency Filters for ROI Finding
```python
# Equation 3.14: High-pass filter for ROI identification
def lf_filter_roi(signal, omega_0, b):
    """
    F_LF(ω) = 1 - exp(-(ω/ω_0)^b)
    High-pass: passes high frequencies, blocks low
    """
    S_fft = fft(signal)
    freqs = fftfreq(len(signal), dt)
    
    # High-pass
    F = zeros_like(freqs)
    mask = (freqs != 0)
    F[mask] = 1 - exp(-(freqs[mask] / omega_0)**b)
    
    S_filtered = ifft(S_fft * F).real
    return S_filtered

# Two variants for robustness
S_roi_loose = lf_filter_roi(S_deconv, omega_0_loose, b)
S_roi_tight = lf_filter_roi(S_deconv, omega_0_tight, b)
```

---

### [ ] Step 10: ROI Application with Baseline Subtraction
```python
def find_rois_induction(signal_loose, signal_tight):
    # Use both filters for robust ROI finding
    rms = calculate_rms_noise(signal_tight)
    threshold = 3.5 * rms  # Looser than collection
    
    # Identify regions with either tight or loose filter above threshold
    roi_mask = (signal_loose > threshold) | (signal_tight > threshold)
    rois = merge_adjacent_hits(roi_mask)
    
    return rois

def extract_charge_induction(deconvolved, rois):
    """Extract charge with linear baseline subtraction"""
    charge_spectrum = []
    
    for roi in rois:
        # Linear baseline correction
        baseline_start = deconvolved[roi.start]
        baseline_end = deconvolved[roi.end]
        
        t_range = arange(roi.start, roi.end)
        baseline = baseline_start + (baseline_end - baseline_start) * (
            (t_range - roi.start) / (roi.end - roi.start)
        )
        
        # Subtract baseline and integrate
        charge = sum(deconvolved[roi.start:roi.end] - baseline)
        charge_spectrum.append(charge)
    
    return charge_spectrum
```

---

## Phase 3: Validation & Optimization

### [ ] Step 11: Charge Resolution Metrics
```python
def evaluate_charge_extraction(extracted_charge, true_charge):
    # Bias: mean extracted - true
    bias = mean(extracted_charge) - mean(true_charge)
    
    # Resolution: RMS of (extracted - true)
    residuals = extracted_charge - true_charge
    resolution = sqrt(mean(residuals**2))
    
    # Relative resolution (%)
    rel_resolution = resolution / mean(true_charge) * 100
    
    print(f"Bias: {bias:.1f} electrons")
    print(f"Resolution: {resolution:.1f} electrons")
    print(f"Relative: {rel_resolution:.1f}%")
    
    return bias, resolution
```

---

### [ ] Step 12: Cross-Plane Consistency Check
```python
def validate_cross_plane(Q_collection, Q_induction_u, Q_induction_v):
    # Collection plane should equal sum of induction contributions
    # (topologically dependent, but should correlate)
    
    # Check correlation
    correlation_u = pearsonr(Q_collection, Q_induction_u)
    correlation_v = pearsonr(Q_collection, Q_induction_v)
    
    print(f"Correlation (Y vs U): {correlation_u:.3f}")
    print(f"Correlation (Y vs V): {correlation_v:.3f}")
    
    # Both should be high (> 0.7-0.8) if extraction is working
```

---

## Parameter Summary Table

| Parameter | Collection | Induction |
|-----------|-----------|-----------|
| **HF-cut Wiener** | | |
| ω_c (Hz) | X | 0.9×X |
| b (exponent) | 1-2 | 1-2 |
| c (amplitude) | fit | fit |
| **Wire Gaussian** | | |
| σ_w (wires) | ~0.5 | ~1-1.5 |
| **ROI Threshold** | | |
| Multiple of RMS | 5× | 3.5× |
| **ROI Type** | simple | loose+tight |
| **Baseline Correct** | no | yes (linear) |

---

## Code Structure Recommendation

```
wiener_filter/
├── __init__.py
├── response_functions.py
│   ├── load_garfield_responses()
│   ├── calculate_rc_impulse()
│   └── position_average_response()
│
├── deconvolution.py
│   ├── deconvolve_2d()
│   └── build_toeplitz_matrix()
│
├── filters.py
│   ├── gaussian_filter()
│   ├── wiener_inspired_filter()
│   ├── wire_dimension_filter()
│   └── lf_filter_roi()
│
├── roi_finding.py
│   ├── find_rois_collection()
│   ├── find_rois_induction()
│   └── extract_charge()
│
├── validation.py
│   ├── evaluate_charge_extraction()
│   └── validate_cross_plane()
│
└── config.py
    ├── FILTER_PARAMS
    ├── ROI_THRESHOLDS
    └── DETECTOR_PARAMS
```

---

## Testing Checklist

- [ ] Gaussian filter produces reasonable charge spectrum
- [ ] Gaussian σ_ω tuned for collection plane
- [ ] Wiener-inspired filter fits well to ideal Wiener
- [ ] 2D deconvolution produces stable results
- [ ] Charge resolution better than Gaussian alone
- [ ] Induction plane LF filters identify signal regions correctly
- [ ] Baseline subtraction removes DC shifts
- [ ] Cross-plane charge correlation > 0.7
- [ ] Performance on real data similar to simulation

---

## Tuning Tips

1. **Start simple**: Gaussian filter first, then move to Wiener-inspired
2. **Single plane validation**: Get collection working before induction
3. **Parameter sensitivity**: Scan ω_c, b over reasonable ranges
4. **Visual inspection**: Plot deconvolved waveforms before/after filtering
5. **Noise floor**: Measure actual RMS from data, not assumptions
6. **ROI threshold**: Use 68% quantile, not raw RMS, for robustness
7. **Baseline shifts**: Watch for charge loss at ROI edges (sign of problems)
