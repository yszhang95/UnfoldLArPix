# Wiener-Like Filter Implementation Guide
## Based on arxiv:1802.08709 - Ionization Electron Signal Processing in Single Phase LArTPCs

---

## 1. LArTPC Signal Formation Overview

### Three Components of TPC Signal:
1. **Field Response** - Induced currents from drifting electrons
2. **Electronics Response** - Amplification and shaping
3. **Initial Distribution** - Charge distribution in detector volume

### Key Detector Properties (MicroBooNE):
- Three wire planes: U (induction, -60°), V (induction, +60°), Y (collection, vertical)
- 3 mm wire pitch
- 273 V/cm drift field
- Electron drift velocity ~1.10 mm/µs toward anode
- ADC sampling: 2 MHz (0.5 µs ticks)

---

## 2. Signal Characteristics by Plane Type

### Induction Planes (U, V Wires)

**Field Response:**
- **Bipolar signal shape** - alternates positive/negative
- Small amplitude compared to collection plane
- **Complex time profile** with multiple peaks
- Origin: Electrons induce current while approaching AND passing induction wires

**Signal Properties:**
- Long-range effects: Signal detectable on wires **10+ cm away**
- Fine structure depends on:
  - Electron drift path geometry
  - Weighting potential of specific wire
  - Local charge density
  
**Mathematical Basis:**
- Induced current on wire from electron at position: `i = -q·Ew·v_d`
  - Ew = weighting field (dimensionless)
  - v_d = drift velocity
  
- Integrated charge proportional to weighting potential difference:
  ```
  ∫i dt ∝ (V_w^end - V_w^start)
  ```

**Signal Distortion Source:**
- Overlapping bipolar signals from multiple electrons
- Long-range coupling between distant charges
- Position-dependent signal spreading

### Collection Plane (Y Wire)

**Field Response:**
- **Unipolar shape** - single polarity
- **Gaussian-like time profile**
- Large amplitude (2-3× induction signals)
- Simple, clean signal

**Signal Properties:**
- Collected charges produce straightforward response
- Minimal long-range coupling (shielded by induction planes)
- More robust to noise

**Comparison:**
- Collection plane: Clean, reconstructable → **primary charge extraction**
- Induction planes: Complex, detailed → **3D reconstruction, topological info**

---

## 3. Electronics Response

### Pre-amplifier Characteristics:
- **Gain settings**: 4.7 mV/fC, 7.8 mV/fC, 14 mV/fC, 25 mV/fC
- **Peaking times**: 0.5 µs, 1.0 µs, 2.0 µs, 3.0 µs (selectable)
- Impulse response approximated by RC filtering:
  ```
  Single RC:   h(t) = δ(t) - (1/τ)·e^(-t/τ)·u(t)
  RC⊗RC:       h(t) = δ(t) + (τ/τ - 2)·(1/τ)·e^(-t/τ)·u(t)
  ```
  where τ = RC time constant ≈ 1 ms in MicroBooNE

### Combined Response:
- Field response convolved with electronics impulse response
- Final signal = measured waveform at ADC input

---

## 4. Deconvolution & Software Filters (Key Section)

The paper describes signal extraction via **Deconvolution and Software Filters** (Section 3.1.1):

### Deconvolution Principle:
**Goal**: Recover original charge distribution Q(t) from measured signal S(t)

**Forward model:**
```
S(t) = Q(t) ⊗ H_total(t) + N(t)
```
where:
- H_total = Field response × Electronics response (convolution)
- N(t) = Noise (electronics + coherent noise)
- ⊗ = convolution operator

**Inverse problem:**
```
Q(t) = S(t) ⊗ H_inverse(t) - optimal filtering
```

### Wiener Filter Concept (appears in Section 3.1.1):

The Wiener filter is an **optimal linear filter** that minimizes noise while deconvolving the signal.

**Basic Wiener filter formulation** (in frequency domain):
```
H_Wiener(f) = H_inverse(f) / [1 + S_noise(f) / S_signal(f)]
```

**Key idea:**
- At frequencies where signal dominates noise: Filter ≈ full deconvolution
- At frequencies where noise dominates: Filter ≈ suppression (low-pass)
- Automatically adapts to signal/noise ratio

**Advantages:**
1. Balances deconvolution with noise suppression
2. No threshold choices (unlike hard deconvolution)
3. Smooth, stable solution
4. Works for both collection and induction planes

---

## 5. ROI (Region of Interest) Identification

### Purpose:
Extract ionization charge signals from complex waveforms (especially for induction planes).

### Challenges:
- **Induction planes**: Bipolar, overlapping signals
- **Collection planes**: Often simpler, but still need baseline and noise filtering
- **Coherent noise**: Common-mode noise across multiple wires

### ROI Definition:
A time window containing significant signal above baseline/noise level.

**Identification Strategy** (Section 3.2.3-3.2.4):
1. **Baseline estimation**: Compute median/mean of early time samples
2. **Threshold detection**: Find samples exceeding baseline + N×σ_noise
3. **Merging**: Combine adjacent hits into continuous ROIs
4. **Refinement**: Expand/contract ROI based on signal shape

### Position-Dependent Response:
- Signal amplitude and shape depend heavily on electron distance from wire
- ROI finding must account for:
  - Variable peak heights (distance-dependent)
  - Complex multipeak structures (induction planes)
  - Time smearing (diffusion during drift)

---

## 6. Collection vs. Induction Processing Strategy

### Collection Plane (Y):
```
Raw signal → Electronics baseline removal → Simple deconvolution/filter 
           → Charge extraction (more straightforward)
```

1. Clean unipolar shape → easier deconvolution
2. Can use simpler filters
3. Primary source for true charge measurement
4. Better charge resolution

### Induction Plane (U, V):
```
Raw signal → Complex deconvolution → Wiener filter for noise suppression
           → ROI identification → 2D projection information
```

1. Bipolar signals require sophisticated filtering
2. **Long-range coupling** must be handled properly
3. Useful for 3D track reconstruction
4. Position resolution along wire direction
5. More sensitive to coherent noise

---

## 7. Wiener Filter Mathematical Formulation (Section 3.1.1)

### Basic Deconvolution Problem:

**Forward model (measured signal):**
```
M(t') = ∫ R(t,t') · S(t) dt + N(t')
```
where:
- M(t') = measured ADC waveform
- R(t,t') = detector response function (field + electronics)
- S(t) = original ionization charge signal
- N(t') = noise

**In frequency domain (time-invariant R):**
```
M(ω) = R(ω) · S(ω) + N(ω)
```

**Naive inverse (problematic):**
```
S(ω) = M(ω) / R(ω)
```
Problem: Noise amplified at high frequencies where R(ω) is small!

### Classical Wiener Filter (Equation 3.4):

**Optimal filter function:**
```
F(ω) = R̄²(ω)·S²(ω) / [R̄²(ω)·S²(ω) + N²(ω)]
```

**Applied filter:**
```
S(ω) = [M(ω) / R(ω)] · F(ω)
```

where:
- R̄(ω) = averaged response function (position-averaged)
- S²(ω) = power spectral density of signal
- N²(ω) = power spectral density of noise
- F(ω) varies 0→1 based on SNR at each frequency

**Behavior:**
- High SNR frequencies: F(ω) ≈ 1 (full deconvolution)
- Low SNR frequencies: F(ω) ≈ 0 (suppress noise)
- Minimizes mean-square error (Wiener optimality)

### Issues with Naive Wiener (Section 3.1.1):

**Problem 1: Signal spectrum varies with topology**
- Different event topologies have different signal spectra
- Cannot use universal S(ω) for all events

**Problem 2: Noise spectrum depends on time window**
- Longer observation windows = more low-frequency noise content
- Cannot achieve universal Wiener filter for all signals

**Problem 3: Charge conservation violated**
- Classical Wiener doesn't conserve total ionization electrons
- Baseline shifts can lose/add charge

**Problem 4: Non-local charge smearing (induction planes)**
- Low-frequency suppression causes charge to smear in time
- Bad for induction planes (already bipolar)
- Direct application would create false charge distributions

### Modified Wiener-Inspired Filter (Equations 3.9-3.10):

To overcome these issues, the paper uses a **parametric functional form** fitted to ideal Wiener results:

**For time domain (temporal deconvolution):**
```
F(ω) = {
  c · e^(-1/2 · (ω/ωc)^b)    for ω > 0
  0                           for ω = 0
}
```

**Parameters:**
- a, b, c = free parameters determined by fitting
- ωc = characteristic frequency cutoff
- e^(-1/2·(...)) ensures smooth rolloff
- F(ω=0) = 0 removes DC baseline component

**Key modification - Zero at DC:**
```
lim F(ω) = 1   (as ω→0⁺)
```
This ensures the integral of time-domain smearing function is unity, preventing charge loss.

**Practical form used:**
```
F(ω) = e^(-1/2 · (ω/ωc)^b)  for ω > 0
```
- Parameters fitted from simulation to match ideal Wiener
- Time window: 100 µs (works well for variety of event topologies)
- Gain/peaking time: 14 mV/fC, 2 µs (nominal MicroBooNE)
- RMS spread in drift: 1 mm (from diffusion)

---

## 8. Key Mathematical Components for Implementation

### Field Response Function:
- Pre-calculated via Garfield simulation (position-dependent)
- Varies with:
  - Electron transverse position relative to wire
  - Wire plane geometry (U: ±60°, V: ±60°, Y: vertical)
  - Drift distance

### Position-Averaged Response (Equation 3.8):

For deconvolution, average over possible electron positions within wire pitch:

```
R_i = [0.5×R_i^0mm + R_i^0.3mm + R_i^0.6mm + R_i^0.9mm + R_i^1.2mm + 0.5×R_i^1.5mm] / 5
```

- 11 discrete positions per wire region
- Spacing: 0.3 mm between positions
- Range: 0 to 1.5 mm transverse distance
- Factor of 0.5 at boundaries = partial weighting

### Electronics Response:
- Impulse response from pre-amplifier + RC filters
- Characterized by peaking time and gain
- Time constant τ ≈ 1 ms (RC filtering)

### Total Response (Time Domain):
```
H_total(t) = H_field(t) ⊗ H_electronics(t)
```

### Frequency Domain Processing:
1. FFT of measured signal: M(ω)
2. FFT of average response: R̄(ω)
3. Apply filter: S(ω) = [M(ω) / R̄(ω)] · F(ω)
4. IFFT to time domain: S(t)

---

## 8. 2D Deconvolution (Section 3.1.2)

### Problem Statement:

Induction plane signals receive contributions from **multiple neighboring wires**, not just one:

```
M_i(t₀) = ∫ [R₋₁(t₀-t)·S₋₁(t) + R₀(t₀-t)·S₀(t) + R₁(t₀-t)·S₁(t) + ...] dt
```

where:
- M_i = measured signal on wire i
- R_j = response function between wire j and target wire
- S_j = true signal from wire j region

### Matrix Formulation (Equation 3.7):

In frequency domain:
```
[ M₁(ω) ]     [ R₀(ω)   R₁(ω)   ... R_{n-1}(ω) ] [ S₁(ω) ]
[ M₂(ω) ]  =  [ R₁(ω)   R₀(ω)   ... R_{n-2}(ω) ] [ S₂(ω) ]
[ ...   ]     [ ...     ...     ... ...         ] [ ...   ]
[ M_n(ω) ]    [ R_{n-1}(ω) ... R₁(ω) R₀(ω)  ] [ S_n(ω) ]
```

**Matrix properties:**
- **Toeplitz matrix** (symmetric, constant diagonals)
- R is symmetric and Toeplitz
- Can be solved efficiently via FFT/DFT

### Solution Method:

1. Apply FFT to measured signals M_i(ω) for all wires
2. Construct response matrix R in frequency domain
3. For each frequency ω:
   ```
   S(ω) = R⁻¹(ω) · M(ω)
   ```
4. Apply Wiener-inspired filter to stabilize solution
5. IFFT back to time domain

### Advantage over 1D:

- Accounts for **long-range coupling** between wires
- Essential for accurate charge extraction on induction planes
- Reduces cross-talk between neighboring signals
- Improves position resolution (can localize in wire dimension)

---

## 9. Wire Dimension Filtering (Section 3.2.2)

### Problem:

2D deconvolution must also suppress noise in the wire direction. Use Gaussian filter:

```
F(ω_w) = e^(-1/2 · (ω_w/ω_{w,c})²)
```

where:
- ω_w = "frequency" in wire number domain (Fourier transform over wire index)
- ω_{w,c} = cutoff "frequency"
- Different parameters for **induction vs collection planes**

**Time domain equivalent (smearing function):**
```
f(i) = e^(-1/2 · (i/σ_w)²)
```
where i is wire index offset, σ_w is smearing width.

---

## 10. Region of Interest (ROI) Strategy (Sections 3.1.3 & 3.2.3-3.2.4)

### The Problem: Low-Frequency Baseline Shifts

Direct application of Wiener filter suppresses low frequencies to remove noise. But:
1. Low-frequency suppression → baseline shifts
2. Baseline shifts → **charge loss/gain**
3. Especially problematic for induction planes (already bipolar → negative baselines)

### The Solution: ROI-Based Processing

**Strategy:**
1. Apply Wiener-inspired filter to entire event
2. Identify signal regions of interest (ROIs) via threshold
3. For each ROI:
   - Extract time window containing signal
   - Apply **linear baseline subtraction** at ROI edges
   - This naturally removes DC shift within ROI

### ROI Identification Steps (Figure 13 flow chart):

**For Collection Planes (Y):**
1. Calculate RMS noise from deconvolved waveform
2. Find samples exceeding threshold = 5 × RMS
3. Merge adjacent hits into continuous ROIs
4. Expand ROI slightly for full signal capture

**For Induction Planes (U, V):**
1. Apply **high-pass filters** to deconvolved signal
2. These high-pass filters identify signal regions without including baseline
3. Define loose and tight ROIs:
   - **Loose ROI**: Broader time window for initial finding
   - **Tight ROI**: Tighter bounds for final charge extraction
4. Two filters used (Figure 17):
   ```
   F_LF,loose(ω) = 1 - e^(-(ω/ω₀)^b)    (loose)
   F_LF,tight(ω) = 1 - e^(-(ω/ω₀')^b)   (tight, ω₀' > ω₀)
   ```

### RMS Threshold Calculation:

- Use **68% quantile range** relative to mean (insensitive to true signals)
- Collection plane threshold: ~300 electrons/tick
- Induction plane threshold: ~350-500 electrons/tick
- Time-dependent: varies with wire length

### ROI Refinement (Section 3.2.4):

**Connectivity check:**
- Ensure adjacent ROIs aren't split
- Merge if within 1-2 samples

**Expansion/contraction:**
- Expand to capture full signal extent
- Contract to avoid noise regions
- Balance sensitivity vs purity

### Baseline Restoration:

For induction planes, **linear baseline subtraction** between ROI edges:
```
baseline(t) = baseline_start + (baseline_end - baseline_start) × (t - t_start)/(t_end - t_start)
charge(t) = deconvolved(t) - baseline(t)
```

This locally removes DC shifts while preserving bipolar structure.

---

## 11. Complete Signal Processing Pipeline (Figure 13)

**Step 1: Excess Noise Filtering**
- Remove coherent noise sources (before deconvolution)
- Previous work [27] - not main focus here

**Step 2: 2D Deconvolution**
- Apply to all wires simultaneously
- Use position-averaged response functions
- Output: deconvolved charge spectrum

**Step 3: HF-cut Wiener Filter**
- Apply modified Wiener-inspired filter (Eq. 3.9-3.10)
- Suppress high-frequency noise
- Preserve charge distribution locally

**Step 4: RMS Calculation**
- Compute noise level from quiet portions
- Used for threshold in ROI finding

**Step 5: ROI Finding (Parallel for induction/collection)**

For **Collection plane**:
- Apply HF-cut Gaussian filter
- Simple threshold at 5× RMS
- Identify continuous signal regions

For **Induction plane**:
- Apply loose LF-cut filter
- Apply tight LF-cut filter
- Combine for robust ROI finding

**Step 6: ROI Application**
- Extract charge within identified ROI windows
- Linear baseline subtraction (induction planes)
- Final deconvolved charge spectrum

---

## 12. Filter Comparison: Wiener vs Gaussian (Section 3.2.2)

### Wiener-Inspired Filter (Modified):
```
F(ω) = e^(-1/2 · (ω/ωc)^b)  for ω > 0
```
**Characteristics:**
- Frequency-dependent suppression
- Tailored to balance SNR adaptively
- Parameters: ωc (cutoff), b (shape exponent)
- Preserves charge locally (DC suppression via ROI edges)

**Advantages:**
- Optimal for minimizing mean-square error
- Data-driven (fitted to ideal Wiener from simulation)
- Adapts to signal characteristics
- Better signal-to-noise ratio

**Time domain equivalent (smearing function):**
- Local support (minimal long-distance spreading)
- Gausssian-like envelope for smoothness

### Gaussian Filter:
```
F(ω) = e^(-1/2 · (ω/ωG)²)
```
**Characteristics:**
- Simple, fixed frequency dependence
- Uniform suppression profile
- One parameter: ωG (width)
- Used in wire dimension (Eq. 3.13)

**Advantages:**
- Simpler to implement
- Well-understood properties
- Good baseline comparison
- Useful for initial prototyping

**Disadvantages:**
- Sub-optimal SNR (not Wiener-optimal)
- Over-smooths at some frequencies
- Under-suppresses at others
- Less well-matched to actual signal statistics

### Comparison (Figure 15):

The paper shows both filters in time and frequency domains:
- **Time domain**: Wiener filters have sharper peaks, Gaussian more gradual
- **Frequency domain**: Wiener suppresses more at intermediate frequencies
- **Result**: Wiener achieves better charge resolution with cleaner waveforms

### Practical Note:
- For initial implementation, **Gaussian filters are simpler** (same form in both time and wire dimensions)
- For production, **use Wiener-inspired** for better performance
- Gaussian useful as sanity check/baseline

---

## 13. Implementation Details & Parameters

### Critical Measurement/Assumption:

**Noise Spectrum N(ω):**
- Electronics noise: relatively white at high frequencies
- **How to measure**: 
  - Extract quiet regions (no signal)
  - Compute power spectrum
  - Fit noise floor
  - Or assume white noise + coherent noise (if known)

**Signal Spectrum S(ω):**
- Depends on charge distribution
- From simulation: use MIP track (known charge deposition)
- Parameter: RMS diffusion during drift = 1 mm (MicroBooNE)

**Regularization (SNR ratio):**
- Implicit in Wiener formula: N²(ω) / [R²(ω)·S²(ω)]
- In practice: tune filter parameters for best empirical performance

### Plane-Specific Tuning:

**Collection Plane (Y):**
```
HF-cut filter parameters:
- ωc = characteristic frequency (tune for this plane)
- b = exponent (typically 1-2)
- Gaussian wire filter: σ_w small (tight spatial smoothing)

ROI threshold:
- 5 × RMS noise
- Simple threshold (signal is unipolar)
- RMS calculated from quiet regions
```

**Induction Planes (U, V):**
```
HF-cut Wiener parameters:
- Same ωc, b as collection (data-driven)
- But stronger effect due to noisier signals

Wire dimension Gaussian:
- σ_w larger (more smoothing, bipolar cancellation)

LF-cut filters for ROI finding:
- Loose: ω₀_loose (gentler high-pass)
- Tight: ω₀_tight (sharper high-pass)
- These are induction-only (not used on collection)

ROI threshold:
- 3.5 × RMS (looser, bipolar nature)
- Two filters for robustness
```

### Parameter Values from MicroBooNE (Table in Section 3.2.2):

**Deconvolution setup:**
- Time window: 100 µs (covers most signals)
- Electronics: 14 mV/fC gain, 2 µs peaking time
- Diffusion: 1 mm RMS in drift dimension

**Noise simulation:**
- Electronics white noise: baseline RMS ~350-500 e⁻/tick
- Coherent noise: modeled from previous work [27]

**Fitted filter parameters:**
- From Figure 15 / 16: Read off ωc, b values
- Collection: one set of parameters
- Induction: possibly adjusted for noisier signals

---

## 14. Practical Implementation Workflow

### Phase 1: Development (Collection Plane Only)

1. **Load/generate response functions**
   - Field responses from Garfield
   - Electronics impulse response
   - Convolve to get total response

2. **Calculate position-averaged response**
   - Average over 11 transverse positions per wire
   - Store R̄(t) and R̄(ω)

3. **Implement 2D deconvolution**
   - FFT of signals from multiple wires
   - Matrix inversion in frequency domain
   - IFFT back to time

4. **Test Gaussian filter first** (simpler)
   - Implement: apply Gaussian in frequency domain
   - Tune σ for collection plane
   - Measure charge resolution

5. **Implement Wiener-inspired filter**
   - Fit parameters to match ideal Wiener results
   - Ensure F(ω=0) = 0 and lim F(ω→0⁺)=1
   - Compare against Gaussian baseline

6. **Validate on truth**
   - Use simulated events with known true charge
   - Measure charge bias and resolution
   - Optimize threshold for ROI finding

### Phase 2: Induction Planes

1. **Apply same HF-cut filters**
   - Use same Wiener-inspired or Gaussian
   - Possibly tune parameters for noisier signals

2. **Implement 2D deconvolution**
   - Accounts for bipolar cross-coupling
   - Handle matrix inversion carefully

3. **Add LF-cut filters for ROI**
   - Tight and loose variants
   - For finding signal regions without baseline shifts

4. **ROI application**
   - Linear baseline subtraction at edges
   - Extract final charge spectrum

5. **Test on induction plane truth**
   - Compare 1D vs 2D deconvolution benefit
   - Measure position resolution

### Phase 3: Integration & Optimization

1. **Cross-plane validation**
   - Ensure Y plane charge matches U+V planes (topologically)
   - Adjust thresholds/parameters for consistency

2. **Real data testing**
   - Measure actual noise spectra
   - Adjust filter parameters if needed

3. **Performance metrics**
   - Charge resolution
   - Position resolution (from 2D deconv)
   - Bias vs true charge
   - Efficiency for signal recovery

---

## 15. Key Equations Summary (For Quick Reference)

**Measured signal (convolution):**
```
M(t) = ∫ R(t-τ) · S(τ) dτ + N(t)
```

**Frequency domain:**
```
M(ω) = R(ω) · S(ω) + N(ω)
```

**Classical Wiener filter:**
```
F(ω) = R²(ω)S²(ω) / [R²(ω)S²(ω) + N²(ω)]
S(ω) = M(ω)/R(ω) · F(ω)
```

**Modified Wiener-inspired (Eq 3.9-3.10):**
```
F(ω) = c·e^(-1/2·(ω/ωc)^b)    for ω > 0
F(ω) = 0                        for ω = 0
```

**2D matrix deconvolution:**
```
M(ω) = R(ω) · S(ω)  → solve for S(ω) = R⁻¹(ω) · M(ω)
```
(R is Toeplitz, solve via FFT)

**Position-averaged response (Eq 3.8):**
```
R_i = [0.5R^0 + R^0.3 + R^0.6 + R^0.9 + R^1.2 + 0.5R^1.5] / 5
```

**Wire dimension Gaussian (Eq 3.13):**
```
F(ω_w) = e^(-1/2·(ω_w/ω_wc)²)
```

**High-pass filter for ROI (Eq 3.14):**
```
F_LF(ω) = 1 - e^(-(ω/ω₀)^b)
```

**ROI baseline correction (linear):**
```
baseline(t) = b_start + (b_end - b_start) × (t - t_start) / (t_end - t_start)
```

---

## 16. Common Pitfalls & Solutions

**Pitfall 1: Naive Wiener amplifies noise**
- *Solution*: Use modified Wiener-inspired filter with appropriate frequency cutoff
- Ensure F(ω=0) = 0 to avoid baseline shifts

**Pitfall 2: Charge non-conservation**
- *Solution*: Apply ROI-based linear baseline correction
- Don't use low-pass filters that alter integrated charge

**Pitfall 3: Induction plane bipolar cancellation**
- *Solution*: Use 2D deconvolution (account for neighboring wires)
- Apply stronger wire-dimension smoothing for induction planes

**Pitfall 4: Threshold sensitivity**
- *Solution*: Use RMS-based thresholds (scale-invariant)
- Measure RMS from actual quiet regions
- May differ per wire length (stronger noise on long wires)

**Pitfall 5: Loss of short signals**
- *Solution*: Tune LF-cut filters carefully for ROI finding
- Use loose filter for initial ROI, tight filter for final charge

**Pitfall 6: Baseline shifts between ROIs**
- *Solution*: Always use linear baseline subtraction at ROI edges
- Critical for bipolar induction signals

---

## 17. Relationship to Your LArPix Work

### Connections to Your Project:

**Burst sequence processing:**
- Each burst integrates for `adc_hold_delay` duration
- Wiener filter works on individual ADC traces
- Consider applying per-burst if bursts are independent

**Deconvolution for smeared reconstruction:**
- Your smeared/true comparison needs charge extraction
- Wiener-like filter → better charge estimate
- 2D deconvolution → improved position resolution

**Cross-plane validation:**
- Use collection plane (simple) for baseline truth
- Compare against induction planes (complex)
- Helps validate 2D deconvolution matrix solution

**ROI concepts:**
- Your burst merging/gap handling → similar to ROI connectivity logic
- Dead-time compensation → requires clean signal extraction first
- Wiener filtering improves input to your merger algorithm

---

## References & Further Reading

**Original paper**: arxiv 1802.08709 (MicroBooNE Collaboration)
- Section 3.1: Deconvolution & software filters
- Section 3.2: Method implementation
- Section 4: Quantitative evaluation (skipped per your request)

**Key references cited in paper**:
- [29], [30]: Prior LArTPC deconvolution work
- [31]: Wiener filter application in data unfolding
- [27]: MicroBooNE electronics noise characterization
- [32]: Wiener filter theory (general reference)

---

## 9. Connection to Gaussian Filter

**Gaussian filtering** (low-pass smoothing):
```
G(t) = (1/(σ√(2π))) · exp(-t²/(2σ²))
```

**Use case:**
- Simple alternative to Wiener for noise suppression
- Less optimal than Wiener (doesn't use signal statistics)
- Useful baseline for comparison
- Parameters: σ = smoothing width

**Wiener vs Gaussian:**
- Gaussian: Fixed, uniform smoothing
- Wiener: Adaptive, frequency-dependent suppression
- Wiener better for optimizing charge extraction
- Gaussian simpler to implement/tune

---

## 10. Next Steps for Implementation

1. **Extract/measure noise spectra** from real or simulated data
2. **Calculate field responses** (use existing Garfield simulations)
3. **Implement Wiener filter** in frequency domain
4. **Test on collection plane first** (validation easier)
5. **Extend to induction planes** (more complex)
6. **Compare against Gaussian filter** (baseline)
7. **Optimize regularization** parameters per plane
8. **Evaluate charge resolution** vs true charge

---

## Reference: Key Equations

**Ramo's Theorem (Field Response):**
```
i = -q·E_w·v_d
```

**Green's Reciprocity (Integrated Charge):**
```
Q_induced = q·(V_w^end - V_w^start)
```

**Convolution Model:**
```
S(t) = Q(t) ⊗ H(t) + N(t)
```

**Wiener Filter (Frequency Domain):**
```
H_W(f) = conj(H(f)) / (|H(f)|² + S_N(f)/S_Q(f))
```

**RC Electronics Response:**
```
h(t) = δ(t) - (1/τ)·exp(-t/τ)·u(t)
```

---

## Document Notes
- **Paper**: arxiv 1802.08709 (MicroBooNE Collaboration)
- **Sections Analyzed**: 1-3 (Introduction through Reconstruction Methods)
- **Sections Skipped**: 4-5 (Performance Evaluation, as requested)
- **Focus**: Algorithm description, not quantitative results
