# Deconvolution Parameter Study: LArPix Charge Reconstruction

## 1. Introduction

This report presents a systematic study of the FFT-based deconvolution
algorithm for LArPix pixel charge reconstruction, applied to a simulated
3 GeV positron event in the 2×2 ND-LAr geometry. The deconvolution
reconstructs the 3D ionisation charge distribution from digitised ADC
hit waveforms by inverting the combined effect of the pixel field
response and the charge-sensitive amplifier (CSA) impulse response.

Two hit-to-waveform reconstruction strategies are compared:

- **V1** (`BurstSequenceProcessor`): uses dead-time compensation with
  `tau = adc_hold_delay + 24 ticks` to merge closely spaced hit
  sequences before deconvolution.
- **V2** (`BurstSequenceProcessorV2`): replaces dead-time compensation
  with template-based gap filling and fractional phase-shift alignment
  to correct sub-sample trigger jitter.

The deconvolution itself applies a 3D Wiener-type regularisation kernel
parameterised by two widths:

- **σ_t** (`sigma`): temporal regularisation width (units: ADC ticks).
  Tested values: 0.005, 0.010.
- **σ_pxl** (`sigma_pxl`): spatial (pixel) regularisation width (units:
  pixels). Tested values: 0.10, 0.15, 0.20.

This gives 6 parameter combinations × 2 processor versions = 12
deconvolution configurations. For each configuration, the resulting
voxelised charge is exported to wire-cell JSON format at six amplitude
thresholds: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 ke⁻. Smeared ground-truth
charge distributions are exported in parallel for direct comparison.

---

## 2. Methodology

### 2.1 Signal chain and deconvolution

Each LArPix pixel records a sequence of ADC values sampled at
`adc_hold_delay = 30` ticks (1.5 µs at 0.05 µs/tick). The hit
processor reconstructs a continuous waveform from the sparse hit list,
and the deconvolution proceeds in the frequency domain:

```
Q(k_x, k_y, ω) = [W(k_x, k_y, ω) / FR(k_x, k_y, ω)] × HWF(k_x, k_y, ω)
```

where FR is the full 3D field response (25×25 pixels × time), HWF is
the reconstructed hit waveform block, and W is the 3D Gaussian
regularisation kernel. The field response is integrated over each
`adc_hold_delay` window before use.

### 2.2 Ground-truth smearing

The effective charge (effq) ground truth is convolved with the same
Gaussian kernel `(σ_pxl, σ_pxl, σ_t)` that defines the deconvolution
regularisation. The resulting `smeared_true` array is then rebinned
from the fine-tick grid to the coarse `adc_hold_delay` grid by summing
all fine-tick contributions within each coarse bin. This ensures a
fair, resolution-matched comparison between deconvolved charge and
ground truth.

### 2.3 Coordinate mapping

Physical coordinates are derived from the coarse voxel indices:

```
x = anode_position − drift_direction × (tick − t_ref) × 0.16 cm/µs × 0.05 µs/tick + drtoa
y = tpc_lower[0] + (pixel_x_offset + i) × 0.4434 cm
z = tpc_lower[1] + (pixel_y_offset + j) × 0.4434 cm
```

---

## 3. Total Charge Conservation

The table below summarises the integrated deconvolved charge
(no threshold) and the reference effective charge for the dominant TPC
event (TPC 0, event 0). All charges are in ke⁻.

| σ_t  | σ_pxl | V1 deconv total | V2 deconv total | effq truth | hits total |
|------|-------|-----------------|-----------------|------------|------------|
| 0.005 | 0.10 | 34 411          | 34 201          | 32 428     | 37 108     |
| 0.005 | 0.15 | 34 509          | 34 299          | 32 428     | 37 108     |
| 0.005 | 0.20 | 34 557          | 34 347          | 32 428     | 37 108     |
| 0.010 | 0.10 | 34 411          | 34 201          | 32 428     | 37 108     |
| 0.010 | 0.15 | 34 509          | 34 299          | 32 428     | 37 108     |
| 0.010 | 0.20 | 34 557          | 34 347          | 32 428     | 37 108     |

**Observations:**

1. The total deconvolved charge is stable to within ±0.1% across both
   σ_t values for fixed σ_pxl — temporal regularisation does not
   affect integrated charge, only its spatial distribution in time.

2. Increasing σ_pxl from 0.10 to 0.20 raises the deconvolved sum by
   ~0.4%, consistent with mild boundary-padding artefacts.

3. Both processors recover ~6% more charge than the effq ground truth
   and ~8% less than the raw hit sum, bracketing the true ionisation
   yield as expected.

4. V1 consistently recovers ~200–250 ke⁻ more than V2 at identical
   σ parameters. This is attributed to the dead-time compensation in
   V1 artificially inflating the first charge of each sequence by
   adding an interpolated "missing" contribution, whereas V2 uses
   template gap-fill only.

---

## 4. Above-Threshold Charge and Voxel Yield

### 4.1 Effect of regularisation on thresholded sum

Because the deconvolution introduces negative ringing, the sum of
voxels above a positive threshold exceeds the total (signed) sum.
The quantity `sum_deconv_gt1` (sum of voxels > 1 ke⁻) provides a
measure of how much charge is scattered into high-amplitude artefacts.

| σ_t  | σ_pxl | V1 gt1 (ke⁻) | V2 gt1 (ke⁻) | smeared_true (ke⁻) |
|------|-------|--------------|--------------|-------------------|
| 0.005 | 0.10 | 36 468       | 36 313       | 32 428            |
| 0.005 | 0.15 | 38 497       | 38 679       | 32 428            |
| 0.005 | 0.20 | 39 550       | 39 934       | 32 428            |
| 0.010 | 0.10 | 44 936       | 45 935       | 32 428            |
| 0.010 | 0.15 | 47 229       | 48 697       | 32 428            |
| 0.010 | 0.20 | 48 455       | 50 377       | 32 428            |

**Key finding:** Doubling σ_t from 0.005 to 0.010 increases the
above-1-ke⁻ sum by ~30–40% relative to ground truth. This indicates
that a narrower temporal regularisation band — whilst introducing
slightly more high-frequency noise — does not produce positive-biased
ringing to the same degree. σ_t = 0.005 is preferred for charge
resolution. The σ_pxl dependence is secondary and monotonically
increasing (~4–8% per 0.05 pixel step).

### 4.2 Voxel counts vs threshold

The following table shows voxel counts after thresholding for the
dominant track event. Smeared-truth counts depend only on (σ_t, σ_pxl)
— they are identical for V1 and V2.

**V2, σ_t = 0.005:**

| Threshold (ke⁻) | sp=0.10 deconv | sp=0.10 truth | sp=0.15 deconv | sp=0.15 truth | sp=0.20 deconv | sp=0.20 truth |
|-----------------|---------------|--------------|----------------|--------------|----------------|--------------|
| 0.5             | 23 681        | 14 312       | 22 576         | 11 753       | 21 980         | 10 079       |
| 1.0             | 13 536        |  8 822       | 13 395         |  8 267       | 13 133         |  7 583       |
| 1.5             |  8 805        |  5 988       |  8 996         |  6 198       |  8 951         |  6 015       |
| 2.0             |  6 264        |  4 328       |  6 625         |  4 785       |  6 659         |  4 817       |
| 2.5             |  4 671        |  3 306       |  5 048         |  3 725       |  5 199         |  3 986       |
| 3.0             |  3 659        |  2 614       |  3 950         |  2 976       |  4 110         |  3 293       |

**V2, σ_t = 0.010:**

| Threshold (ke⁻) | sp=0.10 deconv | sp=0.10 truth | sp=0.15 deconv | sp=0.15 truth | sp=0.20 deconv | sp=0.20 truth |
|-----------------|---------------|--------------|----------------|--------------|----------------|--------------|
| 0.5             | 24 418        | 12 787       | 23 542         | 10 095       | 23 007         |  8 374       |
| 1.0             | 15 144        |  8 243       | 14 772         |  7 362       | 14 602         |  6 472       |
| 1.5             | 10 526        |  5 903       | 10 470         |  5 744       | 10 294         |  5 369       |
| 2.0             |  7 614        |  4 434       |  7 845         |  4 608       |  7 815         |  4 516       |
| 2.5             |  5 875        |  3 494       |  6 095         |  3 817       |  6 129         |  3 868       |
| 3.0             |  4 693        |  2 760       |  4 955         |  3 172       |  4 976         |  3 315       |

**Observations:**

1. At every threshold the deconvolved voxel count exceeds the
   smeared-truth count, reflecting positive-valued ringing artefacts
   that pass threshold.

2. The deconv/truth ratio at threshold 0.5 ke⁻ is 1.65 for
   (σ_t=0.005, σ_pxl=0.10) but rises to 2.75 for
   (σ_t=0.010, σ_pxl=0.20), confirming that wider regularisation
   dramatically inflates the apparent voxel occupancy.

3. At threshold 3.0 ke⁻ the ratio narrows to ~1.40–1.50, suggesting
   that high-confidence voxels (charge >> threshold) are more robust
   to parameter choice.

4. The smeared-truth voxel count *decreases* as σ_pxl increases at
   fixed threshold, because wider spatial smearing reduces peak charge
   below threshold for isolated pixels. Conversely, the deconvolved
   count shows a non-monotonic response reflecting the balance between
   charge spreading and ringing suppression.

---

## 5. V1 vs V2 Processor Comparison

| σ_t  | σ_pxl | thr | V1 voxels | V2 voxels | V2 − V1 |
|------|-------|-----|-----------|-----------|---------|
| 0.005 | 0.10 | 0.5 | 22 143    | 23 681    | +1 538  |
| 0.005 | 0.10 | 1.0 | 13 203    | 13 536    |   +333  |
| 0.005 | 0.10 | 3.0 |  3 691    |  3 659    |    −32  |
| 0.010 | 0.20 | 0.5 | 21 154    | 23 007    | +1 853  |
| 0.010 | 0.20 | 1.0 | 13 258    | 14 602    | +1 344  |
| 0.010 | 0.20 | 3.0 |  4 864    |  4 976    |   +112  |

**Observations:**

1. V2 consistently reconstructs more voxels than V1 at low-to-moderate
   thresholds (0.5–2.0 ke⁻). The fractional phase-shift alignment in
   V2 corrects trigger-jitter misalignment, recovering charge that V1
   spreads or loses during dead-time compensation.

2. At threshold 3.0 ke⁻ the difference becomes small or inverted for
   small σ parameters, implying that the high-charge core of the track
   is recovered equivalently by both processors; the differences arise
   mainly from how each handles low-charge peripheral hits.

3. The improvement of V2 over V1 is larger at σ_t = 0.010 than at
   0.005, consistent with the idea that stronger regularisation in V2
   is better exploited when temporal jitter correction is accurate.

---

## 6. Optimal Configuration

Combining the above findings, the recommended configuration for charge
reconstruction at the present stage is:

| Parameter | Recommended value | Rationale |
|-----------|------------------|-----------|
| Processor | V2 | Better jitter correction, more voxels recovered |
| σ_t       | 0.005 | Minimal ringing inflation above threshold |
| σ_pxl     | 0.10–0.15 | Balances spatial resolution and smearing |
| Threshold | 1.0–1.5 ke⁻ | deconv/truth ratio ~1.55–1.65; suppresses bulk of noise |

At the recommended working point (V2, σ_t=0.005, σ_pxl=0.10,
threshold=1.0 ke⁻), 13 536 voxels are reconstructed in the deconvolved
volume vs 8 822 in the smeared truth — a ratio of 1.53. Raising the
threshold to 1.5 ke⁻ reduces this to 8 805 / 5 988 = 1.47, near the
minimum achievable without significant signal loss.

---

## 7. Output Summary

| Output type | Location | Count |
|-------------|----------|-------|
| Deconvolved JSON (wire-cell format) | `raw_positron/data/0/` | 72 files |
| Smeared-true JSON (wire-cell format) | `raw_positron/data/0/` | 72 files |
| Total JSON files | `raw_positron/data/0/` | 148 (incl. legacy) |
| Histogram plots (per NPZ, per sigma) | `plots/` | 48 PNG files |

Each JSON filename encodes: version (v1/v2), σ_t (s005/s010),
σ_pxl (sp10/sp15/sp20), threshold (t0p5/t1p0/…/t3p0), and
`_smeared` suffix for ground truth files.

---

## 8. Conclusions

1. **Charge conservation** is excellent across all tested configurations:
   the total deconvolved charge lies within 6–7% of the effq ground
   truth for the dominant track, independent of regularisation width.

2. **Temporal regularisation** (σ_t) is the dominant driver of
   ringing artefacts. Doubling σ_t from 0.005 to 0.010 inflates the
   above-threshold voxel count by up to 40% without improving the
   deconv/truth charge ratio.

3. **Spatial regularisation** (σ_pxl) has a secondary, monotonic effect:
   larger values slightly increase ringing and reduce the smeared-truth
   occupancy at a fixed threshold.

4. **V2 processor** recovers ~7–9% more sub-threshold voxels than V1
   at threshold 0.5 ke⁻, due to fractional phase-shift jitter
   correction; at threshold ≥ 3.0 ke⁻ both processors are equivalent.

5. **Working-point recommendation**: V2, σ_t = 0.005, σ_pxl = 0.10,
   threshold = 1.0–1.5 ke⁻, yielding a deconv/truth voxel ratio of
   1.47–1.53 for the 3 GeV positron track.

## 9. Data Consolidation and Workspace Status (2026-03-24)

As of March 24, 2026, the analysis workspace has been manually consolidated to resolve previous organizational inconsistencies. All primary JSON and NPZ results, including the waveform-based **V3** datasets, are now located in the root of the respective analysis directories (`analysis_20260318_tpc0/`, `analysis_20260319_tpc0/`).

### Current Directory Structure:
- **`analysis_20260318_tpc0/`**: Primary repository for March 18 runs, including V2 and V3 results.
- **`analysis_20260319_tpc0/`**: prioritised for the refined low-sigma scan ($\sigma_t=0.001, 0.002$).
- **`report_config.json`**: Updated to map existing naming conventions for the dynamic report viewer.

The workflow for **V3** (waveform-based) analysis has been audited. Future runs using `run_v3_20260318_tpc0.sh` should ensure JSON files are moved from the intermediate `data/0/` subdirectories to the analysis root to maintain consistency with the V2 pipeline.
