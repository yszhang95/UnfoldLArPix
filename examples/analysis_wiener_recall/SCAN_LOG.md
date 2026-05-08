# Wiener ROI Parameter Scan Log

**Run time:** 2026-05-08 15:54:54
**Dataset:** `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz`
**Field response:** `response_44_v2a_full_25x25pixel_tred.npz`
**TPC / event:** 0 / 0

## Fixed Parameters

| Parameter | Value |
|-----------|-------|
| sigma_time (Gaussian) | 0.005 |
| sigma_pixel (Gaussian) | 0.2 |
| adc_hold_delay | 30 ticks |
| direct threshold | 0.5 ke- |
| ghost threshold | 0.1 ke- |
| ROI merge_gap | 2 bins |
| ROI expand | 2 bins |
| Processor | BurstSequenceProcessorV3 |

## Direct Threshold Baseline

| Metric | Value |
|--------|-------|
| Active voxels | 18,038 |
| Ghost count | 6,569 |
| Ghost fraction | 36.418% |
| ΔQ mean | -0.0306 ke- |
| ΔQ std | 0.7495 ke- |

## Grid Scan Results

Threshold sigma at each point = `0.5 / noise_rms` (equivalent
absolute threshold = 0.5 ke-, matching the direct baseline).

### Ghost Fraction

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 14.63% | 16.49% | 17.13% |
| 0.002 | 15.31% | 11.75% | 11.94% |
| 0.003 | 26.62% | 24.85% | 24.99% |
| 0.005 | 35.35% | 33.24% | 33.08% |


### ΔQ Std (truth > 0.5 ke-)

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 0.7781 | 0.7925 | 0.7985 |
| 0.002 | 0.7461 | 0.7510 | 0.7539 |
| 0.003 | 0.7402 | 0.7407 | 0.7409 |
| 0.005 | 0.7379 | 0.7386 | 0.7385 |


### Noise RMS from quiet pixels [ke-]

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 0.01198 | 0.01189 | 0.01201 |
| 0.002 | 0.01930 | 0.01922 | 0.01938 |
| 0.003 | 0.02531 | 0.02522 | 0.02559 |
| 0.005 | 0.03671 | 0.03545 | 0.03573 |


## Recall (true charge / voxels killed by ROI cut)

Reference totals (truth > 0.1 ke-):
- Total true charge: **31614.52 ke-**
- Total voxels truth>0.1 ke-: **17,846**
- Total voxels truth>0.5 ke-: **10,126**
- Total voxels truth>1.0 ke-: **7,574**

Direct-threshold baseline (no ROI): recall = **94.14%**,
killed charge = **1851.32 ke-**, killed voxels (>0.1/0.5/1.0) =
**6,377 / 873 / 152**

### Charge recall after ROI [%]

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 93.72% | 92.90% | 92.61% |
| 0.002 | 96.50% | 96.02% | 95.89% |
| 0.003 | 97.60% | 97.31% | 97.27% |
| 0.005 | 98.57% | 98.24% | 98.26% |


### Killed true charge after ROI [ke-]

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 1986.76 | 2244.62 | 2335.97 |
| 0.002 | 1106.62 | 1257.68 | 1300.48 |
| 0.003 | 759.50 | 851.81 | 864.41 |
| 0.005 | 451.12 | 556.25 | 550.11 |


### Killed voxels (truth > 0.5 ke-, recon = 0)

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 1151 | 1333 | 1391 |
| 0.002 | 437 | 557 | 594 |
| 0.003 | 200 | 260 | 265 |
| 0.005 | 91 | 127 | 123 |


### Killed voxels (truth > 1.0 ke-, recon = 0)

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 338 | 463 | 506 |
| 0.002 | 48 | 79 | 91 |
| 0.003 | 27 | 32 | 35 |
| 0.005 | 27 | 27 | 28 |


### ROI-extra killed charge [ke-] (direct kept, ROI dropped)

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 1150.96 | 1390.05 | 1481.98 |
| 0.002 | 425.79 | 553.42 | 594.35 |
| 0.003 | 173.14 | 234.47 | 244.64 |
| 0.005 | 6.18 | 52.55 | 53.45 |


### ROI-extra killed voxels (truth>0.5 ke-)

| ω_c \ b | 2 | 4 | 6 |
|-----------|---|---|---|
| 0.001 | 848 | 1018 | 1077 |
| 0.002 | 247 | 351 | 385 |
| 0.003 | 65 | 107 | 113 |
| 0.005 | 2 | 17 | 15 |


## Best Configuration

**Lowest ghost fraction:** ω_c = 0.002,  b = 4
Ghost fraction: 11.752%
ΔQ std: 0.7510 ke-
ΔQ mean: -0.0119 ke-
Noise RMS: 0.01922 ke-
Sigma equiv to 0.5 ke-: 26.0

## Output Files

| File | Description |
|------|-------------|
| `heatmaps.png` | Ghost fraction, ΔQ std/mean, noise RMS, sigma equiv, active voxels |
| `recall_heatmaps.png` | Recall, killed charge, killed voxels at 0.1/0.5/1.0 ke- |
| `combo_wc*_b*.png` | Per-combination 2D hist + ΔQ + ghost plots |
| `SCAN_LOG.md` | This file |
