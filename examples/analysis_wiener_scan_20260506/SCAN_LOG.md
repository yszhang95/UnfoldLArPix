# Wiener ROI Parameter Scan Log

**Run time:** 2026-05-06 19:34:16
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
| `combo_wc*_b*.png` | Per-combination 2D hist + ΔQ + ghost plots |
| `SCAN_LOG.md` | This file |
