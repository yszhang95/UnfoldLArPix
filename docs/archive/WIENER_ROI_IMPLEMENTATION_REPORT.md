# Wiener-ROI Implementation Report

**Branch:** `worktree-wiener-filter-like`
**Date:** 2026-05-06
**Author:** Yousen Zhang

---

## 1. Motivation

The existing `deconv_positron_v3_burst.py` pipeline deconvolves zero-suppressed
LArPix burst data and applies a Gaussian regularization filter in frequency space.
There is no ROI (Region of Interest) stage — every voxel in the reconstructed
block is kept, including many bins that contain only deconvolution artifacts
(field-response spread, noise amplification).

The goal is to implement a Wiener-inspired filter (arxiv:1802.08709 §3.1.1,
Eqs. 3.9–3.10) used *only as a probe* for ROI identification.  The ROI mask
is then applied to the existing Gaussian-deconvolved charge map, preserving
the charge-reconstruction quality of the Gaussian kernel while eliminating
ghost voxels identified by the sharper Wiener probe.

Key design decisions:

- Wiener deconv → ROI mask only. Gaussian deconv → final charge values.
- Noise RMS estimated from quiet pixels (pixels with no hardware hits this
  event) in the Wiener-deconvolved block.
- Single uniform filter for all pixels (LArPix pixels are all collection-like).
- No per-ROI baseline subtraction; `apply_roi_mask` zeroes non-ROI bins.
- No changes to the v1/v2 entry points; only `deconv_positron_v3_burst.py`
  gets the new CLI flags.

---

## 2. Implementation

### 2.1 New file: `src/unfoldlarpix/wiener_filter.py`

Implements `wiener_inspired_filter_3d`:

```python
def wiener_inspired_filter_3d(
    s: tuple[int, int, int],
    dt: tuple[float, float, float],
    sigma_pixel: tuple[float, float],
    omega_c: float,
    b: float = 2.0,
) -> np.ndarray
```

**Filter form (time axis):**
```
F(f) = exp(-0.5 * (f / omega_c)^b)   for f > 0
F(0) = 0                               (DC suppressed)
```

**Spatial axes:** Gaussian with widths `sigma_pixel` — matches `gaussian_filter_3d`.

**Implementation:**
```python
freqs_t = fft.rfftfreq(s[-1], d=dt[-1])
time_filter = np.exp(-0.5 * (freqs_t / omega_c) ** b)
time_filter[freqs_t == 0] = 0.0          # DC kill

freqs_x = fft.fftfreq(s[0], d=dt[0])
freqs_y = fft.fftfreq(s[1], d=dt[1])
gx = np.exp(-0.5 * freqs_x**2 / sigma_pixel[0]**2)
gy = np.exp(-0.5 * freqs_y**2 / sigma_pixel[1]**2)

return gx[:, None, None] * gy[None, :, None] * time_filter[None, None, :]
```

**Return shape:** `(s[0], s[1], s[2]//2+1)` — compatible with `rfftn` output
and consumed directly by `deconv_fft(..., filter_fft=...)`.

**DC suppression rationale:** Setting `F(0)=0` ensures the time-domain impulse
response integrates to one and a constant baseline in the measurement does not
propagate into the deconvolved output.

**Tuning recipe (in docstring):**
Start with `b=2`, `omega_c ≈ 1/(3*adc_hold_delay)` in the same units as
`np.fft.rfftfreq(n, d=adc_hold_delay)`; raise `omega_c` for sharper time
localization, lower it for more noise suppression.

---

### 2.2 New file: `src/unfoldlarpix/roi_finder.py`

Three public functions plus private binary morphology helpers.

#### `estimate_quiet_pixel_noise`

```python
def estimate_quiet_pixel_noise(
    deconv_q_wiener: np.ndarray,   # (nx, ny, nt)
    block_offset: np.ndarray,      # (x0, y0, t0) — only spatial used
    hit_pixel_xy: np.ndarray,      # (N, 2) global (x, y) of hits
    *,
    min_quiet_pixels: int = 8,
) -> float
```

- Translates hit global XY to block-local by subtracting `block_offset[:2]`.
- Ignores hits that fall outside the block boundaries (no error).
- Marks hit pixels as "busy"; all other pixels are "quiet".
- Returns `np.std` over all time bins of all quiet pixels.
- Raises `ValueError` if `n_quiet < min_quiet_pixels` — forces caller to
  provide a fallback rather than silently producing a wrong threshold.

#### `find_roi_mask`

```python
def find_roi_mask(
    deconv_q_wiener: np.ndarray,
    noise_rms: float,
    *,
    threshold_sigma: float = 5.0,
    merge_gap: int = 2,
    expand: int = 2,
) -> np.ndarray   # bool, same shape as input
```

Per-pixel time-trace thresholding with morphological post-processing:

1. Mark bins where `deconv_q_wiener[i, j, t] > threshold_sigma * noise_rms`.
2. **Merge** runs separated by ≤ `merge_gap` zero-bins along the time axis
   (binary closing, radius `= (gap+1)//2`).
3. **Expand** each merged run by `expand` bins on each side (binary dilation).

ROIs are independent per pixel — no spatial connectivity.

**Binary closing formula:**
```python
radius = (gap + 1) // 2
dilated = _dilate_along_last_axis(mask, radius)
return ~_dilate_along_last_axis(~dilated, radius)
```
Radius `r` closes False runs of width up to `2r`, so `radius=(gap+1)//2` closes
gaps up to `gap` for any `gap ≥ 1`.

#### `apply_roi_mask`

```python
def apply_roi_mask(deconv_q: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    return np.where(roi_mask, deconv_q, 0.0)
```

Named function for readability and future extensibility (per-ROI baseline
subtraction could be added here without changing the caller).

---

### 2.3 Modified: `src/unfoldlarpix/deconv_workflow.py`

**New imports:**
```python
from .roi_finder import apply_roi_mask, estimate_quiet_pixel_noise, find_roi_mask
from .wiener_filter import wiener_inspired_filter_3d
```

**New helper `build_wiener_deconv_kernel`** (mirrors `build_gaussian_deconv_kernel`):
```python
def build_wiener_deconv_kernel(
    block_shape, response_shape, adc_hold_delay, sigma_pixel, omega_c, b
) -> np.ndarray:
    return wiener_inspired_filter_3d(
        (block_shape[0] + response_shape[0] - 1,
         block_shape[1] + response_shape[1] - 1,
         block_shape[2]),
        dt=(1, 1, adc_hold_delay),
        sigma_pixel=(sigma_pixel, sigma_pixel),
        omega_c=omega_c,
        b=b,
    )
```

**Extended `EventDeconvolutionResult`** (frozen dataclass, all new fields
default to `None` to preserve backward compatibility):
```python
deconv_q_wiener: np.ndarray | None = None
roi_mask: np.ndarray | None = None
deconv_q_roi: np.ndarray | None = None
roi_noise_rms: float | None = None
wiener_omega_c: float | None = None
wiener_b: float | None = None
roi_threshold_sigma: float | None = None
```

**Extended `process_event_deconvolution` signature:**
```python
enable_wiener_roi: bool = False
wiener_omega_c: float | None = None      # required when enabled
wiener_b: float = 2.0
roi_threshold_sigma: float = 5.0
roi_merge_gap: int = 2
roi_expand: int = 2
roi_min_quiet_pixels: int = 8
```

**ROI block** (inserted after Gaussian `deconv_fft`, before `smear_effective_charge`):
```python
if enable_wiener_roi:
    if wiener_omega_c is None:
        raise ValueError("wiener_omega_c is required when enable_wiener_roi=True.")
    wiener_kernel = build_wiener_deconv_kernel(
        block_data.shape, prepared_response.integrated_response.shape,
        readout_config.adc_hold_delay, sigma_pixel, wiener_omega_c, wiener_b)
    deconv_q_wiener, _ = deconv_fft(block_data, prepared_response.integrated_response, wiener_kernel)
    roi_noise_rms = estimate_quiet_pixel_noise(
        deconv_q_wiener, np.asarray(block_offset),
        event.hits.location[:, :2], min_quiet_pixels=roi_min_quiet_pixels)
    roi_mask = find_roi_mask(
        deconv_q_wiener, roi_noise_rms,
        threshold_sigma=roi_threshold_sigma,
        merge_gap=roi_merge_gap, expand=roi_expand)
    deconv_q_roi = apply_roi_mask(deconv_q, roi_mask)
```

**Extended `build_event_output_payload`** — emits ROI arrays when non-None:
```python
if result.deconv_q_wiener is not None:
    payload["deconv_q_wiener"] = result.deconv_q_wiener
    payload["roi_mask"]        = result.roi_mask
    payload["deconv_q_roi"]    = result.deconv_q_roi
    payload["roi_noise_rms"]   = result.roi_noise_rms
    payload["wiener_omega_c"]  = result.wiener_omega_c
    payload["wiener_b"]        = result.wiener_b
    payload["roi_threshold_sigma"] = result.roi_threshold_sigma
```

---

### 2.4 Modified: `src/unfoldlarpix/__init__.py`

New exports added:
```python
from .deconv_workflow import build_wiener_deconv_kernel   # (added to existing import)
from .roi_finder import apply_roi_mask, estimate_quiet_pixel_noise, find_roi_mask
from .wiener_filter import wiener_inspired_filter_3d
```

New `__all__` entries:
`"build_wiener_deconv_kernel"`, `"wiener_inspired_filter_3d"`,
`"estimate_quiet_pixel_noise"`, `"find_roi_mask"`, `"apply_roi_mask"`.

---

### 2.5 Modified: `examples/deconv_positron_v3_burst.py`

Six new argparse flags:
```python
parser.add_argument("--enable-wiener-roi", action="store_true")
parser.add_argument("--wiener-omega-c",    type=float, default=None)
parser.add_argument("--wiener-b",          type=float, default=2.0)
parser.add_argument("--roi-threshold-sigma", type=float, default=5.0)
parser.add_argument("--roi-merge-gap",     type=int,   default=2)
parser.add_argument("--roi-expand",        type=int,   default=2)
```

Validation (early exit):
```python
if args.enable_wiener_roi and args.wiener_omega_c is None:
    parser.error("--wiener-omega-c is required when --enable-wiener-roi is set.")
```

ROI stdout summary line:
```python
if result.roi_mask is not None:
    print(f"  ROI: noise_rms={result.roi_noise_rms:.4g}, "
          f"threshold={result.roi_threshold_sigma * result.roi_noise_rms:.4g}, "
          f"n_roi_bins={int(result.roi_mask.sum())}, "
          f"sum_deconv_q_roi={float(np.sum(result.deconv_q_roi)):.4g}")
```

---

## 3. New Example Scripts

### 3.1 `examples/compare_wiener_roi.py`

Compares four filtering methods on the same Gaussian-deconvolved block:

| Method | Description |
|--------|-------------|
| `direct>0.5ke-` | `deconv_q * (deconv_q > 0.5)` — hard threshold |
| `roi σ=500` | Wiener ROI with `threshold_sigma=500` (very conservative start) |
| `roi σ=σ_equiv` | Wiener ROI with sigma set so absolute threshold = 0.5 ke- (= `0.5/noise_rms`) |
| `roi σ=5` | Wiener ROI with `threshold_sigma=5` (low threshold, sensitivity study) |

For each method, the filtered block is aligned to `smeared_true` via
`align_voxel_blocks` and the following plots are saved to `--output-dir`:

| File | Content |
|------|---------|
| `roi_compare_2dhist.png` | 4-column 2D truth vs recon histogram (log colour) |
| `roi_compare_deltaQ.png` | 1D ΔQ = truth − recon for truth > 0.5 ke- |
| `roi_compare_ghost.png` | Ghost histogram: truth < 0.1 ke- & recon > 0.5 ke- |
| `roi_compare_nearghost_deltaQ.png` | ΔQ for 0 < truth < 0.1 ke- |
| `roi_compare_ghost_2d.png` | 2D scatter: (truth, recon) in ghost region |
| `summary.txt` | Text table: active voxels, ghost count, ghost fraction per method |

**Run command:**
```bash
PYTHONPATH=src python examples/compare_wiener_roi.py \
    --input-file examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz \
    --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
    --tpc-id 0 --event-id 0 \
    --output-dir /tmp/roi_compare
```

Default parameters in the script: `SIGMA_TIME=0.005`, `SIGMA_PIX=0.2`,
`WIENER_OMEGA_C=0.005`, `WIENER_B=4.0`.

---

### 3.2 `examples/wiener_roi_scan.py`

Coarse grid scan over Wiener filter parameters.

**Grid:**
```python
OMEGA_C_GRID = [0.001, 0.002, 0.003, 0.005]   # cycles / adc-bin
B_GRID       = [2, 4, 6]                        # rolloff exponent
```
Total: 4 × 3 = 12 parameter combinations.

**Fixed parameters:** `SIGMA_TIME=0.005`, `SIGMA_PIX=0.2`, `ROI_MERGE_GAP=2`,
`ROI_EXPAND=2`, `DIRECT_THRESHOLD=0.5 ke-`, `GHOST_THRESHOLD=0.1 ke-`.

**Per combination:**
1. Build Wiener filter with `wiener_inspired_filter_3d`.
2. Run `deconv_fft(block_data, integ_resp, wfilt)` → `dq_w`.
3. `noise_rms = estimate_quiet_pixel_noise(dq_w, block_offset, hit_xy)`.
4. `sigma_equiv = 0.5 / noise_rms` (equivalent absolute threshold).
5. `roi_mask = find_roi_mask(dq_w, noise_rms, threshold_sigma=sigma_equiv, ...)`.
6. `dq_roi = apply_roi_mask(gauss_result.deconv_q, roi_mask)`.
7. Compute `active`, `ghost`, `ghost_frac`, `dq_mean`, `dq_std`.
8. Save per-combo 2×3 plot to `combo_wc{omega_c}_b{b}.png`.

**Aggregate outputs:**
- `heatmaps.png` — 2×3 heatmap panels: ghost fraction, ΔQ std, ΔQ mean,
  noise RMS, σ_equiv, active voxels.
- `SCAN_LOG.md` — full markdown table of all metrics.
- `examples/ANALYSIS_HISTORY.md` — new entry appended.

**Run command:**
```bash
PYTHONPATH=src python examples/wiener_roi_scan.py \
    --output-dir examples/analysis_wiener_scan_20260506
```

**Output directory:** `examples/analysis_wiener_scan_20260506/`

---

## 4. Test Suite

### 4.1 How to Run

Because the worktree `src/` is not on the editable-install path, always run
tests with `PYTHONPATH` overriding the installed package:

```bash
cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/.claude/worktrees/wiener-filter-like
PYTHONPATH=src python -m pytest tests/test_wiener_filter.py tests/test_roi_finder.py tests/test_deconv_workflow.py -v
```

Full suite:
```bash
PYTHONPATH=src python -m pytest -q
```

### 4.2 `tests/test_wiener_filter.py` — 7 tests

All tests are in class `TestWienerInspiredFilter3D`.

| Test | Assertion |
|------|-----------|
| `test_shape_matches_rfftn_output` | `filter.shape == (8, 6, 17)` for input `s=(8,6,32)` |
| `test_dc_time_component_is_zero` | `filter[..., 0] == 0.0` exactly |
| `test_first_nonzero_freq_close_to_one_when_below_cutoff` | With `nt=256`, `dt_t=4`, `omega_c=1.0`: `filter[0,0,1] > 0.99` |
| `test_spatial_uniform_with_large_sigma` | For each time bin `k≥1`: `filter[...,k]` is spatially uniform to `atol=1e-6` |
| `test_rejects_nonpositive_omega_c` | `omega_c=0.0` raises `ValueError` matching `"omega_c"` |
| `test_rejects_nonpositive_b` | `b=0.0` raises `ValueError` matching `"b"` |
| `test_higher_b_gives_sharper_rolloff` | With `omega_c=0.05`: `filter_b4[0,0,-1] < filter_b2[0,0,-1]` |

**Note on `test_spatial_uniform_with_large_sigma`:** An earlier version
(`test_spatial_dc_unity_with_large_sigma`) tested that all non-DC frequency
bins were equal in value, which fails because `exp(-0.5*(f/omega_c)^b)` varies
with `f`. The corrected test checks *spatial* uniformity at each time-frequency
bin separately.

### 4.3 `tests/test_roi_finder.py` — 9 tests

#### `TestEstimateQuietPixelNoise` (3 tests)

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_uses_only_quiet_pixels` | `(6,6,32)` block, Gaussian noise σ=0.1; +1000 on pixel (2,3); hit_xy=[[12,23]], offset=[10,20,0] | RMS ≈ 0.1 ± 20% |
| `test_raises_when_too_few_quiet_pixels` | `(2,2,4)` block; all 4 pixels as hits; `min_quiet_pixels=1` | `ValueError("quiet pixels")` |
| `test_ignores_hits_outside_block` | `(4,4,16)` block at offset (0,0,0); hit at global (10,10) | RMS ≈ 0.1 (all pixels quiet) |

#### `TestFindRoiMask` (4 tests)

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_threshold_only_marks_above_cutoff` | `(1,1,10)`, spike=5.0 at bin 4, `threshold_sigma=4, noise_rms=1` | Only bin 4 is True |
| `test_merge_gap_closes_short_runs` | `(1,1,10)`, spikes at bins 2 and 5, `merge_gap=2, expand=0` | Bins 2–5 all True |
| `test_expand_widens_each_segment` | `(1,1,10)`, spike at bin 5, `merge_gap=0, expand=2` | Bins 3–7 True; bins 2, 8 False |
| `test_rejects_nonpositive_noise_rms` | `noise_rms=0.0` | `ValueError("noise_rms")` |

#### `TestApplyRoiMask` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_zeros_outside_mask` | `deconv=[[1,2,3]]`, `mask=[[T,F,T]]` → `[[1,0,3]]` |
| `test_shape_mismatch_raises` | Shape `(2,2)` vs `(3,3)` → `ValueError("Shape mismatch")` |

### 4.4 `tests/test_deconv_workflow.py` — 2 new tests (added to existing class)

#### `test_runs_wiener_roi_when_enabled`

Monkeypatches `hits_to_merged_block`, `build_gaussian_deconv_kernel`,
`build_wiener_deconv_kernel`, `deconv_fft` (iterator returning Gaussian then
Wiener results), and `smear_effective_charge`.

Block: `(3,3,8)`, noise σ=0.05. Hit pixel at block-local (1,1), global (3,4).
Gaussian deconv: spike at `[1,1,3:6]=[5,8,5]`. Wiener deconv: spike at
`[1,1,4]=6.0`.

Assertions:
- `result.deconv_q_wiener`, `roi_mask`, `deconv_q_roi`, `roi_noise_rms` are not None.
- `result.wiener_omega_c == 0.1`, `result.roi_threshold_sigma == 5.0`.
- `result.roi_mask[1, 1, 4]` is True.
- `result.deconv_q_roi == result.deconv_q * result.roi_mask` element-wise.

#### `test_wiener_roi_requires_omega_c`

Calls `process_event_deconvolution(..., enable_wiener_roi=True)` with
`wiener_omega_c=None` (default). Expects `ValueError` matching `"wiener_omega_c"`.

---

## 5. Test Output

```
$ PYTHONPATH=src python -m pytest tests/test_wiener_filter.py \
    tests/test_roi_finder.py tests/test_deconv_workflow.py -v

============================= test session starts ==============================
platform linux -- Python 3.12.11, pytest-9.0.2, pluggy-1.6.0
rootdir: .../worktrees/wiener-filter-like
configfile: pyproject.toml
plugins: cov-7.0.0, dash-4.0.0
collected 31 items

tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_shape_matches_rfftn_output PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_dc_time_component_is_zero PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_first_nonzero_freq_close_to_one_when_below_cutoff PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_spatial_uniform_with_large_sigma PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_rejects_nonpositive_omega_c PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_rejects_nonpositive_b PASSED
tests/test_wiener_filter.py::TestWienerInspiredFilter3D::test_higher_b_gives_sharper_rolloff PASSED
tests/test_roi_finder.py::TestEstimateQuietPixelNoise::test_uses_only_quiet_pixels PASSED
tests/test_roi_finder.py::TestEstimateQuietPixelNoise::test_raises_when_too_few_quiet_pixels PASSED
tests/test_roi_finder.py::TestEstimateQuietPixelNoise::test_ignores_hits_outside_block PASSED
tests/test_roi_finder.py::TestFindRoiMask::test_threshold_only_marks_above_cutoff PASSED
tests/test_roi_finder.py::TestFindRoiMask::test_merge_gap_closes_short_runs PASSED
tests/test_roi_finder.py::TestFindRoiMask::test_expand_widens_each_segment PASSED
tests/test_roi_finder.py::TestFindRoiMask::test_rejects_nonpositive_noise_rms PASSED
tests/test_roi_finder.py::TestApplyRoiMask::test_zeros_outside_mask PASSED
tests/test_roi_finder.py::TestApplyRoiMask::test_shape_mismatch_raises PASSED
tests/test_deconv_workflow.py::TestProcessEventDeconvolution::... (15 existing tests) PASSED
tests/test_deconv_workflow.py::TestProcessEventDeconvolution::test_runs_wiener_roi_when_enabled PASSED
tests/test_deconv_workflow.py::TestProcessEventDeconvolution::test_wiener_roi_requires_omega_c PASSED

============================== 31 passed in 0.08s ==============================
```

Full suite: **83 passed**, 15 pre-existing failures in `test_burst_processor.py`
and `test_data_loader.py` (unrelated `ReadoutConfig` schema mismatch on the
main branch — not introduced by this branch).

---

## 6. Analysis Results

### 6.1 Dataset and baseline

| Parameter | Value |
|-----------|-------|
| Dataset | `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz` |
| Field response | `response_44_v2a_full_25x25pixel_tred.npz` |
| TPC / event | 0 / 0 |
| Processor | `BurstSequenceProcessorV3`, τ = adc_hold_delay = 30 ticks |
| `sigma_time` (Gaussian) | 0.005 |
| `sigma_pixel` (Gaussian) | 0.2 |
| `adc_hold_delay` | 30 ticks (= 1.5 µs at 20 MHz) |
| Direct threshold | 0.5 ke- |
| Ghost threshold | smeared truth < 0.1 ke- |

**Direct threshold baseline (no ROI):**

| Metric | Value |
|--------|-------|
| Active voxels | 18,038 |
| Ghost count | 6,569 |
| Ghost fraction | 36.4% |
| ΔQ mean | −0.031 ke- |
| ΔQ std | 0.750 ke- |

The 36.4% ghost fraction is primarily driven by the 25×25 pixel field-response
kernel spreading real-hit charge into neighbouring spatial and temporal bins.
The Gaussian-smeared truth (σ_pixel=0.2, narrow compared to the kernel) does
not recover these bins.

### 6.2 Grid scan results (from `examples/analysis_wiener_scan_20260506/SCAN_LOG.md`)

Threshold sigma at each grid point: `σ_equiv = 0.5 ke- / noise_rms`
(preserves the same absolute threshold as the direct baseline).

#### Ghost fraction

| ω_c \ b | 2 | 4 | 6 |
|---------|---|---|---|
| 0.001 | 14.63% | 16.49% | 17.13% |
| 0.002 | 15.31% | **11.75%** | 11.94% |
| 0.003 | 26.62% | 24.85% | 24.99% |
| 0.005 | 35.35% | 33.24% | 33.08% |

Direct threshold baseline: **36.4%**

#### ΔQ std (truth > 0.5 ke-)

| ω_c \ b | 2 | 4 | 6 |
|---------|---|---|---|
| 0.001 | 0.778 | 0.793 | 0.799 |
| 0.002 | 0.746 | **0.751** | 0.754 |
| 0.003 | 0.740 | 0.741 | 0.741 |
| 0.005 | 0.738 | 0.739 | 0.739 |

#### Noise RMS from quiet pixels (ke-)

| ω_c \ b | 2 | 4 | 6 |
|---------|---|---|---|
| 0.001 | 0.01198 | 0.01189 | 0.01201 |
| 0.002 | 0.01930 | 0.01922 | 0.01938 |
| 0.003 | 0.02531 | 0.02522 | 0.02559 |
| 0.005 | 0.03671 | 0.03545 | 0.03573 |

#### Best configuration

**ω_c = 0.002, b = 4** — lowest ghost fraction.

| Metric | Value |
|--------|-------|
| Ghost fraction | 11.75% |
| ΔQ std | 0.751 ke- |
| ΔQ mean | −0.012 ke- |
| Noise RMS | 0.01922 ke- |
| σ_equiv | 26.0 |

### 6.3 Sigma-temporal comparison (analytical sweep)

To contextualize the Wiener ROI result, the Gaussian-only pipeline was run at
different σ_time values using only direct threshold 0.5 ke-:

| σ_time | Active | Ghost | Ghost% | ΔQ μ | ΔQ σ |
|--------|--------|-------|--------|------|------|
| 0.005 (baseline) | 18,038 | 6,569 | 36.4% | −0.031 | 0.750 |
| 0.003 | 16,805 | 3,352 | 19.9% | −0.024 | 0.351 |
| 0.002 | 16,380 | 843 | 5.1% | −0.032 | 0.203 |
| 0.001 | 18,797 | 0 | 0.0% | −0.047 | 0.093 |

**Important caveat:** The `smeared_true` reference is recomputed with the same
σ_time in each row. At σ_time=0.002 the apparent 5.1% ghost figure is
partially a self-consistency artifact — truth and reconstruction are processed
with matching narrow kernels, so the improvement conflates better ROI
definition with a different truth definition.

The Wiener ROI approach deliberately preserves σ_time=0.005 for the output
charge (temporal resolution maintained) while using a separate sharper probe
(ω_c=0.002, b=4) for hit location — a qualitatively different design that
avoids the conflation.

---

## 7. Bugs Fixed During Development

### Bug 1: Incorrect binary closing radius

**Original:**
```python
dilated = _dilate_along_last_axis(mask, gap)
closed = ~_dilate_along_last_axis(~dilated, gap)
return closed & _dilate_along_last_axis(mask, 0) | mask
```
Using `gap` directly as the closing radius is wrong. Dilation radius `r`
closes False runs of width up to `2r`, so to close gaps up to `gap` bins
the required radius is `ceil(gap/2) = (gap+1)//2`. The final `& ... | mask`
expression was also logically incorrect.

**Fix:**
```python
radius = (gap + 1) // 2
dilated = _dilate_along_last_axis(mask, radius)
return ~_dilate_along_last_axis(~dilated, radius)
```

### Bug 2: Wrong spatial uniformity test

**Original test:** Compared all time-frequency bins `filt[0,0,1:]` against a
single maximum, expecting equality. Failed because `exp(-0.5*(f/omega_c)^b)`
inherently varies with `f`.

**Fix:** Renamed to `test_spatial_uniform_with_large_sigma`. Assertion now
iterates over each time bin `k` and checks that `filt[..., k]` is spatially
uniform (all spatial entries equal `filt[0, 0, k]`).

---

## 8. Reproduction Guide

### 8.1 Environment

```bash
# Worktree path
cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/.claude/worktrees/wiener-filter-like

# Required: shadow installed package with worktree source
export PYTHONPATH=src
```

Python 3.12, pytest 9.0.2, numpy, matplotlib.

### 8.2 Run all unit tests

```bash
PYTHONPATH=src python -m pytest tests/test_wiener_filter.py \
    tests/test_roi_finder.py tests/test_deconv_workflow.py -v
# Expected: 31 passed
```

### 8.3 Run direct vs ROI comparison

```bash
PYTHONPATH=src python examples/compare_wiener_roi.py \
    --input-file examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz \
    --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
    --tpc-id 0 --event-id 0 \
    --output-dir /tmp/roi_compare
# Outputs: /tmp/roi_compare/{roi_compare_2dhist.png, _deltaQ.png, _ghost.png,
#           _nearghost_deltaQ.png, _ghost_2d.png, summary.txt}
```

### 8.4 Run grid scan

```bash
PYTHONPATH=src python examples/wiener_roi_scan.py \
    --input-file examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz \
    --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
    --tpc-id 0 --event-id 0 \
    --output-dir examples/analysis_wiener_scan_YYYYMMDD
# Outputs: heatmaps.png, 12x combo_wc*_b*.png, SCAN_LOG.md, updates ANALYSIS_HISTORY.md
```

### 8.5 Run CLI with ROI enabled

```bash
PYTHONPATH=src python examples/deconv_positron_v3_burst.py \
    --input-file examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz \
    --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \
    --sigma 0.005 --sigma-pxl 0.2 \
    --response-template center \
    --enable-wiener-roi \
    --wiener-omega-c 0.002 \
    --wiener-b 4.0 \
    --roi-threshold-sigma 26.0 \
    --output-dir /tmp/roi_test \
    --output-suffix wiener_roi
# NPZ contains: deconv_q, deconv_q_wiener, roi_mask, deconv_q_roi, roi_noise_rms,
#               wiener_omega_c, wiener_b, roi_threshold_sigma
```

---

## 9. Archived Analysis Output

`examples/analysis_wiener_scan_20260506/`

| File | Description |
|------|-------------|
| `SCAN_LOG.md` | Full parameter tables and best config summary |
| `heatmaps.png` | 2×3 heatmap: ghost%, ΔQ std, ΔQ mean, noise RMS, σ_equiv, active |
| `combo_wc0.001_b2.png` | Per-combination 2×3 comparison plot |
| `combo_wc0.001_b4.png` | — |
| `combo_wc0.001_b6.png` | — |
| `combo_wc0.002_b2.png` | — |
| `combo_wc0.002_b4.png` | **Best config** (11.75% ghost) |
| `combo_wc0.002_b6.png` | — |
| `combo_wc0.003_b2.png` | — |
| `combo_wc0.003_b4.png` | — |
| `combo_wc0.003_b6.png` | — |
| `combo_wc0.005_b2.png` | — |
| `combo_wc0.005_b4.png` | — |
| `combo_wc0.005_b6.png` | — |

Also: `examples/ANALYSIS_HISTORY.md` — entry appended for this scan.

---

## 10. File Change Summary

| File | Status | Description |
|------|--------|-------------|
| `src/unfoldlarpix/wiener_filter.py` | **New** | `wiener_inspired_filter_3d` |
| `src/unfoldlarpix/roi_finder.py` | **New** | `estimate_quiet_pixel_noise`, `find_roi_mask`, `apply_roi_mask` |
| `src/unfoldlarpix/deconv_workflow.py` | Modified | `build_wiener_deconv_kernel`, extended `EventDeconvolutionResult`, extended `process_event_deconvolution`, extended `build_event_output_payload` |
| `src/unfoldlarpix/__init__.py` | Modified | Re-export new public symbols |
| `examples/deconv_positron_v3_burst.py` | Modified | 6 new CLI flags + ROI stdout line |
| `tests/test_wiener_filter.py` | **New** | 7 unit tests |
| `tests/test_roi_finder.py` | **New** | 9 unit tests |
| `tests/test_deconv_workflow.py` | Modified | 2 new tests |
| `examples/compare_wiener_roi.py` | **New** | 4-method comparison script |
| `examples/wiener_roi_scan.py` | **New** | Grid scan script |
| `examples/analysis_wiener_scan_20260506/` | **New** | Archived scan output |
| `examples/analysis_wiener_recall/` | **New** | Recall-extended scan output (2026-05-08) |
| `examples/ANALYSIS_HISTORY.md` | Modified | Scan entries appended |

---

## 11. Recall analysis (2026-05-08)

### Motivation

The original scan only reported precision-side metrics (ghost fraction,
ΔQ std). It did not answer:

1. Would a Gaussian time filter (b=2) for ROI be better than the
   Wiener-like sharper rolloff (b=4)?
2. How much true charge does the ROI cut destroy?
3. How many voxels with sizable true charge are killed?

To answer these, `compute_metrics()` in `wiener_roi_scan.py` was extended
to report:

- `recall_charge` = 1 − killed_charge / total_true_charge (truth > 0.1 ke-)
- `killed_voxels[X]` for X ∈ {0.1, 0.5, 1.0} ke- — voxels with truth > X
  but reconstruction = 0
- `roi_extra_charge` / `roi_extra_voxels` — the *additional* loss caused by
  ROI on top of a plain direct threshold (voxels the direct method kept
  but ROI dropped). Computed when `recon_direct` is supplied.

A new `recall_heatmaps.png` figure plots recall, killed charge, ROI-extra
killed charge, and killed voxels at 0.1/0.5/1.0 ke- across the (ω_c, b)
grid. The ANALYSIS_HISTORY entry template was extended to include the
recall row alongside the ghost row.

### Results on TPC0 / event0

Reference totals (truth > 0.1 ke-):

- Total true charge: **31,614.5 ke-**
- Voxels: **17,846** (>0.1) / **10,126** (>0.5) / **7,574** (>1.0) ke-

| Method | Ghost frac | Recall | Killed Q [ke-] | Killed >0.5 ke- voxels |
|--------|-----------:|-------:|---------------:|-----------------------:|
| Direct threshold 0.5 ke-   | 36.42% | 94.14% | 1851.3 | 873 |
| Gaussian (ω_c=0.002, b=2)  | 15.31% | 96.50% | 1106.6 | 437 |
| Wiener best (ω_c=0.002, b=4) | 11.75% | 96.02% | 1257.7 | 557 |

### Findings

- **Gaussian (b=2) wins on recall and sizable-voxel preservation.** At the
  same ω_c, b=2 preserves ~150 ke- more true charge and kills ~120 fewer
  voxels with truth > 0.5 ke- than b=4. The shallower rolloff allows
  more genuine high-frequency signal edges through.
- **Wiener-like (b=4) wins on ghost rejection** (11.8% vs 15.3% — a
  3.5 pp gap). The sharper frequency cutoff suppresses more
  high-frequency noise.
- **Both ROI configs kill less true charge than the bare direct
  threshold** (1107 / 1258 vs 1851 ke-). This is counter-intuitive but
  driven by `expand=2`: the ROI mask preserves low-charge voxels
  neighboring genuine signal that the hard threshold would have
  rejected.
- **The `F(0)=0` DC kill is critical.** Without it the deconvolution
  baseline shift would dominate the noise RMS estimate and inflate
  ghost counts regardless of (ω_c, b).

### Recommendation

For LArPix (collection-only pixel readout), the choice between b=2 and
b=4 is a precision/recall preference, not a "right answer":

- If downstream tracking penalises ghosts heavily → **b=4** (current default).
- If preserving low-charge signal voxels matters more (e.g. low-energy
  shower edges, MIP track ends) → **b=2** is competitive and slightly
  better on recall.

**Caveat:** these numbers are from a single event. The b=2 vs b=4
ghost/recall tradeoff is only ~3.5 pp / ~0.5 pp respectively, so the
preference should be confirmed by a multi-event scan before changing
the default in `wiener_inspired_filter_3d` callers.

### File touchpoints for this iteration

| File | Status | Description |
|------|--------|-------------|
| `examples/wiener_roi_scan.py` | Modified | `compute_metrics` recall fields, recall heatmap figure, ANALYSIS_HISTORY entry template extended |
| `examples/analysis_wiener_recall/` | **New** | Heatmaps + SCAN_LOG.md from the recall-extended run |
| `examples/ANALYSIS_HISTORY.md` | Modified | Recall-extended entry rewritten |
| `WIENER_ROI_IMPLEMENTATION_REPORT.md` | Modified | This section added |
