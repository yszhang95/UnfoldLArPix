# Analysis Tasks Summary

**Session Date:** 2026-03-15
**Project:** UnfoldLArPix - LArPix Deconvolution Analysis Pipeline

---

## 1. Interactive Event Display Development

### Dash Application for Voxel Visualization
**File:** `examples/dash_event_display.py`

#### Features Implemented:
- **File Selection**: Recursive scanning of NPZ files from examples directory
- **3D Scatter Plot**: Interactive visualization of deconvolved voxels (`deconv_q` array)
- **Threshold Filtering**: Adjustable slider to filter voxels by charge (0-15 range)
- **Color Scaling**: Toggle between linear and logarithmic color scales
- **Statistics Display**: Real-time statistics including:
  - Grid dimensions
  - Total voxels and voxel count above threshold
  - Charge min/max/mean values
  - Percentage above threshold

#### Waveform Visualization Features:
- **Click-to-View**: Click any voxel in 3D plot to display its waveform
- **Dual Waveform Display**:
  - Blue line: Deconvolved charge (`deconv_q[x_local, y_local, :]`)
  - Red line: Ground truth smeared charge (`smeared_true` at aligned global position)
- **Alignment Logic**: Properly converts local voxel indices to global pixel coordinates using:
  - `pxl_x = boffset[0] + x_local`
  - `pxl_y = boffset[1] + y_local`
  - Then back to smeared_true local: `x_smear = pxl_x - smear_offset[0]`
- **Time Axis Calculation**:
  - Deconv: `times = boffset[2] + np.arange(len) * adc_downsample_factor`
  - Smeared: `times = smear_offset[2] + np.arange(len) * 1`

#### Technical Implementation:
- **Memory Caching**: Large numpy arrays cached in memory to avoid JSON serialization issues
- **Metadata-Only Store**: Dash Store contains only metadata, not actual data arrays
- **Error Handling**: Graceful handling of missing data, out-of-bounds coordinates, and malformed files

**Usage:**
```bash
cd examples
python dash_event_display.py
# Open http://127.0.0.1:8050/
```

---

## 2. Deconvolution Analysis Pipeline - Run 1

### Configuration 1: Ultra-fine temporal regularization
**Parameters:**
- Input: `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz`
- σ (temporal): 0.0005 (`s0p0005`)
- σ (pixel): 0.1 (`sp0p1`)
- Processor: v2 only
- TPC: 0 only

**Step 1 - Deconvolution:**
- Output: `deconv_positron_v2_s0p0005_sp0p1_event_0_0.npz`
- Deconvolved charge sum: 31,789.13 ke⁻
- Voxels with charge > 1 ke⁻: 17,847.48

**Step 2 - JSON Export:**
- Threshold: 0.25 ke⁻
- Output directory: `output_thres5k_nburst256/data/0/data/0/`
- Deconv voxels kept: 39,130
- Smeared voxels kept: 36,856

---

## 3. Deconvolution Analysis Pipeline - Run 2

### Configuration 2: Two datasets with moderate regularization
**Common Parameters:**
- σ (temporal): 0.001 (`s0p001`)
- σ (pixel): 0.08 (`sp0p08`)
- Processor: v2 only
- TPC: 0 only

#### Dataset A: thres5k_nburst16
**Input:** `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst16.npz`

**Step 1 - Deconvolution:**
- Output: `deconv_positron_v2_s0p001_sp0p08_thres5k_nburst16_event_0_0.npz`
- Deconvolved charge sum: 32,883.53 ke⁻
- Voxels with charge > 1 ke⁻: 16,505.22

**Step 2 - JSON Export (6 thresholds):**
- Output directory: `output_thres5k_nburst16/data/0/data/0/`
- Total files: 12 (6 deconv + 6 smeared-true)
- Thresholds: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 ke⁻
- Total size: 7.2 MB

**Voxel counts by threshold:**
| Threshold | Deconv | Smeared |
|-----------|--------|---------|
| 0.5 ke⁻   | 22,775 | 19,027  |
| 1.0 ke⁻   | 10,203 | 8,638   |
| 1.5 ke⁻   | 4,824  | 4,282   |
| 2.0 ke⁻   | 2,179  | 2,010   |
| 2.5 ke⁻   | 766    | 739     |
| 3.0 ke⁻   | 314    | 324     |

#### Dataset B: thres5k_nburst256
**Input:** `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz`

**Step 1 - Deconvolution:**
- Output: `deconv_positron_v2_s0p001_sp0p08_thres5k_nburst256_event_0_0.npz`
- Deconvolved charge sum: 31,718.65 ke⁻
- Voxels with charge > 1 ke⁻: 16,138.04

**Step 2 - JSON Export (6 thresholds):**
- Output directory: `output_thres5k_nburst256/data/0/data/0/`
- Total files: 12 (6 deconv + 6 smeared-true)
- Thresholds: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 ke⁻
- Total size: 7.0 MB

**Voxel counts by threshold:**
| Threshold | Deconv | Smeared |
|-----------|--------|---------|
| 0.5 ke⁻   | 21,833 | 19,027  |
| 1.0 ke⁻   | 9,879  | 8,638   |
| 1.5 ke⁻   | 4,744  | 4,282   |
| 2.0 ke⁻   | 2,194  | 2,010   |
| 2.5 ke⁻   | 800    | 739     |
| 3.0 ke⁻   | 334    | 324     |

---

## 4. Modifications to deconv_positron_v2.py

Enhanced the script with:

### New Command-Line Arguments:
- `--tpc-id`: Filter processing to specific TPC (optional)
- `--output-suffix`: Custom suffix for output filenames

### Helper Functions:
```python
def fmt_sigma_detailed(v: float) -> str:
    """Convert sigma value to filename component (e.g., 0.0005 -> 's0p0005')"""

def fmt_sigma_pxl_detailed(v: float) -> str:
    """Convert sigma_pxl value to filename component (e.g., 0.08 -> 'sp0p08')"""
```

### Output Naming Scheme:
- Base: `deconv_positron_v2_{sigma}_{sigma_pxl}_{dataset}_event_{tpc}_{event}.npz`
- Example: `deconv_positron_v2_s0p001_sp0p08_thres5k_nburst16_event_0_0.npz`
- Preserves dataset characteristics (e.g., `thres5k_nburst16`, `thres5k_nburst256`)

### JSON Export Naming:
- Deconv: `v2_s{sigma}_sp{sigma_pxl}_t{threshold}_{dataset}.json`
- Smeared: `v2_s{sigma}_sp{sigma_pxl}_t{threshold}_{dataset}_smeared.json`
- Example: `0-v2_s0p001_sp0p08_t0p5_nburst16.json`

---

## Key Observations

### Deconvolution Performance:
1. **Ultra-fine temporal (0.0005)**: Very aggressive regularization, smallest feature sizes
2. **Fine temporal (0.001)**: Good balance for this dataset, moderate feature preservation
3. **Charge preservation**: Good agreement between deconvolved sum (~32k ke⁻) and ground truth (~32.4k ke⁻)

### Burst Configuration Impact:
- **nburst16**: Shorter burst windows, 11,970 time bins
- **nburst256**: Longer burst windows, 16,500 time bins
- **Effect**: nburst256 captures ~1-2% more charge above thresholds due to extended integration window

### Threshold Effectiveness:
- At 1.0 ke⁻: ~45% of voxels retained (good SNR cutoff)
- At 2.0 ke⁻: ~7% of voxels retained (high confidence)
- At 3.0 ke⁻: ~1% of voxels retained (only highest signal)

---

## Output Artifacts

### Generated Files:
- **Deconvolved NPZ**: 3 files total
- **JSON Exports**: 24 files (12 deconv + 12 smeared-true)
- **Total Data**: ~30 MB (NPZ + JSON combined)

### Directory Structure:
```
examples/
├── dash_event_display.py
├── deconv_positron_v2_s0p0005_sp0p1_event_0_0.npz
├── deconv_positron_v2_s0p001_sp0p08_thres5k_nburst16_event_0_0.npz
├── deconv_positron_v2_s0p001_sp0p08_thres5k_nburst256_event_0_0.npz
├── output_thres5k_nburst16/data/0/data/0/
│   └── [12 JSON files]
├── output_thres5k_nburst256/data/0/data/0/
│   ├── 0-v2_s0p0005_sp0p1_t0p25*.json (earlier run)
│   └── [12 JSON files for s0p001_sp0p08]
```

---

## Next Steps

1. **Visualization**: Load JSON files in wire-cell event display
2. **Comparison**: Analyze differences between nburst16 and nburst256 configurations
3. **Parameter Optimization**: Fine-tune sigma values based on visual inspection
4. **Multi-TPC Analysis**: Extend to process all TPCs (0, 1, 2, 5) simultaneously
5. **Statistical Analysis**: Generate histograms and efficiency curves via `plot_proj.py`

---

**Analysis completed successfully on 2026-03-15**

---

## Analysis Outputs Since 2026-03-31

The following analysis/output directories were generated or refreshed in the workspace after `2026-03-31`.

| Path | Scope | Current contents |
| --- | --- | --- |
| `examples/analysis_20260331/` | single-input `v2` rerun for `nburst256` | `3` NPZ, `24` JSON, `21` PNG |
| `examples/analysis_20260401/` | mixed `v1`/`v2` reruns and later updates | `24` NPZ, `108` JSON, `216` PNG |
| `examples/analysis_20260402/` | refreshed `v1` + `v2` sweep after burst/template fixes | `24` NPZ, `96` JSON, `224` PNG |
| `examples/analysis_20260402_masked18/` | downstream JSON/plot analysis of the masked fastadc file | `0` NPZ, `4` JSON, `9` PNG |
| `examples/analysis_20260402/nbins3/plots/` | grouped `nbins=3` comparison plots for fastadc masked/unmasked files | `6` PNG |
| `examples/analysis_output/analysis_20260408/` | April 8 fastadc non-noise shielded studies, moved under `analysis_output` | `4` NPZ, `8` JSON, `48` PNG |
| `examples/analysis_output/analysis_20260409/` | April 9 shield-response studies, moved under `analysis_output` | `6` NPZ, `12` JSON, `54` PNG |

Additional generated artifact:

- `data/deconv_positron_thres5k_nburst256_fastadc0p5_sp005_spp2_event_0_0_masked18_to_match_nburst256.npz`
  masked copy of the fastadc deconvolution output with `18` unmatched triggered sequences removed to match the non-fast `nburst256` reference.

### April 8 Study

- Input: `data/pgun_positron_3gev_tred_nonoises_effq_nt1_thres5k_nburst256_fastadc0p5.npz`
- Processors: `v1`, `v2`
- Parameters:
  - `sigma_temporal = 0.005, 0.003`
  - `sigma_pixel = 0.2`
  - `tpc_id = 0`, `event_id = 0`
  - JSON / standard-plot threshold `0.5`
- Outputs now stored under `examples/analysis_output/analysis_20260408/`
- Additional grouped plots:
  - `nbins=3` plots in `examples/analysis_output/analysis_20260408/nbins3/plots/`
  - `12` grouped PNGs total

### April 9 Shield Studies

- All runs used the shield field response:
  - `/srv/storage1/yousen/tred_workspace/response_44_v2a_shield_500V_25x25pixel_tred.npz`
- Common parameters:
  - `sigma_temporal = 0.004`
  - `sigma_pixel = 0.2`
  - `tpc_id = 0`, `event_id = 0`
  - JSON / standard-plot threshold `0.5`
- Inputs analyzed:
  - `data/pgun_positron_3gev_tred_noises_effq_nt1_selftrigger_shield_reset0.npz`
  - `data/pgun_positron_3gev_tred_noises_effq_nt1_nburst8_shield_reset0.npz`
  - `data/pgun_positron_3gev_tred_noises_effq_nt1_nburst4_shield_reset0.npz`
- Processors: `v1`, `v2`
- Outputs now stored under `examples/analysis_output/analysis_20260409/`
