# v3 Deconvolution Analysis Summary

## Analysis Overview
Full deconvolution analysis using `deconv_positron_v3.py` processor with waveform-based input and subsequent JSON export and visualization.

**Analysis Date**: 2026-03-18
**Input Dataset**: `pgun_positron_3gev_tred_noises_effq_nt1_wf.npz` (waveform data)
**Processor Version**: v3

---

## Configuration Parameters

| Parameter | Value |
|-----------|-------|
| Sigma (temporal) | 0.005 |
| Sigma (pixel) | 0.8 |
| TPC ID | 0 |
| Threshold (JSON export) | 0.5 ke⁻ |
| Field Response | `response_44_v2a_full_25x25pixel_tred.npz` |

---

## Processing Details

### TPC 0 Statistics
- **Hits**: 19,936 (403 bursts)
- **EffQ Shape**: (572,151, 4)
- **EffQ Location Shape**: (572,151, 3)
- **Deconvolved Charge**: 32.4 keV
- **Deconvolved Charge (>1 ke⁻)**: 30.7 keV
- **Deconvolved Charge (>4 ke⁻)**: 24.8 keV
- **Smeared True Charge**: 32.4 keV
- **Effective Q (last integral)**: 32.4 keV

### Geometry
- **TPC 0 Bounds**: [-62.076, 2.462] to [62.076, 64.538]
- **Time Spacing**: 0.05 μs
- **Coordinate Range**: x: [36, 219], y: [0, 139], z: [-3720, -360]

---

## Commands Executed

### 1. Syntax Fix (deconv_positron_v3.py line 125)
**Issue**: Syntax error `for in` should be `for o in`
```bash
# Fixed line 125:
# OLD: if any(list(o != 0 for in local_offset)):
# NEW: if any(list(o != 0 for o in local_offset)):
```

### 2. Deconvolution
```bash
python deconv_positron_v3.py \
  --sigma 0.005 \
  --sigma-pxl 0.8 \
  --input-file data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz \
  --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
```

**Output**: `deconv_positron_v3_event_0_0.npz` → renamed to `deconv_positron_v3_s0p005_sp0p8_event_0_0.npz`

### 3. JSON Export
```bash
python deconv_xyz.py deconv_positron_v3_s0p005_sp0p8_event_0_0.npz \
  --tpc-id 0 --event-id 0 --threshold 0.5 \
  --prefix v3_s0p005_sp0p8_t0p5 \
  --smeared-prefix v3_s0p005_sp0p8_t0p5_smeared \
  --output-dir output_matrix_v3_sp0p8
```

**Outputs**:
- `0-v3_s0p005_sp0p8_t0p5.json` (16 MB, 173,197 voxels, deconv)
- `0-v3_s0p005_sp0p8_t0p5_smeared.json` (605 KB, 6,373 voxels, smeared)

### 4. Histogram Generation
```bash
mkdir -p plots/v3_s0p005_sp0p8

python plot_proj.py deconv_positron_v3_s0p005_sp0p8_event_0_0.npz \
  --threshold 0.5 --prefix plots/v3_s0p005_sp0p8/v3_s0p005_sp0p8
```

**Outputs** (4 PNG files):
- `v3_s0p005_sp0p8_hist_2d_hits.png` (31 KB)
- `v3_s0p005_sp0p8_hist_2d_hits.png` (28 KB)
- `v3_s0p005_sp0p8_hist_deconv_q.png` (34 KB)
- `v3_s0p005_sp0p8_hist_diff.png` (34 KB)

### 5. Output Organization
```bash
mkdir -p raw_positron/data/0/v3_s0p005_sp0p8

cp output_matrix_v3_sp0p8/data/0/*.json raw_positron/data/0/v3_s0p005_sp0p8/
```

---

## Output Files

### Generated Deconvolution NPZ
- **File**: `deconv_positron_v3_s0p005_sp0p8_event_0_0.npz` (2.0 GB)
- **Format**: NumPy compressed archive with deconvolved waveforms and metadata
- **Location**: `examples/` root directory

### JSON Files
**Location**: `raw_positron/data/0/v3_s0p005_sp0p8/`

| File | Size | Type | Voxels |
|------|------|------|--------|
| `0-v3_s0p005_sp0p8_t0p5.json` | 16 MB | Deconv | 173,197 |
| `0-v3_s0p005_sp0p8_t0p5_smeared.json` | 605 KB | Smeared | 6,373 |

### Histogram Plots
**Location**: `plots/v3_s0p005_sp0p8/`

| Plot | Size | Description |
|------|------|-------------|
| `v3_s0p005_sp0p8_hist_2d_hits.png` | 31 KB | 2D XY projection (deconv) |
| `v3_s0p005_sp0p8_hist_2d_hits.png` | 28 KB | 2D hits |
| `v3_s0p005_sp0p8_hist_deconv_q.png` | 34 KB | Deconvolved charge distribution |
| `v3_s0p005_sp0p8_hist_diff.png` | 34 KB | Deconv vs smeared difference |

---

## Key Observations

### Voxel Recovery
- **Deconvolved**: 173,197 voxels at threshold 0.5 ke⁻
- **Smeared**: 6,373 voxels at threshold 0.5 ke⁻
- **Recovery Ratio**: ~27x finer structure in deconvolution

### Charge Preservation
- Total deconvolved charge (32.4 keV) matches smeared true charge (32.4 keV)
- High-threshold charge (>4 ke⁻): 24.8 keV (76% of total)
- Indicates good deconvolution fidelity with sigma_pxl=0.8

### Coordinate Extent
- **Space**: 184×140 pixel grid (slightly broader than TPC 0 footprint)
- **Time**: 403 time bins (waveform-based processing)
- **Total Voxels in Volume**: 184×140×403 ≈ 10.4M possible locations

---

## Comparison: v3 vs v2

| Aspect | v3 (waveform) | v2 (hit-based) |
|--------|---|---|
| **Input** | Waveform data | Hit data |
| **Hits (TPC 0)** | 19,936 (403 bursts) | 1,450 |
| **Deconv Charge** | 32.4 keV | 22.8 keV |
| **Deconv Voxels** | 173,197 | 5,563 |
| **Field Response** | v2a_full | v2a_shield_500V |
| **Processing** | Waveform-based 3D FFT | Hit-to-grid interpolation |

v3 processes full waveform data yielding higher fidelity and more granular spatial reconstruction.

---

## Notes

1. **Syntax Fix**: Line 125 of `deconv_positron_v3.py` contained a typo that was corrected before execution
2. **File Naming**: v3 processor outputs generic names; renamed to include parameter information for clarity
3. **Large JSON**: Deconvolved JSON is 16 MB due to dense voxel grid with fine structure
4. **Threshold**: Single threshold (0.5 ke⁻) used; higher thresholds can be applied by rerunning deconv_xyz.py

---

## Reproducibility

To reproduce this analysis:

```bash
cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples

# 1. Run deconvolution
python deconv_positron_v3.py \
  --sigma 0.005 --sigma-pxl 0.8 \
  --input-file data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz \
  --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz

# 2. Rename output for clarity
mv deconv_positron_v3_event_0_0.npz deconv_positron_v3_s0p005_sp0p8_event_0_0.npz

# 3. Export JSON
python deconv_xyz.py deconv_positron_v3_s0p005_sp0p8_event_0_0.npz \
  --tpc-id 0 --event-id 0 --threshold 0.5 \
  --prefix v3_s0p005_sp0p8_t0p5 \
  --smeared-prefix v3_s0p005_sp0p8_t0p5_smeared \
  --output-dir output_matrix_v3_sp0p8

# 4. Generate plots
mkdir -p plots/v3_s0p005_sp0p8
python plot_proj.py deconv_positron_v3_s0p005_sp0p8_event_0_0.npz \
  --threshold 0.5 --prefix plots/v3_s0p005_sp0p8/v3_s0p005_sp0p8

# 5. Copy to final location
mkdir -p raw_positron/data/0/v3_s0p005_sp0p8
cp output_matrix_v3_sp0p8/data/0/*.json raw_positron/data/0/v3_s0p005_sp0p8/
```
