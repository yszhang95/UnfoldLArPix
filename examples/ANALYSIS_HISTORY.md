# LArPix Deconvolution Analysis History

This document summarizes the deconvolution analysis runs for TPC 0 data performed in March 2026.

## 1. Batch Analysis Scan (2026-03-18)
**Output Directory:** `analysis_20260318_tpc0`  
**Processor Version:** v2  
**Datasets:**
- `pgun_positron_3gev_tred_noises_effq_nt1_thres1k_nburst256.npz`
- `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz`
- `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz`
- `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger.npz`

### Configurations ($\sigma_{temporal}$, $\sigma_{pixel}$)
1. (0.005, 0.1) -> `sp005_spp1`
2. (0.005, 0.2) -> `sp005_spp2`
3. (0.004, 0.2) -> `sp004_spp2`
4. (0.003, 0.2) -> `sp003_spp2`
5. (0.003, 0.15) -> `sp003_spp15`
6. (0.004, 0.1) -> `sp004_spp1`
7. (0.002, 0.2) -> `sp002_spp2`

**Command/Script:** `redo_analysis_tpc0.sh` calling `run_analysis.py`.
```bash
python run_analysis.py \
  --sigmas 0.005 0.004 0.003 0.002 \
  --sigma-pxls 0.1 0.15 0.2 \
  --thresholds 0.5 \
  --versions v2 \
  --input-files [4 files] \
  --dest-dir analysis_20260318_tpc0 \
  --plot-dir analysis_20260318_tpc0
```

---

## 2. Waveform-Based Analysis (v3) (2026-03-18)
**Output Directory:** `analysis_20260318_tpc0`  
**Processor Version:** v3 (manual runs)  
**Datasets:** `_wf.npz` (noises and nonoises variants), `_wf_5tks.npz`

### Parameters:
- Various manual tags: `v3_wf_sp005_spp2`, `v3_wf_nonoises_sp003_spp1`, `v3_wf_5tks_nonoises_sp005_spp2`.
- Thresholds: 0.5 ke⁻ (standard) and 0.05 ke⁻ (5-tick study).

**Command/Script:** `run_v3_20260318_tpc0.sh` and `run_v3_special.sh`.
```bash
python deconv_positron_v3.py --sigma ${SigmaT} --sigma-pxl $SigmaP --input-file $InputFile
python deconv_xyz.py --threshold 0.5 --output-dir analysis_20260318_tpc0/ --prefix "${TAG}_t0p5" ...
python plot_proj.py --prefix "analysis_20260318_tpc0/${TAG}" --threshold 0.5 ...
```

---

## 3. Refined Low-Sigma Scan (2026-03-19)
**Output Directory:** `analysis_20260319_tpc0`  
**Processor Version:** v2  
**Datasets:** `thres5k_nburst256`, `thres5k_nburst4`

### Configurations:
- (0.002, 0.2) -> `sp002_spp2` (Redundant with Mar 18 Run 7)
- (0.001, 0.2) -> `sp001_spp2`

**Command/Script:** `redo_analysis_tpc0_new_test.sh`.

---

## 4. Manual Cleanup and Consolidation (2026-03-24)
**Status:** Completed  
**Action:** Manually consolidated analysis results and resolved organizational inconsistencies.

### Key Actions:
1. **JSON Consolidation:** Moved all missing or nested JSON files (including `v3_wf` variants) into their respective analysis root directories (`analysis_20260318_tpc0/`, etc.).
2. **Directory Cleanup:** Removed empty or redundant intermediate directories (e.g., `data/0/` subfolders within analysis directories).
3. **Configuration Audit:** Verified that `report_config.json` correctly maps to the existing file naming conventions.

## Resolved Inconsistencies

1. **JSON File Locations:** 
   - [Fixed] All `v2` and `v3_wf` JSON files are now located in the root of the analysis directories.
2. **Naming Conventions:**
   - [Standardized] `v2` and `v3` filenames follow a predictable pattern for the dynamic report viewer.
3. **Redundant Runs:**
   - [Resolved] Conflicting $(\sigma_t=0.002, \sigma_p=0.2)$ runs have been audited; the most recent (March 19) results are prioritized for the final report.
4. **Script Documentation:**
   - [Updated] `run_v3_20260318_tpc0.sh` identified as the primary source for waveform-based results.

---

## 5. Comprehensive Run Inventory (As of 2026-03-24)

This inventory lists all unique deconvolution runs identified across the project workspace, including exploratory and legacy results.

### 5.1 Refined Parameter Study (TPC 0, Event 0)
Located in `analysis_20260318_tpc0/` and `analysis_20260319_tpc0/`.

| Version | Dataset | $\sigma_t$ (ticks) | $\sigma_{pxl}$ (px) | NPZ Status | JSON Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | thres1k_nburst256 | 0.002 - 0.005 | 0.1, 0.15, 0.2 | Available | t=0.5 |
| v2 | thres5k_nburst256 | 0.001 - 0.005 | 0.1, 0.15, 0.2 | Available | t=0.5 |
| v2 | thres5k_nburst4 | 0.001 - 0.005 | 0.1, 0.15, 0.2 | Available | t=0.5 |
| v2 | thres5k_selftrigger | 0.002 - 0.005 | 0.1, 0.15, 0.2 | Available | t=0.5 |
| v3_wf | noises/nonoises | 0.003, 0.005 | 0.1, 0.15, 0.2 | Available | Standardized in root |

### 5.2 Coarse exploratory Scans (Root Directory)
Larger spatial regularization used for stability baseline.

| Version | Dataset | $\sigma_t$ | $\sigma_{pxl}$ | Filename Pattern |
| :--- | :--- | :--- | :--- | :--- |
| v2 | thres5k_nburst256 | 0.001 - 0.005 | 0.8 | `deconv_positron_v2_s{sig}_sp0p8_...` |
| v2 | thres5k_nburst4 | 0.002 | 0.8 | `deconv_positron_v2_s0p002_sp0p8_burst4_...` |
| v3 | baseline | 0.003 - 0.005 | 0.1 - 0.8 | `deconv_positron_v3_s{sig}_sp{sig}_...` |

### 5.3 Special/Shielding Studies
Studies investigating specific hardware or trigger conditions.

| Version | Study | Parameters | Status |
| :--- | :--- | :--- | :--- |
| v2 | `selftrigger_shield` | $\sigma_t=0.005, \sigma_p=0.2$ | NPZ in root |
| v2 | `shield_correct` | Coarse scans (sp=0.8) | JSONs in `output_matrix_shield_correct/` |
| v3_wf | `5tks_nonoises` | $\sigma_t=0.005, \sigma_p=0.2$ | 5-tick study baseline |

### 5.4 Multi-TPC Expansion (Event 0)
Batch runs covering multiple TPC IDs for geometry consistency.

*   **Dataset:** `thres1k_nburst256`, `thres5k_nburst256`, `thres5k_nburst4`, `thres5k_selftrigger`.
*   **TPCs:** 0, 1, 2, 3, 5 (Event ID: 0).
*   **Status:** Focused exclusively on **TPC 0** for final analysis; other TPC setups are ignored.
