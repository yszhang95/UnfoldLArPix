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

## Identified Inconsistencies

1. **JSON File Locations:** 
   - `v2` JSON files are located in the root of the analysis directories (copied by `run_analysis.py`).
   - `v3_wf` JSON files are missing from the root and were likely never generated successfully or were left in temporary subdirectories (e.g., `data/0/`) that are not present in the final directory structure.
2. **Naming Conventions:**
   - `v2` filenames include dataset labels extracted from input (e.g., `thres5k_nburst256`), while `v3` filenames use manual tags (e.g., `v3_wf_nonoises`).
   - JSON filenames for `v2` have a `0-` prefix (e.g., `0-v2_...json`), while diagnostic plots do not.
3. **Redundant Runs:**
   - Configuration $(\sigma_t=0.002, \sigma_p=0.2)$ for `thres5k` datasets was run on both March 18 and March 19, with results stored in different directories.
4. **Script Documentation:**
   - `run_v3_20260318_tpc0.sh` has redundant combo numbering (two different runs labeled as "combo 6").
   - `run_v3_special.sh` contains commented-out blocks and seems to overlap with testing performed in the primary v3 script.
