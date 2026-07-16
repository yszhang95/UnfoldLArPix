# Archive notes — muon-derived time-axis filter study

The large deconvolution outputs from this study are **git-ignored** (`*.npz`,
`*.png`) and total ~10 GB. They are not committed; archive them to shared
storage under the lab convention `analysis_<YYYYMMDD>_<label>`.

- **Generated:** 2026-06-07
- **Produced by commit:** `55068dc` (feat: muon-derived time-axis filter to correct template compensation)
- **Scripts:** `examples/deconv_positron_v3_burst.py`, `examples/build_muon_filter.py`, `examples/muon_filter_report.py`
- **Naming:** `s0p005` = sigma_time 0.005, `sp0p2` = sigma_pxl 0.2

## Destination

```
/srv/storage1/yousen/analysis/charge_unfolading_ndlar/analysis_20260607_muon_time_filter/
├── muon_out/   # muon deconv — INPUTS used to build the filter |H(f)|   (956 MB)
├── pos_out/    # positron deconv — RESULTS incl. muon-filter variants   (9.1 GB)
└── report/     # filter artifacts (*.npz), validation figures, write-up (2.1 MB)
```

Grouped under one dated study dir because the three directories are one
coherent analysis. (Alternative: split into `..._muon_timefilter_build` and
`..._positron_muonfilt` if you prefer per-directory dates.)

## Contents

### muon_out/ — filter-build inputs (from muon pgun data)
| file | size | what it is |
|---|---|---|
| `deconv_positron_v3_event_0_0.npz` | 342 MB | continuous readout (v3, no template) → reference `S_v3` |
| `deconv_positron_v3_burst_s0p005_sp0p2_event_0_0.npz` | 328 MB | template-compensated (v3_burst), nogrid field response → `S_v3burst` |
| `deconv_positron_v3_burst_s0p005_sp0p2_v2a_event_0_0.npz` | 332 MB | v3_burst, v2a field-response variant |

### pos_out/ — positron results being evaluated
| file | size | what it is |
|---|---|---|
| `deconv_positron_v3_event_0_0.npz` | 2.06 GB | continuous reference (v3) |
| `deconv_positron_v3_burst_s0p005_sp0p2_event_0_0.npz` | 1.93 GB | compensated, **no** filter |
| `deconv_positron_v3_burst_s0p005_sp0p2_muonfilt_event_0_0.npz` | 1.93 GB | compensated + muon filter (readout-mode) |
| `deconv_positron_v3_burst_s0p005_sp0p2_truth_muonfilt_event_0_0.npz` | 1.93 GB | compensated + muon filter (truth-mode) |
| `deconv_positron_v3_burst_s0p005_sp0p2_truthv2a_muonfilt_event_0_0.npz` | 1.93 GB | compensated + muon filter (truth-mode, v2a response) |

### report/ — deliverables (keep these; the *.npz are the actual filters)
| file | what it is |
|---|---|
| `muon_time_filter{,_truth,_truth_v2a}.npz` | the derived `|H(f)|` filters (readout / truth / truth-v2a modes) |
| `muon_time_filter*.png` | filter plots |
| `muonfilt_fig{1..4}*.png` | validation figures (readout-mode) |
| `muonfilt_truth_fig{1..4}*.png` | validation figures (truth-mode) |
| `muonfilt_truthv2a_fig{1..4}*.png` | validation figures (truth-mode, v2a) |
| `MUON_FILTER_REPORT.md` | write-up |
| `ARCHIVE_NOTES.md` | this file |

## Archive commands

Source and destination are on different filesystems (`/home` vs `/srv`), so
copy-verify-then-delete rather than `mv`. Run from the worktree root:

```bash
DEST=/srv/storage1/yousen/analysis/charge_unfolading_ndlar/analysis_20260607_muon_time_filter
mkdir -p "$DEST"

# 1. copy (report/ excludes the git-tracked .md docs — keep those in the repo)
rsync -ah --progress muon_out "$DEST"/
rsync -ah --progress pos_out  "$DEST"/
rsync -ah --progress --exclude='MUON_FILTER_REPORT.md' --exclude='ARCHIVE_NOTES.md' report "$DEST"/

# 2. verify (dry-run should report no differences)
rsync -ahn --delete --itemize-changes muon_out "$DEST"/muon_out/../  # inspect output is empty
diff <(cd muon_out && sha1sum * | sort) <(cd "$DEST"/muon_out && sha1sum * | sort) && echo "muon_out OK"
diff <(cd pos_out && sha1sum * | sort) <(cd "$DEST"/pos_out && sha1sum * | sort) && echo "pos_out OK"

# 3. only after verification passes, reclaim local space
rm -rf pos_out muon_out
# keep report/ locally (it's small and holds the committed .md + the filter npz)
```
