# Examples Scripts Guide

This file summarizes the purpose, typical usage, and current status of the
scripts under `examples/`.

## How To Read This Folder

The scripts fall into four broad groups:

1. Main pipeline entry points
2. Conversion and plotting utilities
3. Interactive visualization and diagnostics
4. Historical wrappers and exploratory scripts

If you only need the current workflow, start with:

- `run_analysis.py`
- `deconv_positron_v2.py`
- `deconv_xyz.py`
- `plot_proj.py`

## Main Pipeline Entry Points

### `run_analysis.py`

Purpose:
- Main batch driver for the deconvolution workflow.
- Runs parameter scans over multiple input files, temporal sigmas, pixel sigmas,
  thresholds, and processor versions.

Functionality:
- Step 1: run `deconv_positron_v1.py` and/or `deconv_positron_v2.py`
- Step 2: convert NPZ outputs to JSON with `deconv_xyz.py`
- Step 3: copy NPZ/JSON files into a destination directory
- Step 4: generate QA plots with `plot_proj.py`

Typical usage:

```bash
uv run python run_analysis.py \
  --sigmas 0.001 0.002 0.005 \
  --sigma-pxls 0.2 \
  --thresholds 0.5 0.2 \
  --versions v2 \
  --input-files data/file1.npz data/file2.npz \
  --dest-dir analysis_out \
  --plot-dir analysis_out/plots \
  --steps 1 2 3 4
```

Status:
- Current main orchestration script.

### `deconv_positron_v2.py`

Purpose:
- Current main deconvolution script for positron datasets.

Functionality:
- Loads a `tred` NPZ file
- Loads the field response
- Uses `BurstSequenceProcessorV2`
- Builds a 3D deconvolution block
- Runs FFT deconvolution
- Smears the true charge for comparison
- Saves a deconvolved event NPZ

Inputs:
- `--input-file`
- `--field-response`
- `--sigma`
- `--sigma-pxl`
- optional `--tpc-id`
- optional `--output-suffix`

Outputs:
- `deconv_positron_v2_<suffix>_event_<tpc>_<event>.npz`

Status:
- Current preferred event-level deconvolution path.

### `deconv_positron_v1.py`

Purpose:
- Older deconvolution script using the V1 burst processor.

Functionality:
- Same broad workflow as `deconv_positron_v2.py`
- Uses `BurstSequenceProcessor` instead of V2

Outputs:
- `deconv_positron_event_<tpc>_<event>.npz`

Status:
- Maintained for comparison against V2.

### `deconv_positron_v3.py`

Purpose:
- Experimental alternative deconvolution path using waveform-like or
  interval-averaged input instead of the main burst-processor workflow.

Functionality:
- Builds blocks directly from hit burst intervals
- Runs deconvolution and truth smearing
- Saves `deconv_positron_v3_event_<tpc>_<event>.npz`

Status:
- Experimental. Useful for comparison studies, not the main production path.

## Conversion And Plotting Utilities

### `deconv_xyz.py`

Purpose:
- Convert deconvolution NPZ outputs into Wire-Cell-style voxel JSON files.

Functionality:
- Computes physical `x,y,z` voxel coordinates from detector metadata
- Writes a thresholded JSON for `deconv_q`
- Optionally writes a second JSON for the smeared-truth volume

Typical usage:

```bash
uv run python deconv_xyz.py deconv_file.npz \
  --threshold 0.5 \
  --tpc-id 0 \
  --event-id 0 \
  --prefix v2_test \
  --smeared-prefix v2_test_smeared \
  --output-dir output_matrix
```

Status:
- Core utility used by `run_analysis.py`.

### `plot_proj.py`

Purpose:
- Main offline QA plot generator for a deconvolution NPZ.

Functionality:
- Aligns deconvolved and smeared-truth voxel blocks
- Produces histograms of differences and masked differences
- Produces 2D comparisons and peak-distance diagnostics

Typical usage:

```bash
uv run python plot_proj.py deconv_file.npz \
  --threshold 0.5 \
  --prefix plots/run1
```

Status:
- Core utility used by `run_analysis.py`.

### `plot_proj_nbins.py`

Purpose:
- Alternate plotting/validation script that groups consecutive coarse time bins.

Functionality:
- Similar role to `plot_proj.py`
- Adds `--nbins` to sum every `n` consecutive coarse time bins before plotting
- Produces grouped-bin histogram and 2D comparison plots

Status:
- Specialized comparison tool, not the default plot script.

### `plot_proj_2bin.py`

Purpose:
- Compatibility wrapper for `plot_proj_nbins.py`.

Functionality:
- Forwards the same CLI arguments to `plot_proj_nbins.py`
- Defaults `--nbins 2`

Status:
- Legacy alias kept for convenience.

### `organize_json_files.py`

Purpose:
- Reorganize output JSON files into grouped directories by filename pattern.

Functionality:
- Scans a directory tree for JSON files
- Groups regular and `_smeared` pairs
- Copies them into structured output folders
- Generates helper Python scripts for downstream event-set creation/deletion

Typical usage:

```bash
uv run python organize_json_files.py --source-dir analysis_out --dest-dir grouped_json
```

Status:
- Utility for post-processing large JSON collections.

## Interactive Visualization And Diagnostics

### `dash_event_display.py`

Purpose:
- Interactive Dash event viewer for deconvolved NPZ outputs.

Functionality:
- Browse NPZ files in `examples/`
- View 3D voxel clouds above a threshold
- Inspect residual and correlation histograms
- Plot waveforms for selected global pixels

Typical usage:

```bash
uv run python dash_event_display.py
```

Status:
- Interactive inspection tool.

### `dash_compare_hwf.py`

Purpose:
- Interactive Dash comparison of `hwf_block` waveforms from two NPZ files.

Functionality:
- Select two NPZ files containing `hwf_block`
- Align them with `hwf_block_offset`
- Compare the same global pixel across both files

Typical usage:

```bash
uv run python dash_compare_hwf.py
```

Status:
- Comparison/debug tool for waveform-block outputs.

### `burst_processing_example.py`

Purpose:
- Educational example for the burst-processing algorithm.

Functionality:
- Demonstrates manual sequence merging
- Demonstrates template compensation
- Demonstrates processing on real hit data
- Saves illustrative plots

Status:
- Best script to read when learning how burst processing works.

### `snr.py`

Purpose:
- Power-spectrum and SNR study for noisy versus true interval-averaged data.

Functionality:
- Builds measurement blocks from hit and true-hit data
- Computes 3D power spectra
- Produces SNR ratio plots and projections

Status:
- Research/diagnostic script.

### `snr2.py`

Purpose:
- Extended SNR and power-spectrum analysis script.

Functionality:
- Similar to `snr.py`
- Accepts the input filename on the command line
- Adds active-region selection and more detailed spectral projections

Typical usage:

```bash
uv run python snr2.py data/some_interval_average_file.npz
```

Status:
- More flexible diagnostic variant of `snr.py`.

### `fr_expand_from_average.py`

Purpose:
- Study the semi-reverse of field-response averaging.

Functionality:
- Expands an averaged response back onto a sub-pixel grid
- Compares reconstructed and raw quarter responses
- Saves a comparison plot and expanded NPZ

Status:
- Field-response validation/debug tool.

### `toy2d/deconv2d_demo.py`

Purpose:
- Self-contained 2D toy model for deconvolution.

Functionality:
- Builds an asymmetric synthetic kernel
- Convolves a known signal
- Demonstrates deconvolution under ideal, noisy, misaligned, and combined cases
- Saves figures into `examples/toy2d/`

Typical usage:

```bash
uv run python toy2d/deconv2d_demo.py
```

Status:
- Conceptual/demo script, independent of detector data files.

## Exploratory Deconvolution Scripts

These scripts are mostly one-off studies with hard-coded inputs and plotting.
They are useful as historical notes and for understanding how the workflow
evolved, but they are not the main production entry points.

### `deconv_example.py`

Purpose:
- Early minimal deconvolution example on `two_point_charges.npz`.

Functionality:
- Uses `hit_to_wf`
- Integrates the field-response kernel
- Runs deconvolution on a simple toy dataset

Status:
- Historical prototype.

### `deconv_example2.py`

Purpose:
- Early noisy two-point example with Gaussian regularization and waveform plots.

Functionality:
- Uses `hit_to_wf`
- Compares waveform blocks and original current
- Produces diagnostic plots

Status:
- Historical exploratory script.

### `deconv_example3.py`

Purpose:
- Refactored example using the shared workflow helpers in `src/unfoldlarpix/`.

Functionality:
- Loads muon data
- Uses `BurstSequenceProcessor`
- Runs deconvolution and truth smearing
- Saves one NPZ per event

Status:
- Useful reference for the new shared workflow.

### `deconv_example4.py`

Purpose:
- Two-point noisy example using burst processing and 3D Gaussian filtering.

Status:
- Exploratory comparison script.

### `deconv_example5.py`

Purpose:
- Muon example using burst processing with hard-coded settings.

Status:
- Exploratory comparison script.

### `deconv_example6.py`

Purpose:
- Interval-averaged muon study with alternate input files and offsets.

Status:
- Research/debug script.

## Historical Wrapper Scripts

These shell scripts mostly encode specific dated runs or parameter scans. They
are useful for reproducibility of prior analyses, but they do not introduce new
processing logic beyond calling the Python scripts above.

### `run_deconv_positron.sh`

Purpose:
- Simple wrapper that runs V1 and V2 once, exports JSON, and copies outputs.

Status:
- Early convenience wrapper.

### `run_deconv_matrix.sh`

Purpose:
- Older matrix scan over sigma and sigma-pixel combinations.

Status:
- Predecessor to `run_analysis.py`.

### `run_full_analysis.sh`

Purpose:
- Older all-in-one loop over V1/V2, plotting, JSON export, and copy steps.

Status:
- Historical predecessor to `run_analysis.py`.

### `redo_analysis_tpc0.sh`

Purpose:
- Re-run a fixed set of TPC-0 parameter combinations for several datasets.

Status:
- Dated reproducibility wrapper.

### `redo_analysis_tpc0_new_test.sh`

Purpose:
- Re-run a smaller TPC-0 test matrix on newer datasets.

Status:
- Dated reproducibility wrapper.

### `redo_analysis_tpc0_plotting.sh`

Purpose:
- Rebuild only the plotting stage for an existing TPC-0 analysis set.

Status:
- Dated plotting-only wrapper.

### `run_shield.sh`

Purpose:
- Run `run_analysis.py` on a shield-response dataset with a shield-specific
  field-response file.

Status:
- Specialized wrapper for one detector configuration.

### `run_v3_20260318_tpc0.sh`

Purpose:
- Run several `deconv_positron_v3.py` parameter points, export JSON, plot, and
  rename outputs into a dated analysis directory.

Status:
- Dated wrapper for the V3 workflow.

### `run_v3_special.sh`

Purpose:
- Similar to `run_v3_20260318_tpc0.sh`, but for a small special-case V3 scan.

Status:
- Dated/specialized wrapper.

### `analysis_20260327/run_analysis_20260327.sh`

Purpose:
- Saved command used for one specific `analysis_20260327` run.

Functionality:
- Calls `run_analysis.py` with fixed input files, sigmas, thresholds, and
  output locations.

Status:
- Reproducibility artifact for one completed analysis, not a general-purpose
  script.

## Practical Recommendation

For current work, prefer this stack:

1. `run_analysis.py` for scans and end-to-end runs
2. `deconv_positron_v2.py` for event-level deconvolution
3. `deconv_xyz.py` for JSON export
4. `plot_proj.py` for QA plots
5. `dash_event_display.py` for interactive inspection

For understanding the algorithms, read:

1. `burst_processing_example.py`
2. `deconv_example3.py`
3. `toy2d/deconv2d_demo.py`
