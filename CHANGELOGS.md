# Change Logs

## Since 2026-03-31

Analysis-output summaries for this period are collected in [ANALYSIS_SUMMARY.md](/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/ANALYSIS_SUMMARY.md).

### Repository Changes

Repository changes since `2026-03-31` are concentrated in deconvolution diagnostics, plotting, waveform comparison, and output-layout cleanup.

#### Burst Processor V3

- `src/unfoldlarpix/burst_processor_v3.py`
  - requires both `template_coll` and `template_indu`
  - selects the compensation template from the merged-group max accumulated charge
- `src/unfoldlarpix/deconv_workflow.py`
  - treats `center_response` as the V3 collection response and adds `response_indu`
  - derives `response_indu` automatically from the `+/-1` pixel neighborhood around the center response, excluding the center pixel
  - defaults V3 event processing to the prepared `response_indu` when no override is supplied
  - applies the `1.2 * readout_config.threshold` default only to V3
- `src/unfoldlarpix/README_burst_processor.md`
  - documents the two-template V3 compensation flow

#### Deconvolution / Data Products

- `src/unfoldlarpix/deconv_workflow.py`
  - saves template-compensation diagnostics into the output NPZ
  - saves `hwf_block_offset` using the original merged-block offset
  - preserves the shifted `boffset` convention separately
- `src/unfoldlarpix/burst_processor.py`
  - records template-compensation anchors and debug quantities
  - fixes template-view mutation/state contamination (`view -> copy` style fix)
  - leaves an explicit `FIXME` on the V1 overlap boundary heuristic
- `src/unfoldlarpix/burst_processor_v2.py`
  - records the same template-compensation diagnostics as V1
- `examples/deconv_positron_v1.py`
  - writes `hwf_block` into the V1 NPZ
  - saves deconvolution output with the one-`adc_hold_delay` time shift by default
- `examples/deconv_positron_v3.py`
  - supports output-directory selection
  - supports TPC filtering from the CLI

#### Plotting

- `examples/plot_proj.py`
  - adds template-compensation diagnostic plots:
    - `*_hist_template_comp_peak_distance.png`
    - `*_hist2d_template_comp_peak_distance_vs_effq_peak.png`
  - adds inset statistics on the restricted peak-distance range
- `examples/plot_proj_nbins.py`
  - introduced as the configurable replacement for `plot_proj_2bin.py`
  - now rebins first and applies threshold only after rebinning
  - matches the `plot_proj.py` 1D histogram range `[-5, 5]`
  - matches the `plot_proj.py` 2D axis range `[0, 10] x [0, 10]`
  - includes a colorbar for the 2D histogram
  - now writes:
    - `*_hist_diff.png`
    - `*_hist_diff_deconv_mask.png`
    - `*_hist_2d.png`

#### Dash Comparison UI

- `examples/dash_compare_hwf.py`
  - added explicit two-file selection from the UI
  - added refresh button for the dropdown file list
  - compares:
    - merged burst sequence per channel
    - `deconv_q` plus rebinned/fine `smeared_true`
    - raw saved hit waveforms
  - adds trigger-stamp overlays on the merged plot
  - adds saved template-compensation comparison panels
  - changes transit-fraction and threshold-index views to histograms
  - removes the bottom delta-threshold plot
  - adds independent scale controls for:
    - merged burst
    - `deconv_q` / rebinned truth
    - fine `smeared_true`
    - hits
  - adds per-channel integrated true-charge display in the status card

#### Utilities / Helpers

- `examples/mask_sequences_to_reference.py`
  - utility to remove unmatched sequence-aligned rows from a source NPZ using a reference NPZ
- `examples/plot_trigger_comp_reference.py`
  - helper for comparing saved template-compensation trigger quantities against the `wf_5tks` reference file

### Commit Summary

Relevant commits in this period:

- `9f0dbc9` new plotting script; save waveform block now in `v1`
- `366188f` keep analysis artifacts in per-run directories
- `58fcf3c` shift deconvolution output back by one `adc_hold_delay`
- `a5ce937` `v3` output path and TPC filter
- `e405af2` template-compensation debug information saved
- `2485fed` temporary V1 workaround
- `2065168` align burst waveform offset with raw
- `2996591` prevent template contamination by avoiding view mutation
- `e50a9f0` waveform comparison UI
- `f9d3df0` threshold logic cleanup and extra grouped plot
- `7873585` integrated charge information in Dash
- `f2e5d57` histogram-based comparison panels
- `3736b35` separate truth scaling in visualization

### Notes

- Counts above reflect the current workspace state on `2026-04-03`.
- Several analysis directories are untracked workspace artifacts rather than versioned source files.
