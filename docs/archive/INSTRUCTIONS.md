# Analysis Records

`ANALYSIS_SUMMARY.md` is the canonical run log for this repository. Use it to track analysis outputs, archive locations, and cleanup decisions so the workspace does not depend on loose `examples/*.npz` files.

## Summary Format

- `Analysis Outputs Since 2026-03-31`: table of dated analysis directories, their scope, and current file counts.
- Per-study sections: input file, processor version, sigma values, threshold, `tpc_id`, `event_id`, field response, and output location.
- Loose NPZ audit: duplicate detection, intentional discards, legacy moves, and plot mappings for files that used to live directly under `examples/`.
- Archive notes: canonical long-term paths are under `/srv/storage1/yousen/analysis/charge_unfolading_ndlar/`.

## How To Use It

- Record the exact analysis command in the relevant `commands.sh` next to the output directory.
- Update `ANALYSIS_SUMMARY.md` whenever you add, rerun, move, compare, or delete analysis outputs.
- Prefer dated output directories over top-level `examples/*.npz` files.
- If a run is archived to `/srv`, keep the `/srv` path in the summary and delete the duplicate workspace copy when it is no longer needed.

## Run A Job

- Use `examples/run_analysis.py` for the standard deconvolution pipeline.
- Run it from `examples/` so relative input and output paths stay consistent.
- Pass the input NPZ, processor versions, sigma values, threshold, destination directory, plot directory, and pipeline steps explicitly.
- Save the exact command in `commands.sh` inside the output directory.

Example:

```bash
cd examples
uv run python run_analysis.py \
  --versions v1 v2 \
  --sigmas 0.001 0.002 0.005 \
  --sigma-pxls 0.2 \
  --thresholds 0.5 \
  --plot-threshold 0.5 \
  --input-files data/your_input.npz \
  --dest-dir analysis_YYYYMMDD \
  --output-matrix analysis_YYYYMMDD/output_matrix \
  --plot-dir analysis_YYYYMMDD/plots \
  --steps 1 2 3 4
```
