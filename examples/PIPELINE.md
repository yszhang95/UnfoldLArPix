# ZS charge-unfolding pipeline — local run guide

End to end: **input NPZ → constrained-solver unfold → truth/reco
metrics → correlation plot → event-display projection.**

## Quick start

```bash
cd examples
./run_pipeline.sh data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz 0 0 nb4
```

Arguments: `INPUT.npz [TPC] [EVENT] [TAG]`. Outputs land in
`analysis_output/pipeline_<TAG>/`. One event takes ~30 s on GPU.

## What runs, step by step

The wrapper is a thin shell over four Python entry points. Run them by
hand if you want to change knobs:

```bash
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
IN=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz
OUT=analysis_output/pipeline_nb4
```

### 1. Unfold (the solver)

```bash
PYTHONPATH=../src $PY deconv_positron_solver.py \
  --input-file $IN --field-response $FR --tpc-id 0 \
  --alpha-ladder 1.0 0.5 0.3 --seed-cut 0.5 --soft-seed-len 2 \
  --split-trigger --pad-pixels 12 --support-eps 0.3 --support-dilate 1 \
  --beta-quiet 1.0 --ladder-iters 150 --centroid-window 1 \
  --backend torch --device cuda \
  --output-dir $OUT --output-suffix nb4
```

This is the **adopted default configuration** (rationale in
`report/FINDINGS.md`). What each knob does:

| flag | meaning |
|---|---|
| `--alpha-ladder 1.0 0.5 0.3` | strong-charge-first L1 homotopy (3 stages) |
| `--seed-cut 0.5 --soft-seed-len 2` | strong-q skeleton + soft exponential position prior |
| `--split-trigger` | model the trigger-window overshoot as its own row |
| `--pad-pixels 12` | spatial pad so ±12-pixel response coupling has room (no edge ghosts) |
| `--support-eps 0.3 --support-dilate 1` | ROI from the smoothed FFT-deconv warm start |
| `--beta-quiet 1.0` | quiet-window inequality penalty (silence = charge stayed below threshold) |
| `--ladder-iters 150` | FISTA iterations per ladder stage |
| `--centroid-window 1` | sub-bin time position = local reco centroid (halves ghost) |
| `--backend torch --device cuda` | GPU (float32, ~20× CPU); drop to `--backend numpy` if no GPU |

The output NPZ is **self-contained**: it holds the smeared truth
(`smeared_true`) alongside the reconstruction (`deconv_q`,
`deconv_q_sharp`, `deconv_q_offsets`), so the analysis steps need no
external reference. Per-charge results are in the `charges` array
(`pixel_x pixel_y t_center_tick charge_ke on_skeleton`) with the
sub-bin position already folded into `t_center_tick`.

### 2. Metrics (truth vs reco)

```bash
SOLVED=$OUT/deconv_positron_solver_nb4_event_0_0.npz
PYTHONPATH=../src $PY eval_deconv_metrics.py $SOLVED --labels nb4 \
  --universal-grid --deposit-shape gaussian --use-fitted-offsets \
  --json $OUT/metrics_nb4.json
```

Truth and reco are each rebinned onto the **universal grid** (bin edges
at global multiples of `adc_hold_delay` — reconstruction-independent, so
numbers compare across configs/events). Reco charges are deposited as
Gaussian shapes at their regressed sub-bin centres. Columns:
`int%` (integral bias), `r`/`slope` (correlation), `ghost%` split into
`gAdj` (one-voxel offsets of truth, tolerated) / `gIso` (isolated,
unphysical) with `gIsoQ` the isolated ghost charge, and `killed`
(truth charge dropped below the cut).

### 3. Correlation plot

```bash
PYTHONPATH=../src $PY corr2d_report.py $SOLVED --labels nb4 \
  --out $OUT/corr2d_nb4.png
```

2D truth-vs-reco histogram with the reco cut line and the truth-smearing
/ readout parameters annotated. Add `--group-pixels N --group-time N`
to pool into N-voxel groups, `--hist-max V` to zoom.

### 4. Event display

```bash
PYTHONPATH=../src $PY event_display_3d.py $SOLVED --labels nb4 \
  --out $OUT/event_nb4.html
```

Writes a 3-view projection PNG (truth row + reco row, all above the cut)
always; the interactive 3D HTML only if `plotly` is installed
(`pip install plotly`).

## Notes

- **Field response**: positron datasets use the `v2a` response above.
  Muon datasets use the `nogrid` response — do not mix them. Do NOT use
  the `_shield` selftrigger file (different field response).
- **Disk**: the self-contained (non-lean) output is ~1.9 GB per event
  (the dense smeared-truth array). For batch runs add `--lean-output`
  (few MB) to the solver and pass one non-lean file as
  `--truth-npz <ref>` to the metrics/plot steps.
- **No GPU?** Use `--backend numpy` (≈8 min/event, identical results).
  Needs numpy + scipy + matplotlib; the solver's numpy backend does not
  require torch.
- **Datasets** in `data/`: `..._nburst{4,16,64,256}.npz` (increasing
  window density) and `..._selftrigger.npz` (single latch, nb1).
