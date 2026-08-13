# ZS charge-unfolding pipeline — local run guide (framework edition)

End to end: **input NPZ → config-driven unfold → truth/reco metrics →
correlation plot → event display.**

## Quick start

```bash
cd examples
./run_pipeline.sh data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz 0 nb4
```

Outputs land in `analysis_output/pipeline_<TAG>/`.  One event runs in
~10 s on GPU (framework unfold ~9 s + evaluation).

## The framework

Reconstruction is a **configured sequence of algorithms** exchanging
data through a write-once event store (Gaudi-style).  A job is a YAML
file; run it directly with:

```bash
PYTHONPATH=src <tred-venv>/python -m unfoldlarpix.fwk.runner configs/adopted_nb4.yaml
```

The two reference configs (also the golden-regression configs):

- `configs/adopted_nb4.yaml` — dense regime: soft-ladder (α 1.0/0.5/0.3),
  no explicit term (`terms: []`), 150 iters, centroid w1.
- `configs/sparse_nb1.yaml` — sparse regime: + censor term (L2 hinge,
  margin 3 ke), 600 iters, centroid w2.

Anatomy of a config:

```yaml
services:
  compute:  {device: cuda, dtype: float32}   # or cpu / float64
  detector: {response: /srv/.../response_44_v2a_full_25x25pixel_tred.npz}
  rng:      {seed: 0}
sequence:
  - LoadEvent:        {input: <data.npz>, tpc: 0, max_events: 1}
  - FFTWarmStart:     {sigma_time: 0.005, sigma_pixel: 0.2, pad_pixels: 12}
  - BuildMeasurement: {split_trigger: true}
  - BuildSupport:     {eps: 0.3, dilate: 1, smooth_first: true}
  - Solve:
      engine:   {iters: 150}
      strategy: {type: ladder, alphas: [1.0, 0.5, 0.3], seed_cut: 0.5, soft_len: 2.0}
      terms: []                               # DataFidelity is implicit
        # - {type: censor, beta: 1.0, margin: 3.0, norm: l2}   # sparse regime
  - CentroidPositions: {window: 1}
  - WriteCharges:      {out_dir: ..., prefix: ..., embed_truth: true}
```

Every output NPZ embeds the resolved config, the git commit, and the
store provenance — results are self-describing.

`embed_truth: true` makes the output self-contained (smeared truth
included; ~1.9 GB/event) so metrics and plots need no external
reference.  Omit it for lean production outputs and pass
`--truth-npz <ref>` to the evaluation instead.

## Evaluation and plots (unchanged CLIs)

```bash
SOLVED=analysis_output/pipeline_nb4/nb4_event_0_0.npz
PYTHONPATH=../src $PY eval_deconv_metrics.py $SOLVED --labels nb4 \
  --universal-grid --deposit-shape gaussian --use-fitted-offsets
PYTHONPATH=../src $PY corr2d_report.py $SOLVED --labels nb4 --out corr2d.png
PYTHONPATH=../src $PY event_display_3d.py $SOLVED --labels nb4 --out event.html
```

The universal-grid protocol lives in the package
(`unfoldlarpix.eval`); the examples CLIs are thin shims.  Metrics:
`int%` (integral bias), `r`/`slope`, `ghost%` split into `gAdj`
(one-voxel offsets, tolerated) / `gIso` (isolated, unphysical) with
`gIsoQ` the isolated charge, `killed` (truth dropped below the cut).

## Golden regression

`tests/golden/golden_metrics.json` pins the headline metrics of both
reference configs.  After any solver-affecting change, re-run the two
configs and check with `tests/golden/golden_gate.compare_to_golden`.

## Notes

- **Field response**: positron datasets use the v2a response; muons use
  nogrid; never use the `_shield` selftrigger file (different response).
- **No GPU**: set `compute: {device: cpu, dtype: float64}` (slower;
  identical numerics were verified for the operator and warm start).
- **Legacy**: the pre-framework driver and its study scripts live in
  git history and `examples/legacy/` (provenance for the FINDINGS
  ledger); the numpy solver was removed in the torch-only refactor.
