# UnfoldLArPix

Reconstruction of 3D ionization charge from zero-suppressed LArPix burst
readout (ND-LAr-like pixel TPC, tred simulation).

Zero suppression breaks conventional FFT deconvolution: silence is
inequality information and the recorded data are window integrals, not
dense waveforms.  The current method is a **constrained sparse solver**
in measurement space (FISTA; positivity + weighted L1 + honest
latch-window forward model), with the legacy compensated FFT
deconvolution retained as warm start and cheap first pass.

## Quick start

```bash
cd examples
./run_pipeline.sh data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz 0 0 nb4
```

One event end to end (unfold → truth/reco metrics → correlation plot →
event display) in ~30 s on GPU.  **`examples/PIPELINE.md`** documents
every stage and knob.

## Layout

- `src/unfoldlarpix/`
  - `constrained_solver.py` / `constrained_solver_torch.py` — the ZS
    solver (numpy / GPU backends): latch-window forward operator, FISTA,
    soft-ladder homotopy, censoring, sub-bin centroid estimator.
  - `burst_processor*.py`, `deconv_workflow.py`, `deconv.py` — burst
    merging, template compensation, FFT deconvolution (warm-start path;
    see `README_burst_processor.md`).
  - `data_loader.py`, `smear_truth.py` — tred NPZ IO, truth smearing.
- `examples/` — drivers, evaluation (`eval_deconv_metrics.py`,
  universal-grid protocol), plotting, `PIPELINE.md`.
- `tests/` — unit tests (`pytest`; the solver suites are current).
- `docs/archive/` — superseded design
  notes and analysis-session reports (see its README).

## Results & provenance

Findings, adopted configuration, and the complete study ledger live in
the analysis archive (not in this repo):
`/srv/storage1/yousen/analysis/charge_unfolading_ndlar/analysis_20260716_zs_fixes/report/FINDINGS.md`.

## Environments

- Repo venv (`.venv`): numpy/matplotlib — enough for tests and most
  evaluation scripts.
- GPU runs use tred's venv (torch 2.6 + CUDA):
  `PYTHONPATH=src /home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python …`
  Do **not** install torch into the repo venv.

## Development

```bash
pytest             # run tests
ruff check . --fix # lint
```
