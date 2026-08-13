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

## Layout (torch-only, Gaudi-style framework)

- `src/unfoldlarpix/`
  - `fwk/` — write-once EventStore, Algorithm/Service bases, registry,
    YAML job runner.
  - `algs/`, `services/` — the pipeline components (LoadEvent,
    FFTWarmStart, BuildMeasurement, BuildSupport, Solve,
    CentroidPositions, WriteCharges; compute/detector/rng services).
  - `model/` — the single (torch) ZS operator, GPU FFT warm start,
    `conventions.py` (every tick/bin/phase convention, one place).
  - `terms/`, `solve/` — objective terms (data, censor + the
    coordinatewise prox) with autograd-checked gradients; FISTA engine;
    Ladder/FinalRefit strategies on an explicit SolveState.
  - `eval/` — the universal-grid evaluation protocol.
  - `io/` — typed hits accessors (column semantics validated);
    `data_loader.py` tred NPZ IO.
  - `constrained_solver.py` — measurement building + numpy utilities
    (slimmed; the numpy solver was removed).
  - `burst_processor*.py`, `deconv_workflow.py` — template compensation
    (CPU part of the warm start).
- `configs/` — YAML job configs (`adopted_nb4.yaml`, `sparse_nb1.yaml`
  = the golden-regression references).
- `examples/` — thin CLIs (evaluation, plots), `PIPELINE.md`,
  `legacy/` (pre-framework study scripts, provenance only).
- `tests/` — unit + golden-regression tests (`pytest`; needs torch —
  run under the tred venv).
- `docs/archive/` — superseded design notes (see its README).

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
PYTHONPATH=src <tred-venv>/python -m pytest   # tests need torch
ruff check . --fix                            # lint
```
