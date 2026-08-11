# iso50: isochronous-track lifetime study

Ten isochronous 3 GeV muon tracks (along +z at fixed drift depth, angle 0),
depths d = 1.5 .. 28.5 cm from the anode in 3 cm steps, 50 independent
copies per depth, electron lifetime 1.0 ms. Question: does the unfolded
charge (`deconv_q_sharp`) give a better lifetime estimate than raw hits?

## Inputs / provenance

| Stage | Path | Revision |
|---|---|---|
| Geant4 primaries | `~/Documents/NDLAr2x2/MuonLArSim/pgun_mu_3gev_iso50_d*.mac` + `merge_iso50.py` + `iso50_list.txt` | MuonLArSim `c6b245c` |
| tred fullsim | `tred_worktree/pgun_far_field/tests/pgun_farfield/config_iso_lifetime1ms.yaml` (study) / `config_iso_lifetime20ms.yaml` (amplitude null) | pgun_farfield `c7f17cc` |
| Solver | this repository, branch `zs-solver` | `2d1d853` (includes the hits-source support) |

tred runs execute on wcgpu0 (dual 4090; the local 4070Ti OOMs), e.g.

    uv run tred -c config_iso_lifetime1ms.yaml fullsim \
        -i ~/Documents/NDLAr2x2/MuonLArSim/pgun_mu_3gev_iso50_d01p5.hdf5 \
        -o pgun_mu_3gev_iso50_d01p5_tred_nb1.npz

`effq_out_nt: 30` stores effq per 1.5 us. That is lossless for everything
here: every dQ/dx quantity is time-integrated per pixel, and time smearing
commutes out of the integral.

## Order of operations

Interpreter: `tred_worktree/pgun_far_field/.venv/bin/python` (needs
sklearn + scipy). Run everything from this directory.

1. `run_iso50_solve.py W NW [response_override] [outdir_override]` --
   sharded solver driver, arms B (ladder + fractional censor) and
   C (B + refit eps 0.5). Outputs
   `analysis_output/iso50/{B,C}/<tag>/<tag>_event_0_<ev>.npz`
   (wcgpu0 shards write to `~/iso50_staging/{B,C}`; the analysis searches
   both).
2. `iso50_analyse.py` -- THE result of record: track-PCA + 3 cm projected
   segments, pooled MPV per depth, ln(MPV) vs drift time, bootstrap over
   events. Writes `analysis_output/iso50_lifetime_eval.json`.
3. `rebin4.py` -- 3 cm vs 4 cm segment-length robustness.
4. `shapes_alldepth.py` -- pooled dQ/dx shape panels at every depth.

Mechanism / elimination chain (each script one question):

- `null_readout.py`, `null_deconv.py` -- 20 ms amplitude null: the deconv
  tilt is attenuation-independent; the hits tilt is not.
- `op_closure.py`, `op_closure20.py` -- data-operator closure vs depth.
- `raw_sums.py`, `paired_reg.py`, `walkthrough.py` -- trunk uniformity,
  segment-level regression, single-event walkthrough.
- `smear_scan.py`, `smear_overlay.py` -- is the dQ/dx width a resolution
  (smearing) effect? No: no pixel-plane Gaussian reproduces it, and it
  grows with drift. Writes `analysis_output/iso50_smear_scan.json`.
- `corr4x4.py G SM [sym]`, `reconcile1x1.py` -- group-level 2D
  correlations under four comparison conventions (one-sided smear, both
  sharp, fig-11 symmetric); slope-convention reconciliation.
- `fig11_dqdx.py` -- the note's app_corr2d symmetric smearing applied to
  the dQ/dx pipeline (tau-neutral). Writes
  `analysis_output/iso50_fig11_dqdx.json`.
- `reco_cut_scan.py`, `qcut_dqdx.py`, `iso50_tcut.py` -- minimum-q /
  reco-side / time cuts: none rescues tau; fixed thresholds are
  themselves a lifetime-bias source (the hits mechanism).

## Headline numbers (tau in ms, truth 1.0; bootstrap over events)

| estimator | 3 cm MPV | 4 cm MPV |
|---|---|---|
| effq (truth) | 1.033 +- 0.031 | 1.073 +- 0.042 |
| hits | 1.094 +- 0.049 | 1.106 +- 0.054 |
| decB | 0.710 +- 0.028 | -- |
| decC | 0.727 +- 0.027 | 0.752 +- 0.025 |

The deconv deficit is a depth-dependent amplitude-scale slide
(sum ratio 1.05 -> 0.98 over 170 us, ~ -7.7%), invariant under all
comparison conventions and cuts tested, and attenuation-independent
(identical at 20 ms) -- i.e. calibratable with one lifetime-independent
capture curve.
