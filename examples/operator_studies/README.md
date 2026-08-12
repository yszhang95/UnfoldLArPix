# Operator and system studies (angle-scan samples)

Diagnostics of the forward operator and of the linear system it defines.
They run on the **angle-scan** samples (`mu_a{00,25,50,75}_nb*`,
`pos_a{00,25,50,75}_nb*`), reading each run's own stored `job_config`
from `analysis_output/nb1_fraccensor/{A,B}/` and replaying only
LoadEvent / FFTWarmStart / BuildMeasurement / BuildSupport — nothing is
re-solved unless a script says so.

Nothing here belongs to the isochronous lifetime study; that lives in
`../iso50_analysis/` (10 depths x 50 seeded muon copies at fixed drift
depth — hence "iso50").

| script | question it answers | technote |
|---|---|---|
| `channel_coupling.py <tag> [arm]` | Which channels see the same charge, over what range in pixel space, and how many independent constraints survive the support/active-set restriction? Exact row Gram `G = A P A^T`, coupling profiles, spectra, near-null localisation. | §4.4 |
| `weight_proto.py` | Window-sampling weights against the exact waveform: box vs piecewise-linear. (Linear is worse.) | §4.3 |
| `phase_exact_proto.py` | Same target, adding the phase-exact edges; also the alignment and global-shift scans that separate sampling error from the within-bin floor and the time-base offset. | §4.3 |
| `slope_a75.py [tag]` | Which objective term controls the regression slope? Ablations (l1 ladder, censor, refit, warm-start time regularisation) plus per-pixel time spread truth vs reco. Solves into `analysis_output/slope_probe/`. | §5.2 |
| `slope_origin.py [tags…]` | Is the slope deviation selection, time placement, or amplitude? Re-regresses with different selections and after time integration. | §5.2 |
| `perfect_reco.py [tags…]` | The metric's own slope floor: the truth's exact bin integrals pushed through the same evaluation chain. | §3.6 |

Interpreter: `~/Documents/NDLAr2x2/tred/.venv/bin/python` with
`PYTHONPATH=<repo>/src` (torch + numpy 2). The GPU ones honour
`CUDA_VISIBLE_DEVICES`.

Outputs: `analysis_output/channel_coupling/`, `analysis_output/slope_probe/`.
