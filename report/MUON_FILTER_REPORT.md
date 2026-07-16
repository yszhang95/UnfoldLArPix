# Report: Muon-derived time-axis correction filter (Option 1)

## Goal

Apply a new time-axis filter to correct for imperfect template compensation in
burst-sequence deconvolution. The filter is the **inverse ratio** of the muon's
continuous-readout spectrum to its template-compensated spectrum, derived from
the **muon** (long track, many pixels → good statistics) and applied to the
**positron** (sparse, the test case):

```
desired_positron(f) = |H(f)| · S_positron_compensated(f),
|H(f)| ≈ sqrt( P_muon_continuous / P_muon_compensated )   (per-pixel temporal power, averaged)
```

## Inputs (as specified)

| role | file | field response | readout |
|---|---|---|---|
| muon continuous | `pgun_muplus_3gev_noises_interval_average.npz` | nogrid | nburst1 / interval-avg (v3) |
| muon compensated | `pgun_muplus_3gev_tred_nburst4_noises_nd_readout.npz` | nogrid | nburst4 (v3_burst) |
| positron test | `pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz` | v2a_full 25×25 | nburst256 (v3_burst) |

- Muon FR = `fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz` (matches muon generation).
- Positron FR = `response_44_v2a_full_25x25pixel_tred.npz` (matches positron generation).
- Muon continuous/compensated confirmed same event (`sum_effq` = 4464.59).
- No continuous positron is needed: Fig 2 uses the positron's own smeared truth as reference.

## Method

- **Filter form — Option 1 (SNR-regularized):**
  `|H(f)| = 1 + W(f)·(sqrt(P_v3 / P_v3burst) − 1)`, Wiener gain `W = S/(S+N)`,
  `S = max(P_v3 − N, 0)`, noise floor `N(f)` measured from below-threshold
  pixels. `W → 0` (no correction) in noise bands; `W → 1` (full inverse ratio)
  in high-SNR bands. No `h_max` clip.
- Applied on the time axis only, multiplied on top of the existing 3-D Gaussian
  regularization kernel inside `deconv_fft`
  (`src/unfoldlarpix/deconv_workflow.py:apply_time_filter`).
- Result: `|H| ∈ [1.00, 1.24]` — compensation *under*-recovers power on the muon,
  so the filter is a gentle boost.

## Figures

### Fig 1 — MUON power-spectra ratio (`muonfilt_fig1_muon_ratio.png`)
Compensated/continuous (green) sits below 1 and falls to ~0.4 at high frequency
(template compensation under-recovers power). Applying `|H|²` (red) lifts it
toward 1 across the band — the filter does what it is built to do (self-consistency).

### Fig 2 — POSITRON reco/truth power ratio (`muonfilt_fig2_positron_ratio.png`)
Reference is the positron's own smeared truth (target = 1). The ratio is ≈1 at
DC but rises steeply with frequency (to ~10–15×) for both curves, because the
deconvolved charge retains far more high-frequency content than the
heavily Gaussian-smoothed truth — i.e. this band is dominated by the
deconv-vs-smoothing spectral mismatch, not by the compensation. The muon
correction (red) adds further high-frequency power, moving *away* from truth at
high frequency. Only the low-frequency band (≲0.05) is a fair comparison, where
both are near 1.

### Fig 3 — POSITRON truth vs reco 2-D correlation (`muonfilt_fig3_corr2d.png`)
The Gaussian filter is applied to the smeared truth; the spectral `|H|`
correction is applied **only to the reconstruction**, never to the truth.

| | total charge | vs truth | slope (deconv_q>0.5) | Pearson r |
|---|---|---|---|---|
| truth | 32428 | — | 1 | 1 |
| uncorrected | 31938 | −1.5% | 0.899 | 0.9710 |
| muon-corrected | 33374 | **+2.9%** | **0.966** | 0.9681 |

### Fig 4 — readout-level deficit, muon vs positron (`muonfilt_fig4_deficit.png`)
The apples-to-apples quantity the filter is actually built from:
`P_compensated_readout / P_continuous_readout` (hwf_block power), plotted for
both particles. The **muon (nburst4)** drops from ~0.92 at DC to ~0.4 at high
frequency — a large broadband power **deficit** (dead-time loss), so `|H| > 1`
everywhere. The **positron (nburst256)** stays ≈ 1 through low frequency and only
mildly droops at high frequency — little deficit. Mean `|comp/cont − 1|`:
**muon 0.375 vs positron 0.126** (≈ 3× smaller). This is the direct answer to
"why does the muon show a deficit but the positron barely does": the deficit is
set by burst density, and nburst4 ≫ nburst256 in dead-time loss.

## Assessment — the filter OVER-corrects here

The per-voxel amplitude slope improves (0.899 → 0.966), but the total charge
**over-shoots** (−1.5% → +2.9%) and the high-frequency band is over-boosted
(Fig 2), with Pearson r essentially flat (slightly down).

Root cause: **a burst-density mismatch between the muon and the positron.** The
muon filter is built from an `nburst4` muon — sparse readout with large
dead-time loss, so `|H|` is a sizeable boost. The positron test is `nburst256` —
a dense readout that loses little charge to dead time (uncorrected already only
−1.5%). Applying the large nburst4-derived boost to the nearly-complete nburst256
positron therefore over-corrects. The transfer function `|H|` is **not
nburst-invariant**: it must be derived from a muon with the *same* readout
configuration as the test particle to be quantitatively correct.

### Recommended next step
Build the filter from a muon recorded with the **same nburst/threshold settings
as the positron** (here: `thres5k_nburst256`). No such muon file currently
exists — it would need to be simulated. With a matched-readout muon, `|H|` would
reflect the positron's actual (small) dead-time loss and should improve rather
than over-shoot the integral.

---

# Solution 2 (recommended): truth/deconv_q filter — `--mode truth`

The readout filter above can only **boost** (`|H| ≥ 1`), so it cannot remove the
positron's high-frequency over-power and it over-shoots the integral. A better
definition uses the muon's **deconvolution→truth transfer function**:

```
|H(f)| = sqrt( P_smeared_truth(f) / P_deconv_q(f) )     (muon, aligned per-pixel)
```

then boxcar-smoothed over frequency (9 bins) to remove statistical oscillation.
This `|H|` is ≈1 at low frequency and **rolls below 1** where `deconv_q`
over-shoots the (Gaussian-smoothed) truth, so it *attenuates* the high-frequency
excess instead of amplifying it. Filter shape: `|H| ∈ [0.18, 1.02]`, flat at DC,
smooth monotone roll-off (`report/muon_time_filter_truth.png`).

Figures: `muonfilt_truth_fig{1,2,3,4}_*.png`.

### Three-way comparison (positron, thres5k_nburst256)

| | integral vs truth | Fig 2 mean \|ratio−1\| | Pearson r | slope (deconv_q>0.5) |
|---|---|---|---|---|
| uncorrected | −1.5% | 2.95 | 0.9710 | 0.899 |
| readout filter (boost-only) | +2.9% | 4.11 (worse) | 0.9681 | 0.966 |
| **truth filter (`--mode truth`)** | **+0.3%** | **0.20** (≈15× better) | **0.9745** | 0.882 |

- **Fig 2** (`muonfilt_truth_fig2_positron_ratio.png`): the corrected reco/truth
  ratio hugs 1 from DC to ~0.27 cyc/sample (uncorrected blows up to ~10×). The
  high-frequency over-power is removed.
- **Fig 3** (`muonfilt_truth_fig3_corr2d.png`): the truth-corrected scatter is
  visibly tighter around `y = x`; Pearson r improves (0.971 → 0.975).
- The only minor cost is slope 0.899 → 0.882: the high-frequency attenuation
  slightly lowers the sharpest peaks. The integral and the spectral/correlation
  agreement are all better.

### Why it succeeds where the readout filter failed
The readout filter encodes only "compensation lost power, boost it" — a
broadband `|H| ≥ 1` tuned to the sparse nburst4 muon, which over-shoots the
dense nburst256 positron. The truth filter instead measures the **full
reconstruction error** `deconv_q → truth` on the muon, which is dominated by the
common high-frequency over-response of the deconvolution (shared across
particles and largely nburst-independent), so it transfers well and can correct
in both directions. Build it with `--mode truth` (default) in `build_muon_filter.py`.

## Files

Code:
- `examples/build_muon_filter.py` — filter estimator (Option 1)
- `examples/muon_filter_report.py` — this report's figures
- `src/unfoldlarpix/deconv_workflow.py` — `apply_time_filter` + `time_filter` kwarg
- `examples/deconv_positron_v3_burst.py` — `--time-filter-npz`

Outputs (in `report/`):
- `muon_time_filter.npz`, `muon_time_filter.png`
- `muonfilt_fig1_muon_ratio.png`, `muonfilt_fig2_positron_ratio.png`,
  `muonfilt_fig3_corr2d.png`, `muonfilt_fig4_deficit.png`

## Reproduce

```bash
FR_MU=examples/data/fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npz
FR_POS=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
MU_CONT=/nfs/home/yousen/Documents/NDLAr2x2/tred/tests/pgun_signal_processing/pgun_muplus_3gev_noises_interval_average.npz
MU_COMP=/nfs/home/yousen/Documents/NDLAr2x2/tred/tests/pgun_signal_processing/pgun_muplus_3gev_tred_nburst4_noises_nd_readout.npz
POS=examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz

# muon (continuous + compensated), build filter
uv run python examples/deconv_positron_v3.py       --input-file "$MU_CONT" --field-response "$FR_MU" --tpc-id 0 --output-dir muon_out
uv run python examples/deconv_positron_v3_burst.py --input-file "$MU_COMP" --field-response "$FR_MU" --tpc-id 0 --output-dir muon_out
uv run python examples/build_muon_filter.py \
  --muon-v3 muon_out/deconv_positron_v3_event_0_0.npz \
  --muon-v3-burst muon_out/deconv_positron_v3_burst_s0p005_sp0p2_event_0_0.npz \
  --out report/muon_time_filter.npz

# positron (continuous reference + compensated uncorrected + corrected)
POS_CONT=examples/data/pgun_positron_3gev_tred_noises_effq_nt1_wf.npz   # same event/FR
uv run python examples/deconv_positron_v3.py       --input-file "$POS_CONT" --field-response "$FR_POS" --tpc-id 0 --output-dir pos_out
uv run python examples/deconv_positron_v3_burst.py --input-file "$POS" --field-response "$FR_POS" --tpc-id 0 --output-dir pos_out
uv run python examples/deconv_positron_v3_burst.py --input-file "$POS" --field-response "$FR_POS" --tpc-id 0 --output-dir pos_out \
  --time-filter-npz report/muon_time_filter.npz

# report
uv run python examples/muon_filter_report.py \
  --muon-v3 muon_out/deconv_positron_v3_event_0_0.npz \
  --muon-v3-burst muon_out/deconv_positron_v3_burst_s0p005_sp0p2_event_0_0.npz \
  --filter-npz report/muon_time_filter.npz \
  --pos-v3burst-uncorr pos_out/deconv_positron_v3_burst_s0p005_sp0p2_event_0_0.npz \
  --pos-v3burst-corr pos_out/deconv_positron_v3_burst_s0p005_sp0p2_muonfilt_event_0_0.npz \
  --pos-continuous pos_out/deconv_positron_v3_event_0_0.npz \
  --out-prefix report/muonfilt
```
