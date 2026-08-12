# Plan: the measurement error model, and what an anisotropic prior can and cannot do

Status: analysis done, nothing implemented in the solver yet.
Repo commit at time of writing: `d5a91ec` (UnfoldLArPix).
Numbers below are measured, not assumed — every table names the script
that produced it.

---

## 1. The readout error covariance (row space)

### 1.1 Generative model

Inside one trigger sequence on one pixel, with `y_j` the *j*-th latched
cumulative reading:

    y_j = Q(t_j) + eps_j + beta

* `eps_j` — per-latch uncorrelated noise, `s_u = 0.5 ke`, independent
  across every read;
* `beta` — the kTC baseline draw of that reset epoch, `s_r = 0.9 ke`.
  **One draw per sequence**, shared by every row in it, absent for a
  pixel's first (virgin) sequence;
* `eta` — threshold dispersion of that trigger, nominal `s_t = 0.65 ke`,
  drawn once per trigger in `tred/readout.py:77`
  (`thres = threshold + normal(0, thres_noise)`).

**`eta` never appears in any ADC word.** That is a fair objection to
calling it a measurement error, and it is literally true. It enters the
residual for a different reason: `constrained_solver.py:176` writes the
*nominal* threshold as the pseudo row's right-hand side, while the charge
that actually crossed is `thr_nom + eta`. So `eta` is an error of a
right-hand side **we impose**, not of anything the chip reported — which
is exactly why it must stay in `Sigma`, and why no amount of looking at
the data will reveal it row by row.

It is nevertheless measurable in aggregate, because it is the only term
in three combinations:

    Var(pseudo | virgin)   = s_t^2 +   s_u^2
    Var(remainder)         = s_t^2 + 2 s_u^2       (beta cancels)
    -Cov(pseudo,remainder) = s_t^2 +   s_u^2

| sample | from Var(pseudo,virgin) | from Var(remainder) | from Cov | combined |
|---|---|---|---|---|
| pos_a50_nb4 | 0.450 | 0.278 | 0.436 | **0.395** |
| mu_a50_nb4 | 0.456 | 0.441 | 0.442 | **0.446** |
| mu_a75_nb4 | 0.460 | 0.384 | 0.390 | **0.413** |
| mu_a25_nb4 | 0.288 | — | 0.195 | **0.246** |

`s_t_eff = 0.25-0.45 ke` against a nominal 0.65: **not zero, but about
60% of nominal.** The three estimators agree inside each sample (mu_a50:
0.456 / 0.441 / 0.442), which is what makes the deficit credible rather
than an artefact of one combination. The cause is the crossing selection
— the trigger fires on the first tick whose noisy accumulation exceeds
the noisy threshold, and that selection compresses the realised spread.
`mu_a25` is the low outlier and deserves a look.

Consequence for §1.4: use `s_t_eff`, fitted per sample, not the nominal
0.65. A GLS built on 0.65 over-trusts the pseudo/remainder split
direction by `(0.65/0.40)^2 = 2.6x`.

The four row kinds are then

| row | error | variance |
|---|---|---|
| `diff_j = y_j - y_{j-1}` | `eps_j - eps_{j-1}` | `2 s_u^2` (beta cancels) |
| `lumped` | `eps_1 + beta` | `s_u^2 + s_r^2` |
| `pseudo` | `-eta + eps_* + beta` | `s_t^2 + s_u^2 + s_r^2` |
| `remainder` | `eps_1 + eta - eps_*` | `2 s_u^2 + s_t^2` |

### 1.2 The couplings the current diagonal weighting throws away

Three off-diagonal terms follow directly, and all three are inside a
single sequence — sequences share nothing, so `Sigma` is exactly
**block diagonal by trigger sequence**:

1. `Cov(diff_j, diff_{j+1}) = -s_u^2`, i.e. correlation **−1/2**. The
   shared `eps_{j-1}` enters the two rows with opposite sign, so *that
   component* is 100% anti-correlated; after the independent halves
   dilute it the row-level correlation is −1/2.
2. `Cov(remainder, diff_2) = -s_u^2` — the remainder carries `+eps_1`
   and the first diff carries `-eps_1`. **This one is not in the
   `noise.py` docstring.** It is what turns the block from
   "two special rows plus an MA(1) chain" into a single tridiagonal
   chain running `pseudo — remainder — diff_2 — diff_3 — …`.
3. `Cov(pseudo, remainder) = -(s_t^2 + s_u^2)`, and crucially

       pseudo + remainder = eps_1 + beta = the lumped row's error

   so **splitting a trigger adds no information in the sum direction**;
   it only resolves the split direction, where the noise is `s_t`
   dominated. A diagonal weight treats the two halves as independent,
   which double-counts the sum and mis-weights the difference.

### 1.3 Measured (`row_covariance.py`, four nb4 waveform samples)

`n_r = d_r - d_exact_r` is known exactly per row from the waveform files,
so every entry above is directly checkable. `s_u=0.5, s_t=0.65, s_r=0.9 ke`.

| quantity | model | pos_a50 | mu_a50 | mu_a75 | mu_a25 |
|---|---|---|---|---|---|
| Var(diff) | 0.500 | 0.674 | 0.536 | 0.553 | 0.516 |
| Corr(diff_j, diff_j+1) | **−0.500** | −0.378 | −0.535 | −0.466 | −0.463 |
| Corr(remainder, diff_2) | −0.368 | −0.281 | −0.391 | −0.252 | −0.364 |
| Corr(lumped, diff_2) | −0.384 | −0.075 | −0.429 | −0.322 | −0.357 |
| Corr(pseudo, remainder) | −0.770 | −0.707 | −0.777 | −0.742 | −0.689 |
| Var(pseudo+remainder) | 0.250 (virgin) | 0.366 | 0.277 | 0.294 | 0.260 |
| Corr across sequences | **0** | +0.209 | +0.045 | −0.052 | −0.094 |

Reading:

* the **−1/2 diff correlation is confirmed** (−0.46 … −0.54 on the three
  muon samples);
* the **remainder↔diff link is confirmed** and is real, not a rounding
  artefact — it must go into the block;
* `Var(pseudo+remainder)` lands on the *virgin* prediction `s_u^2 = 0.25`,
  and the average model variance of the `pseudo` rows (0.69–0.83 vs
  0.6725 virgin / 1.4825 post-reset) says these samples are dominated by
  first-trigger sequences. The kTC term is therefore **untested** by
  these events; it needs a multi-trigger sample before it is trusted;
* `Corr(pseudo, remainder)` matches (−0.69…−0.78 vs −0.77) but both
  *variances* are 30–40% below model — this is the `s_t_eff` deficit
  quantified in §1.1;
* the cross-sequence null passes on the muon samples. The `pos_a50`
  +0.209 is the one thing that does not fit; the likely cause is that
  window edges are derived from the (noisy) hit list, so a mis-placed
  edge leaks *signal* into what we are calling noise — an effect that
  scales with local rate, which is also why `Var(diff)` is 35% high only
  there. Worth isolating before whitening a shower.

### 1.4 What to implement

Per-sequence GLS, not global whitening:

```
Sigma_seq = tridiag block, size (#latches + 2) at most
          = s_u^2 * T  +  s_r^2 * 11^T (post-reset only)  +  s_t^2 * ww^T
```
with `T` the MA(1) pattern above and `w` the ±1 pseudo/remainder
contrast. Blocks are ≤ 6×6 for nb4, so a Cholesky per sequence costs
nothing and the whitened forward operator is `L^-1 A`. This is a strict
generalisation of `row_weights(mode='diag')` and reduces to it when the
off-diagonals are zeroed.

Two guards, both learned from the earlier attempt at full whitening
(which traded ghost fraction and muon r/slope for integral):

* use the fitted `s_t_eff ~ 0.40 ke` of §1.1, not the nominal 0.65;
* keep the kTC term switchable until a multi-trigger sample confirms it.

---

## 2. The operator error is a *different* object and must not be folded into the same diagonal

`||e|| / ||n||` = 2.3 … 6.8 across the six samples (`anisotropy.py`).
The operator error dominates the readout error everywhere. It is not
block diagonal by sequence, and shrinking it with sigma-based filters is
wrong (`picard.py`: the truth's modal coefficients are flat in sigma, so
the Picard condition is violated and the ideal Wiener filter is
non-monotonic in sigma).

### 2.1 The "long window ⇒ small structural error" idea is mostly a proxy

Raw correlation of the relative operator error with window length is
negative in all six configurations, `spearman(|e|/q, dt) = −0.16 … −0.47`,
which looks like a clean law. It is not — **it is largely a row-kind
effect**. Broken down by kind (`anisotropy.py`):

| kind | typical `<dt>` | `<|e|/q>` | `rho(dt)` within kind |
|---|---|---|---|
| pseudo | 800–1900 | 0.07–0.13 | −0.22 … −0.45 |
| lumped | 190–1170 | 0.04–0.19 | −0.40 … −0.75 (mu_a75 +0.30) |
| diff | 30 | 0.17–0.49 | +0.00 … +0.19 |
| remainder | 30 | 0.19–0.46 | −0.21 … +0.08 |

Within the `diff` rows — which are all the same length — the `dt` trend
is **zero or the wrong sign**. The apparent law is "pseudo rows are
accurate and diff rows are not", dressed up as a time-extent law. A
weight built on `dt` would encode a coincidence of these topologies.

This is exactly the objection raised about multi-prong events, and it is
correct: `dt` is not the causal variable.

### 2.2 The causal variable: charge in the partially covered q-bins

The operator has to guess how charge is distributed *within* a q-grid
bin. A bin fully inside a readout window contributes no ambiguity — its
whole content is measured. Only the bins the window **cuts** are guessed.
So the natural predictor is

    q_part(r) = charge in the first and last q-bin the window touches
              (= the whole window if it fits inside one bin)

and the model is `|e_r| ~ kappa * q_part(r)`. Measured:

| sample | `rho(|e|, q_part)` | `kappa = rms|e|/rms q_part` | median `q_part/q` |
|---|---|---|---|
| mu_a00_nb1 | +0.488 | 0.577 | 0.40 |
| mu_a50_nb1 | +0.598 | 0.512 | 0.47 |
| mu_a25_nb4 | +0.680 | 0.518 | 1.00 |
| mu_a50_nb4 | +0.549 | 0.422 | 1.00 |
| mu_a75_nb4 | +0.112 | 0.287 | 1.00 |
| pos_a50_nb4 | +0.718 | 0.413 | 1.00 |

`kappa` sits in 0.29–0.58 (±30% around 0.45) across a sparse 0-degree
muon, a steep muon and a dense positron shower, while the naive relative
error `|e|/q` varies by a factor 4–5 between bins of the same sample.
That stability is the whole point: **`kappa * q_part` is a topology-
transferable error scale; `dt` is not.**

It also explains the nb4 result in one line: `median q_part/q = 1.00`
means the readout window is shorter than one q-bin, so *all* of the row's
charge is model-guessed. That is why nb4 diff rows are the worst rows in
the problem, and it is a statement about the grid, not about the angle.

The multi-prong worry is handled automatically by construction: a long
window that happens to start inside a second prong's pulse has a large
`q_part` and gets down-weighted, whereas a `dt`-based weight would call
it trustworthy. `q_part` is a linear functional of `q`, so it is
computable from the current iterate — the weight becomes an IRLS-style
adaptive weight rather than a fixed table.

`mu_a75` is the honest outlier (`kappa` 0.287, `rho` +0.11) and needs a
look before this is adopted.

### 2.3 What to implement

    Sigma_op = diag( (kappa * q_part(q_hat))^2 ),  kappa ~ 0.45
    data term: (A q - d)^T (Sigma_readout + Sigma_op)^-1 (A q - d)

re-evaluated every outer iteration. `Sigma_readout` is the exact block
matrix of §1.4.

Whether `Sigma_op` may be taken diagonal was the open question, and it is
now measured (`anisotropy.py`, `seq_structure`). The expectation was that
it could not be — consecutive windows on one pixel see the same
mis-modelled pulse. **The expectation was wrong:**

| sample | `corr(e_j, e_j+1)` in sequence | `corr(n_j, n_j+1)` (control) | sign coherence |
|---|---|---|---|
| mu_a00_nb1 | −0.096 | −0.674 | 0.742 |
| mu_a50_nb1 | −0.159 | −0.692 | 0.516 |
| mu_a25_nb4 | −0.253 | −0.477 | 0.628 |
| mu_a50_nb4 | +0.128 | −0.539 | 0.574 |
| mu_a75_nb4 | −0.031 | −0.444 | 0.025 (across seq) |
| pos_a50_nb4 | +0.125 | −0.358 | 0.653 |

The readout column is the control and behaves exactly as §1.3 requires.
Against it, the operator error's sequence correlation is small and does
not keep a consistent sign, so **a diagonal `Sigma_op` is defensible**
and the block structure is needed only for the readout part. Sign
coherence does sit above 0.5, so a weak shared bias exists; it belongs in
the operator's mean, not its covariance.

This does *not* say `e` is unstructured — the SVD study shows its power
concentrated in specific mid-sigma mode deciles (48.4% in one decile for
mu_a50). That structure lives across pixels, not along a sequence, and a
diagonal `Sigma_op` will not capture it. It is the reason step 4 below is
gated on the full metric set rather than on the integral alone.

Trade-off to measure, not assume: the rows with large `q_part` are the
short windows, and those are the rows carrying the time-localisation
information. Down-weighting them buys charge-scale accuracy and may cost
time resolution. The acceptance metrics (ghost, iso-ghost, killed truth,
slope, integral) all have to be re-run, not just the integral.

---

## 3. The anisotropic penalty

### 3.1 What the current prior cannot express

`l1` and `l2` as used are separable per voxel: they see a voxel's
magnitude and nothing else. They cannot say "charge should continue
along the track and stop across it". The truth is locally a 1-D curve
convolved with diffusion — a rank-1 structure with a strong direction
and two weak ones. The null space is anisotropic to match: the measured
weak-direction spread runs 26.5 px at 0 degrees down to 0.6 px at 75
degrees (`channel_coupling.py`), so the direction in which the data
cannot constrain the answer rotates with the track.

### 3.2 The right form is not "one penalty for long-t, another for long-xy"

Smoothness and sparsity are the two things wanted, and they want
*different axes*:

* **along** the local track tangent, `dE/dx` varies slowly — that is an
  `l2` smoothness prior on the derivative;
* **across** it, the profile should be narrow — that is not smoothness at
  all, it is sparsity, i.e. `l1` in the transverse plane.

So the anisotropic prior is one object with a direction field, not two
penalties selected by a case analysis:

    R(q) = sum_v [ w_par * (t_v . grad q)^2 ]  +  mu * sum_v || P_perp,v q ||_1

with `t_v` the local tangent from the structure tensor of the current
iterate. In group form: group voxels along the tangent, take the group
`l2`, sum with `l1` across groups (a `||.||_{2,1}` norm, standard prox).
Multi-prong is handled by the same mechanism as §2.2 — the direction
field is *local*, so each prong carries its own tangent and a vertex is
simply a place where the structure tensor is isotropic and the prior
falls back to the current isotropic behaviour.

Note this also settles the earlier `l1`-on-`u`-vs-`G u` question in the
smeared-operator arm: sparsity belongs on the *unsmeared* `u` and in the
*transverse* directions, which is where it is not fighting the filter.

### 3.3 The caveat that must stay attached

The under-determination result stands: `A` restricted to the support has
full row rank, `L_min ~ 0`, and 100% of the truth's residual lies in the
range — the truth is never the minimiser of the data term. A prior
selects a point inside a feasible set that is already *offset* by the
operator error. Therefore:

* an anisotropic prior can improve resolution, ghost and iso-ghost —
  it shapes the null-space component;
* it **cannot** fix the charge-scale slide, because that lives in the
  range component;
* only the weighting of §2.3 acts on the range component.

Both are worth doing. They are not substitutes, and only the second one
can move the slope/integral numbers. Anyone reading a slope improvement
after a prior change should suspect a retuned `alpha` first — that is
what the earlier `alpha` rescan showed.

---

## 4. Order of work

1. ~~Measure `corr(e_r, e_s)` within a sequence.~~ **Done** — §2.3: it is
   weak, so `Sigma_op` may be diagonal.
2. ~~Measure `s_t_eff`.~~ **Done** — 0.25-0.45 ke vs nominal 0.65 (§1.1).
   Still needed: a
   multi-trigger sample to test the kTC term. Decides §1.4's two guards.
3. Implement per-sequence Cholesky whitening behind a flag; re-run the
   full acceptance metric set on the 18-system angle scan. Accept only
   if ghost / iso-ghost / killed truth do not regress.
4. Implement the `kappa * q_part` adaptive weight, IRLS style; same
   acceptance gate. This is the one expected to move slope and integral.
5. Only then the anisotropic prior, judged on resolution metrics with
   `alpha` re-scanned at each setting so the comparison is honest.

## Reproduce

```
cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix
PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
PYTHONPATH=src $PY examples/operator_studies/row_covariance.py \
    pos_a50_nb4 mu_a50_nb4 mu_a75_nb4 mu_a25_nb4
PYTHONPATH=src $PY examples/operator_studies/anisotropy.py
```

Outputs: `examples/analysis_output/channel_coupling/row_covariance.json`,
`.../anisotropy.json`. Both need the `*_wf.npz` waveform companions in
`/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield/`,
which exist for `mu ang00/ang50 nb1` and `mu ang25/ang50/ang75 nb4`,
`positron ang50 nb4`.

### 3.4 The soft-seed alpha field is already an anisotropic prior — with the wrong anisotropy

Audit of what is actually implemented (`solve/strategy.py:95`,
`constrained_solver.py:307,326`), because §3 must not propose something
the code already does under another name.

**Where the field comes from.** Not from the support. `reco_algs.py:268`
starts with `SolveState(q=q0)`, whose `skeleton` is `None`, so stage 0
runs a **uniform scalar** `alpha = a` over the whole support. Each stage
then sets `skeleton = q > seed_cut` (0.5 ke) from *its own solution*, and
the next stage weights by `alpha_v = a * exp((d_v / soft_len)^p)` with
`d_v` the Manhattan distance to that skeleton. The initial support never
shapes the field; it only masks the feasible set in `CoordProx`.

**There is no superposition.** `manhattan_distance_from` is a multi-source
BFS returning the distance to the *nearest* seed. Verified: two seeds ten
voxels apart give the midpoint exactly the same alpha as a single seed
would (ratio 1.000). Consequences:

* a bridge between two prongs is penalised identically to a dangling
  tail — the field carries no "charge is likely on the line between two
  seeds" information;
* the seed amplitude is discarded (`q > 0.5` is boolean), so a 100 ke
  voxel and a 0.51 ke voxel generate identical fields;
* far from everything the distance saturates at
  `d_max = ceil(8*soft_len) = 16`, i.e. a multiplier `exp(8) = 2981`.
  With `a = 0.3` that is `alpha ~ 894`, effectively infinite against
  `|A^T r|`. "Inside the support but far from all seeds" is therefore
  **excluded**, and the cut-off distance is set by an implementation
  constant, not a physics choice.

**The ladder tightens rather than relaxes beyond ~2 voxels.** Nominally
`a` descends 1.0 -> 0.5 -> 0.3, but the field simultaneously raises the
penalty away from the skeleton:

| d | stage0 (a=1.0) | stage1 (a=0.5) | stage2 (a=0.3) |
|---|---|---|---|
| 0 | 1.000 | 0.500 | 0.300 |
| 1 | 1.000 | 0.824 | 0.495 |
| 2 | 1.000 | 1.359 | 0.815 |
| 3 | 1.000 | **2.241** | **1.345** |
| 4 | 1.000 | **3.695** | **2.217** |

Stage 1 exceeds the stage-0 uniform level at `d >= 2`, stage 2 at
`d >= 3`. The ladder genuinely relaxes only in a one-voxel shell around
what stage 0 already found; charge that stage 0 missed at distance >= 3
becomes monotonically harder to activate. That is the intended
strong-charge-first homotopy, but it means the ladder progressively locks
in stage 0's topology rather than admitting more of the solution.

**The metric is grid-index Manhattan, so the anisotropy is accidental.**
One time bin is `adc_hold_delay * time_spacing = 30 * 0.05 us = 1.5 us`
= **2.395 mm** at `vdrift = 1.59645 mm/us`; one pixel is **4.434 mm**.
The same physical displacement therefore costs 1.85x more in the time
direction than in the pixel direction, and Manhattan (not Euclidean)
adds a further sqrt(2)-sqrt(3) penalty on diagonals. So the prior is
already directional — just along the grid axes, for no physical reason.

Three fixes, cheapest first:

1. rescale the time axis by 1.85 so the metric is physically isotropic —
   a constant, no new machinery;
2. replace nearest-seed distance by an amplitude-weighted soft minimum
   over seeds, which makes an inter-prong bridge cheaper than a dangling
   tail and restores the information the boolean skeleton throws away;
3. the track-tangent anisotropy of §3.2, which is the real change.

### 3.5 Should L1 be weaker or stronger where the charge is large?

Two knobs answer this, and they answer differently, which is the point.

**`soft_len` — the near/far contrast during selection.**
`alpha_v = a * exp(d_v / soft_len)` with `d_v` measured from the previous
stage's skeleton, so `soft_len -> 0` is maximal relaxation ON large
charge and `soft_len -> inf` is uniform alpha. Scanned over
{0.5, 1, 2, 4, 8, 1000} (`softlen_scan.py`, record value 2.0):

| tag | integ% at 0.5 | at 2 (record) | at 1000 | ghost% 0.5 -> 1000 | killed 0.5 -> 1000 |
|---|---|---|---|---|---|
| mu_a75_nb1 | −6.30 | −5.49 | −5.16 | 7.55 -> 7.39 | 132.0 -> 132.2 |
| mu_a00_nb1 | −9.33 | −9.09 | −8.06 | 11.48 -> 12.34 | 202.1 -> 191.2 |
| pos_a00_nb1 | −5.98 | −5.73 | −4.88 | 5.68 -> 6.07 | 473.0 -> 446.9 |

`r` is flat to 4 decimals throughout; slope moves by <0.005. The whole
knob is worth ~1.1 pp of integral out of a 5-9 pp deficit, and the sign
is against the adaptive-lasso intuition: **relaxing L1 near large charge
buys nothing**, and the more uniform the field the better the integral
and the killed truth (at 0.4-0.9 pp more ghost).

**`refit` — no L1 at all on the large charges.** The direct form of the
question. `FinalRefit(eps, alpha=0)` freezes the strong support and
re-solves the amplitudes unpenalised (`refit_test.py`):

| tag | integ% none -> refit | slope | r | ghost% | iso-ghost | killed |
|---|---|---|---|---|---|---|
| mu_a75_nb1 | −5.49 -> **−2.46** | 0.874 -> 0.896 | 0.8971 -> 0.9037 | 7.57 -> 8.29 | 0.00 -> 0.00 | 132.6 -> 122.8 |
| mu_a00_nb1 | −9.09 -> **−4.74** | 0.816 -> 0.836 | 0.9336 -> 0.9378 | 11.54 -> 12.91 | 0.53 -> 2.68 | 196.7 -> 166.4 |
| mu_a50_nb1 | −7.20 -> **−3.21** | 0.675 -> 0.696 | 0.8246 -> 0.8227 | 19.41 -> 20.36 | 0.00 -> 0.00 | 290.3 -> 264.2 |
| pos_a00_nb1 | −5.73 -> **−2.83** | 0.948 -> 0.962 | 0.9754 -> 0.9754 | 5.89 -> 6.86 | 5.56 -> 8.60 | 461.5 -> 413.1 |

Roughly **half the integral deficit is L1 shrinkage on the strong
voxels**, and removing it also moves the slope towards 1 and cuts killed
truth in all four. `eps = 0.5` versus `0.2` changes the integral by
0.2 pp at most, so the gain comes from dropping the penalty on the strong
voxels, not from freezing more of them -- a clean control.

**Resolution.** The two results are consistent once selection and
amplitude are separated:

* during selection alpha must stay finite (it *is* the selector), so
  reshaping its near/far contrast is nearly inert -- hence the 1.1 pp;
* the amplitude bias is a separate stage, and there the answer is not
  "smaller" but **zero**;
* the price is ghost, +0.7 to +1.4 pp, and iso-ghost specifically on the
  two 0-degree topologies (0.53 -> 2.68 and 5.56 -> 8.60). That is the
  prior doing real work: with no penalty the strong support absorbs
  operator error into amplitude, exactly where §2.2 measured the model to
  be least trustworthy (`|e| ~ kappa * q_part`, and `q_part` is largest
  where the charge is).

So the answer to "weaker or stronger near large charge" is: **weaker for
amplitude (zero, via a refit), and the resulting ghost should be paid for
with the data weight of §2.3 rather than with L1** -- because L1 charges
the signal for the operator's error, while the weight charges the rows
whose error it actually is.

This re-measures a decision that is already open: the `refit:
{eps: 0.5, alpha: 0}` stage has been the measured candidate default for
nb1 since 2026-08-10 and adoption is still with the user. Nothing here is
new physics; it is the same conclusion on four more configurations, plus
the `soft_len` scan showing that the alternative route (reshaping alpha)
cannot substitute for it.
