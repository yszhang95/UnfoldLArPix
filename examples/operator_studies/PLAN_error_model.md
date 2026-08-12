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
* `eta` — threshold dispersion of that trigger, `s_t = 0.65 ke`. One draw
  per sequence, and it is *invisible*: no measurement reveals it.

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
  *variances* are 30–40% below model. That is the **crossing selection**:
  the trigger fires on the first tick above threshold, which truncates
  the `eta` distribution. The analytic `s_t^2` overstates the realised
  dispersion, and a GLS built on the analytic value will over-trust the
  split direction;
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

* fit `s_t_eff` from the data (`Var(pseudo)+Var(remainder)-2Var(sum)`)
  rather than using the nominal 0.65, because of the selection effect;
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
2. **Measure** `s_t_eff` from the pseudo/remainder pair, and get a
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
