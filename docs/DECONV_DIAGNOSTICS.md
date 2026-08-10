# Deconvolution Diagnostics — a reusable toolkit

Data-driven probes for debugging zero-suppressed (ZS) charge deconvolution:
*why* a solve under/over-recovers, and *where* the error lives (measurement
model vs. solver vs. penalty vs. genuine readout loss).

The value here is the **methods** (portable to any deconv problem) and the
**disciplines** (measurement hygiene). Case-specific numbers from the ZS
large-angle study are kept in a separate section at the end and are **not** the
point — they only show each probe in action.

---

## A. Methods — reusable probes

Each probe uses a known quantity (usually the simulation truth) as an
instrument to isolate one failure mode. State the **criterion** up front so the
outcome is a decision, not a vibe.

### M1 · Plug truth into the objective
Evaluate `OBJ(q_truth)` and `OBJ(q_solved)` on the *same* objective, decomposed
term by term (data-residual vs. penalty).
- **Criterion:** if `OBJ(truth)` is *lower* than the solver's output, the solver
  did not reach the true minimizer → over-regularized or constrained/under-
  converged away from truth. Read which term (data vs. penalty) carries the gap
  to see *who* pulled it away.
- **Use when:** you suspect the penalty is over-correcting, or the solve
  converged to the wrong place.

### M2 · Truth as seed
Initialize the solver at `q_truth` and let it run.
- **Criterion:** if the solution walks away from truth, the bias lives in the
  objective itself (not the initialization); if it stays, the problem is the
  warm-start / a local minimum.
- **Use when:** ruling out "the warm-start biased the answer."

### M3 · Forward-projection `A · q_truth` vs. `d`
Push truth through the measurement operator and compare to the actual data.
- **Criterion:** if `A·q_truth` already fails to match `d`, the error is in the
  operator / forward model (not the solver); if it matches but the solve does
  not, the error is in the solver / penalty.
- **Use when:** separating "model is wrong" from "solve is wrong."

### M4 · Penalty-strength ablation, read position and amplitude separately
Scan the penalty weight (full → refit → weak → off) and read two *independent*
axes: position metrics (`purity`, `ghost`) vs. amplitude metrics
(`matched-charge`, `integral`).
- **Criterion:** if ghost/purity stay flat while integral moves, the penalty's
  two roles (support selection vs. amplitude shrinkage) are separable → keep the
  good role, relax the bad one (e.g. freeze support, then refit amplitude).
- **Use when:** deciding which role of a sparsity penalty to keep vs. loosen.

### M5 · Attribute charge loss across stages
Split the total shortfall `reco/truth` into serial stages: readout capture ×
fit recovery. **The threshold acts on the *channel*, and a channel sees the
field-response-weighted *sum* of many nearby voxels' charge** — so a pixel being
sub-threshold does **not** mean its charge is lost: charge sharing (the wide
field response) lets the deconvolution recover it from any firing neighbor its
charge reached, and two individually sub-threshold voxels can jointly trip a
channel and both be recovered.
- **Clean probe:** sum `reco` charge in voxels whose own pixel **never latched**
  → the charge recovered purely via sharing from firing neighbors.
- **Criterion:** if that fraction is appreciable, sub-threshold charge is
  *recoverable*, not physically lost → attribute the shortfall to the fit /
  penalty, not the readout. Genuinely unrecoverable = only charge whose *entire*
  induced footprint stays below threshold on *every* channel it couples to.
- **Pitfall:** do **not** use a raw sum of the hit charges as the "recoverable
  ceiling" — sharing makes it undercount, and cumulative per-burst latches
  double-count re-triggers (use differenced per-burst increments if you need the
  captured charge at all).
- **Use when:** deciding "truly lost at readout" vs. "recoverable but shrunk by
  the penalty."

### M6 · Neighborhood-sum collapse
Before comparing per-voxel, sum charge over a small local voxel neighborhood and
compare *that* to truth.
- **Criterion:** if the neighborhood sum matches truth while per-voxel does not,
  the error is migration / mis-placement, not charge non-conservation.
- **Use when:** distinguishing a position-resolution error from a charge-loss
  error.

### M7 · Event display in physical coordinates — localize before theorizing
When an integral/metric is off, do not argue mechanisms first: map truth, reco,
and (reco − truth) in *physical detector coordinates* (e.g. drift depth from
the anode × pixel index, with the anode/cathode drawn in), plus a 1D charge
profile along the physically meaningful axis.
- **Criterion:** *where* the difference clusters names the mechanism class.
  Clustered at a detector/acquisition boundary (anode, cathode, window edge)
  → boundary/response-edge effect, not a solver property; uniform along the
  track → normalization or penalty bias; displaced into adjacent voxels →
  resolution/migration (hand off to M6).
- **Use when:** any unexplained charge deficit or excess. Convert grid/time
  bins into detector coordinates first — a deficit that looks like "large-angle
  tracks are worse" in metric space can read as "the anode-side track end is
  missing" the moment it is drawn where the detector lives.

---

## B. Disciplines — measurement hygiene (not problem-solving probes)

These are guards that keep every measurement valid. They are not "analyses that
cracked a problem" — they are the conditions under which the probes above mean
anything.

- **G1 · Select on the observable, never on truth.** Condition every selection
  and every efficiency/bias metric on `reco > cut`, never on smeared `truth >
  cut`. Truth-conditioning hides the selection/Eddington bias that only shows up
  at high reco — the real world has no truth to cut on.
- **G2 · Verify with data, not reasoning.** Every *explanation* must come with a
  measurement that could falsify it; if the measurement kills it, retract the
  explanation. Plausible-sounding stories (e.g. "the truth leaks out of the
  fiducial volume," "the warm-start over-recovers") have been wrong here and
  were only caught by a probe.

> A note on what does **not** belong here: "the change reproduces the golden /
> S=1 baseline bit-for-bit" is change-correctness hygiene, not a diagnostic —
> and a regression baseline built on buggy data actively *hides* the bug (it
> certifies the broken behavior as "correct"). Never mistake a green regression
> check for a validated result.

---

## C. Worked results from the ZS large-angle study (case-specific — illustration only)

These are the outputs of the probes above on one study; they are here to show
the method in action, not as portable conclusions.

| Probe | What it found here |
|---|---|
| **M1** | Large angle: `data(truth) < data(q_hat)` **and** `OBJ(truth) < OBJ(q_hat)` — the returned solve both fits the data worse and is over-sparse relative to truth → over-regularized / constrained away from the true minimizer. |
| **M2** | Shielded self-trigger `+10.5%` persisted from *both* the warm-start seed and the truth seed → not a warm-start artifact. |
| **M3** | The coarse operator time bin (`B=30`) averages the sharp shielded response into a charge-additive near-null direction that the data-term fills → root cause of the shield over-recovery; halving the bin (`B/2`) removes it (`+11.9% → −3.9%` at the objective, `+10.9% → −2.6%` in the pipeline). |
| **M4** | L1's *position* role is good (purity ≈ 98%, ghost ≈ 2%); its *amplitude* role over-shrinks (integral `−22% → −18%` as L1 weakens, while ghost rises `1.9% → 5.6%`) → decouple via a support-frozen amplitude refit. |
| **M5** | *Pending* — the never-latched recovered-charge probe has not been run cleanly; the earlier raw-hit-sum proxy was unreliable (gave `>100%` capture at 0°). |
| **M6** | Large-angle charge migrates to adjacent along-track voxels rather than being lost; a finer fit bin (`S=2`) tightens it (spec_dev: μ 50° 3.04→2.24, 75° 2.36→1.67; e⁺ 75° 7.09→5.06). |
| **M7** | The large-angle integral deficit, drawn in (drift depth, pixel) with the anode marked: 91% of the missing charge sits within 5 cm of the anode — instantly redirecting the investigation from "solver/ZS" to the near-anode prompt-induction physics (Ramo: a deposit at depth d promptly induces only 1−w(d); the rest is locked in the static ion image) and to the operator's first-window edge (`acq_start`). |

---

*Written in English for shareability with the collaboration; ask if a Chinese
version is wanted. Section C is a moving target — update or drop it as the study
evolves; Sections A/B are the durable content.*
