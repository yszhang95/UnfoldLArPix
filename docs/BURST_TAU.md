# `burst_tau` — gating the split-trigger pseudo-measurement

## What it fixes

`split_trigger` splits each trigger sequence's first window at the trigger
instant into two measurement rows:

| row | window | value |
|---|---|---|
| pseudo-measurement | `(restart, trigger]` | `threshold` |
| remainder | `(trigger, first latch]` | `q_burst0 - threshold` |

The pseudo row's value is not a digitised sample. It is *inferred* from the
discriminator having fired: "at the trigger the accumulator equalled the
threshold". That inference is only valid when the trigger was
**threshold-limited**.

After a latch the CSA is dead for `adc_down_time`, and the accumulator keeps
integrating throughout (`nd_readout` gates *triggering*, not accumulation).
If the pixel is still above threshold when the discriminator re-arms, it fires
immediately — the trigger is **dead-time-limited**. The pre-trigger window then
holds the whole dead-time pile-up, not the threshold, and the pseudo row
asserts a value far below the truth.

`burst_tau` is the gap below which a re-trigger is treated as immediate. Its
physical floor is `adc_hold_delay + adc_down_time + one_tick`
(`burst_tau_min`) — the shortest possible gap of a continuous re-trigger.
Sequences below it are emitted as ONE lumped window with no split.

## Measured

`pos_a50 nb4`, post-acq-fix, window integrals compared against the **true
current waveform** (`_wf` dataset, noiseless):

| pseudo rows | n | asserted | true integral | error | per row |
|---|---:|---:|---:|---:|---:|
| immediate re-trigger (gap < 56) | 101 | 505.0 | 1476.9 | **+971.9** | +9.62 |
| threshold-limited (gap >= 56) | 205 | 1025.0 | 1006.9 | -18.1 | -0.09 |

33% of the pseudo rows carried **102%** of the total pseudo-row error; the
threshold-limited rows were accurate to -0.09 ke/row. The spurious +972 ke
matches the +1060 ke charge excess the solver produced at 5-10 cm depth.

Reconstruction, same event:

| config | int% | r | slope | ghost% | killed |
|---|---:|---:|---:|---:|---:|
| split on, no gate | +8.23 | 0.9468 | 1.090 | 5.71 | 91.6 |
| split off | -3.83 | 0.9951 | 1.031 | 2.14 | 114.1 |
| split on + `burst_tau: auto` | -4.52 | 0.9951 | 1.029 | **1.55** | **105.2** |

The gate beats *both* baselines on ghosting and killed truth: it keeps the 205
accurate constraints that `split_trigger: false` throws away, and drops the 78
harmful ones that the ungated split emits.

The immediate-re-trigger fraction is a property of the data alone (hits only,
no fit). Across the 48-configuration scan it rises with angle to the anode
plane and falls to exactly zero at `nburst = 64`:

| | a00 | a25 | a50 | a75 |
|---|---:|---:|---:|---:|
| positron nb4 | 16.4% | 18.7% | 23.7% | 30.9% |
| muon nb4 | 8.7% | 10.4% | 10.9% | 47.0% |
| positron nb1 | 29.0% | 32.8% | 44.7% | 52.9% |
| any particle nb64 | 0.0% | 0.0% | 0.0% | 0.0% |

## Measuring the fraction — `ImmediateFraction`

`algs/readout_algs.ImmediateFraction` computes it. It reads `hits_view` and
`readout_config` and nothing else, so it runs on a sequence truncated at
`LoadEvent`: no response file, no operator, no solve, no GPU. That is
deliberate — a claim about the *readout* should not be derivable only from a
reconstruction of it.

```yaml
- LoadEvent:          {input: …, tpc: 0, max_events: 1}
- ImmediateFraction:  {tau: auto}
```

The gap is measured from the previous sequence's **last** latch on the same
pixel, which `HitsView.last_latch` defines as `trigger + nburst *
adc_hold_delay`:

```
gap_j = trigger_j - last_latch_{j-1}
```

**Three populations have been called "immediate" and they differ by up to a
factor of three.** All of them are reported so that none is the default
reading by accident:

| key | definition | `mu_a00_nb1` |
|---|---|---:|
| `frac_immediate` | `gap < tau`, over **all** sequences | **29.4%** |
| `frac_immediate_inclusive` | `gap <= tau`, over all sequences | 30.3% |
| `frac_immediate_of_retriggers` | `gap < tau`, over sequences that *have* a previous latch | 69.7% |
| `frac_at_rearm` | fired at the earliest instant allowed (`trigger` = previous `rearm`) | 10.1% |
| `frac_lumped_in_B` | `(gap < tau) OR (c0 < threshold)` — what the split gate leaves lumped | 39.9% |

`frac_immediate` is the published one. Three things about the others are worth
stating once:

- The gate is **strict** (`>= tau` is threshold-limited), so
  `frac_immediate_inclusive` is a reading of prose, never of the code.
- The shortest gap the readout allows is `adc_down_time + one_tick` = 26 ticks,
  not `tau` = 56, so the immediate band `[26, 56)` is one `adc_hold_delay`
  wide. "Fires immediately" read *literally* is `frac_at_rearm`, and that is
  about a third of `frac_immediate` — the gate is deliberately the wider,
  conservative window.
- `frac_lumped_in_B` is **not** an immediate fraction. Its second branch is a
  sub-threshold first window; it survives to `nburst = 64` where the immediate
  population is exactly zero. `ab_anatomy/ab_rows.py`'s `imm_frac` is this
  quantity, which is why it disagrees with the table above on every cell —
  reproduced here on all 12 of its cells.

`tests/test_immediate_fraction.py` pins the distinctions on hand-counted
synthetic hits, and asserts the gate against `build_latch_rows` rather than
restating it.

## Usage

```yaml
- BuildMeasurement: {split_trigger: true, acq_start: event, burst_tau: auto}
```

| value | meaning |
|---|---|
| absent | legacy — every trigger treated as threshold-limited (bit-identical to pre-`burst_tau`) |
| `auto` | the physical floor from the readout config |
| *number* | ticks, clamped to `[floor, 2*adc_hold_delay]` with a warning |

## Layer separation

The feature is deliberately split across the four layers; nothing reaches
across them.

| layer | where | responsibility |
|---|---|---|
| **data / convention** | `model/conventions.py` | `burst_tau_min(readout_config)` and `resolve_burst_tau(readout_config, tau)` — pure functions of the readout config, no I/O, no state. Convention lives here by this package's own rule. |
| **model** | `constrained_solver.build_latch_windows` | takes `burst_tau: int \| None` — an already-resolved integer. Never reads `ReadoutConfig`, never touches a service. |
| **algo** | `algs/reco_algs.BuildMeasurement` | owns the *policy*: reads the `burst_tau` prop, resolves it against the readout config, passes the integer down. Mirrors how `FFTWarmStart` handles its `tau`. |
| **service** | — | untouched. `tau` is a per-event readout convention, not shared infrastructure, so it has no business in a service. |

`resolve_burst_tau` was moved from `model/warm_start.py` to
`model/conventions.py` when it acquired a second consumer: leaving it in
`warm_start` would have made `BuildMeasurement` import from an unrelated model
module. `warm_start` re-exports it, so existing imports and
`tests/test_burst_tau.py` are unaffected.

## Regression safety

- `burst_tau=None` (the default) produces bit-identical windows to the
  pre-feature code — asserted directly, not just tested by metric.
- Enabling the gate conserves charge: `sum(value)` is unchanged
  (10569.6 ke before and after on `pos_a50 nb4`), because the pseudo row and
  its remainder merge back into one lumped row.
- Full suite passes (185 tests).

## See also

- `model/conventions.py` — every tick/bin/phase convention, with its measured
  justification.
- `docs/DECONV_DIAGNOSTICS.md` — the diagnostic metrics used in the table above.
