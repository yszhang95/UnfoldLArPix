# Gaussian-Smoothed Template Compensation — Implementation Proposal

**Status:** future work — design proposal, not yet implemented.
**Owner:** TBD
**Motivation snapshot:** `output_spectra_compare/diag_residual.png` shows the
v2 / v3_burst residual spectrum (compensated − sampling baseline) has
peaks at f ≈ 0.13 and f ≈ 0.20 cycles per (30-tick) bin, corresponding to
periods 5–8 bins (150–240 ticks). These peaks track the sharp edges in
the inserted template differentials — high-frequency spectral content
the underlying sampling data does not carry. A Gaussian low-pass on the
template, applied at injection time, kills that excess without per-event
calibration.

---

## 1. Where the high-frequency content enters

`src/unfoldlarpix/burst_processor.py:342` — `_template_compensation()` is
the canonical injection point used by all three processors:

```python
# line 424
template_section = template_section * (threshold / template_section[-1])
# line 425
template_section = np.diff(template_section, prepend=0)   # ← per-bin charge
# line 428
chgs = template_section[1:].tolist() + [next_seq.charges[0] - threshold] \
       + next_seq.charges[1:].tolist()
```

The differential `template_section` at line 425 is the array of per-bin
charges we want to smooth. It is anchored such that its **last element**
sits at `template_times[-1] == threshold_time == next_seq.trigger_time_idx`
(verified at line 415–416). That is the "peak / last point" the design
discussion settled on as the Gaussian-kernel centre.

Downstream classes that reuse this entry point:

| Processor | File | Notes |
|---|---|---|
| `BurstSequenceProcessor` (v1) | `burst_processor.py` | Base class — owns `_template_compensation`. |
| `BurstSequenceProcessorV2` | `burst_processor_v2.py` | Tweaks gap policy; injection path inherited / mirrored. |
| `BurstSequenceProcessorV3` (v3_burst) | `burst_processor_v3.py` | Two-pass; chooses collection vs induction template, then injects. |

Smoothing therefore needs to be done **once** at the level of
`template_section` (post-diff, pre-injection). If the base method gets
the smoothing hook, v2 and v3_burst inherit it for free.

---

## 2. Smoothing scheme

### 2.1 Kernel — numpy only, no scipy

A normalised Gaussian kernel `g[k]`, length `2R + 1`, centred at the
**right edge** (= last point of the differential = peak anchor):

```python
def _make_one_sided_gaussian(sigma_bins: float, n_sigma: float = 4.0) -> np.ndarray:
    """One-sided Gaussian whose peak sits at index 0 (rightmost element).

    Returns a length-(R+1) array g such that g[0] is the peak and g[-1]
    is the leftmost tail, with all weight on offsets <= 0 (no forward
    leak past the peak anchor). Normalised so g.sum() == 1.
    """
    R = int(np.ceil(n_sigma * sigma_bins))
    offsets = np.arange(R + 1, dtype=np.float64)        # 0, 1, ..., R
    g = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    g /= g.sum()                                         # normalise
    return g                                             # g[0] = peak
```

`sigma_bins` is in units of `adc_hold_delay` windows — the same units
used in `diag_residual.png`. Default `sigma_bins = 1.0` (covers the
f ≈ 0.1–0.2 cycles/bin band we want to attenuate). `n_sigma = 4` keeps
≥99.99 % of the kernel mass.

### 2.2 Peak-anchored convolution (back-direction only)

Given the per-bin differential `dC = template_section` of length `N`,
build the smoothed differential `dC_s` so that `dC_s[N-1]` is *exactly*
the original peak position (the trigger anchor), and the kernel folds
**leftward** into earlier bins. Numpy implementation:

```python
def _smooth_template_diff(dC: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve dC with a right-edge-anchored Gaussian (kernel[0] is the peak).

    kernel must satisfy kernel.sum() == 1 and represent a half-Gaussian
    extending into negative offsets (i.e. earlier bins).
    """
    # Each entry dC[i] gets spread back into dC_s[i-k] using kernel[k].
    # Equivalent to convolution with the reversed kernel + 'full' mode,
    # then clipping to the original length.
    N = dC.size
    R = kernel.size - 1                                  # tail length
    pad = np.concatenate([np.zeros(R), dC])              # left-pad for tails
    full = np.convolve(pad, kernel[::-1], mode="valid")  # length N
    return full
```

Properties this gives us by construction:
- `dC_s[-1]` collects `kernel[0] * dC[-1]` plus the smeared tails of
  earlier-but-near entries — i.e. the peak bin stays the peak (slightly
  reduced in amplitude, as designed).
- All other smoothed mass migrates to bins with index `< N-1`
  (earlier in time) — **no forward leak past the trigger anchor**, so
  the next burst's first recorded sample is never modified.
- `dC_s.sum()` equals `dC.sum()` only if the tails do not fall outside
  the array. With left-padding by `R` and `valid` convolution we instead
  drop the leftward tails — those tails are what the renormalise/leak
  modes (§3) decide what to do with.

### 2.3 Worked example

For `dC = [5, 4, 3, 2]` (peak at index 3, the trigger anchor), σ = 1
bin, R = 4, the one-sided kernel is roughly `[0.398, 0.242, 0.054,
0.004, 0.000]` (peak-first). Applying §2.2:

```
dC_s ≈ [3.50, 3.16, 2.39, 1.51]
sum(dC_s) ≈ 10.56     # original sum was 14.0
```

The missing 3.44 ke is what the §3 modes account for.

---

## 3. Edge-handling modes (option-controlled)

Both modes must be available, selectable per-processor:

### 3.1 `edge_mode = "renormalize"` (preserve gap integral)

After computing `dC_s` as in §2.2:

```python
target_total = dC.sum()
scale = target_total / dC_s.sum() if dC_s.sum() > 0 else 1.0
dC_s *= scale
```

Recovers the original integrated charge for the gap exactly. Recorded
hits are untouched. This is the conservative default — recommended.

### 3.2 `edge_mode = "leak"` (let tails spill into recorded bins)

Replace the `valid` convolution with a `full` convolution and additively
deposit the leftward overflow into the bins **before** the gap (which
belong to the previous burst's recorded samples). Numpy form:

```python
def _smooth_template_diff_leak(
    dC: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Like §2.2 but also returns the leftward overflow that should be
    added into the prior-burst region.

    Returns (dC_s_in_gap, dC_overflow_left) where
      len(dC_s_in_gap)   == len(dC)
      len(dC_overflow_left) == kernel.size - 1
    The caller is responsible for adding dC_overflow_left into the
    block at indices [trigger_bin - N - R + 1 ... trigger_bin - N].
    """
    N = dC.size
    R = kernel.size - 1
    full = np.convolve(dC, kernel[::-1], mode="full")    # length N + R
    dC_s_in_gap = full[R:]                               # last N samples (the gap)
    dC_overflow_left = full[:R]                          # the leftward tails
    return dC_s_in_gap, dC_overflow_left
```

The caller then injects `dC_s_in_gap` into the gap (no scaling) and
**adds** `dC_overflow_left` to the block at the bins immediately
preceding the gap. This preserves the convolved mass exactly and lets
the smear physically extend onto the previous burst's recorded samples.

Both modes share §2.1 (kernel construction) and §2.2 (peak anchoring);
the only difference is whether the leftward tails are recovered into
the gap (renormalise) or deposited outside it (leak).

---

## 4. API surface

### 4.1 Processor constructor

Add three keyword arguments to `BurstSequenceProcessor.__init__`
(`burst_processor.py:145`). V2 and V3 inherit / forward them:

```python
def __init__(
    self,
    ...,
    template_smooth_sigma_bins: float | None = None,   # None = disable
    template_smooth_edge_mode: str = "renormalize",    # or "leak"
    template_smooth_n_sigma: float = 4.0,
):
    ...
    self._template_smooth_sigma_bins = template_smooth_sigma_bins
    self._template_smooth_edge_mode = template_smooth_edge_mode
    self._template_smooth_kernel = (
        _make_one_sided_gaussian(template_smooth_sigma_bins,
                                  template_smooth_n_sigma)
        if template_smooth_sigma_bins
        else None
    )
```

`None` (default) ⇒ smoothing disabled, behaviour bit-identical to today.

### 4.2 Injection point

Inside `_template_compensation()` (`burst_processor.py:425`):

```python
template_section = np.diff(template_section, prepend=0)   # existing

if self._template_smooth_kernel is not None:
    if self._template_smooth_edge_mode == "renormalize":
        template_section = _smooth_template_diff(
            template_section, self._template_smooth_kernel
        )
        # restore the original gap integral
        target = float(template_section.sum())   # before smoothing? see note
        # NOTE: capture target BEFORE the smoothing call (refactor as helper).
    elif self._template_smooth_edge_mode == "leak":
        template_section, overflow = _smooth_template_diff_leak(
            template_section, self._template_smooth_kernel
        )
        # overflow is returned out-of-band so the caller can write it
        # into the bins preceding `template_times[0]`.
    else:
        raise ValueError(f"unknown edge_mode {self._template_smooth_edge_mode}")
```

For the `"leak"` branch, `_template_compensation()` must return the
overflow array up the call stack so that `merged_sequences_to_block()`
can add it to the correct bins of the dense block (those bins exist —
they're the previous burst's recorded samples). Concretely, return a
new optional field on `MergedSequence` (`burst_processor.py:117`),
say `template_overflow: list[tuple[int, np.ndarray]]` listing
`(start_bin_index, overflow_array)` pairs, and consume it in
`merged_sequences_to_block` (`burst_processor.py:598`) right after the
main `block_charges[..., tinds] = charges` loop.

### 4.3 CLI surface (deconv scripts)

`examples/deconv_positron_v1.py`, `_v2.py`, `_v3_burst.py` already accept
many `--`-flags. Add three:

```
--template-smooth-sigma SIGMA_BINS   (default: unset = disabled)
--template-smooth-edge {renormalize,leak}   (default: renormalize)
--template-smooth-n-sigma N         (default: 4.0)
```

These pass straight through to the processor constructor via
`process_event_deconvolution()` / `hits_to_merged_block()` (extend the
existing kwargs plumbing in `deconv_workflow.py`).

---

## 5. Step-by-step implementation order

1. **Helpers, isolated, with unit tests.** Add `_make_one_sided_gaussian`
   and `_smooth_template_diff` / `_smooth_template_diff_leak` to
   `burst_processor.py` near the existing template helpers (lines
   ~14–95). Add `tests/test_template_smoothing.py` with:
   - kernel sums to 1 and peaks at index 0
   - identity case (`sigma → 0`) returns input unchanged
   - renormalise preserves `sum(dC)` to machine precision
   - leak: `sum(dC_s_in_gap) + sum(dC_overflow_left) == sum(dC)`
   - peak bin (`dC_s[-1]`) is strictly less than original `dC[-1]` for
     any sigma > 0 (because the kernel spreads mass leftward)
2. **Wire into `_template_compensation`.** Capture `target_total` before
   smoothing. Add the if/elif block. With `sigma=None` the test suite
   for existing v1/v2/v3_burst behaviour must pass unchanged.
3. **Plumb the overflow path.** Extend `MergedSequence`,
   `merged_sequences_to_block`, and `hits_to_merged_block` to carry
   `template_overflow`. Skip this step if you only need renormalise mode
   in the short term — it's the bigger refactor.
4. **Expose CLI flags** in the three `deconv_positron_*` scripts and
   forward through `deconv_workflow.process_event_deconvolution`.
5. **Re-run the diagnostics.** Run all four pipelines on TPC 0 with
   `--template-smooth-sigma 1.0` for v2 and v3_burst, then re-run
   `examples/spectra_diagnose.py` and `examples/spectra_compare.py`.
   Expected qualitative outcomes:
   - `diag_residual.png`: the f ≈ 0.13 / 0.20 peaks in the v2 / v3_burst
     residual spectra flatten by an order of magnitude or more.
   - `spectra_compare_ratio.png`: the medium-frequency excess (ratio
     bumps above 1 in 0.15–0.25 cycles/bin) shrinks toward the v1 curve.
   - Total deconvolved charge per event (printed by each script) stays
     within a few-percent of the unfiltered run when `edge_mode =
     renormalize`.

---

## 6. Open design questions (resolve before coding)

- **σ sweep.** What sigma grid to test? Proposed: 0.5, 1.0, 1.5, 2.0 bins.
- **Should the bootstrap (`first_seq`) branch of `_template_compensation`
  also smooth?** It uses a different template slice (line 395). Easiest:
  yes, treat both branches identically.
- **Anchor for v3_burst's selective branch.** v3 picks collection or
  induction template based on charge; both go through the same
  `_template_compensation` path, so one σ controls both. Is that
  desirable, or should collection and induction have their own σ?
- **Default `edge_mode`.** Renormalise is conservative and recommended.
  Leak is physically truer (no charge invented or destroyed at the
  boundary) but mutates recorded-bin amplitudes. Confirm renormalise
  ships as default.

---

## 7. References / verification artefacts

- `output_spectra_compare/diag_residual.png` — residual peaks identified.
- `output_spectra_compare/diag_wiener.png` — confirms that a calibrated
  spectral correction *does* flatten the ratio; the Gaussian template
  smoothing is the in-processor counterpart that achieves the same
  effect without a post-hoc Wiener filter.
- `output_spectra_compare/diag.npz` keys `H_v1`, `H_v2`, `H_v3_burst`
  store the Wiener filter magnitude; useful as a target to compare
  against once smoothing is in place.
