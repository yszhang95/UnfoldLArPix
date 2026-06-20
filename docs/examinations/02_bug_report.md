# Bug Report: UnfoldLArPix

**Date:** 2026-04-13  
**Source revision:** `main` branch (commit d8ea6df)  
**Examination method:** Static analysis — all claims hand-verified by reading source files directly.  
**No code was modified.**

Severity scale:
- **HIGH** — could cause incorrect results silently, or will raise an unhandled exception at runtime.
- **MED** — incorrect in specific conditions, latent mutation bug, or design flaw that could cause issues.
- **LOW** — dead code, style issue, performance-correctness trap, or risk only in edge cases.

---

## HIGH Severity

---

### H1 — Unreachable return + fall-through IndexError in V2 template compensation

**File:** `src/unfoldlarpix/burst_processor_v2.py:303-319`

```python
if not np.any(valid_mask):
    # No template points fit in the gap — append fractional-shifted sequence only
    ...compute updated_times, updated_cumulative...
    if len(updated_cumulative) != len(updated_times) + 1:
        raise ValueError(...)
        return updated_times, ...   # <--- LINE 317: UNREACHABLE (after raise)
    # NO return outside the length-check if-block!

# falls through to:
if not np.isclose(candidate_times[valid_mask][-1], threshold_time):  # LINE 319
```

**What goes wrong:** When `valid_mask` is all-False (no template points fit in the gap), the code correctly computes `updated_times` and `updated_cumulative`. It then checks the length invariant. If the check passes, execution exits the `if len(...)` block — but there is **no `return` statement outside** it. Execution falls through to line 319 where `candidate_times[valid_mask]` is an empty array and `[-1]` raises `IndexError`.

If the length check fails, `raise ValueError` fires — making the `return` on line 317 dead code.

**Impact:** Any event where a pixel's template compensation yields zero valid template points (gap after deadtime is at most one ADC window) crashes with `IndexError` rather than silently falling through to the intended "append sequence only" path.

**How to reproduce:** Create two burst sequences for the same pixel with a gap exactly equal to `tau + adc_hold_delay` and `deadtime` close to `adc_hold_delay`, so `valid_mask` ends up empty after the collision filter.

---

### H2 — Time axis not FFT-padded or rolled: silent circular wrap-around assumption

**Files:** `src/unfoldlarpix/deconv.py:50-52` (padding), `deconv.py:73-74` (rolling)

```python
shape = np.array(measurement.shape)
shape[0] = measurement.shape[0] + (kernel.shape[0] - 1)  # spatial x
shape[1] = measurement.shape[1] + (kernel.shape[1] - 1)  # spatial y
# shape[2] = measurement.shape[2]  ← time axis NOT extended
```

And:
```python
signal = np.roll(signal, (kernel.shape[0] - 1) // 2, axis=0)
signal = np.roll(signal, (kernel.shape[1] - 1) // 2, axis=1)
# axis=2 NOT rolled
```

**What goes wrong:** A linear convolution in the FFT domain requires zero-padding each dimension to at least `len(measurement) + len(kernel) - 1`. The spatial dimensions are padded, but the time axis is not. This means the time-axis deconvolution is a **circular** (not linear) convolution: energy from the kernel tail wraps around from the end of the time window back to the beginning.

The docstring acknowledges this: *"Assume measurement is len(m) = len(s) + len(k) - 1 in time."* The caller in `merged_sequences_to_block` uses `npadbin=50` coarse-bin padding on each end, which partially mitigates the wrap-around, but the adequacy depends on the integrated response kernel's time extent and is not verified.

Additionally, the spatial-axis roll re-centers the output to compensate for the circular-shift introduced by placing the kernel origin at index 0. The time axis is not rolled. This is correct **only if** the integrated response kernel is strictly causal (all mass at `t >= 0`). If the kernel has any non-causal component (pre-trigger charge), the deconvolved output will be time-shifted by that amount.

**Impact:** Potential circular-convolution artifact in time for events where the kernel tail exceeds the 50-bin padding. Potential systematic time-shift if the kernel is not exactly causal.

---

### H3 — Author-flagged bug: first-interval scaling is not cumulative in V2 `_append_shifted`

**File:** `src/unfoldlarpix/burst_processor_v2.py:466`

```python
if gap < self.tau:
    # Signal dipped below threshold — no template needed.
    # FIXME: BUG! this is not cumulative because we correct the first charge. It is an average...
    times, cumulative = self._append_shifted(cumulative, times, curr_seq)
```

Inside `_append_shifted` (lines 364-403):
```python
charges[0] *= self.adc_hold_delay / active_time_first
seq_cumulative = np.cumsum(charges) + cumulative[-1]
```

**What goes wrong:** `seq_cumulative` is a cumulative sum of individually corrected charges. The first charge is rescaled by `adc_hold_delay / active_time_first` to normalize for a shorter-than-normal first integration window. But this rescaling should be applied to the charge **density** (charge per tick), not to the cumulative. The author's own comment flags that the result is effectively an average rather than a proper cumulative integral. The downstream `np.diff(cumulative)` then recovers a corrected charge-per-bin array that may not accurately represent the physical charge over the gap period.

**Impact:** Charge accounting in `gap < tau` transitions is approximate and potentially biased, particularly when `active_time_first` differs significantly from `adc_hold_delay`.

---

### H4 — V1 dead-time compensation rebuilds times uniformly, discarding non-uniform template times

**File:** `src/unfoldlarpix/burst_processor.py:205-261`

In `process_pixel_sequences` (lines 431-489), when `0 < gap <= tau`, a `temp_seq` is built from all previously-merged charges via `np.diff(cumulative)`:

```python
temp_seq = BurstSequence(
    t_first=times[0],
    t_last=times[-1],
    charges=prev_charges,   # all prior merged charges
    ...
)
times, cumulative = self._dead_time_compensation(temp_seq, curr_seq, self.deadtime)
```

Inside `_dead_time_compensation`, new times are regenerated uniformly:
```python
for i in range(len(seq_a.charges)):
    times.append(seq_a.t_first + i * self.adc_hold_delay)   # line 250
```

**What goes wrong:** After one or more template-compensation steps, `times` is **non-uniformly** spaced (template points may be sub-sampled, first times may be at non-uniform offsets). Building `temp_seq` with `t_first=times[0]` and then regenerating as `t_first + i * adc_hold_delay` re-spaces all prior charges uniformly, discarding the original non-uniform time positions. This misaligns the reconstructed waveform against the block grid.

**Impact:** Any V1 pixel that undergoes at least one template compensation followed by a dead-time compensation has systematically incorrect time assignments for the first portion of its merged sequence.

---

## MED Severity

---

### M1 — Even-shaped field response kernel: center pixel is off by one

**Files:** `src/unfoldlarpix/field_response.py:66-83`, `src/unfoldlarpix/deconv_workflow.py:83-85`

`_quadrant_copy` doubles both spatial dimensions:
```python
shape[:2] = shape[:2] * 2   # field_response.py:66
```
If the input quadrant has spatial shape `(r*npath, r*npath)`, the doubled shape is `(2r*npath, 2r*npath)`. After downsampling by `npath`, the final shape is `(2r, 2r)` — **even**.

The center-response is then extracted as:
```python
center_response = full_response[full_response.shape[0] // 2,
                                full_response.shape[1] // 2, :]  # deconv_workflow.py:84
```

For shape `(2r, 2r)`, `shape//2 = r`, so this picks pixel `(r, r)`. The geometric center of the even grid lies between indices `(r-1, r-1)` and `(r, r)` — pixel `(r, r)` is **one pixel toward positive** in both x and y.

The docstring of `_downsample_by_averaging` claims the output is `(2r+1, 2r+1, Nt)` (odd-symmetric), but the implementation produces even spatial extent.

**Impact:** The template used for bootstrap and gap-fill is slightly off-center. This affects the phase of the rising-edge interpolation but is unlikely to cause large charge errors.

---

### M2 — Silent event data drop when location/data batch counts disagree

**File:** `src/unfoldlarpix/data_loader.py:207, 231, 255`

```python
if effq_arrays and len(effq_location_arrays) == len(effq_arrays):
    # create container
    ...
# else: silently does nothing — no warning, no exception
```

Identical pattern for `current` (line 231) and `hits` (line 255).

**What goes wrong:** If the number of data batches collected does not match the number of location batches (e.g., due to a missing `_location` key in the NPZ, a batch ID mismatch, or a partially written file), the entire data type for that event is silently dropped. The event will then either fail downstream when hits are expected, or worse, process with `None` hits/effq and produce empty or incorrect output.

**Impact:** Silent data loss; hard to diagnose because no warning is emitted.

---

### M3 — Ambiguous truthiness test on numpy scalar arrays in `_parse_readout_config`

**File:** `src/unfoldlarpix/data_loader.py:135-137`

```python
uncorr_noise=float(data["uncorr_noise"]) if data["uncorr_noise"] else None,
thres_noise =float(data["thres_noise"])  if data["thres_noise"]  else None,
reset_noise =float(data["reset_noise"])  if data["reset_noise"]  else None,
```

**What goes wrong:** `data["uncorr_noise"]` is a numpy array (0-d or shape `(1,)`). The truthiness test `if data["uncorr_noise"]` is:
- Valid for a 0-d scalar: `bool(np.array(0.0))` is `False`, `bool(np.array(1.5))` is `True`.
- Ambiguous for shape `(1,)`: raises `ValueError: The truth value of an array with more than one element is ambiguous` on numpy versions ≥ 1.25.
- Also a logic error: a noise value of `0.0` is a valid measurement but will be treated as `None`.

**Impact:** May crash during config loading on some file formats; silently stores `None` for a legitimate zero noise value.

---

### M4 — Magnitude-only epsilon clamp injects phase errors into kernel FFT

**File:** `src/unfoldlarpix/deconv.py:62`

```python
epsilon = 1e-10
kernel_fft = np.where(np.abs(kernel_fft) < epsilon, epsilon, kernel_fft)
```

**What goes wrong:** For frequency bins where `|R(k)| < ε`, the complex kernel is replaced with a real positive scalar `ε`. This changes both the magnitude **and the phase** of those bins. For bins that are legitimately small but have a well-defined phase (e.g., bins near a null in the response), this injects an arbitrary phase rotation into the inverse-filtered signal.

A proper regularization would use `max(|R(k)|, ε) * R(k) / |R(k)|` — clamping the magnitude while preserving phase — or a Tikhonov-style `R* / (|R|² + λ)` denominator.

Additionally, `ε = 1e-10` is not scaled by `max|R|`, so its relative significance depends on the normalization of the response.

---

### M5 — `smear_truth.py` padding formula collapses to n_single_side ≈ 1 for typical parameters

**File:** `src/unfoldlarpix/smear_truth.py:20, 49`

```python
n_single_side = int((8 * 1/2/np.pi/width) // n + 1)   # gaus_smear_true, line 20
n_single_side = int((8 * 1/2/np.pi/width[-1]) // n + 1)  # gaus_smear_true_3d, line 49
```

The apparent intent is to pad the time axis by at least `8σ` (8 standard deviations) of the Gaussian in time, so the periodic boundaries of the FFT don't introduce ringing. The formula computes `8 / (2π * width * n)`. For `width = sigma_time ≈ 0.001 – 0.005` and `n = Nt ≈ 10,000+`, this evaluates to `≈ 0` — floor-dividing 0 and adding 1 gives `n_single_side = 1`. Thus `ktimes = 3`, i.e., 3× padding regardless of `sigma_time`.

For small `sigma_time` (sharp Gaussian) the 3× padding may be insufficient; for large `sigma_time` it may be excessive. The formula does not implement the stated 8σ intent.

---

### M6 — `burst_processor.py` dead code before bootstrap

**File:** `src/unfoldlarpix/burst_processor.py:408-410`

```python
# Initialize cumulative with first sequence
cumulative = np.concatenate([[0], np.cumsum(first_seq.charges)])
times = np.array([first_seq.t_first + i * self.adc_hold_delay
                 for i in range(len(first_seq.charges))])

# hypothetical zeroth sequence
times, cumulative, threshold_idx, transit_fraction = self._template_compensation(
    None, None, 0, first_seq, self.threshold, self.template
)
```

The first four lines initialize `cumulative` and `times` from `first_seq.charges`, but both are immediately overwritten by the `_template_compensation` call (with `None, None` as first arguments). The initialization is dead code.

**Impact:** No correctness issue, but creates confusion about the intended initial state.

---

### M7 — `template_section[-1]` divide-by-zero in template normalization

**File:** `src/unfoldlarpix/burst_processor.py:349`

```python
template_section = template_section * (threshold / template_section[-1])  # FIXME
```

**What goes wrong:** If `template_section` is all-zero or `template_section[-1] == 0` (can happen if `threshold_idx == 0` and the template is flat-zero at the start), this divides by zero, producing NaN or Inf. The `FIXME` comment acknowledges the assumption that the cumulative template saturates at 1, but does not guard against the zero case.

**Impact:** NaN propagation through subsequent `np.diff`, `np.cumsum`, and eventually into `block_charges`.

---

### M8 — `hit_to_wf.py:convert_bin_wf_to_blocks` mutates input `wf.location` in place

**File:** `src/unfoldlarpix/hit_to_wf.py:102-108`

```python
wfloc = wf.location   # NOT a copy — shares buffer with wf.location
if shift_to_center:
    wfloc[:, 2] = (wfloc[:, 2] + 0.5*bin_size) // bin_size
else:
    wfloc[:, 2] = wfloc[:, 2] // bin_size
```

`wfloc` is a direct reference to `wf.location`. The division modifies the time column of the caller's `Current` object in place. Any subsequent use of `wf.location` will see the modified (coarse-bin) times, not the original fine-grained times.

---

### M9 — `TrueHits.location` shares object reference with `Hits.location`

**File:** `src/unfoldlarpix/data_loader.py:272-277`

```python
event_data.truehits = TrueHits(
    data=merged_data,
    location=event_data.hits.location,   # same array, not a copy
    ...
)
```

Any code that modifies `truehits.location` in place will silently also modify `hits.location`. This is documented in the code but has no guard.

---

### M10 — `allow_pickle=True` blocks memory-mapped NPZ loading

**File:** `src/unfoldlarpix/data_loader.py:38`, `src/unfoldlarpix/field_response.py:35`

```python
self._data = dict(np.load(self.npz_path, allow_pickle=True))
```

`np.load(..., allow_pickle=True)` and `mmap_mode='r'` are mutually exclusive. With `dict(...)` wrapping, all arrays are immediately materialized into RAM regardless. For large NPZ files (all batches, all TPCs), this causes large peak RAM usage at startup before any event processing begins. See memory audit for details.

---

### M11 — Stray `print()` calls in production path

**Files:**  
- `src/unfoldlarpix/burst_processor.py:544` — `print(pmin, pmax)` (in `merged_sequences_to_block`)  
- `src/unfoldlarpix/burst_processor.py:551` — `print(offset)` (in `merged_sequences_to_block`)  
- `src/unfoldlarpix/hit_to_wf.py:44` — `print('fracs', fracs)`  
- `src/unfoldlarpix/hit_to_wf.py:57` — `print(ind_max, indices, len(template))`  
- `src/unfoldlarpix/hit_to_wf.py:103` — `print('--- loc', wfloc)`  

These are debug prints left in production code. They produce per-event console output that pollutes batch-job logs and can significantly slow down runs processing thousands of events (stdout I/O is not free).

---

## LOW Severity

---

### L1 — `totq_per_pix` never cleared between events (if processor is reused)

**Files:** `src/unfoldlarpix/burst_processor.py:100`, `src/unfoldlarpix/burst_processor_v2.py:68`

```python
self.totq_per_pix: Dict[Tuple[int, int], float] = {}
```

`process_hits` resets `template_compensation_anchors` at the start (`burst_processor.py:512`), but does **not** reset `totq_per_pix`. If the same processor instance is reused across events, this dict accumulates entries for every pixel ever processed, growing unboundedly across a long job.

---

### L2 — `bata` dead variable in `convert_bin_wf_to_blocks`

**File:** `src/unfoldlarpix/hit_to_wf.py:135`

```python
bata = np.add.at(bdata, indices, wf.data.flatten())
```

`np.add.at` returns `None`; `bdata` is modified in place. The variable `bata` is assigned `None` and never used. The confusion between `bata` and `bdata` suggests this was a transcription error.

---

### L3 — `hits_to_bin_wf` divide-by-zero on zero-signal pixels

**File:** `src/unfoldlarpix/hit_to_wf.py:43`

```python
fracs = threshold / np.max(recorded_charges[:, :], axis=1)
```

For a pixel with all-zero charges, `np.max = 0`, giving `inf`. The following `np.where(fracs < 1, fracs, 1.0)` clamps `inf → 1.0`, which then causes `np.searchsorted(cum_template, 1.0)` to return the full template length — inserting a full template for a zero-signal pixel. A RuntimeWarning is also raised for the divide.

---

### L4 — Python for-loops in hot code paths

The following loops are correct but sequential and not vectorized. They are performance bottlenecks that also prevent GPU batching (detailed in efficiency report):

- `burst_processor.py:145` / `burst_processor_v2.py:152` — per-hit loop in `extract_sequences_from_hits`
- `burst_processor.py:517` / `burst_processor_v2.py:538` — per-pixel loop in `process_hits`
- `burst_processor.py:308` / `burst_processor_v2.py:278` — linear search for `threshold_idx` in `_template_compensation`; replaceable with `np.argmax(np.convolve(...) >= transit)`
- `smear_truth.py:17, 46` — per-tick scatter loop in `gaus_smear_true(_3d)`
- `burst_processor.py:555` — per-pixel scatter in `merged_sequences_to_block`

---

### L5 — `_quadrant_copy` reasserts equal spatial dims but silently accepts unequal input

**File:** `src/unfoldlarpix/field_response.py:62-65`

The validation `if shape[0] != shape[1]: raise` comes after `shape = np.array(raw.shape)`. If the caller passes a quadrant that is not square, the error fires. The documentation says "one quadrant of a symmetric 2D plane", implying square input. This is adequate but relies on the caller enforcing it.

---

### L6 — Commented-out monotonicity check in V1

**File:** `src/unfoldlarpix/burst_processor.py:297-298`

```python
# if not np.all(np.diff(template_cumulative) >= 0):
#     raise ValueError("Template must be monotonically increasing.")
```

The check is performed once at `__init__` (line 97) on `self.template`, but inside `_template_compensation` the template is converted via `np.asarray(template_cumulative, dtype=float)` (line 294) and the monotonicity check is commented out. If a caller passes a non-monotone template at the call site it would not be caught.

---

## Ruled-Out False Positive

The following was flagged during an automated audit but verified by hand-tracing to be **correct code**:

### NOT A BUG — `gaussian_filter_3d` shape is correct

**File:** `src/unfoldlarpix/deconv.py:21-29`

```python
def gaussian_filter_3d(s, dt, sigma):
    freqs = fft.rfftfreq(s[-1], d=dt[-1])              # shape: (s[-1]//2+1,)
    gaussian = np.exp(-0.5 * freqs**2/sigma[-1]**2)    # shape: (s[-1]//2+1,)
    for i in range(len(s[:-1])):                        # i = 0, 1 for 3D
        freqs_i = fft.fftfreq(s[i], d=dt[i])           # shape: (s[i],)
        gaussian_i = np.exp(-0.5 * freqs_i**2/sigma[i]**2)
        gaussian = gaussian_i[None, :] * gaussian[..., None]
        # i=0: (1, s[0]) * (s[-1]//2+1, 1) → (s[-1]//2+1, s[0])
        # i=1: (1, s[1]) * (s[-1]//2+1, s[0], 1) → (s[-1]//2+1, s[0], s[1])
    gaussian = np.moveaxis(gaussian, 0, -1)
    # → (s[0], s[1], s[-1]//2+1)   ✓ correct rfftn shape
    return gaussian
```

After the loop, `gaussian` has shape `(s[-1]//2+1, s[0], s[1])`. After `moveaxis(0, -1)` it becomes `(s[0], s[1], s[-1]//2+1)`, which is exactly the rfft half-spectrum shape for a real 3D array of shape `(s[0], s[1], s[-1])`. The same logic in `gaus_smear_true_3d` (`smear_truth.py:56-62`) is identical and also correct.

**Do not re-open this item.**
