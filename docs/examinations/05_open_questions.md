# Open Questions: UnfoldLArPix

**Date:** 2026-04-13  
**Purpose:** Items where static code analysis reaches its limits. These require either domain knowledge, a running test, or access to real input data to resolve definitively.

---

## Q1 — Is the integrated field response kernel strictly causal?

**Relevant code:** `deconv.py:73-74` (no time-axis `np.roll`); `field_response.py:154-207`

The `deconv_fft` function rolls the output along axes 0 and 1 (spatial) by `(kernel.shape[i]-1)//2` to compensate for the circular-shift introduced by kernel origin at index 0. The time axis is **not rolled**.

This is correct *if and only if* the integrated response kernel has all of its mass at `t ≥ 0` (i.e., no pre-trigger component). From physics: charge induced on the collecting pixel rises after the drifting charge arrives (causal). However:

- The raw field response is a time derivative of the induced current profile. After integration over `adc_hold_delay` ticks and time-reversal for convolution (`_flip_kernel_for_convolution`), does the time origin align exactly with index 0?
- If the raw response NPZ contains a pre-trigger baseline or the flip shifts the origin, the kernel will have non-causal content and the deconvolved output will be systematically time-shifted.

**To resolve:** Print `np.argmax(integrated_response[r, r, :])` for the center pixel. If it is at index 0 (or very close), the kernel is effectively causal. If the peak is at a large index, a time-axis roll of `peak_index` would be needed.

---

## Q2 — Is the 50-bin pre-pad sufficient to prevent time-axis circular wrap-around?

**Relevant code:** `merged_sequences_to_block` (`burst_processor.py:540`, `npadbin=50`); `deconv.py:50-52`

The time axis is not FFT-padded in `deconv_fft`. The docstring says the caller must ensure `len(measurement_t) = len(signal_t) + len(kernel_t) - 1`. With `npadbin=50` coarse bins of padding on each end, the available "slack" is `50 * adc_hold_delay` fine ticks on each side.

The question is: how many coarse bins does the integrated response kernel span in time (`Kt_coarse = integrated_response.shape[-1]`)? If `Kt_coarse ≤ 50`, the 50-bin pad is sufficient to avoid wrap-around on both sides. If events can have response kernels longer than 50 coarse bins, artifacts will appear.

**To resolve:** Print `prepare_field_response(...).integrated_response.shape[-1]` for the actual field response file. Compare against `npadbin=50`.

---

## Q3 — Does `template_section[-1]` ever equal zero?

**Relevant code:** `burst_processor.py:349`, `burst_processor_v2.py:327`

```python
template_section = template_section * (threshold / template_section[-1])
```

This normalizes the template section so its last value equals `threshold`. If `template_section[-1] == 0` (the last element of the downsampled template section is zero), a divide-by-zero produces NaN.

When can this happen? The template is `np.cumsum(center_response)`. Since `center_response` is the center pixel's field response (always ≥ 0 with sum = 1), its cumsum is monotonically non-decreasing. The downsampled slice `template_section = template_cumulative[a:b:step]` at the end of the template will be non-zero if the template has any content after position `a`. However, if `a > len(template_cumulative)` or if the downsampling step skips all non-zero entries, `template_section` could end up all-zero.

**To resolve:** Add an assertion or print in a test run: `assert template_section[-1] > 0` before the normalization, and observe if it ever fires.

---

## Q4 — What does `true_charge[i, -1]` represent in `gaus_smear_true(_3d)`?

**Relevant code:** `smear_truth.py:18, 47`

```python
data[tuple(ticks[i] - loc_min)] += true_charge[i, -1]
```

This uses only the **last column** of `true_charge` for each hit. The assumption is that `true_charge[:, -1]` holds the total integrated charge for that hit. 

`TrueHits.data` is described as "each column is a charge value" (a burst-format: `(N, T)` where T is the number of burst time bins). `true_charge[i, -1]` would then be the charge in the **last burst** only — not the total charge.

Alternatively, if the NPZ stores cumulative charges in the last column (matching `Hits.data` column convention where the last of `N+3` columns is the cumulative total), then `[:, -1]` would be the correct total.

**To resolve:** Print `event.effq.data[0, :]` for a real event and check whether the last column is monotonically increasing (cumulative) or a burst increment. If it's a burst increment, `true_charge[i, -1]` takes only the last burst, not the total charge, which would undercount truth charge for multi-burst pixels.

---

## Q5 — How often does `gap < tau` trigger vs. `gap >= tau` in real data?

**Relevant code:** `burst_processor_v2.py:464`, `burst_processor.py:438`

The processing branch taken (dead-time scaling vs. template compensation) depends on the ratio of `gap` to `tau`. `tau` defaults to `adc_hold_delay` (`deconv_workflow.py:104`). For real LArPix data, if most gaps are larger than one ADC window (i.e., `gap >= tau` dominates), then bug H3 (the author-flagged bug in `_append_shifted`) rarely fires and its impact is small. If `gap < tau` is common (dense ionization, frequent below-threshold dips), bug H3 is a major charge-accounting error.

**To resolve:** Add a counter inside `process_pixel_sequences` that tracks how many times each branch is taken per event. Print the ratio after processing a representative event.

---

## Q6 — What is the actual time extent of the integrated response kernel?

**Relevant code:** `field_response.py:56-71`, `deconv_workflow.py:56-71`

The fine-tick field response can span hundreds to thousands of fine ticks (depending on the drift velocity and detector geometry). After integration by `adc_hold_delay`, the coarse kernel extent is `Nt_fine / adc_hold_delay`. The deconvolution quality and the validity of the time-padding assumption (Q2) both depend on this.

**To resolve:** Print `prepared_response.integrated_response.shape` and `np.sum(prepared_response.integrated_response[r, r, :] > 1e-4)` to see how many coarse bins have significant content.

---

## Q7 — Is `data["event_id"]` consistently stored as a 0-d array or shape (1,)?

**Relevant code:** `data_loader.py:164-165`

```python
event_id = data[event_key]       # 0-d or (1,) numpy array
group_key = (tpc_id, int(event_id))
```

`int(np.array(5))` works. `int(np.array([5]))` also works. But `data[event_key] == event_id` (line 204) compares a loaded array against an int: for a 0-d array this returns a 0-d bool (truthy); for shape `(1,)` it returns a `(1,)` bool array — and `if np.array([True])` raises `DeprecationWarning` and evaluates as `True` in older numpy but raises `ValueError` in ≥ 1.25.

**To resolve:** Print `data[event_key].shape` and `data[event_key].dtype` for your actual NPZ file. If shape is `()` (0-d), the code is fine. If `(1,)`, add an explicit `int(data[event_key].flat[0])` conversion.

---

## Q8 — Is `sigma_time` / `sigma_pixel` a time-domain or frequency-domain width?

**Relevant code:** `deconv.py:8-19` (docstring says "time domain"), `deconv.py:18` (uses `exp(-f²/σ²)` directly)

The Gaussian filter is `exp(-0.5 * freqs² / sigma²)` where `freqs = rfftfreq(n, d=dt)`. If σ is a time-domain standard deviation (units: ticks), then the frequency-domain Gaussian should be `exp(-0.5 * (2π * freqs)² * sigma²)` — there is a factor of `2π` difference. The current code treats σ as a frequency-domain width.

In the example scripts, `sigma_time ≈ 0.001 – 0.005`. Whether this should be interpreted as "0.001 per tick" (frequency units) or "1/0.001 = 1000 ticks standard deviation in time" (time units) affects the actual resolution. The comparison with `smeared_true` (which uses the same sigma and the same formula) is self-consistent either way, but the *absolute* resolution of the deconvolution depends on the correct interpretation.

**To resolve:** Check whether the effective resolution (width of the deconvolved point-spread function) matches expectations from the sigma value in physical units.
