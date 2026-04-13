# Efficiency and GPU Analysis: UnfoldLArPix

**Date:** 2026-04-13  
**Source revision:** `main` branch (commit d8ea6df)  
**Status:** Pure numpy (no cupy/torch/numba/jax anywhere in `src/unfoldlarpix/`)  
**All GPU discussion is prospective — describing what a port would require.**

---

## 1. Work Mix and Dominant Costs

### 1.1 For Large Events (many pixels, long time window)

The dominant compute cost is the **three rFFTn / irFFTn calls** inside `deconv_fft` (`deconv.py:55, 58, 70`) and the rFFTn / irFFTn pair inside `gaus_smear_true_3d` (`smear_truth.py:64`).

Each operates on a volume of `(Nx+2r-1) × (Ny+2r-1) × Nt` real points, producing a half-complex array of `(Nx+2r-1) × (Ny+2r-1) × (Nt/2+1)` complex points. For a 30×30 pixel event with Nt = 16,500:

| Operation | Approx work | Comments |
|-----------|-------------|---------|
| `rfftn(measurement)` | O(N log N) | N ≈ 34 × 34 × 16500 ≈ 19 M |
| `rfftn(kernel)` | O(N log N) | Same shape, kernel is compact |
| Element-wise `/`, `*` (complex) | O(N/2) | N/2 complex points |
| `np.where` ε-clamp | O(N/2) | Full complex-array copy (see memory) |
| `irfftn` | O(N log N) | Heaviest: inverse + normalization |
| Two `np.roll` | O(N) copies | Avoidable (see §5) |
| `rfftn / irfftn` (smear_truth) | O(M log M) | M = ktimes * N_effq, often ≈ 3 × N |

Roughly 5 FFTs dominate at ~80–90% of wall time for large events.

### 1.2 For Small / Sparse Events

The FFTs are fast (small N). The dominant cost shifts to the **Python-level burst-processing loops**:

- `extract_sequences_from_hits` — Python `for i in range(len(hits))` (`burst_processor.py:145`, `v2:152`)
- `process_pixel_sequences` — Python `while i < len(sequences)` or `for curr_seq in sequences[1:]` (`burst_processor.py:431`, `v2:459`), each calling `_template_compensation` or `_append_shifted`
- `_template_compensation` inner search — Python `for jidx in range(tlength, len(template_cumulative))` (`burst_processor.py:308`, `v2:278`)
- `merged_sequences_to_block` scatter — Python `for pixel_key, merged_seq in merged_seqs.items()` (`burst_processor.py:555`)
- `gaus_smear_true_3d` scatter — Python `for i in range(ticks.shape[0])` (`smear_truth.py:46`)

---

## 2. FFT Plan-Reuse and Shape Instability

### 2.1 Variable FFT Shapes

`deconv_fft` computes its shape as:
```python
shape[0] = measurement.shape[0] + (kernel.shape[0] - 1)   # deconv.py:51
shape[1] = measurement.shape[1] + (kernel.shape[1] - 1)   # deconv.py:52
shape[2] = measurement.shape[2]                            # unchanged
```

`measurement.shape` is the event bounding box — different for every event. Both `pocketfft` (numpy's FFT backend) and `cuFFT` (cupy's backend) cache twiddle factors keyed on shape. With per-event variable shapes, every event incurs a **plan build cost** rather than reusing a cached plan.

### 2.2 Non-Smooth FFT Sizes

`Nt = ceil((tmax - tmin + 2*pad_length) / bin_size) + 1`

This is an arbitrary integer that is almost never a product of small primes (2, 3, 5). Both `pocketfft` and `cuFFT` are slowest for large prime factors. For example, `Nt = 16,501 = 7 × 2357` is significantly slower than `Nt = 16,500 = 2² × 3 × 5³ × 11` or the next power-of-2, `Nt = 16,384`.

**Recommendation (CPU win, no code structure change needed):** Before calling `deconv_fft`, pad `Nt` up to the next 2-3-5 smooth integer. This can yield 2–5× FFT speedup on CPU and significantly better GPU plan efficiency. See open questions for whether this changes the output.

### 2.3 Gaussian Kernel Rebuilt Per Event

`build_gaussian_deconv_kernel` is called inside `process_event_deconvolution` (`deconv_workflow.py:289`). Because `block_shape` varies per event, a new Gaussian filter tensor is built every event — this is an element-wise computation over the full `(Nx+2r-1, Ny+2r-1, Nt/2+1)` half-spectrum, not free for large events.

If events were padded to a fixed `(Nx_max, Ny_max, Nt_max)` envelope, the Gaussian kernel could be built once and reused across all events.

---

## 3. Hot Python Loops to Vectorize Before Any GPU Port

The following loops are in CPU-critical paths. Vectorizing them will improve CPU performance and is a prerequisite for effective GPU use, because Python loops cannot be dispatched to a GPU.

### 3.1 `extract_sequences_from_hits` — per-hit loop

**Location:** `burst_processor.py:145`, `burst_processor_v2.py:152`

Current code iterates over every hit row individually with Python, building numpy arrays inside the loop. The entire function could be vectorized:

```python
# Current:
for i in range(len(hits)):
    charges = np.array([raw[0]] + np.diff(raw).tolist())
    ...build BurstSequence...

# Vectorizable:
raw = hits.data[:, 3:]                          # (N, nburst)
charges_2d = np.diff(raw, prepend=raw[:, :1], axis=1)  # (N, nburst)
pixel_keys = hits.location[:, :2]               # (N, 2)
trigger_times = hits.location[:, 2]             # (N,)
# Then group by pixel_key using np.unique or a dict comprehension over arrays
```

### 3.2 `_template_compensation` threshold search

**Location:** `burst_processor.py:308`, `burst_processor_v2.py:278`

Current code:
```python
for jidx in range(tlength, len(template_cumulative)):
    if template_cumulative[jidx] - template_cumulative[jidx - tlength] >= transit:
        threshold_idx = jidx
        break
```

This is a linear search over the cumulative template. It can be replaced with:
```python
diffs = template_cumulative[tlength:] - template_cumulative[:len(template_cumulative)-tlength]
valid = np.where(diffs >= transit)[0]
threshold_idx = int(valid[0]) + tlength if len(valid) > 0 else len(template_cumulative) - 1
```

A one-line O(len(template)) numpy operation instead of a Python loop.

### 3.3 `merged_sequences_to_block` scatter

**Location:** `burst_processor.py:555-563`

Current code scatters per-pixel in Python:
```python
for pixel_key, merged_seq in merged_seqs.items():
    block_charges[pix_inds[0], pix_inds[1], tinds.astype(int)] = charges
```

Can be vectorized by concatenating all pixel sequences and using `np.add.at` on raveled indices — or even better, `np.ravel_multi_index` + `np.bincount`:
```python
flat_inds = np.ravel_multi_index((all_px, all_py, all_tinds), shape)
block_flat = np.bincount(flat_inds, weights=all_charges, minlength=np.prod(shape))
block_charges = block_flat.reshape(shape)
```
`np.bincount` is internally compiled and significantly faster than a Python loop over pixels.

### 3.4 `gaus_smear_true_3d` scatter

**Location:** `smear_truth.py:46-47`

```python
for i in range(ticks.shape[0]):
    data[tuple(ticks[i] - loc_min)] += true_charge[i, -1]
```

This scatter can be written as:
```python
indices = tuple((ticks[:, j] - loc_min[j]) for j in range(ticks.shape[1]))
np.add.at(data, indices, true_charge[:, -1])
```
`np.add.at` handles repeated indices correctly (unlike direct fancy indexing) and is vectorized over the truth points.

---

## 4. cupy Drop-In Compatibility

The majority of the numerical computation uses standard numpy APIs that cupy mirrors. A rough port could be done by replacing `import numpy as np` with `import cupy as np` in `deconv.py`, `smear_truth.py`, and `deconv_workflow.py`. However, the following specific issues need attention:

| Location | numpy call | cupy situation |
|----------|-----------|----------------|
| `hit_to_wf.py:135` | `np.add.at(bdata, indices, ...)` | cupy does not implement `ufunc.at`; use `cupyx.scatter_add` instead |
| `hit_to_wf.py:109` | `np.unique(wfloc, axis=0, return_inverse=True, return_counts=True)` | cupy's `unique` supports `axis` but complex multi-return on 2D rows has been incomplete in some versions; verify |
| `burst_processor*.py` all | Per-pixel Python control flow | Cannot run on GPU. Must restructure as batched array operations first |
| `data_loader.py:38-39` | `np.load(..., allow_pickle=True)` | Must stay host-side; arrays to be processed on GPU need explicit `.get()` / `.put()` transfers |
| `deconv.py:62` | `np.where(np.abs(kernel_fft) < epsilon, epsilon, kernel_fft)` | Works in cupy but creates a full host-sized temporary; see §5 for a better approach |
| `np.roll` × 2 | `deconv.py:73-74` | Works in cupy but copies the full volume twice |

For the FFT path specifically (`deconv_fft`, `gaus_smear_true_3d`), the port to cupy is essentially drop-in once arrays are on the device.

**Biggest obstacle:** The burst processing (`BurstSequenceProcessor`, `BurstSequenceProcessorV2`) is heavily Python-loop-based and cannot benefit from GPU dispatch in its current form. The sequence merge itself (`_template_compensation`) depends on cumulative running state that is inherently sequential within a pixel. A GPU-friendly formulation would require re-expressing the merge as a vectorized scan (prefix-sum style) across all pixels simultaneously — a significant algorithmic restructure.

---

## 5. Low-Hanging CPU Wins (No GPU Required)

### 5.1 Absorb `np.roll` into a frequency-domain phase shift

**Location:** `deconv.py:73-74`

The two `np.roll` calls each copy the full `(Nx+2r-1, Ny+2r-1, Nt)` real array. This is avoidable. Instead of rolling in the spatial domain, apply a linear-phase correction to `signal_fft` before `irfftn`:

```python
# Instead of np.roll(signal, n, axis=0), multiply signal_fft by exp(2πi * n * kx / Nx)
n0 = (kernel.shape[0] - 1) // 2
n1 = (kernel.shape[1] - 1) // 2
kx = np.fft.fftfreq(fft_shape[0])[:, None, None]   # (Nx, 1, 1)
ky = np.fft.fftfreq(fft_shape[1])[None, :, None]   # (1, Ny, 1)
phase = np.exp(2j * np.pi * (n0 * kx + n1 * ky))
signal_fft *= phase   # applied before irfftn
```

This eliminates two full array copies at the cost of one multiply on the half-spectrum (already in memory for the Gaussian filter step).

### 5.2 Pad Nt to next 2-3-5 smooth size

Compute `Nt_smooth = next_smooth_size(Nt)` before calling `deconv_fft`. Can give 2–5× speedup on the FFT calls with zero impact on results (the extra output samples are cropped away).

```python
def next_smooth_size(n, max_factor=5):
    """Return smallest integer >= n whose only prime factors are 2, 3, 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1
```

### 5.3 Multiprocessing over events

Events are fully independent. A `multiprocessing.Pool` (or `concurrent.futures.ProcessPoolExecutor`) over `loader.iter_events()` would parallelize across CPU cores with no algorithmic changes.

The main concern is memory: if each worker loads its own copy of the NPZ (due to `dict(np.load(...))` materializing all data), RAM could be a constraint. Using `mmap_mode='r'` for NPZ loading would allow workers to share physical pages.

### 5.4 Reuse the Gaussian kernel when event shape is fixed

If processing multiple events from the same acquisition and events are padded to a common bounding box, call `build_gaussian_deconv_kernel` once and pass the cached result to all `process_event_deconvolution` calls.

### 5.5 Delete FFT intermediates before irfftn

```python
measurement_fft = fft.rfftn(measurement, s=shape)
kernel_fft = fft.rfftn(kernel, s=shape)
# epsilon guard...
signal_fft = measurement_fft / kernel_fft
del measurement_fft, kernel_fft   # free ~2 × complex128 half-spectra here
signal_fft *= filter_fft
signal = fft.irfftn(signal_fft, s=shape)
del signal_fft
```

This reduces peak complex memory from 3× to 1× the half-spectrum size during `irfftn` (see memory audit).

---

## 6. Natural GPU Batch Dimensions

From coarsest to finest:

| Level | Dimension | Independent? | Current status |
|-------|-----------|-------------|----------------|
| Event batch | Multiple events | Yes | Serial Python loop in example scripts |
| TPC-within-event | Multiple TPCs per event | Yes | Serial in `iter_events` |
| Pixel-within-event | Multiple pixels per event | Yes (per-pixel merging) | Serial Python loop |
| Burst-within-pixel | Multiple bursts per pixel | **No** — sequential cumulative | Not parallelizable directly |

**Recommended GPU batch strategy:**

1. Pad all events in a batch to the same `(Nx_max, Ny_max, Nt_max)` bounding box.
2. Stack as a batch dimension: `measurement_batch` shape `(B, Nx_max, Ny_max, Nt_max)`.
3. Call `cupy.fft.rfftn(measurement_batch, s=(Nx_fft, Ny_fft, Nt_fft), axes=(1,2,3))` — a single batched 3D FFT over B events.
4. The kernel FFT is identical for all events in a batch: precompute once.
5. Element-wise ops broadcast across the batch dimension.
6. `irfftn` — again batched.
7. The burst-processor loop still runs on CPU per pixel, but its output (the dense block) is transferred to GPU for FFT.

For a batch of B=8 events on a GPU with 40 GB VRAM, each event's FFT working set of ~600 MB × 3 complex arrays = 1.8 GB fits (at float32 this drops to ~900 MB for 8 events).

---

## 7. Cross-Reference: Bug H1 and Parallelism

Bug H1 (`burst_processor_v2.py:312-317`, unreachable return + IndexError) will be triggered at higher rates if the burst processor is run with multi-threaded or multi-processed per-pixel parallelism, because the edge case (zero valid template points) becomes more likely as more pixels and events are processed. This bug must be fixed before any parallelization effort — otherwise parallel workers will crash non-deterministically.

---

## 8. Summary: Recommended Priority Order

| Priority | Action | Benefit |
|----------|--------|---------|
| 1 | Fix bug H1 (V2 IndexError) | Correctness prerequisite for all further work |
| 2 | Vectorize `merged_sequences_to_block` scatter | CPU speedup for medium/small events |
| 3 | Vectorize `gaus_smear_true_3d` scatter | CPU speedup |
| 4 | Vectorize `_template_compensation` threshold search | CPU speedup per-pixel |
| 5 | Pad `Nt` to next smooth size | 2–5× FFT speedup, zero output change |
| 6 | Replace `np.roll` with frequency-domain phase | Saves 2 large array copies |
| 7 | `del` FFT intermediates before `irfftn` | Halves peak complex memory |
| 8 | Multiprocess over events | Near-linear CPU core scaling |
| 9 | Switch to float32 throughout FFT path | Halves FFT memory, ≈2× GPU throughput |
| 10 | Event-batch GPU port (cupy + fixed envelope) | Orders-of-magnitude speedup for large events |
