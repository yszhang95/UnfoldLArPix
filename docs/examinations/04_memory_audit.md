# Memory Audit: UnfoldLArPix

**Date:** 2026-04-13  
**Source revision:** `main` branch (commit d8ea6df)  
**Measurement basis:** Static analysis of numpy allocations; no profiling run.  
**All GPU memory estimates are prospective.**

---

## 1. Peak Memory Formula (Single Event)

The following arrays are live simultaneously at peak inside `process_event_deconvolution` → `deconv_fft`:

| Array | Shape | dtype | Size (float64) |
|-------|-------|-------|----------------|
| `block_data` (measurement) | `(Nx, Ny, Nt)` | float64 | `Nx·Ny·Nt·8` B |
| `measurement_fft` | `(Nx+2r-1, Ny+2r-1, Nt/2+1)` | complex128 | `(Nx+2r-1)·(Ny+2r-1)·(Nt/2+1)·16` B |
| `kernel_fft` | same | complex128 | same |
| `signal_fft` = `Y/R` | same | complex128 | same |
| `filter_fft` (Gaussian) | same | float64 (broadcast) | `(Nx+2r-1)·(Ny+2r-1)·(Nt/2+1)·8` B |
| `integrated_response` | `(2r, 2r, Nt_coarse)` | float64 | small (compact kernel) |
| `np.where` temporary on `kernel_fft` | same as `kernel_fft` | complex128 | same |
| `deconv_q` (output) | `(Nx, Ny, Nt)` | float64 | `Nx·Ny·Nt·8` B |

The `np.where` call at `deconv.py:62` allocates a fourth complex half-spectrum copy before `signal_fft` is computed. So peak complex memory is effectively **4 × (Nx+2r-1)·(Ny+2r-1)·(Nt/2+1)·16 B**.

### 1.1 Concrete Estimates

Using `2r = 6` (typical 3-pixel-each-side kernel), `Nt ≈ 16,500` coarse bins:

**30×30 pixel event:**

| Item | Size |
|------|------|
| `block_data` | 30 × 30 × 16500 × 8 = **119 MB** |
| Each complex half-spectrum | 35 × 35 × 8251 × 16 = **162 MB** |
| 4 complex arrays (peak) | 4 × 162 = **648 MB** |
| Gaussian filter (real) | 35 × 35 × 8251 × 8 = **81 MB** |
| `deconv_q` (output) | 119 MB |
| **Peak inside deconv_fft** | ≈ **1.0 GB** |

**64×64 pixel event:**

| Item | Size |
|------|------|
| `block_data` | 64 × 64 × 16500 × 8 = **541 MB** |
| Each complex half-spectrum | 69 × 69 × 8251 × 16 = **629 MB** |
| 4 complex arrays (peak) | 4 × 629 = **2.5 GB** |
| Gaussian filter (real) | 314 MB |
| `deconv_q` (output) | 541 MB |
| **Peak inside deconv_fft** | ≈ **4.0 GB** |

These numbers are for CPU RAM. On GPU (VRAM), the same tensors would occupy the same space; the advantage of GPU is throughput, not smaller tensors.

### 1.2 Additional Peaks from Ground-Truth Smearing

`gaus_smear_true_3d` (`smear_truth.py:34-69`) allocates:
- `data`: scatter grid over `effq` bounding box — typically similar to or smaller than `block_data`
- `oshape[-1] = ktimes * Nt_effq` with `ktimes ≈ 3` (see bug M5) → padded FFT input ~3× time extent
- `rfftn(data, s=oshape)` → complex half-spectrum at the padded size

This is an additional ~1.5–3× of the effq volume in complex memory, alive simultaneously with `EventDeconvolutionResult`.

---

## 2. Unnecessary Copies

### 2.1 `np.where` on `kernel_fft` (deconv.py:62)

```python
kernel_fft = np.where(np.abs(kernel_fft) < epsilon, epsilon, kernel_fft)
```

Creates a **new full complex128 array** equal in size to `kernel_fft`. The original `kernel_fft` is then orphaned until GC. This is the fourth complex copy at peak. Better: use in-place masking or absorb the guard into the division step.

### 2.2 Two `np.roll` calls (deconv.py:73-74)

```python
signal = np.roll(signal, (kernel.shape[0] - 1) // 2, axis=0)
signal = np.roll(signal, (kernel.shape[1] - 1) // 2, axis=1)
```

Each `np.roll` creates a **full copy** of the real-valued `signal` array (shape `(Nx+2r-1, Ny+2r-1, Nt)`). These are back-to-back copies adding ~2 × `block_data` size to peak RAM after `irfftn`. Can be eliminated by applying a phase shift to `signal_fft` before `irfftn` (see efficiency report §5.1).

### 2.3 `_flip_kernel_for_convolution` (field_response.py:149)

```python
reshaped = np.flip(reshaped, axis=(0, 2)).reshape(...)
```

`np.flip` returns a view, but the subsequent `.reshape(...)` forces a copy if the flipped strides are not contiguous (and a `flip` of a C-contiguous array along axis 0 is not C-contiguous). This creates one copy of `expanded_response` at shape `(2r*npath, 2r*npath, Nt_fine)`. This happens only once at startup during `prepare_field_response`, not per-event.

### 2.4 Per-hit list→numpy round-trip (burst_processor.py:155)

```python
charges = [charges[0],] + np.diff(charges).tolist()
charges = np.array(charges)
```

Converts numpy → Python list → numpy. For N hits per pixel and many pixels, this is O(N * nburst) Python object allocations. Small but done in a tight loop.

### 2.5 `broadcast_to(...).copy()` in hit_to_wf.py:122

```python
local_ind_full = np.broadcast_to(local_ind[:, None, :],
                                  (local_ind.shape[0], nt, 3)).copy()
```

Explicitly materializes the broadcast view. For a large event with many pixels and many time bins, `local_ind_full` has shape `(N_pix, nt, 3)` and is a concrete copy in RAM, used only for indexing.

### 2.6 `_fractional_shift` padding (burst_processor_v2.py:120)

```python
padded = np.concatenate([np.zeros(pad), charges, np.zeros(pad)])
```

Creates a 3× sized copy per per-pixel contiguous block per Phase-2 pass. Small per-call but called once per unique delta_T block per pixel — potentially many calls.

---

## 3. Long-Lived Intermediates

### 3.1 Three FFT intermediates live until function return (deconv.py)

Inside `deconv_fft`, `measurement_fft`, `kernel_fft`, and `signal_fft` are all bound as locals and remain allocated until the function exits. In Python, local variables keep their reference alive until the frame is popped.

After `signal_fft = measurement_fft / kernel_fft` (line 65), both `measurement_fft` and `kernel_fft` are dead. Adding `del measurement_fft, kernel_fft` immediately after line 65 would free ~2 × complex half-spectrum before `irfftn`, reducing peak by roughly one-third.

After `signal_fft *= filter_fft` (line 67), `filter_fft` is dead. It is passed as an argument so Python won't GC it until the caller releases it, but the signal_fft processing is done — no further use. Adding `del signal_fft` after `irfftn` (line 70) and before the `np.roll` calls frees the half-spectrum before the two roll copies.

### 3.2 `EventDeconvolutionResult` holds all big arrays simultaneously

```python
@dataclass(frozen=True)
class EventDeconvolutionResult:
    hwf_block: np.ndarray      # (Nx, Ny, Nt) — full merged block
    deconv_q: np.ndarray       # (Nx, Ny, Nt) — deconvolved output
    smeared_true: np.ndarray   # (Nx_eff, Ny_eff, ktimes*Nt_eff) — truth smear
    ...
```

All three large arrays are alive simultaneously while the result object exists. In typical usage (deconv_example scripts), the result is unpacked into a `.npz` and then the loop continues. If the GC does not collect the result before the next event begins, both events' peak arrays can overlap in RAM.

### 3.3 `PreparedFieldResponse` double-stores the field response

```python
@dataclass(frozen=True)
class PreparedFieldResponse:
    processor: FieldResponseProcessor   # holds self._processed_response (full_response)
    full_response: np.ndarray           # second reference to the same array
    integrated_response: np.ndarray     # a different array (coarse bins)
    center_response: np.ndarray         # 1D slice
```

`processor._processed_response` and `PreparedFieldResponse.full_response` reference the **same** numpy array (no copy, confirmed by `deconv_workflow.py:82-89`). So there is no double-count. However, both `full_response` (fine-tick, shape `(2r, 2r, Nt_fine)`) and `integrated_response` (coarse, shape `(2r, 2r, Nt_coarse)`) are kept for the entire job lifetime. Only `integrated_response` is used in `deconv_fft`; `full_response` is used only for `center_response` extraction and inspection. For a long job, these can be released after initialization.

---

## 4. dtype Analysis

**Everything is float64 / complex128.** There is no float32 usage in the numerical computation.

Switching the FFT path to float32 / complex64 would:

- Cut the three FFT working arrays from 16 B/element to 8 B/element (complex128 → complex64) — **halves peak complex memory**
- Cut `block_data` from 8 B/element to 4 B/element — **halves block memory**
- On GPU: complex64 FFTs run significantly faster than complex128 (cuFFT optimizes for 32-bit)
- The Gaussian filter (currently float64 real) would also drop to float32

Numerical precision: deconvolution amplifies noise at high spatial/temporal frequencies; the dominant precision concern is the ratio of `measurement_fft / kernel_fft` near the regularization cutoff. At the signal levels and sigma values used in practice (sigma_time ≈ 0.001–0.005), the deconvolution SNR is ≫ float32's relative precision (~1e-7). Float32 should be adequate.

**Note:** The ground-truth smearing (`gaus_smear_true_3d`) uses `true_charge.dtype`, which depends on the input — if float64 input, the smear is float64. This could independently be switched to float32.

---

## 5. Global-Volume FFT: No Tiling

`deconv_fft` operates on the **full event bounding box** in a single FFT call. There is no overlap-save or overlap-add tiling along any axis.

This means memory grows linearly with event duration:
- `Nt = 16,500` → ~1 GB peak (30×30 pixels, float64)
- `Nt = 33,000` → ~2 GB peak
- `Nt = 100,000` → ~6 GB peak

For very long events (e.g., long cosmic-ray tracks), this will exceed available RAM (CPU or GPU). An overlap-save approach along the time axis would cap peak time-memory at a fixed tile size (e.g., 2× the kernel's time extent plus the desired output length per tile), at the cost of multiple FFT calls per event.

**Overlap-save sketch:**
- Tile `Nt` into windows of `W + Kt - 1` points (overlap `Kt - 1` from previous tile)
- Each window FFT is size `W + Kt - 1`
- Keep only the central `W` output points per tile
- Peak memory per tile: `Nx_fft × Ny_fft × (W + Kt - 1) × 16` B instead of full `Nt`

For `W = 512`, `Kt ≈ 100`, the tile FFT is 612 points — far more cache-friendly.

---

## 6. Data Loader Memory

`DataLoader._load_data` (`data_loader.py:36-40`):
```python
self._data = dict(np.load(self.npz_path, allow_pickle=True))
```

`dict(...)` forces numpy's lazy NpzFile to materialize every array immediately. For a file containing 10 TPCs × 100 events × 4 data types × 2 arrays (data + location) = 8,000 arrays, all are loaded upfront and held in `self._data` for the entire loader lifetime.

`allow_pickle=True` prevents `mmap_mode='r'` (the two cannot be combined). With mmap mode, the OS would page in only the actually-accessed regions on demand, dramatically reducing resident RAM for large files accessed sparsely.

**Blocked by:** `allow_pickle=True` is needed if any array in the NPZ was saved with `pickle` (object dtype). If locations and data can be guaranteed as plain numeric arrays, removing `allow_pickle=True` would unlock `np.load(..., mmap_mode='r')`.

---

## 7. Potential Memory Leaks

### 7.1 `totq_per_pix` grows unboundedly across events

`self.totq_per_pix` is a dict keyed by `(pixel_x, pixel_y)` that accumulates total charge per pixel. `process_hits` (`burst_processor.py:512`) resets `template_compensation_anchors`, but **not** `totq_per_pix`:

```python
def process_hits(self, hits: Hits):
    self.template_compensation_anchors = []   # reset
    # self.totq_per_pix is NOT reset
    sequences_by_pixel = self.extract_sequences_from_hits(hits)
```

If the same processor instance processes many events (e.g., in a long batch loop), `totq_per_pix` grows with every new pixel seen. For a file with 1000 events × 200 pixels = 200,000 entries, each storing an `int` key tuple and a `float` value ≈ ~30 bytes → ~6 MB. Mild, but technically unbounded.

---

## 8. Suggested Memory Reductions (Ranked by Payoff)

| Rank | Action | Files | Expected saving |
|------|--------|-------|-----------------|
| 1 | Switch FFT path to float32/complex64 | `deconv.py`, `smear_truth.py`, callers | Halves all FFT peak memory; 2× GPU throughput |
| 2 | `del measurement_fft, kernel_fft` after division | `deconv.py:65-66` | Frees 2× complex half-spectra at peak |
| 3 | Replace `np.roll` with frequency-domain phase | `deconv.py:73-74` | Saves 2× full real-array copies |
| 4 | Replace `np.where` ε-clamp with in-place mask | `deconv.py:62` | Saves 1× complex half-spectrum copy |
| 5 | Remove `allow_pickle`, use `mmap_mode='r'` | `data_loader.py:38` | On-demand page-in; large RAM saving for big files |
| 6 | Pad Nt to smooth size before FFT | Callers of `deconv_fft` | Reduces FFT time (not memory, but improves throughput) |
| 7 | Implement overlap-save tiling along time | `deconv.py` | Caps peak memory for very long events |
| 8 | Clear `totq_per_pix` in `process_hits` | `burst_processor.py`, `v2.py` | Eliminates mild long-run leak |
| 9 | Event-batched fixed-envelope GPU execution | Architecture change | Amortizes VRAM for batch GPU runs |
