# UnfoldLArPix Code Examination

**Examination date:** 2026-04-13  
**Source revision:** `main` branch (commit d8ea6df)  
**Method:** Static analysis — all source files under `src/unfoldlarpix/` read directly. No code was modified.  
**No profiling was run.** Memory and compute estimates are derived analytically from array shapes.

---

## How to Read These Reports

The reports are independent but cross-referenced. Suggested reading order:

1. **Algorithm overview** — read first to understand what the code is supposed to do before reading what might be wrong with it.
2. **Bug report** — read next; HIGH items should be investigated before any optimization work begins.
3. **Efficiency and GPU** — read after the bug report; several optimizations interact with bugs that must be fixed first.
4. **Memory audit** — complements the efficiency report; provides concrete size formulas.
5. **Open questions** — read last; these are the items where you need to run the code or apply domain knowledge.

---

## Report Index

| File | Contents | Key takeaway |
|------|----------|--------------|
| [`01_algorithm_overview.md`](01_algorithm_overview.md) | Forward model, solver, kernel construction, burst processing, call graph, tensor shapes, V1 vs V2 comparison | One-pass pseudo-Wiener deconvolution; no iteration; burst processor does the heavy pre-processing |
| [`02_bug_report.md`](02_bug_report.md) | 4 HIGH, 9 MED, 6 LOW findings with `file:line` references, descriptions, and reproduction hints | **H1 is a confirmed crash** (IndexError); other HIGHs are charge-accounting correctness issues |
| [`03_efficiency_and_gpu.md`](03_efficiency_and_gpu.md) | Work mix, FFT plan-reuse, hot Python loops, cupy compatibility, GPU batch strategy, CPU quick wins | Large events are FFT-bound; small events are Python-loop-bound; GPU port is mostly feasible but needs loop vectorization first |
| [`04_memory_audit.md`](04_memory_audit.md) | Peak memory formulas, unnecessary copies, long-lived intermediates, dtype, global-volume FFT, loader memory, suggested reductions | Peak = ~1 GB (30×30 pix), ~4 GB (64×64 pix) at float64; switching to float32 halves these |
| [`05_open_questions.md`](05_open_questions.md) | 8 questions where static analysis is insufficient; each includes a concrete "to resolve" step | Causality of kernel (Q1), padding adequacy (Q2), `true_charge[:,-1]` meaning (Q4) are highest priority |

---

## Bug Severity Summary

### HIGH (4 findings)

| ID | Location | Summary |
|----|----------|---------|
| H1 | `burst_processor_v2.py:312-317` | **Confirmed crash**: unreachable `return` inside length-check block; execution falls through to `candidate_times[valid_mask][-1]` on empty mask → `IndexError`. Fix before any other work. |
| H2 | `deconv.py:50-52, 73-74` | Time axis not FFT-padded and not rolled; relies silently on `npadbin=50` and causal kernel. Verify with Q1 and Q2 in open questions. |
| H3 | `burst_processor_v2.py:466` | Author-flagged bug: first-interval scaling in `_append_shifted` is not a proper cumulative correction. Charge accounting for `gap < tau` transitions is approximate. |
| H4 | `burst_processor.py:205-261` | V1 `_dead_time_compensation` regenerates times uniformly, discarding non-uniform template-compensated times from prior merges. Time/charge misalignment for mixed-compensation pixels. |

### MED (9 findings)

| ID | Location | Summary |
|----|----------|---------|
| M1 | `field_response.py:66`, `deconv_workflow.py:84` | Even-shaped kernel after `_quadrant_copy`; center pixel off by one toward positive |
| M2 | `data_loader.py:207, 231, 255` | Silent drop of event sub-arrays when batch counts disagree |
| M3 | `data_loader.py:135-137` | Ambiguous numpy truthiness test for noise values (crashes on shape `(1,)`) |
| M4 | `deconv.py:62` | Epsilon clamp changes phase of small kernel FFT bins |
| M5 | `smear_truth.py:20, 49` | Padding formula collapses to n_single_side=1 regardless of sigma |
| M6 | `burst_processor.py:408-410` | Dead code before bootstrap template compensation |
| M7 | `burst_processor.py:349` / `v2:327` | Divide-by-zero if `template_section[-1] == 0` |
| M8 | `hit_to_wf.py:102-108` | `convert_bin_wf_to_blocks` mutates input `wf.location` in place |
| M9 | `data_loader.py:272-277` | `TrueHits.location` shares buffer with `Hits.location` |
| M10 | `data_loader.py:38` | `allow_pickle=True` blocks memory-mapped loading |
| M11 | Multiple files | Stray `print()` in production paths (burst_processor + hit_to_wf) |

### LOW (6 findings)

| ID | Location | Summary |
|----|----------|---------|
| L1 | `burst_processor.py:100`, `v2:68` | `totq_per_pix` not cleared between events |
| L2 | `hit_to_wf.py:135` | `bata = np.add.at(...)` assigns `None` — dead variable |
| L3 | `hit_to_wf.py:43` | Zero-signal pixels get `frac = 1.0` due to divide-by-zero clamp |
| L4 | Multiple files | Python for-loops in hot code paths (see efficiency report §3) |
| L5 | `field_response.py:62` | Square-input validation is adequate but relies on caller convention |
| L6 | `burst_processor.py:297` | Commented-out monotonicity check not applied inside `_template_compensation` |

---

## Confirmed False Positive (Do Not Reopen)

**`gaussian_filter_3d` shape is correct** (`deconv.py:21-29`, `smear_truth.py:56-62`)

An automated audit agent flagged the broadcast-accumulation loop as producing a malformed shape. Hand-tracing shows:
- After the loop: shape is `(s[-1]//2+1, s[0], s[1])` (time-rfft axis first, two spatial axes)
- After `moveaxis(0, -1)`: shape is `(s[0], s[1], s[-1]//2+1)` — the correct rfftn half-spectrum shape for a real 3D array of shape `(s[0], s[1], s[-1])`

The same pattern in `gaus_smear_true_3d` is identical and also correct. Do not open an issue for this.

---

## Memory Quick Reference

| Event size | Peak RAM (float64) | Peak RAM (float32) |
|------------|-------------------|-------------------|
| 30×30 pixels, Nt=16,500 | ~1.0 GB | ~0.5 GB |
| 64×64 pixels, Nt=16,500 | ~4.0 GB | ~2.0 GB |
| 64×64 pixels, Nt=33,000 | ~8.0 GB | ~4.0 GB |

These are peak inside `deconv_fft`. Additional memory is consumed by `smear_truth` and the `EventDeconvolutionResult` that keeps all three large arrays alive.

---

## GPU Port Feasibility

| Component | Port difficulty | Notes |
|-----------|----------------|-------|
| `deconv_fft` | **Low** — mostly drop-in cupy | Replace `np.fft` with `cupy.fft`; fix `np.where` |
| `gaus_smear_true_3d` | **Low** — cupy-compatible | Same |
| `merged_sequences_to_block` scatter | **Medium** — vectorize first | Needs `np.bincount` / `cupyx.scatter_add` |
| `gaus_smear_true_3d` scatter | **Low** — use `np.add.at` → `cupyx.scatter_add` | |
| `BurstSequenceProcessor[V2]` | **High** — needs algorithmic restructure | Per-pixel sequential state machine; not GPU-friendly in current form |
| NPZ data loader | **N/A** — must stay host-side | `allow_pickle=True` incompatible with mmap and GPU streaming |
