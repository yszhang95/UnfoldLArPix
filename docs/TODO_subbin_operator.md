# DONE: sub-bin (finer-time) operator, and censor sub-bin awareness

**Implemented** as the `time_subbin: S` prop on `BuildMeasurement` (default 1 =
unchanged, regression-safe). BuildMeasurement builds the operator at bin `B/S`
(finer kernel + `nt*S` block shape + `B/S` window sampling); BuildSupport and
Solve lift the B-grid warm seed/support to `B/S`; `CensorRunningMax.from_hits`
now takes `bin_ticks` (Solve passes `B/S`); Solve sums the sub-bins back to the
B grid on output so downstream eval is unchanged. Tests: `tests/test_subbin.py`
+ golden/burst_tau all pass; S=1 reproduces the baseline bit-for-bit.

Finding: a finer time bin (S=2) improves the large-angle **shape/resolution**
(spec_dev: mu 50deg 3.04->2.24, mu 75deg 2.36->1.67, pos 75deg 7.09->5.06) and
visibly tightens the along-track charge, but does not improve r/integral (those
are dominated by containment) and hurts small-angle (trivial time structure).
Original TODO text below.

---



**Motivation.** The shielded single-burst deconvolution over-recovers the
integral by $+10$–$12\%$. Root cause: the coarse operator time bin
($B=\texttt{adc\_hold\_delay}=30$ ticks) averages the sharp shielded field
response into a charge-additive near-null direction; the data-term least
squares fills it. Halving the operator time bin ($B/2=15$) removes the
degeneracy (integral $+11.9\% \to -3.9\%$ on the objective; full pipeline
$+10.9\% \to -2.6\%$ with a correctly-binned censor). This also shares the
origin of the high-charge $\pm1$ time-bin misalignment.

**Bug to fix.** `terms/censor.py::CensorRunningMax.from_hits` computes the
reset/arm bin boundaries as `ceil(latch_tick / hits.adc_hold_delay)` — in
physical-$B$ units. On a $B/2$ operator grid this is off by a factor of 2 and
over-constrains the solve (integral collapses to $-61\%$). Boundaries must be
computed in the *operator's* bin (`ceil(tick / bin_ticks)`), not
`adc_hold_delay`.

**Proposed change.** Thread a `time_subbin` (or `bin_ticks`) factor through:
- `model/warm_start.py::fft_warm_start` — block time axis, FFT-deconv kernel,
  and the Gaussian filter units (`dt`, `sigma_time`) at $B/\text{subbin}$.
- `algs/reco_algs.py::BuildMeasurement` / `model/operator.py::ZSOperator` —
  build the operator at the finer bin.
- `terms/censor.py::CensorRunningMax.from_hits` — accept the operator bin and
  use it for the boundaries.
Then the finer bin is a config knob (default 1 = current behaviour).

**Interim.** A validated manual B/2 + Bp-corrected-censor path exists in the
session scratchpad (`cens_fix.py`); use it if a corrected shielded self-trigger
number is needed before the architecture change.
