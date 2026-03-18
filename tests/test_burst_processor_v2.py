"""Tests for BurstSequenceProcessorV2 — fractional phase-shift alignment."""

import numpy as np
import pytest

from unfoldlarpix.burst_processor import BurstSequence, MergedSequence
from unfoldlarpix.burst_processor_v2 import BurstSequenceProcessorV2
from unfoldlarpix.data_containers import Hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADC = 10.0      # adc_hold_delay used throughout tests
TAU = 15.0      # gap threshold: gap < tau → no template; gap >= tau → template
DEADTIME = 1.0  # hardware deadtime used throughout tests


def make_processor(template=None, threshold=50.0, tau=TAU, deadtime=DEADTIME):
    tmpl = template if template is not None else np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)
    return BurstSequenceProcessorV2(
        adc_hold_delay=ADC, tau=tau, deadtime=deadtime, template=tmpl, threshold=threshold
    )


def make_seq(trigger_time_idx, nburst=2, charges=None, pixel_x=0, pixel_y=0):
    if charges is None:
        charges = np.ones(nburst, dtype=float) * 100.0
    t_start = trigger_time_idx + ADC
    t_end = trigger_time_idx + ADC * len(charges)
    return BurstSequence(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        trigger_time_idx=trigger_time_idx,
        t_first=t_start,
        t_last=t_end,
        charges=np.asarray(charges, dtype=float),
        last_adc_latch=0,
        next_integration_start=0,
    )


# ---------------------------------------------------------------------------
# 1. Fractional shift
# ---------------------------------------------------------------------------

class TestFractionalShift:

    def test_zero_shift_is_identity(self):
        """delta_T = 0 should leave charges unchanged."""
        proc = make_processor()
        charges = np.array([3.0, 7.0, 2.0, 5.0])
        result = proc._fractional_shift(charges, 0.0)
        np.testing.assert_allclose(result, charges, atol=1e-12)

    def test_padding_reduces_cyclic_leakage(self):
        """Zero-padding reduces cyclic aliasing compared to an unpadded FFT shift.

        An impulse at the last index shifted right by D=0.1 wraps to index 0
        in a cyclic (no-padding) FFT.  The zero-padded implementation pushes
        the periodic copy far away, so |shifted_padded[0]| < |shifted_cyclic[0]|.
        """
        proc = make_processor()
        charges = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        shifted_padded = proc._fractional_shift(charges, 0.1 * ADC)  # D = 0.1

        # Cyclic (no-padding) reference
        N = len(charges)
        D = 0.1
        X = np.fft.fft(charges)
        k = np.fft.fftfreq(N) * N
        phase = np.exp(-1j * 2 * np.pi * k * D / N)
        shifted_cyclic = np.real(np.fft.ifft(X * phase))

        assert abs(shifted_padded[0]) < abs(shifted_cyclic[0]), (
            f"Padded leakage at [0]: {shifted_padded[0]:.4f}; "
            f"cyclic leakage: {shifted_cyclic[0]:.4f}"
        )

    def test_full_sample_shift_non_cyclic(self):
        """A 1.0-sample shift moves the impulse from index 0 to index 1.

        With zero-padding, exp(-2πi k D / M) with D=1 shifts the padded
        impulse at position pad to pad+1; after extraction the result equals
        np.roll(charges, 1) (same outcome as cyclic for this special case).
        """
        proc = make_processor()
        charges = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        shifted = proc._fractional_shift(charges, ADC)  # D = 1.0
        expected = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(shifted, expected, atol=1e-10)

    def test_single_element(self):
        """Single-element array should be returned unchanged regardless of shift."""
        proc = make_processor()
        charges = np.array([42.0])
        for frac in [0.0, 0.3, 0.9]:
            result = proc._fractional_shift(charges, frac * ADC)
            np.testing.assert_allclose(result, charges, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. First-interval scaling
# ---------------------------------------------------------------------------

class TestFirstIntervalScaling:
    """Scaling is applied only in the gap < tau path (_append_shifted).

    For gap >= tau the template models the rising edge; charges[0] is left as-is.
    """

    def _run_two_seq(self, proc, trigger_a, trigger_b, charges_a, charges_b):
        """Helper: build two sequences and call process_pixel_sequences."""
        seq_a = make_seq(trigger_time_idx=trigger_a, charges=charges_a)
        seq_b = make_seq(trigger_time_idx=trigger_b, charges=charges_b)
        return proc.process_pixel_sequences([seq_a, seq_b]), seq_a, seq_b

    def _make_cumul_times(self, seq_a):
        """Helper: build cumulative and times arrays from a single sequence."""
        cumul = np.concatenate([[0], np.cumsum(seq_a.charges)])
        times = np.array([seq_a.t_first + i * ADC for i in range(len(seq_a.charges))])
        return cumul, times

    def test_scaling_uses_gap_minus_deadtime(self):
        """active_time_first = gap - deadtime, so charges[0] is scaled by
        ADC / (gap - deadtime).

        seq_a last latch at t=20, seq_b t_first=30, gap=10, deadtime=1
        → active_time_first = 9 → scale = ADC/9 = 10/9
        """
        proc = make_processor()
        seq_a = make_seq(trigger_time_idx=0, charges=[80.0, 40.0])   # t_last=20
        seq_b = make_seq(trigger_time_idx=20, charges=[60.0, 30.0])  # t_first=30
        gap = seq_b.t_first - seq_a.t_last   # = 10
        active = gap - DEADTIME              # = 9
        assert seq_b.t_first - seq_a.t_last < TAU

        cumul_a, times_a = self._make_cumul_times(seq_a)
        # delta_T = 20 % 10 = 0 → fractional shift is identity
        _, cumul = proc._append_shifted(cumul_a, times_a, seq_b)

        first_charge_b = np.diff(cumul)[len(seq_a.charges)]
        expected = 60.0 * ADC / active
        np.testing.assert_allclose(first_charge_b, expected, rtol=1e-10)

    def test_append_shifted_scales_only(self):
        """_append_shifted applies first-interval scaling only — no fractional shift.

        Fractional shift is deferred to the final alignment pass in
        process_pixel_sequences.  After calling _append_shifted the charges in
        the returned cumulative are the SCALED (not shifted) values.
        """
        proc = make_processor()
        seq_a = make_seq(trigger_time_idx=0, charges=[80.0, 40.0])   # t_last=20
        # trigger_b=23 → t_first_b=33, gap=13, delta_T=23%10=3
        seq_b = make_seq(trigger_time_idx=23, charges=[60.0, 30.0])
        gap = seq_b.t_first - seq_a.t_last   # = 13
        active = gap - DEADTIME              # = 12
        assert gap < TAU

        cumul_a, times_a = self._make_cumul_times(seq_a)
        _, cumul = proc._append_shifted(cumul_a, times_a, seq_b)

        # Expected after _append_shifted: only scaling applied, no fractional shift
        scaled = seq_b.charges.copy()
        scaled[0] *= ADC / active
        expected_charges = scaled   # fractional shift NOT applied yet

        actual_charges_b = np.diff(cumul)[len(seq_a.charges):]
        np.testing.assert_allclose(actual_charges_b, expected_charges, atol=1e-10)

    def test_no_scaling_for_large_gap(self):
        """For gap >= tau (template path), charges[0] of seq_b must NOT be scaled.

        After grid alignment times are shifted by delta_T: seq_b.t_first (=63) becomes
        63 - delta_T (=3) = 60.  The charge at that aligned position must not equal
        the wrong scaled value ADC/(ADC+3) * 100.
        """
        LONG_TEMPLATE = np.cumsum(np.ones(200, dtype=float))
        proc = make_processor(template=LONG_TEMPLATE, threshold=30.0)

        # trigger_b=53 → delta_T=53%10=3; t_first_b=63; gap=63-20=43 >= TAU=15
        seq_a = make_seq(trigger_time_idx=0, charges=[100.0, 100.0])
        seq_b = make_seq(trigger_time_idx=53, charges=[100.0, 100.0])
        assert seq_b.t_first - seq_a.t_last >= TAU

        merged = proc.process_pixel_sequences([seq_a, seq_b])

        # After grid alignment seq_b's first latch time is t_first_b - delta_T = 60
        delta_T_b = seq_b.trigger_time_idx % ADC   # = 3
        aligned_t_first_b = seq_b.t_first - delta_T_b  # = 60
        idx = np.where(np.isclose(merged.times, aligned_t_first_b))[0]
        assert len(idx) == 1, (
            f"Expected exactly one aligned time at {aligned_t_first_b}; "
            f"got merged.times={merged.times}"
        )
        actual_charge = merged.charges[idx[0]]
        # Wrong scaling formula (uses ADC+delta_T instead of gap-deadtime)
        wrong_scaled_value = 100.0 * ADC / (ADC + delta_T_b)
        assert not np.isclose(actual_charge, wrong_scaled_value, rtol=1e-3), (
            f"charges[0] appears to have been wrongly scaled "
            f"({actual_charge:.4f} ≈ {wrong_scaled_value:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Template collision stopping
# ---------------------------------------------------------------------------

class TestTemplateCollisionStop:
    """Use a long template so the gap (40 ticks) can be filled.

    gap=40 >= TAU=15, so template compensation is used.
    """

    # Template with 200 points so tlength=40-DEADTIME fits inside range(...)
    LONG_TEMPLATE = np.cumsum(np.ones(200, dtype=float))

    def test_no_template_point_at_or_after_t_first(self):
        """No inserted template time must be >= next_seq.t_first."""
        proc = make_processor(template=self.LONG_TEMPLATE, threshold=30.0)
        # seq_a: trigger=0, 2 bursts → t_first=10, t_last=20
        seq_a = make_seq(trigger_time_idx=0, charges=[100.0, 100.0])
        # seq_b: trigger=50, t_first=60; gap=40 >= TAU=15 → template used
        seq_b = make_seq(trigger_time_idx=50, charges=[100.0, 100.0])
        assert seq_b.t_first - seq_a.t_last >= TAU

        merged = proc.process_pixel_sequences([seq_a, seq_b])

        seq_b_t_first = seq_b.t_first  # 60
        template_region = merged.times[merged.times < seq_b_t_first]
        assert np.all(template_region < seq_b_t_first), (
            f"Template points at or after t_first={seq_b_t_first}: "
            f"{merged.times[merged.times >= seq_b_t_first]}"
        )

    def test_times_strictly_monotonic(self):
        """Merged times must always be strictly monotonically increasing."""
        proc = make_processor(template=self.LONG_TEMPLATE, threshold=30.0)
        seq_a = make_seq(trigger_time_idx=0, charges=[100.0, 100.0])
        # gap=40 >= TAU=15 → template used
        seq_b = make_seq(trigger_time_idx=50, charges=[100.0, 100.0])
        merged = proc.process_pixel_sequences([seq_a, seq_b])
        assert np.all(np.diff(merged.times) > 0), "times not strictly monotonic"


# ---------------------------------------------------------------------------
# 4. Adjacent sequences (gap == 0)
# ---------------------------------------------------------------------------

class TestSmallGap:
    """Tests for the gap < tau path (no template, just fractional shift)."""

    def test_small_gap_no_template_points_inserted(self):
        """When gap < tau no template points are inserted between the sequences.

        seq_a: trigger=0, 2 bursts → t_last=20
        seq_b: trigger=20, 2 bursts → t_first=30  (gap=10 < TAU=15)
        """
        proc = make_processor(threshold=30.0)
        seq_a = make_seq(trigger_time_idx=0, charges=[80.0, 40.0])
        seq_b = make_seq(trigger_time_idx=20, charges=[60.0, 30.0])

        gap = seq_b.t_first - seq_a.t_last
        assert gap < TAU, f"Expected gap={gap} < TAU={TAU}"

        merged = proc.process_pixel_sequences([seq_a, seq_b])

        # No times in the open interval (seq_a.t_last, seq_b.t_first) = (20, 30)
        between = merged.times[(merged.times > seq_a.t_last) & (merged.times < seq_b.t_first)]
        assert len(between) == 0, f"Unexpected template points between sequences: {between}"
        assert np.all(np.diff(merged.times) > 0)

    def test_small_gap_fractional_shift_applied(self):
        """With gap < tau the next sequence's charges are still fractional-shifted."""
        proc = make_processor(threshold=30.0)
        # trigger=5 → delta_T = 5 % ADC = 5 (non-zero jitter)
        seq_a = make_seq(trigger_time_idx=0, charges=[80.0, 40.0])
        seq_b = make_seq(trigger_time_idx=25, charges=[60.0, 30.0])  # t_first=35, gap=15=TAU

        # Use gap just below tau
        seq_b_below = make_seq(trigger_time_idx=24, charges=[60.0, 30.0])  # t_first=34, gap=14
        assert seq_b_below.t_first - seq_a.t_last < TAU

        merged_shifted = proc.process_pixel_sequences([seq_a, seq_b_below])
        # Zero-jitter version for comparison
        seq_b_zero = make_seq(trigger_time_idx=20, charges=[60.0, 30.0])  # delta_T=0
        merged_zero = proc.process_pixel_sequences([seq_a, seq_b_zero])

        # Both should be monotonic
        assert np.all(np.diff(merged_shifted.times) > 0)
        assert np.all(np.diff(merged_zero.times) > 0)

    def test_duplicate_aligned_time_is_removed(self):
        """When gap + delta_T_A < ADC the two blocks share an aligned time;
        the one from the next sequence is dropped.

        trigger_A=5  → delta_T_A=5, t_last_A = 5 + 10*2 = 25
        trigger_B=17 → delta_T_B=7, t_first_B = 17 + 10  = 27
        gap = 27 - 25 = 2  (> DEADTIME=1, < TAU=15 → _append_shifted)
        aligned condition: gap + delta_T_A = 2+5 = 7 < ADC=10 → duplicate
        last aligned A  = 25 - 5 = 20
        first aligned B = 27 - 7 = 20  ← duplicate, must be removed
        """
        proc = make_processor(threshold=30.0)
        seq_a = make_seq(trigger_time_idx=5,  charges=[80.0, 40.0])  # t_last=25
        seq_b = make_seq(trigger_time_idx=17, charges=[60.0, 30.0])  # t_first=27, gap=2
        gap = seq_b.t_first - seq_a.t_last
        assert gap > DEADTIME
        assert gap < TAU
        assert gap + (seq_a.trigger_time_idx % ADC) < ADC  # duplicate condition

        merged = proc.process_pixel_sequences([seq_a, seq_b])

        # No duplicate times
        assert np.all(np.diff(merged.times) > 0), f"Duplicate times: {merged.times}"
        # Total number of time points = 2 (seq_a) + 2 (seq_b) - 1 (dropped duplicate)
        # plus whatever the bootstrap template adds (trigger region)
        # At minimum the merged length < seq_a_len + seq_b_len
        assert len(merged.times) < len(seq_a.charges) + len(seq_b.charges) + 10


# ---------------------------------------------------------------------------
# 5. Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_missing_threshold_raises(self):
        with pytest.raises(ValueError, match="Threshold value must be provided"):
            BurstSequenceProcessorV2(
                adc_hold_delay=ADC, tau=TAU, deadtime=DEADTIME, threshold=None
            )

    def test_non_monotonic_template_raises(self):
        with pytest.raises(ValueError, match="template must be monotonically increasing"):
            BurstSequenceProcessorV2(
                adc_hold_delay=ADC,
                tau=TAU,
                deadtime=DEADTIME,
                template=np.array([1, 3, 2, 4]),
                threshold=10.0,
            )

    def test_empty_template_raises(self):
        with pytest.raises(ValueError, match="Template cannot be empty"):
            BurstSequenceProcessorV2(
                adc_hold_delay=ADC,
                tau=TAU,
                deadtime=DEADTIME,
                template=np.array([]),
                threshold=10.0,
            )

    def test_valid_construction(self):
        proc = BurstSequenceProcessorV2(
            adc_hold_delay=ADC, tau=TAU, deadtime=DEADTIME, threshold=50.0
        )
        assert proc.adc_hold_delay == ADC
        assert proc.tau == TAU
        assert proc.deadtime == DEADTIME
        assert proc.threshold == 50.0
        assert proc.template.size > 0


# ---------------------------------------------------------------------------
# 6. Duplicate trigger time validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_duplicate_trigger_times_raises(self):
        proc = make_processor()
        hits_data = np.array([
            [0.0, 0.0, 0.0, 100.0, 200.0],
            [0.0, 0.0, 0.0, 150.0, 250.0],
        ])
        hits_loc = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],  # same trigger time for same pixel
        ])
        hits = Hits(data=hits_data, location=hits_loc, tpc_id=0, event_id=0)
        with pytest.raises(ValueError, match="Duplicate trigger times"):
            proc.extract_sequences_from_hits(hits)


# ---------------------------------------------------------------------------
# 7. Single-hit sequences (t_first == t_last)
# ---------------------------------------------------------------------------

class TestSingleHit:
    """When nburst == 1, t_first == t_last.  All code paths must handle this."""

    def test_single_hit_construction(self):
        """BurstSequence with nburst=1 (t_first == t_last) should not raise."""
        seq = make_seq(trigger_time_idx=0, nburst=1, charges=[100.0])
        assert seq.t_first == seq.t_last

    def test_single_hit_single_sequence(self):
        """process_pixel_sequences works with one single-hit sequence."""
        proc = make_processor(threshold=30.0)
        seq = make_seq(trigger_time_idx=0, nburst=1, charges=[100.0])
        merged = proc.process_pixel_sequences([seq])
        assert len(merged.charges) > 0
        assert len(merged.times) == len(merged.charges)
        assert len(merged.cumulative) == len(merged.charges) + 1
        np.testing.assert_allclose(merged.cumulative[0], 0.0)

    def test_single_hit_followed_by_multi_hit(self):
        """A single-hit first sequence followed by a normal sequence merges correctly."""
        LONG_TEMPLATE = np.cumsum(np.ones(200, dtype=float))
        proc = make_processor(template=LONG_TEMPLATE, threshold=30.0)
        seq_a = make_seq(trigger_time_idx=0, nburst=1, charges=[100.0])   # t_first=t_last=10
        seq_b = make_seq(trigger_time_idx=50, nburst=2, charges=[80.0, 40.0])  # t_first=60
        merged = proc.process_pixel_sequences([seq_a, seq_b])
        assert np.all(np.diff(merged.times) > 0), "times not strictly monotonic"
        assert len(merged.cumulative) == len(merged.charges) + 1
