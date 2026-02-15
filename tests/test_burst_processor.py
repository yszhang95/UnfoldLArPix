"""Tests for burst sequence processor."""

import numpy as np
import pytest

from unfoldlarpix.burst_processor import (
    BurstSequence,
    BurstSequenceProcessor,
    MergedSequence,
)
from unfoldlarpix.data_containers import Hits


class TestBurstSequence:
    """Tests for BurstSequence dataclass."""

    def test_valid_sequence(self):
        """Test creation of valid burst sequence."""
        seq = BurstSequence(
            pixel_x=0,
            pixel_y=1,
            trigger_time_idx=10,
            t_start=20.0,
            t_end=40.0,
            charges=np.array([10.0, 20.0]),
            last_adc_latch=0,
            next_integration_start=0,
        )
        assert seq.pixel_x == 0
        assert seq.pixel_y == 1
        assert len(seq.charges) == 2

    def test_invalid_time_ordering(self):
        """Test that t_end must be > t_start."""
        with pytest.raises(ValueError, match="t_end.*must be.*t_start"):
            BurstSequence(
                pixel_x=0, pixel_y=0, trigger_time_idx=0,
                t_start=20.0, t_end=10.0,
                charges=np.array([10.0]),
                last_adc_latch=0, next_integration_start=0,
            )

    def test_empty_charges(self):
        """Test that charges cannot be empty."""
        with pytest.raises(ValueError, match="charges.*cannot be empty"):
            BurstSequence(
                pixel_x=0, pixel_y=0, trigger_time_idx=0,
                t_start=0.0, t_end=10.0,
                charges=np.array([]),
                last_adc_latch=0, next_integration_start=0,
            )


class TestBurstSequenceProcessor:
    """Tests for BurstSequenceProcessor."""

    def test_processor_initialization(self):
        """Test processor initialization with valid parameters."""
        processor = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            delta_t=1.0,
        )
        assert processor.adc_hold_delay == 10.0
        assert processor.tau == 5.0
        assert processor.delta_t == 1.0

    def test_invalid_template(self):
        """Test that template must be monotonically increasing."""
        with pytest.raises(ValueError, match="template must be monotonically increasing"):
            BurstSequenceProcessor(
                adc_hold_delay=10.0,
                tau=5.0,
                delta_t=1.0,
                template=np.array([1, 3, 2, 4]),  # Not monotonic
            )

    def test_dead_time_compensation_example(self):
        """Test dead-time compensation with CLAUDE.md example."""
        # Setup from CLAUDE.md
        adc_hold_delay = 10.0  # ms
        tau = 5.0              # ms
        delta_t = 1.0          # ms

        processor = BurstSequenceProcessor(
            adc_hold_delay=adc_hold_delay,
            tau=tau,
            delta_t=delta_t,
        )

        # Sequence A: t_start=0, t_end=10, charges=[90, 100]
        seq_a = BurstSequence(
            pixel_x=0, pixel_y=0, trigger_time_idx=-10,
            t_start=0.0, t_end=10.0,
            charges=np.array([90.0, 100.0]),
            last_adc_latch=0, next_integration_start=0,
        )

        # Sequence B: t_start=13, charges=[130, 10]
        seq_b = BurstSequence(
            pixel_x=0, pixel_y=0, trigger_time_idx=3,
            t_start=13.0, t_end=33.0,
            charges=np.array([130.0, 10.0]),
            last_adc_latch=0, next_integration_start=0,
        )

        # Gap = 13 - 10 = 3ms
        gap = seq_b.t_start - seq_a.t_end
        assert gap == 3.0
        assert gap <= tau

        # Expected: slope = 130 / (3 - 1) = 65
        # compensated_value = 65 * 1 = 65
        # Merged charges: [90, 100, 130+65, 10] = [90, 100, 195, 10]
        # Cumulative (with prepended 0): [0, 90, 190, 385, 395]

        merged = processor.process_pixel_sequences([seq_a, seq_b])

        # Check cumulative
        expected_cumulative = np.array([0, 90, 190, 385, 395])
        np.testing.assert_allclose(merged.cumulative, expected_cumulative, rtol=1e-10)

        # Check charges (differentiation of cumulative)
        expected_charges = np.array([90, 100, 195, 10])
        np.testing.assert_allclose(merged.charges, expected_charges, rtol=1e-10)

        # Check times
        expected_times = np.array([0, 10, 13, 23])
        np.testing.assert_allclose(merged.times, expected_times, rtol=1e-10)

    def test_single_sequence(self):
        """Test processing a single sequence (no merging needed)."""
        processor = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            delta_t=1.0,
        )

        seq = BurstSequence(
            pixel_x=0, pixel_y=0, trigger_time_idx=0,
            t_start=10.0, t_end=30.0,
            charges=np.array([10.0, 20.0]),
            last_adc_latch=0, next_integration_start=0,
        )

        merged = processor.process_pixel_sequences([seq])

        # Should just have original charges
        np.testing.assert_allclose(merged.charges, seq.charges)
        expected_times = np.array([10.0, 20.0])
        np.testing.assert_allclose(merged.times, expected_times)

    def test_extract_sequences_from_hits(self):
        """Test extracting burst sequences from Hits container."""
        processor = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            delta_t=1.0,
        )

        # Create synthetic hits data
        # hits_data: (x, y, z, charge1, charge2, ...)
        hits_data = np.array([
            [1.0, 2.0, 3.0, 10.0, 20.0],  # pixel (0, 0), 2 bursts
            [1.0, 2.0, 4.0, 15.0, 25.0],  # pixel (0, 0), 2 bursts (different trigger)
            [2.0, 3.0, 5.0, 30.0, 40.0],  # pixel (1, 1), 2 bursts
        ])

        # hits_location: (pixel_x, pixel_y, trigger_time_idx, last_adc_latch, next_integration_start)
        hits_location = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0],  # Same pixel, different trigger time
            [1, 1, 0, 0, 0],   # Different pixel
        ])

        hits = Hits(
            data=hits_data,
            location=hits_location,
            tpc_id=0,
            event_id=1,
        )

        sequences = processor.extract_sequences_from_hits(hits)

        # Should have 2 pixels
        assert len(sequences) == 2
        assert (0, 0) in sequences
        assert (1, 1) in sequences

        # Pixel (0, 0) should have 2 sequences
        assert len(sequences[(0, 0)]) == 2

        # Pixel (1, 1) should have 1 sequence
        assert len(sequences[(1, 1)]) == 1

        # Check ordering
        seq1, seq2 = sequences[(0, 0)]
        assert seq1.trigger_time_idx < seq2.trigger_time_idx

    def test_duplicate_trigger_times_raises_error(self):
        """Test that duplicate trigger times raise an error."""
        processor = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            delta_t=1.0,
        )

        # Create hits with duplicate trigger times for same pixel
        hits_data = np.array([
            [1.0, 2.0, 3.0, 10.0, 20.0],
            [1.0, 2.0, 4.0, 15.0, 25.0],
        ])

        hits_location = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],  # Same pixel, same trigger time!
        ])

        hits = Hits(
            data=hits_data,
            location=hits_location,
            tpc_id=0,
            event_id=1,
        )

        with pytest.raises(ValueError, match="Duplicate trigger times"):
            processor.extract_sequences_from_hits(hits)

    def test_template_compensation(self):
        """Test template compensation for non-close sequences."""
        adc_hold_delay = 10.0
        tau = 5.0
        delta_t = 1.0
        template = np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)

        processor = BurstSequenceProcessor(
            adc_hold_delay=adc_hold_delay,
            tau=tau,
            delta_t=delta_t,
            template=template,
            template_spacing=adc_hold_delay,
        )

        # Create two sequences with large gap
        seq_a = BurstSequence(
            pixel_x=0, pixel_y=0, trigger_time_idx=0,
            t_start=10.0, t_end=20.0,
            charges=np.array([100.0]),
            last_adc_latch=0, next_integration_start=0,
        )

        # Large gap: seq_b starts at 50, gap = 50 - 20 = 30 > tau
        seq_b = BurstSequence(
            pixel_x=0, pixel_y=0, trigger_time_idx=40,
            t_start=50.0, t_end=60.0,
            charges=np.array([50.0]),
            last_adc_latch=0, next_integration_start=0,
        )

        gap = seq_b.t_start - (seq_a.t_end + adc_hold_delay)
        assert gap > tau, "Gap should be larger than tau for template compensation"

        # Process sequences
        merged = processor.process_pixel_sequences([seq_a, seq_b])

        # Should have template points inserted
        assert len(merged.times) > 2, "Template points should be inserted"

        # Times should be strictly increasing
        assert np.all(np.diff(merged.times) > 0)

        # Cumulative should be monotonically increasing
        assert np.all(np.diff(merged.cumulative) >= 0)


class TestMergedSequence:
    """Tests for MergedSequence dataclass."""

    def test_valid_merged_sequence(self):
        """Test creation of valid merged sequence."""
        merged = MergedSequence(
            pixel_x=0,
            pixel_y=1,
            times=np.array([0, 10, 20]),
            charges=np.array([10, 20, 30]),
            cumulative=np.array([0, 10, 30, 60]),
        )
        assert merged.pixel_x == 0
        assert len(merged.times) == 3
        assert len(merged.charges) == 3

    def test_mismatched_lengths(self):
        """Test that times and charges must have same length."""
        with pytest.raises(ValueError, match="times and charges must have the same length"):
            MergedSequence(
                pixel_x=0, pixel_y=0,
                times=np.array([0, 10]),
                charges=np.array([10, 20, 30]),
                cumulative=np.array([0, 10, 30, 60]),
            )

    def test_non_monotonic_times(self):
        """Test that times must be strictly monotonically increasing."""
        with pytest.raises(ValueError, match="times must be strictly monotonically increasing"):
            MergedSequence(
                pixel_x=0, pixel_y=0,
                times=np.array([0, 20, 10]),  # Not monotonic
                charges=np.array([10, 20, 30]),
                cumulative=np.array([0, 10, 30, 60]),
            )
