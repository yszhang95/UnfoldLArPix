"""Tests for burst sequence processor."""

import numpy as np
import pytest

from unfoldlarpix.burst_processor import MergedSequence


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
