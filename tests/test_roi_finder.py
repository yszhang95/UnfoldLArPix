"""Tests for ROI identification helpers."""

import numpy as np
import pytest

from unfoldlarpix.roi_finder import (
    apply_roi_mask,
    estimate_quiet_pixel_noise,
    find_roi_mask,
)


class TestEstimateQuietPixelNoise:
    def test_uses_only_quiet_pixels(self):
        rng = np.random.default_rng(0)
        nx, ny, nt = 6, 6, 32
        block = rng.normal(0.0, 0.1, size=(nx, ny, nt))
        # Inject a huge "signal" on one pixel.
        block[2, 3, :] += 1000.0

        block_offset = np.array([10, 20, 0])
        # Hit pixel global index = block-local (2, 3) + offset (10, 20).
        hit_xy = np.array([[12, 23]])

        rms = estimate_quiet_pixel_noise(block, block_offset, hit_xy)
        assert rms == pytest.approx(0.1, rel=0.2)

    def test_raises_when_too_few_quiet_pixels(self):
        block = np.zeros((2, 2, 4))
        block_offset = np.array([0, 0, 0])
        # All four pixels marked as hits -> zero quiet pixels.
        hit_xy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        with pytest.raises(ValueError, match="quiet pixels"):
            estimate_quiet_pixel_noise(
                block, block_offset, hit_xy, min_quiet_pixels=1
            )

    def test_ignores_hits_outside_block(self):
        rng = np.random.default_rng(1)
        block = rng.normal(0.0, 0.1, size=(4, 4, 16))
        block_offset = np.array([0, 0, 0])
        # Hit at (10, 10) lies outside the block; should be ignored, all 16
        # block pixels are treated as quiet.
        rms = estimate_quiet_pixel_noise(
            block, block_offset, np.array([[10, 10]]), min_quiet_pixels=8
        )
        assert rms == pytest.approx(0.1, rel=0.2)


class TestFindRoiMask:
    def test_threshold_only_marks_above_cutoff(self):
        block = np.zeros((1, 1, 10))
        block[0, 0, 4] = 5.0
        mask = find_roi_mask(
            block, noise_rms=1.0, threshold_sigma=4.0, merge_gap=0, expand=0
        )
        expected = np.zeros((1, 1, 10), dtype=bool)
        expected[0, 0, 4] = True
        np.testing.assert_array_equal(mask, expected)

    def test_merge_gap_closes_short_runs(self):
        block = np.zeros((1, 1, 10), dtype=float)
        block[0, 0, 2] = 10.0
        block[0, 0, 5] = 10.0  # gap of 2 below-threshold bins
        mask = find_roi_mask(
            block, noise_rms=1.0, threshold_sigma=4.0, merge_gap=2, expand=0
        )
        assert mask[0, 0, 2:6].all()

    def test_expand_widens_each_segment(self):
        block = np.zeros((1, 1, 10), dtype=float)
        block[0, 0, 5] = 10.0
        mask = find_roi_mask(
            block, noise_rms=1.0, threshold_sigma=4.0, merge_gap=0, expand=2
        )
        assert mask[0, 0, 3:8].all()
        assert not mask[0, 0, 2]
        assert not mask[0, 0, 8]

    def test_rejects_nonpositive_noise_rms(self):
        with pytest.raises(ValueError, match="noise_rms"):
            find_roi_mask(np.zeros((1, 1, 4)), noise_rms=0.0)


class TestApplyRoiMask:
    def test_zeros_outside_mask(self):
        deconv = np.array([[1.0, 2.0, 3.0]])
        mask = np.array([[True, False, True]])
        out = apply_roi_mask(deconv, mask)
        np.testing.assert_array_equal(out, np.array([[1.0, 0.0, 3.0]]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_roi_mask(np.zeros((2, 2)), np.zeros((3, 3), dtype=bool))
