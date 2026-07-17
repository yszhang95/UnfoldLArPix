"""Tests for iterative self-consistent template recompensation."""

import numpy as np
import pytest

from unfoldlarpix.deconv import deconv_fft
from unfoldlarpix.iterative_recomp import (
    forward_model_block,
    iterative_recompensation,
    measured_bin_mask,
    refine_unmeasured_segments,
)


def _make_kernel(kx=3, ky=3, kt=5, seed=0):
    """Small centered/causal positive kernel normalised to unit sum."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.1, 1.0, size=(kx, ky, kt))
    # make it peaked at the spatial center like a field response
    k[kx // 2, ky // 2] += 3.0
    return k / k.sum()


def _linear_convolve(q, kernel, block_shape):
    """Reference O(N^2) linear convolution with deconv_fft's alignment."""
    nx, ny, nt = block_shape
    kx, ky, kt = kernel.shape
    cx, cy = (kx - 1) // 2, (ky - 1) // 2
    out = np.zeros(block_shape)
    qx, qy, qt = q.shape
    for i in range(qx):
        for j in range(qy):
            for s in range(qt):
                if q[i, j, s] == 0.0:
                    continue
                for a in range(kx):
                    for b in range(ky):
                        x = i + a - cx
                        y = j + b - cy
                        if 0 <= x < nx and 0 <= y < ny:
                            t0 = s
                            t1 = min(s + kt, nt)
                            out[x, y, t0:t1] += q[i, j, s] * kernel[a, b, : t1 - t0]
    return out


class TestForwardModel:
    def test_roundtrip_deconv_forward(self):
        """forward_model(deconv(m)) reproduces m for a well-formed block."""
        rng = np.random.default_rng(1)
        kernel = _make_kernel()
        nx, ny, nt = 7, 7, 64
        q = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        # place sparse charges away from the edges
        for _ in range(6):
            q[rng.integers(2, nx - 2), rng.integers(2, ny - 2),
              rng.integers(5, 40)] = rng.uniform(1, 10)
        block = _linear_convolve(q, kernel, (nx, ny, nt))

        deconv_q, _ = deconv_fft(block, kernel, None)
        assert deconv_q.shape == q.shape
        np.testing.assert_allclose(deconv_q, q, atol=1e-8)

        pred = forward_model_block(deconv_q, kernel, block.shape)
        np.testing.assert_allclose(pred, block, atol=1e-8)

    def test_forward_of_truth_matches_convolution(self):
        kernel = _make_kernel(seed=3)
        q = np.zeros((5, 5, 30))
        q[2, 2, 10] = 4.0
        q[1, 3, 12] = 2.0
        block_shape = (5, 5, 34)
        expected = _linear_convolve(q, kernel, block_shape)
        pred = forward_model_block(q, kernel, block_shape)
        np.testing.assert_allclose(pred, expected, atol=1e-10)


class TestMeasuredBinMask:
    def test_marks_latch_bins_floor(self):
        B = 30
        loc = np.array([[10, 20, 300]])  # trigger at t=300
        offset = np.array([10, 20, 300 - 5 * B])
        mask = measured_bin_mask(loc, nburst=4, adc_hold_delay=B,
                                 block_offset=offset, block_shape=(1, 1, 20))
        # latches at trigger + B..4B -> fpos = 6..9
        assert set(np.flatnonzero(mask[0, 0])) == {6, 7, 8, 9}

    def test_marks_split_bins_linear(self):
        B = 30
        loc = np.array([[0, 0, 300]])
        offset = np.array([0, 0, 300 - 5 * B])
        mask = measured_bin_mask(loc, nburst=1, adc_hold_delay=B,
                                 block_offset=offset, block_shape=(1, 1, 20),
                                 deposit_mode="linear", deposit_phase=-0.5)
        # latch fpos = 6 - 0.5 = 5.5 -> bins 5 and 6
        assert set(np.flatnonzero(mask[0, 0])) == {5, 6}

    def test_out_of_block_pixels_skipped(self):
        B = 30
        loc = np.array([[99, 99, 300]])
        offset = np.array([0, 0, 0])
        mask = measured_bin_mask(loc, nburst=2, adc_hold_delay=B,
                                 block_offset=offset, block_shape=(2, 2, 40))
        assert not mask.any()


class TestRefineSegments:
    def test_replaces_model_bins_conserving_integral(self):
        block = np.zeros((1, 1, 10))
        block[0, 0] = [0, 1, 2, 3, 10, 8, 0, 0, 0, 0]
        meas = np.zeros((1, 1, 10), dtype=bool)
        meas[0, 0, 4:6] = True  # bins 4,5 measured; 1..3 are template
        pred = np.zeros((1, 1, 10))
        pred[0, 0] = [0, 3, 3, 6, 99, 99, 0, 0, 0, 0]
        out = refine_unmeasured_segments(block, pred, meas)
        assert out[0, 0, 4] == 10 and out[0, 0, 5] == 8  # data untouched
        seg = out[0, 0, 1:4]
        assert seg.sum() == pytest.approx(6.0)  # integral preserved
        np.testing.assert_allclose(seg, [1.5, 1.5, 3.0])  # pred shape

    def test_negative_prediction_clipped(self):
        block = np.zeros((1, 1, 6))
        block[0, 0] = [0, 2, 2, 5, 0, 0]
        meas = np.zeros((1, 1, 6), dtype=bool)
        meas[0, 0, 3] = True
        pred = np.zeros((1, 1, 6))
        pred[0, 0] = [0, -1, 4, 0, 0, 0]
        out = refine_unmeasured_segments(block, pred, meas)
        np.testing.assert_allclose(out[0, 0, 1:3], [0.0, 4.0])

    def test_zero_prediction_leaves_segment(self):
        block = np.zeros((1, 1, 6))
        block[0, 0] = [0, 2, 2, 5, 0, 0]
        meas = np.zeros((1, 1, 6), dtype=bool)
        meas[0, 0, 3] = True
        pred = np.zeros((1, 1, 6))
        out = refine_unmeasured_segments(block, pred, meas)
        np.testing.assert_allclose(out[0, 0], block[0, 0])


class TestIterativeLoop:
    def test_perfect_data_is_fixed_point(self):
        """If the block is already consistent, iteration changes nothing."""
        kernel = _make_kernel(seed=5)
        q = np.zeros((5, 5, 40))
        q[2, 2, 15] = 5.0
        block = forward_model_block(q, kernel, (5, 5, 44))
        meas = np.ones(block.shape, dtype=bool)  # everything measured
        dq, refined = iterative_recompensation(block, kernel, None, meas, n_iter=2)
        np.testing.assert_allclose(refined, block, atol=1e-10)
        np.testing.assert_allclose(dq, q, atol=1e-7)

    def test_refinement_recovers_distorted_segment(self):
        """A wrongly-shaped (but integral-correct) segment is repaired."""
        kernel = _make_kernel(seed=7)
        q = np.zeros((5, 5, 40))
        q[2, 2, 18] = 6.0
        truth_block = forward_model_block(q, kernel, (5, 5, 44))

        # Corrupt an unmeasured window on the center pixel: keep its
        # integral but flatten its shape (like a bad template).
        corrupted = truth_block.copy()
        window = slice(14, 20)
        seg = truth_block[2, 2, window]
        corrupted[2, 2, window] = seg.sum() / seg.size

        meas = np.ones(truth_block.shape, dtype=bool)
        meas[2, 2, window] = False

        dq0, _ = deconv_fft(corrupted, kernel, None)
        err0 = np.abs(dq0 - q).max()
        dq2, refined = iterative_recompensation(
            corrupted, kernel, None, meas, n_iter=3)
        err2 = np.abs(dq2 - q).max()
        # The positivity projection is a weak contraction on this benign
        # synthetic (little negative ringing); require monotone improvement
        # of both the charge error and the segment shape, not a big factor.
        assert err2 < 0.99 * err0
        seg_err0 = np.abs(corrupted[2, 2, window] - seg).sum()
        seg_err2 = np.abs(refined[2, 2, window] - seg).sum()
        assert seg_err2 < 0.99 * seg_err0

    def test_positivity_off_is_noop(self):
        """Without the nonlinearity the loop must be an exact fixed point.

        Only holds for compactly-supported content (the deconv crop must be
        lossless), which the real pipeline guarantees via padding.  The
        underlying charge is given NEGATIVE parts so the block is genuinely
        'unphysical', yet the linear loop cannot know: pred == block.
        """
        kernel = _make_kernel(seed=9)
        q = np.zeros((7, 7, 40))
        q[3, 3, 15] = 5.0
        q[3, 4, 17] = -2.0  # unphysical content, invisible to a linear loop
        block = forward_model_block(q, kernel, (7, 7, 44))
        meas = np.zeros(block.shape, dtype=bool)
        meas[:, :, ::3] = True
        dq0, _ = deconv_fft(block, kernel, None)
        dq1, refined = iterative_recompensation(
            block, kernel, None, meas, n_iter=2, positivity=False)
        np.testing.assert_allclose(refined, block, atol=1e-8)
        np.testing.assert_allclose(dq1, dq0, atol=1e-8)
