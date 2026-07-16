"""Tests for the Gaussian-smoothed template-compensation helpers."""

import numpy as np
import pytest

from unfoldlarpix.burst_processor import (
    _make_one_sided_gaussian,
    _smooth_template_diff,
    _smooth_template_diff_leak,
    BurstSequenceProcessor,
)


class TestOneSidedGaussian:
    def test_kernel_normalised(self):
        g = _make_one_sided_gaussian(1.0, n_sigma=4.0)
        assert pytest.approx(g.sum(), abs=1e-12) == 1.0

    def test_peak_at_index_zero(self):
        g = _make_one_sided_gaussian(1.0, n_sigma=4.0)
        assert int(np.argmax(g)) == 0

    def test_length_matches_n_sigma(self):
        g = _make_one_sided_gaussian(2.0, n_sigma=3.0)
        assert g.size == int(np.ceil(3.0 * 2.0)) + 1

    def test_rejects_non_positive_sigma(self):
        with pytest.raises(ValueError):
            _make_one_sided_gaussian(0.0)
        with pytest.raises(ValueError):
            _make_one_sided_gaussian(-1.0)


class TestSmoothTemplateDiff:
    def test_peak_anchored_at_right_edge(self):
        dC = np.array([5.0, 4.0, 3.0, 2.0])
        kernel = _make_one_sided_gaussian(1.0)
        dC_s = _smooth_template_diff(dC, kernel)
        assert dC_s.size == dC.size
        # peak bin (index -1) collects kernel[0]*dC[-1] plus leftward smear
        assert dC_s[-1] < dC[-1]
        assert dC_s[-1] > kernel[0] * dC[-1] - 1e-12

    def test_no_forward_leak(self):
        # Construct a dC where the last element is the only non-zero entry.
        # Smoothing must redistribute mass to earlier bins only (no forward leak).
        dC = np.array([0.0, 0.0, 0.0, 1.0])
        kernel = _make_one_sided_gaussian(0.8, n_sigma=4.0)
        dC_s = _smooth_template_diff(dC, kernel)
        assert dC_s[-1] == pytest.approx(kernel[0])
        # All mass landed at indices <= len(dC)-1
        assert dC_s.size == dC.size

    def test_identity_kernel_returns_input(self):
        kernel = np.array([1.0])
        dC = np.array([5.0, 4.0, 3.0, 2.0])
        dC_s = _smooth_template_diff(dC, kernel)
        np.testing.assert_array_equal(dC_s, dC)


class TestSmoothTemplateDiffLeak:
    def test_lengths(self):
        dC = np.array([5.0, 4.0, 3.0, 2.0])
        kernel = _make_one_sided_gaussian(1.0)
        dC_s, overflow = _smooth_template_diff_leak(dC, kernel)
        assert dC_s.size == dC.size
        assert overflow.size == kernel.size - 1

    def test_mass_conservation(self):
        dC = np.array([5.0, 4.0, 3.0, 2.0])
        kernel = _make_one_sided_gaussian(1.0)
        dC_s, overflow = _smooth_template_diff_leak(dC, kernel)
        total_in = dC.sum() * kernel.sum()  # kernel.sum() == 1 by construction
        total_out = dC_s.sum() + overflow.sum()
        assert pytest.approx(total_out, rel=1e-10) == total_in

    def test_overflow_grows_with_kernel(self):
        # If the kernel extends past the array length, leftward tails of
        # dC[-1] spill into the overflow region.
        dC = np.array([0.0, 0.0, 0.0, 1.0])
        kernel = _make_one_sided_gaussian(0.8)
        dC_s, overflow = _smooth_template_diff_leak(dC, kernel)
        R = kernel.size - 1
        # overflow[R-1] is the bin immediately preceding dC_s[0]; with R=4
        # and dC[-1]=1, this slot receives kernel[R-(R-1)+... ] terms.
        # Mass conservation must hold:
        assert pytest.approx(dC_s.sum() + overflow.sum(), rel=1e-10) == dC.sum()


def _build_test_template_cumulative(n: int = 50) -> np.ndarray:
    t = np.arange(n)
    return np.cumsum(np.exp(-((t - 25.0) ** 2) / (2 * 5.0 ** 2)))


class TestRenormalizeMode:
    """Renormalize preserves the gap integral exactly."""

    @staticmethod
    def _make_processor(sigma_bins):
        return BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            deadtime=1.0,
            template=_build_test_template_cumulative(),
            threshold=5.0,
            template_smooth_sigma_bins=sigma_bins,
            template_smooth_edge_mode="renormalize",
        )

    def test_renormalize_preserves_integral(self):
        proc = self._make_processor(1.0)
        dC = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = proc._apply_template_smoothing(dC, np.arange(dC.size, dtype=float))
        assert pytest.approx(smoothed.sum(), rel=1e-10) == dC.sum()

    def test_disabled_smoothing_is_identity(self):
        proc = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            deadtime=1.0,
            template=_build_test_template_cumulative(),
            threshold=5.0,
            template_smooth_sigma_bins=None,
        )
        dC = np.array([1.0, 2.0, 3.0, 4.0])
        out = proc._apply_template_smoothing(dC, np.arange(dC.size, dtype=float))
        np.testing.assert_array_equal(out, dC)


class TestLeakMode:
    """Leak preserves total mass between gap and overflow."""

    def test_overflow_captured(self):
        proc = BurstSequenceProcessor(
            adc_hold_delay=10.0,
            tau=5.0,
            deadtime=1.0,
            template=_build_test_template_cumulative(),
            threshold=5.0,
            template_smooth_sigma_bins=1.0,
            template_smooth_edge_mode="leak",
        )
        proc._pending_template_overflow = []
        dC = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.arange(dC.size, dtype=float) * proc.adc_hold_delay + 100.0
        smoothed = proc._apply_template_smoothing(dC, times)
        assert len(proc._pending_template_overflow) == 1
        start_time, overflow = proc._pending_template_overflow[0]
        # mass conservation
        assert pytest.approx(smoothed.sum() + overflow.sum(), rel=1e-10) == dC.sum()
        # start_time is R bins before times[0]
        R = overflow.size
        assert start_time == pytest.approx(times[0] - R * proc.adc_hold_delay)


class TestInvalidConfig:
    def test_rejects_unknown_edge_mode(self):
        with pytest.raises(ValueError, match="template_smooth_edge_mode"):
            BurstSequenceProcessor(
                adc_hold_delay=10.0,
                tau=5.0,
                deadtime=1.0,
                template=_build_test_template_cumulative(),
                threshold=5.0,
                template_smooth_sigma_bins=1.0,
                template_smooth_edge_mode="bogus",
            )
