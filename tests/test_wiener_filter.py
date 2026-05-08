"""Tests for the Wiener-inspired regularization filter."""

import numpy as np
import pytest

from unfoldlarpix.wiener_filter import wiener_inspired_filter_3d


class TestWienerInspiredFilter3D:
    def test_shape_matches_rfftn_output(self):
        s = (8, 6, 32)
        filt = wiener_inspired_filter_3d(
            s,
            dt=(1, 1, 4),
            sigma_pixel=(0.2, 0.2),
            omega_c=0.05,
            b=2.0,
        )
        assert filt.shape == (s[0], s[1], s[2] // 2 + 1)

    def test_dc_time_component_is_zero(self):
        filt = wiener_inspired_filter_3d(
            (4, 4, 16),
            dt=(1, 1, 4),
            sigma_pixel=(0.2, 0.2),
            omega_c=0.05,
            b=2.0,
        )
        np.testing.assert_array_equal(filt[..., 0], 0.0)

    def test_first_nonzero_freq_close_to_one_when_below_cutoff(self):
        nt = 256
        dt_t = 4.0
        # rfftfreq(nt, d=dt_t)[1] = 1/(nt*dt_t); pick omega_c much higher.
        filt = wiener_inspired_filter_3d(
            (4, 4, nt),
            dt=(1, 1, dt_t),
            sigma_pixel=(10.0, 10.0),
            omega_c=1.0,
            b=2.0,
        )
        assert filt[0, 0, 1] > 0.99

    def test_spatial_uniform_with_large_sigma(self):
        filt = wiener_inspired_filter_3d(
            (8, 8, 16),
            dt=(1, 1, 4),
            sigma_pixel=(1e6, 1e6),
            omega_c=1.0,
            b=2.0,
        )
        # At each non-DC time bin the spatial pattern should be essentially
        # uniform when sigma_pixel is huge (Gaussian -> 1 everywhere).
        for k in range(1, filt.shape[-1]):
            np.testing.assert_allclose(
                filt[..., k], filt[0, 0, k], atol=1e-6
            )

    def test_rejects_nonpositive_omega_c(self):
        with pytest.raises(ValueError, match="omega_c"):
            wiener_inspired_filter_3d(
                (4, 4, 8),
                dt=(1, 1, 4),
                sigma_pixel=(0.2, 0.2),
                omega_c=0.0,
                b=2.0,
            )

    def test_rejects_nonpositive_b(self):
        with pytest.raises(ValueError, match="b"):
            wiener_inspired_filter_3d(
                (4, 4, 8),
                dt=(1, 1, 4),
                sigma_pixel=(0.2, 0.2),
                omega_c=0.5,
                b=0.0,
            )

    def test_higher_b_gives_sharper_rolloff(self):
        s = (4, 4, 64)
        common = dict(s=s, dt=(1, 1, 4), sigma_pixel=(10.0, 10.0), omega_c=0.05)
        filt_b2 = wiener_inspired_filter_3d(b=2.0, **common)
        filt_b4 = wiener_inspired_filter_3d(b=4.0, **common)
        # Far above cutoff: b=4 should be smaller (sharper rolloff at high freq).
        idx_high = -1
        assert filt_b4[0, 0, idx_high] < filt_b2[0, 0, idx_high]
