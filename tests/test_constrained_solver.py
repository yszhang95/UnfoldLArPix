"""Measurement-building and numpy-utility tests (post torch-only refactor).

Solver tests moved to tests/test_terms_engine.py (new engine) and
tests/test_model_core.py (torch operator).
"""

import numpy as np
import pytest

from unfoldlarpix.constrained_solver import (
    LatchWindow,
    build_latch_windows,
    centroid_bin_offsets,
    exponential_alpha_field,
    manhattan_distance_from,
    split_adjoint,
    split_deposit,
    wiener_spectral_weight,
    windows_to_sampling,
)


class TestBuildLatchWindows:
    def test_single_sequence_windows(self):
        B = 30
        loc = np.array([[5, 6, 300, 330, 400]])
        dat = np.array([[0.0, 0.0, 0.0, 10.0, 14.0, 15.0]])  # cumulative
        offset = np.array([5, 6, 0])
        ws = build_latch_windows(loc, dat, B, offset)
        assert len(ws) == 3
        # first burst: from -inf to trigger + B
        assert ws[0].t_lo == -np.inf and ws[0].t_hi == 330.0
        assert ws[0].value == pytest.approx(10.0)
        # subsequent exact windows with diffed charges
        assert (ws[1].t_lo, ws[1].t_hi, ws[1].value) == (330.0, 360.0, 4.0)
        assert (ws[2].t_lo, ws[2].t_hi, ws[2].value) == (360.0, 390.0, 1.0)

    def test_second_sequence_starts_at_restart(self):
        B = 30
        loc = np.array([
            [5, 6, 300, 330, 400],
            [5, 6, 600, 630, 700],
        ])
        dat = np.array([
            [0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0, 8.0],
        ])
        offset = np.array([5, 6, 0])
        ws = build_latch_windows(loc, dat, B, offset)
        assert len(ws) == 2
        assert ws[1].t_lo == 400.0  # fallback: previous next_integration_start
        assert ws[1].t_hi == 630.0

    def test_restart_uses_csa_reset_when_given(self):
        B = 30
        loc = np.array([
            [5, 6, 300, 330, 400],
            [5, 6, 600, 630, 700],
        ])
        dat = np.array([
            [0.0, 0.0, 0.0, 10.0, 12.0],  # two bursts: latches 330, 360
            [0.0, 0.0, 0.0, 8.0, 9.0],
        ])
        offset = np.array([5, 6, 0])
        ws = build_latch_windows(loc, dat, B, offset, csa_reset_time=2)
        # windows: seq1 first, seq1 burst2, seq2 first, seq2 burst2
        assert len(ws) == 4
        # seq2 first window starts at seq1 last latch (360) + csa reset (2)
        assert ws[2].t_lo == 362.0
        assert ws[2].t_hi == 630.0


class TestCumulativeWindows:
    def test_rows_share_restart_and_carry_cumulative_values(self):
        from unfoldlarpix.constrained_solver import build_cumulative_windows

        B = 30
        loc = np.array([[5, 6, 300, 330, 400]])
        dat = np.array([[0.0, 0.0, 0.0, 10.0, 14.0, 15.0]])
        offset = np.array([5, 6, 0])
        ws, pseudo = build_cumulative_windows(loc, dat, B, offset,
                                              split_threshold=5.0)
        assert len(ws) == 4 and pseudo.tolist() == [True, False, False, False]
        assert (ws[0].t_lo, ws[0].t_hi, ws[0].value) == (-np.inf, 300.0, 5.0)
        # cumulative rows all start at the same restart, values cumulative
        for k, (hi, val) in enumerate(((330.0, 10.0), (360.0, 14.0),
                                       (390.0, 15.0))):
            w = ws[1 + k]
            assert (w.t_lo, w.t_hi, w.value) == (-np.inf, hi, val)


class TestSoftSeedField:
    def test_manhattan_distance_and_alpha_field(self):
        seed = np.zeros((5, 5, 5), dtype=bool)
        seed[2, 2, 2] = True
        d = manhattan_distance_from(seed, d_max=10)
        assert d[2, 2, 2] == 0
        assert d[2, 2, 3] == 1
        assert d[2, 3, 3] == 2
        assert d[0, 0, 0] == 6
        field = exponential_alpha_field(seed, alpha=0.1, decay_len=2.0)
        assert field[2, 2, 2] == pytest.approx(0.1)
        assert field[2, 2, 4] == pytest.approx(0.1 * np.e)
        assert np.all(np.diff(field[2, 2, 2:]) >= 0)

    def test_manhattan_distance_is_periodic(self):
        """Pins the wrap-around: _dilate_mask uses np.roll, so the seed
        distance is periodic on every axis.  Documented, not endorsed --
        the record pipeline was produced with this behaviour, and
        weighted_l1_distance_from is the open-boundary alternative."""
        from unfoldlarpix.constrained_solver import weighted_l1_distance_from
        seed = np.zeros((1, 1, 21), dtype=bool)
        seed[0, 0, 1] = True
        d = manhattan_distance_from(seed, d_max=16)
        assert d[0, 0, 20] == 2          # wraps: 2 steps, not 19
        d_open = weighted_l1_distance_from(seed, 16, (1, 1, 1))
        assert d_open[0, 0, 20] == 16    # open boundary, capped at d_max

    def test_weighted_l1_matches_manhattan_at_unit_cost(self):
        from unfoldlarpix.constrained_solver import weighted_l1_distance_from
        seed = np.zeros((11, 13, 15), dtype=bool)
        seed[5, 6, 7] = True
        seed[4, 4, 4] = True
        # both seeds are more than d_max from every edge, so the wrap
        # never binds and the two metrics must agree voxel by voxel
        a = manhattan_distance_from(seed, d_max=3).astype(float)
        b = weighted_l1_distance_from(seed, 3, (1, 1, 1))
        assert np.allclose(a, b)

    def test_axis_cost_scales_the_field_along_that_axis(self):
        seed = np.zeros((1, 1, 21), dtype=bool)
        seed[0, 0, 10] = True
        plain = exponential_alpha_field(seed, 0.3, 2.0)
        cheap = exponential_alpha_field(seed, 0.3, 2.0,
                                        axis_cost=(1.0, 1.0, 0.5))
        assert plain[0, 0, 10] == pytest.approx(cheap[0, 0, 10])
        # a half-cost time step is a half-distance step
        assert cheap[0, 0, 14] == pytest.approx(plain[0, 0, 12])


class TestSplitAndCentroid:
    def test_split_identity_sum_and_adjoint(self):
        from unfoldlarpix.constrained_solver import split_adjoint, split_deposit

        rng = np.random.default_rng(0)
        q = rng.uniform(0.0, 5.0, size=(4, 4, 12))
        u = rng.uniform(-0.5, 0.5, size=(4, 4, 12))
        u[:, :, 0] = np.clip(u[:, :, 0], 0.0, 0.5)
        u[:, :, -1] = np.clip(u[:, :, -1], -0.5, 0.0)
        # u = 0 is the identity
        np.testing.assert_allclose(split_deposit(q, np.zeros_like(u)), q)
        # sum-preserving (edge charges cannot spill outside)
        out = split_deposit(q, u)
        assert out.sum() == pytest.approx(q.sum(), rel=1e-12)
        # adjoint dot test: <S q, g> == <q, S^T g>
        g = rng.standard_normal(q.shape)
        lhs = float((split_deposit(q, u) * g).sum())
        rhs = float((q * split_adjoint(g, u)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_centroid_bin_offsets(self):
        from unfoldlarpix.constrained_solver import centroid_bin_offsets

        q = np.zeros((3, 3, 10))
        # a 0.65/0.35 split between bins 4 and 5 = charge at 4 + 0.35
        q[1, 1, 4], q[1, 1, 5] = 6.5, 3.5
        u = centroid_bin_offsets(q, window_bins=1)
        assert u[1, 1, 4] == pytest.approx(0.35, abs=1e-9)
        assert u[1, 1, 5] == pytest.approx(-0.5, abs=1e-9)  # clipped
        # an isolated single-bin charge keeps offset 0
        q2 = np.zeros((3, 3, 10))
        q2[0, 0, 3] = 4.0
        u2 = centroid_bin_offsets(q2, window_bins=2)
        assert u2[0, 0, 3] == 0.0
