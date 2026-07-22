"""Phase-1 core: hits accessors, torch operator, torch deconv parity."""
import numpy as np
import pytest
import torch

from unfoldlarpix.constrained_solver import LatchWindow
from unfoldlarpix.io.hits import HitsView
from unfoldlarpix.model.conventions import solver_time_shift
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.model.warm_start import (deconv_fft_torch,
                                           gaussian_filter_3d_torch)

B = 30


def _hits(nburst, n=3):
    rng = np.random.default_rng(0)
    trig = rng.integers(300, 900, n)
    loc = np.stack([np.arange(n), np.arange(n) + 5, trig, trig + B,
                    trig + nburst * B + 26], axis=1).astype(np.int64)
    dat = np.concatenate(
        [rng.normal(size=(n, 3)),
         np.cumsum(rng.uniform(1, 5, (n, nburst)), axis=1)], axis=1)
    return loc, dat


class TestHitsView:
    def test_accessors_nb4_style(self):
        loc, dat = _hits(nburst=4)
        hv = HitsView(loc, dat, B)
        assert hv.nburst == 4
        np.testing.assert_array_equal(hv.first_latch, hv.trigger + B)
        np.testing.assert_array_equal(hv.last_latch, hv.trigger + 4 * B)
        np.testing.assert_array_equal(hv.latch(2), hv.trigger + 2 * B)
        np.testing.assert_allclose(
            hv.burst_charges.sum(axis=1), hv.cumulative_charges[:, -1])

    def test_nb1_style_first_equals_last(self):
        loc, dat = _hits(nburst=1)
        hv = HitsView(loc, dat, B)
        np.testing.assert_array_equal(hv.first_latch, hv.last_latch)

    def test_semantics_violation_rejected(self):
        loc, dat = _hits(nburst=4)
        loc[0, 3] = loc[0, 2] + 4 * B     # the col3 bug, injected
        with pytest.raises(ValueError, match="col3"):
            HitsView(loc, dat, B)


class TestTorchOperator:
    def _op(self):
        rng = np.random.default_rng(1)
        k = rng.uniform(0.1, 1.0, (3, 3, 5)); k[1, 1] += 3.0
        windows = [LatchWindow(1, 1, 300.0, 330.0, 5.0),
                   LatchWindow(2, 2, 150.0, 175.0, 2.0)]
        return ZSOperator(k / k.sum(), (5, 5, 20), windows, B,
                          device="cpu", dtype=torch.float64)

    def test_adjoint_dot(self):
        op = self._op()
        g = torch.Generator().manual_seed(2)
        q = torch.randn(op.q_shape, generator=g, dtype=torch.float64)
        r = torch.randn(op.n_data, generator=g, dtype=torch.float64)
        lhs = float(torch.dot(op.forward(q), r))
        rhs = float((q * op.adjoint(r)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_lipschitz_cached_and_positive(self):
        op = self._op()
        l1 = op.lipschitz
        assert l1 > 0 and op.lipschitz == l1     # cached


class TestTorchDeconvParity:
    def test_matches_numpy_deconv(self):
        from unfoldlarpix.deconv import deconv_fft, gaussian_filter_3d
        rng = np.random.default_rng(3)
        block = rng.normal(size=(6, 6, 24))
        kernel = rng.uniform(0.1, 1.0, (3, 3, 5)); kernel[1, 1] += 3.0
        shape = (block.shape[0] + 2, block.shape[1] + 2, block.shape[2])
        filt_np = gaussian_filter_3d(shape, dt=(1, 1, B),
                                     sigma=(0.2, 0.2, 0.005))
        q_np, _ = deconv_fft(block, kernel, filt_np)
        q_t = deconv_fft_torch(
            torch.as_tensor(block, dtype=torch.float64),
            torch.as_tensor(kernel, dtype=torch.float64),
            gaussian_filter_3d_torch(shape, (1, 1, B), (0.2, 0.2, 0.005),
                                     "cpu", torch.float64))
        np.testing.assert_allclose(q_t.numpy(), q_np, atol=1e-10)


def test_solver_time_shift_convention():
    assert solver_time_shift(30) == -15
