"""Equivalence tests for the torch solver backend (skipped without torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from unfoldlarpix.constrained_solver import (  # noqa: E402
    LatchWindow,
    ZSOperator,
    solve_fista,
)
from unfoldlarpix.constrained_solver_torch import (  # noqa: E402
    TorchZSOperator,
    solve_fista as solve_fista_torch,
)
from unfoldlarpix.iterative_recomp import forward_model_block  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_problem(seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.1, 1.0, size=(3, 3, 5))
    k[1, 1] += 3.0
    k /= k.sum()
    q_true = np.zeros((5, 5, 36))
    q_true[2, 2, 12] = 8.0
    q_true[1, 3, 20] = 3.0
    block = forward_model_block(q_true, k, (5, 5, 40))
    windows = [
        LatchWindow(px, py, b * 30, (b + 1) * 30, float(block[px, py, b]))
        for px in range(5) for py in range(5) for b in range(40)
    ]
    return k, block, windows, q_true


class TestTorchBackend:
    def test_forward_adjoint_match_numpy(self):
        k, block, windows, _ = _make_problem()
        op_np = ZSOperator(k, block.shape, windows, adc_hold_delay=30)
        op_t = TorchZSOperator(k, block.shape, windows, adc_hold_delay=30,
                               device=DEVICE)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(op_np.q_shape)
        f_np = op_np.forward(x)
        f_t = op_t.forward(op_t.to_tensor(x)).cpu().numpy()
        np.testing.assert_allclose(f_t, f_np, rtol=1e-4, atol=1e-4)

        r = rng.standard_normal(op_np.n_data)
        a_np = op_np.adjoint(r)
        a_t = op_t.adjoint(op_t.to_tensor(r)).cpu().numpy()
        np.testing.assert_allclose(a_t, a_np, rtol=1e-4, atol=1e-4)

    def test_solve_matches_numpy(self):
        k, block, windows, q_true = _make_problem()
        op_np = ZSOperator(k, block.shape, windows, adc_hold_delay=30)
        op_t = TorchZSOperator(k, block.shape, windows, adc_hold_delay=30,
                               device=DEVICE)
        kwargs = dict(alpha=1e-3, n_iter=300)
        q_np = solve_fista(op_np, **kwargs)
        q_t = solve_fista_torch(op_t, **kwargs)
        np.testing.assert_allclose(q_t, q_np, atol=5e-3)
        np.testing.assert_allclose(q_t, q_true, atol=0.05)
