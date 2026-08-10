"""Phase-2: term gradients vs autograd; engine + strategy behavior."""
import numpy as np
import pytest
import torch

from unfoldlarpix.constrained_solver import LatchWindow
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.terms.base import CoordProx, IterCtx
from unfoldlarpix.terms.data import DataFidelity
from unfoldlarpix.terms.censor import CensorRunningMax
from unfoldlarpix.solve.engine import Fista
from unfoldlarpix.solve.strategy import Ladder, SolveState

B = 30


def make_op(seed=1, nx=5, ny=5, nt=40, dense=False):
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.1, 1.0, (3, 3, 5)); k[1, 1] += 3.0
    k = k / k.sum()
    if dense:
        # dense coverage windows for recovery tests (values set later)
        windows = [LatchWindow(px, py, b * B, (b + 1) * B, 0.0)
                   for px in range(nx) for py in range(ny)
                   for b in range(nt)]
    else:
        windows = [LatchWindow(1, 1, 300.0, 330.0, 5.0),
                   LatchWindow(2, 3, 150.0, 195.0, 2.5)]
    return ZSOperator(k, (nx, ny, nt), windows, B, device="cpu",
                      dtype=torch.float64)


def autograd_check(term, op, seed=7, atol=1e-8):
    g = torch.Generator().manual_seed(seed)
    q = torch.rand(op.q_shape, generator=g, dtype=torch.float64) * 2.0
    # hand-written gradient
    grad = torch.zeros_like(q)
    term.grad_into(IterCtx(q, op), grad)
    # autograd through value()
    q2 = q.clone().requires_grad_(True)
    val = term.value(IterCtx(q2, op))
    val.backward()
    assert torch.allclose(grad, q2.grad, atol=atol), \
        f"max diff {(grad - q2.grad).abs().max()}"


class TestGradientsVsAutograd:
    def test_data_fidelity(self):
        op = make_op()
        autograd_check(DataFidelity(op), op)

    @pytest.mark.parametrize("norm", ["l2", "l1"])
    def test_censor_running_max(self, norm):
        op = make_op()
        nx, ny, nt = op.block_shape
        reset = np.zeros((nx, ny), np.int64)
        arm = np.full((nx, ny), 2, np.int64)
        term = CensorRunningMax(op, reset, arm, censor_end=nt - 2,
                                threshold=0.3, beta=1.5, norm=norm)
        autograd_check(term, op)

    def test_censor_l1_zero_curvature(self):
        op = make_op()
        nx, ny, nt = op.block_shape
        t = CensorRunningMax(op, np.zeros((nx, ny), np.int64),
                             np.zeros((nx, ny), np.int64), nt, 0.3,
                             norm="l1")
        assert t.curvature() == 0.0


class TestIterCtx:
    def test_conv_computed_once(self):
        op = make_op()
        calls = {"n": 0}
        orig = op.conv
        op.conv = lambda q: (calls.__setitem__("n", calls["n"] + 1), orig(q))[1]
        ctx = IterCtx(torch.rand(op.q_shape, dtype=torch.float64), op)
        _ = ctx.block_pred; _ = ctx.block_pred; _ = ctx.block_pred
        assert calls["n"] == 1


class TestEngineRecovery:
    def test_recovers_sparse_charge_dense_windows(self):
        """Port of the classic recovery test onto the new engine."""
        op = make_op(seed=4, dense=True)
        q_true = torch.zeros(op.q_shape, dtype=torch.float64)
        q_true[2, 2, 12] = 8.0
        q_true[2, 3, 20] = 3.0
        d = op.forward(q_true)
        op.d = d          # set ONCE at build time (test-only shortcut)
        q = Fista(n_iter=400).minimize(
            op, [DataFidelity(op)], CoordProx(1e-4))
        assert float((q - q_true).abs().max()) < 0.05


class TestLadder:
    def test_alpha_field_from_skeleton(self):
        """Stage transition logic, engine mocked out."""
        op = make_op()
        lad = Ladder([1.0, 0.5], seed_cut=0.5, soft_len=2.0)
        skel = torch.zeros(op.q_shape, dtype=torch.bool)
        skel[2, 2, 10] = True
        f = lad.alpha_field(op, 0.5, skel)
        assert float(f[2, 2, 10]) == pytest.approx(0.5, rel=1e-6)
        # one Manhattan step away: alpha * exp(1/2)
        assert float(f[2, 2, 11]) == pytest.approx(0.5 * np.exp(0.5), rel=1e-5)

    def test_ladder_runs_and_records(self):
        op = make_op(seed=4, dense=True)
        q_true = torch.zeros(op.q_shape, dtype=torch.float64)
        q_true[2, 2, 12] = 8.0
        op.d = op.forward(q_true)
        state = Ladder([0.5, 0.1, 0.02], n_iter=200).run(
            Fista(), op, [DataFidelity(op)], None,
            SolveState(q=torch.zeros(op.q_shape, dtype=torch.float64)))
        assert len(state.history) == 3
        assert float(state.q[2, 2, 12]) == pytest.approx(8.0, rel=0.1)


class TestCensorFractionalBoundaries:
    def test_fractional_reset_weights_boundary_bin(self):
        """A half-bin reset counts half of the boundary bin's charge."""
        op = make_op()
        nx, ny, nt = op.block_shape
        reset = np.full((nx, ny), 1.5)          # restart mid-bin-1
        arm = np.zeros((nx, ny))
        term = CensorRunningMax(op, reset, arm, censor_end=nt,
                                threshold=0.0, beta=1.0, norm="l1")
        w = term.w[0, 0].cpu().numpy()
        assert w[0] == 0.0 and w[1] == 0.5 and w[2] == 1.0

    def test_integer_reset_matches_hard_mask(self):
        op = make_op()
        nx, ny, nt = op.block_shape
        reset = np.full((nx, ny), 2.0)
        arm = np.zeros((nx, ny))
        term = CensorRunningMax(op, reset, arm, censor_end=nt,
                                threshold=0.0, beta=1.0, norm="l1")
        w = term.w[0, 0].cpu().numpy()
        assert (w[:2] == 0.0).all() and (w[2:] == 1.0).all()

    def test_arm_checks_first_valid_bin_end(self):
        """Re-arm inside bin k -> end of bin k (at (k+1)B) is armed."""
        op = make_op()
        nx, ny, nt = op.block_shape
        reset = np.zeros((nx, ny))
        arm = np.full((nx, ny), 2.3)            # re-arm inside bin 2
        term = CensorRunningMax(op, reset, arm, censor_end=nt,
                                threshold=0.0, beta=1.0, norm="l1")
        armed = term.armed[0, 0].cpu().numpy()
        assert not armed[1] and armed[2]        # end of bin 2 = 3.0 >= 2.3

    def test_fractional_autograd(self):
        op = make_op()
        nx, ny, nt = op.block_shape
        reset = np.random.default_rng(3).uniform(0, 1.8, (nx, ny))
        arm = reset + 0.7
        term = CensorRunningMax(op, reset, arm, censor_end=nt - 1,
                                threshold=0.2, beta=1.5, norm="l2")
        autograd_check(term, op)
