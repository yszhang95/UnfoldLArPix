"""Tests for the constrained linear ZS solver."""

import numpy as np
import pytest

from unfoldlarpix.constrained_solver import (
    LatchWindow,
    ZSOperator,
    build_latch_windows,
    exponential_alpha_field,
    manhattan_distance_from,
    smear_kernel_gaussian,
    solve_fista,
    solve_fista_ladder,
)
from unfoldlarpix.iterative_recomp import forward_model_block


def _make_kernel(kx=3, ky=3, kt=5, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.1, 1.0, size=(kx, ky, kt))
    k[kx // 2, ky // 2] += 3.0
    return k / k.sum()


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

    def test_row_weights_fold_into_operator(self):
        kernel = _make_kernel()
        windows = [LatchWindow(1, 1, 0.0, 30.0, 4.0),
                   LatchWindow(1, 1, 30.0, 60.0, 2.0)]
        w = np.array([4.0, 1.0])
        op0 = ZSOperator(kernel, (3, 3, 20), windows, adc_hold_delay=30)
        opw = ZSOperator(kernel, (3, 3, 20), windows, adc_hold_delay=30,
                         row_weights=w)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(op0.q_shape)
        np.testing.assert_allclose(opw.forward(x),
                                   np.sqrt(w) * op0.forward(x), atol=1e-12)
        np.testing.assert_allclose(opw.d, np.sqrt(w) * op0.d, atol=1e-12)


class TestOperatorAdjoint:
    def test_dot_product_identity(self):
        """<A x, y> == <x, A^T y> to machine precision."""
        rng = np.random.default_rng(2)
        kernel = _make_kernel()
        block_shape = (6, 6, 40)
        windows = []
        for _ in range(15):
            px, py = rng.integers(0, 6, size=2)
            lo = float(rng.uniform(0, 900))
            hi = lo + float(rng.uniform(20, 120))
            windows.append(LatchWindow(int(px), int(py), lo, hi, 0.0))
        op = ZSOperator(kernel, block_shape, windows, adc_hold_delay=30)
        x = rng.standard_normal(op.q_shape)
        y = rng.standard_normal(op.n_data)
        lhs = float(np.dot(op.forward(x), y))
        rhs = float(np.sum(x * op.adjoint(y)))
        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_sampling_covers_full_window(self):
        """A window spanning whole bins sums the bin integrals exactly."""
        kernel = _make_kernel()
        block_shape = (3, 3, 20)
        B = 30
        windows = [LatchWindow(1, 1, 60.0, 150.0, 0.0)]  # bins 2,3,4
        op = ZSOperator(kernel, block_shape, windows, adc_hold_delay=B)
        block = np.zeros(block_shape)
        block[1, 1, :] = np.arange(20.0)
        sampled = op.sample(block)
        assert sampled[0] == pytest.approx(2.0 + 3.0 + 4.0)


class TestTriggerSplit:
    def test_first_window_split_at_trigger(self):
        B = 30
        loc = np.array([[5, 6, 300, 330, 400]])
        dat = np.array([[0.0, 0.0, 0.0, 12.0, 15.0]])
        offset = np.array([5, 6, 0])
        ws = build_latch_windows(loc, dat, B, offset, split_threshold=5.0)
        assert len(ws) == 3
        assert (ws[0].t_lo, ws[0].t_hi, ws[0].value) == (-np.inf, 300.0, 5.0)
        assert (ws[1].t_lo, ws[1].t_hi, ws[1].value) == (300.0, 330.0, 7.0)
        assert (ws[2].t_lo, ws[2].t_hi, ws[2].value) == (330.0, 360.0, 3.0)

    def test_no_split_below_threshold(self):
        B = 30
        loc = np.array([[5, 6, 300, 330, 400]])
        dat = np.array([[0.0, 0.0, 0.0, 3.0]])
        offset = np.array([5, 6, 0])
        ws = build_latch_windows(loc, dat, B, offset, split_threshold=5.0)
        assert len(ws) == 1
        assert ws[0].value == pytest.approx(3.0)


class TestTVGradient:
    def test_matches_numerical_gradient(self):
        from unfoldlarpix.constrained_solver import _tv_gradient

        rng = np.random.default_rng(5)
        x = rng.standard_normal((4, 4, 6))
        val, grad = _tv_gradient(x)
        h = 1e-6
        for idx in [(0, 0, 0), (2, 1, 3), (3, 3, 5)]:
            xp = x.copy()
            xp[idx] += h
            vp, _ = _tv_gradient(xp)
            num = (vp - val) / h
            assert grad[idx] == pytest.approx(num, rel=1e-3, abs=1e-6)


class TestSpectralWienerPrior:
    def test_flat_weight_equals_l2(self):
        """w == 1 must reproduce the flat ridge exactly (Parseval identity)."""
        from unfoldlarpix.constrained_solver import solve_fista

        kernel = _make_kernel(seed=12)
        q_true = np.zeros((5, 5, 36))
        q_true[2, 2, 12] = 8.0
        block = forward_model_block(q_true, kernel, (5, 5, 40))
        windows = [
            LatchWindow(px, py, b * 30, (b + 1) * 30, float(block[px, py, b]))
            for px in range(5) for py in range(5) for b in range(40)
        ]
        op = ZSOperator(kernel, (5, 5, 40), windows, adc_hold_delay=30)
        w = np.ones(36 // 2 + 1)
        q_a = solve_fista(op, alpha=1e-3, n_iter=150, lam_l2=0.05)
        q_b = solve_fista(op, alpha=1e-3, n_iter=150,
                          lam_spectral=0.05, spectral_weight=w)
        # identical up to FFT round-trip rounding accumulated over iterations
        np.testing.assert_allclose(q_a, q_b, atol=1e-6)

    def test_weight_construction(self):
        from unfoldlarpix.constrained_solver import wiener_spectral_weight

        freqs = np.linspace(0, 0.5, 50)
        P_truth = np.exp(-freqs / 0.05)          # falls steeply
        P_deconv = P_truth + 0.01                # flat noise on top
        w = wiener_spectral_weight(freqs, P_truth, P_deconv, n_time=64,
                                   cap=100.0)
        assert w.shape == (33,)
        assert w[0] == pytest.approx(0.01, rel=0.1)   # ~N/S at DC
        assert w[-1] == 100.0                          # capped in the tail
        assert np.all(np.diff(w) >= -1e-9)             # monotone rise


class TestGaussianBasis:
    def test_integral_shape_and_shift(self):
        kernel = _make_kernel()
        kg, shift = smear_kernel_gaussian(
            kernel, adc_hold_delay=30, sigma_time=0.005, sigma_pixel=0.2,
            pad_pixel=3, pad_time=8,
        )
        assert shift == 4
        assert kg.shape == (kernel.shape[0] + 6, kernel.shape[1] + 6,
                            kernel.shape[2] + 8)
        assert kg.shape[0] % 2 == 1  # odd spatial size preserved
        assert kg.sum() == pytest.approx(kernel.sum(), rel=1e-9)

    def test_operator_adjoint_with_smeared_kernel(self):
        rng = np.random.default_rng(3)
        kernel, _ = smear_kernel_gaussian(
            _make_kernel(), adc_hold_delay=30, sigma_time=0.005,
            sigma_pixel=0.2, pad_pixel=2, pad_time=6,
        )
        block_shape = (9, 9, 40)
        windows = [
            LatchWindow(int(rng.integers(0, 9)), int(rng.integers(0, 9)),
                        float(lo := rng.uniform(0, 900)), lo + 60.0, 0.0)
            for _ in range(10)
        ]
        op = ZSOperator(kernel, block_shape, windows, adc_hold_delay=30)
        x = rng.standard_normal(op.q_shape)
        y = rng.standard_normal(op.n_data)
        assert float(np.dot(op.forward(x), y)) == pytest.approx(
            float(np.sum(x * op.adjoint(y))), rel=1e-10)

    def test_blob_fit_localizes_at_shifted_index(self):
        """Data built from one blob is recovered at fit index = phys - shift."""
        base = _make_kernel(seed=4)
        kg, shift = smear_kernel_gaussian(
            base, adc_hold_delay=30, sigma_time=0.005, sigma_pixel=0.3,
            pad_pixel=2, pad_time=6,
        )
        B = 30
        nx, ny, nt = 7, 7, 50
        phys_t = 20
        c_true = np.zeros((nx, ny, nt - kg.shape[2] + 1))
        c_true[3, 3, phys_t - shift] = 6.0
        block = forward_model_block(c_true, kg, (nx, ny, nt))
        windows = [
            LatchWindow(px, py, b * B, (b + 1) * B, float(block[px, py, b]))
            for px in range(nx) for py in range(ny) for b in range(nt)
        ]
        op = ZSOperator(kg, (nx, ny, nt), windows, adc_hold_delay=B)
        c_hat = solve_fista(op, alpha=1e-4, n_iter=400)
        peak = np.unravel_index(np.argmax(c_hat), c_hat.shape)
        assert peak[:2] == (3, 3)
        assert abs(peak[2] - (phys_t - shift)) <= 1
        assert c_hat.sum() == pytest.approx(6.0, rel=0.05)


class TestSolverRecovery:
    def test_recovers_sparse_charge_from_complete_windows(self):
        """With dense window coverage the solver recovers the true charge."""
        kernel = _make_kernel(seed=4)
        B = 30
        nx, ny, nt = 5, 5, 40
        q_true = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        q_true[2, 2, 12] = 8.0
        q_true[2, 3, 20] = 3.0
        block = forward_model_block(q_true, kernel, (nx, ny, nt))

        windows = []
        for px in range(nx):
            for py in range(ny):
                for b in range(nt):
                    windows.append(
                        LatchWindow(px, py, b * B, (b + 1) * B,
                                    float(block[px, py, b]))
                    )
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        q_hat = solve_fista(op, alpha=1e-4, n_iter=400)
        assert np.abs(q_hat - q_true).max() < 0.05

    def test_zero_suppressed_windows_with_positivity(self):
        """Sparse (ZS) window coverage + positivity still localizes charge."""
        kernel = _make_kernel(seed=6)
        B = 30
        nx, ny, nt = 5, 5, 40
        q_true = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        q_true[2, 2, 12] = 8.0
        block = forward_model_block(q_true, kernel, (nx, ny, nt))

        # only the center pixel is read out, and only around the signal
        windows = []
        for b in range(10, 20):
            windows.append(
                LatchWindow(2, 2, b * B, (b + 1) * B, float(block[2, 2, b]))
            )
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        q_hat = solve_fista(op, alpha=1e-3, n_iter=300)
        # data on the read-out pixel is reproduced
        pred = op.forward(q_hat)
        np.testing.assert_allclose(pred, op.d, atol=0.05)
        # positivity + L1 keep the solution concentrated: total charge less
        # than or close to the truth (no wild ghosts)
        assert q_hat.sum() <= q_true.sum() * 1.2

    def test_ladder_recovers_strong_and_weak(self):
        """Homotopy: a strong and a weak charge are both recovered, the
        weak one entering only in later (small-alpha) stages near the
        strong-charge skeleton."""
        kernel = _make_kernel(seed=10)
        B = 30
        nx, ny, nt = 5, 5, 40
        q_true = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        q_true[2, 2, 12] = 10.0
        q_true[2, 3, 14] = 0.8    # weak neighbor charge
        block = forward_model_block(q_true, kernel, (nx, ny, nt))
        windows = [
            LatchWindow(px, py, b * B, (b + 1) * B, float(block[px, py, b]))
            for px in range(nx) for py in range(ny) for b in range(nt)
        ]
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)

        # stage-0 (alpha=0.3) alone keeps the strong charge, kills the weak
        q_strong = solve_fista(op, alpha=0.3, n_iter=300)
        assert q_strong[2, 2, 12] > 2.0
        assert q_strong[2, 3, 14] < 0.1

        # The weak charge sits at Manhattan distance 3 from the seed; the
        # dilation radius bounds how far small charges may enter, so
        # dilate=2 excludes it and dilate=3 admits it.
        q_narrow = solve_fista_ladder(
            op, [0.3, 0.05, 0.002], seed_cut=0.5, seed_dilate=2,
            n_iter_per_stage=300,
        )
        assert q_narrow[2, 3, 14] == 0.0

        q_hat = solve_fista_ladder(
            op, [0.3, 0.05, 0.002], seed_cut=0.5, seed_dilate=3,
            n_iter_per_stage=300,
        )
        assert q_hat[2, 2, 12] == pytest.approx(10.0, rel=0.05)
        assert q_hat[2, 3, 14] == pytest.approx(0.8, rel=0.3)

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

    def test_soft_prior_admits_distant_weak_charge(self):
        """The soft exponential prior admits the weak charge the hard
        dilate=2 mask permanently excludes (Manhattan distance 3), while
        still suppressing it in the strong-alpha stage."""
        kernel = _make_kernel(seed=10)
        B = 30
        nx, ny, nt = 5, 5, 40
        q_true = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        q_true[2, 2, 12] = 10.0
        q_true[2, 3, 14] = 0.8
        block = forward_model_block(q_true, kernel, (nx, ny, nt))
        windows = [
            LatchWindow(px, py, b * B, (b + 1) * B, float(block[px, py, b]))
            for px in range(nx) for py in range(ny) for b in range(nt)
        ]
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        q_hat = solve_fista_ladder(
            op, [0.3, 0.05, 0.002], seed_cut=0.5, soft_decay_len=2.0,
            n_iter_per_stage=300,
        )
        assert q_hat[2, 2, 12] == pytest.approx(10.0, rel=0.05)
        assert q_hat[2, 3, 14] == pytest.approx(0.8, rel=0.3)

    def test_deghost_regress_alternation(self):
        """D/R alternation recovers strong + weak charges with the weak
        amplitude coming from the (near-unbiased) regression phase."""
        from unfoldlarpix.constrained_solver import solve_deghost_regress

        kernel = _make_kernel(seed=10)
        B = 30
        nx, ny, nt = 5, 5, 40
        q_true = np.zeros((nx, ny, nt - kernel.shape[2] + 1))
        q_true[2, 2, 12] = 10.0
        q_true[2, 3, 14] = 0.8
        block = forward_model_block(q_true, kernel, (nx, ny, nt))
        windows = [
            LatchWindow(px, py, b * B, (b + 1) * B, float(block[px, py, b]))
            for px in range(nx) for py in range(ny) for b in range(nt)
        ]
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        q_hat = solve_deghost_regress(
            op, n_rounds=3, alpha_deghost=0.3, alpha_regress=0.002,
            seed_cut=0.5, decay_len=2.0, n_iter_deghost=250,
            n_iter_regress=250,
        )
        assert q_hat[2, 2, 12] == pytest.approx(10.0, rel=0.05)
        # The weak charge is below the deghost threshold, so it never joins
        # the skeleton permanently: it survives only through the exponential
        # prior tail and pays the distance-dependent shrinkage — by design.
        assert q_hat[2, 3, 14] == pytest.approx(0.8, rel=0.35)

    def test_ladder_rejects_empty_alphas(self):
        kernel = _make_kernel()
        op = ZSOperator(kernel, (3, 3, 20),
                        [LatchWindow(1, 1, 0.0, 30.0, 1.0)], adc_hold_delay=30)
        with pytest.raises(ValueError, match="ladder"):
            solve_fista_ladder(op, [])

    def test_quiet_penalty_suppresses_charge(self):
        """The quiet-bin inequality pulls predicted charge below threshold."""
        kernel = _make_kernel(seed=8)
        B = 30
        nx, ny, nt = 3, 3, 20
        windows = [LatchWindow(1, 1, 300.0, 330.0, 5.0)]
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        quiet = np.ones((nx, ny, nt), dtype=bool)
        quiet[1, 1] = False
        q_free = solve_fista(op, alpha=0.0, n_iter=200)
        q_con = solve_fista(op, alpha=0.0, n_iter=200,
                            beta_quiet=10.0, quiet_mask=quiet,
                            quiet_threshold=0.5)
        pred_free = op.conv(q_free)
        pred_con = op.conv(q_con)
        assert pred_con[quiet].max() <= pred_free[quiet].max() + 1e-9
        assert pred_con[quiet].max() < 1.0


class TestSubbinPositions:
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

    def test_recovers_subbin_offset(self):
        """Truth deposited between two bins (u=0.35) is recovered as a
        single skeleton charge with the right amplitude AND offset."""
        from unfoldlarpix.constrained_solver import (
            solve_subbin_positions,
            split_deposit,
        )

        kernel = _make_kernel(seed=4)
        B = 30
        nx, ny, nt = 5, 5, 40
        ntq = nt - kernel.shape[2] + 1
        q_true = np.zeros((nx, ny, ntq))
        q_true[2, 2, 12] = 8.0
        u_true = np.zeros((nx, ny, ntq))
        u_true[2, 2, 12] = 0.35
        block = forward_model_block(
            split_deposit(q_true, u_true), kernel, (nx, ny, nt))
        windows = []
        for px in range(nx):
            for py in range(ny):
                for b in range(nt):
                    windows.append(
                        LatchWindow(px, py, b * B, (b + 1) * B,
                                    float(block[px, py, b]))
                    )
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        skel = q_true > 0
        q0 = np.where(skel, 8.0, 0.0)
        q_hat, u_hat = solve_subbin_positions(
            op, q0, skeleton=skel, n_rounds=3, q_iters=80, u_iters=12,
            alpha=1e-4)
        assert q_hat[2, 2, 12] == pytest.approx(8.0, rel=0.03)
        assert u_hat[2, 2, 12] == pytest.approx(0.35, abs=0.03)

    def test_aligned_charge_keeps_zero_offset(self):
        from unfoldlarpix.constrained_solver import solve_subbin_positions

        kernel = _make_kernel(seed=5)
        B = 30
        nx, ny, nt = 5, 5, 40
        ntq = nt - kernel.shape[2] + 1
        q_true = np.zeros((nx, ny, ntq))
        q_true[2, 2, 12] = 8.0
        block = forward_model_block(q_true, kernel, (nx, ny, nt))
        windows = [
            LatchWindow(px, py, b * B, (b + 1) * B, float(block[px, py, b]))
            for px in range(nx) for py in range(ny) for b in range(nt)
        ]
        op = ZSOperator(kernel, (nx, ny, nt), windows, adc_hold_delay=B)
        skel = q_true > 0
        q_hat, u_hat = solve_subbin_positions(
            op, q_true.copy(), skeleton=skel, n_rounds=2, q_iters=60,
            u_iters=8, alpha=1e-4)
        assert abs(u_hat[2, 2, 12]) < 0.02
        assert q_hat[2, 2, 12] == pytest.approx(8.0, rel=0.03)

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
