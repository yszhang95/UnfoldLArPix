"""Tests for the fine-binned operator and its closed-form Tikhonov inverse.

CPU only, float64, small synthetic grids.  Every test states the identity it
checks; nothing here loads the response file or touches the GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from unfoldlarpix.algs.finebasis_algs import (CellGrid, FineOperator,
                                              cell_charge_model_taps,
                                              cell_window_kernel,
                                              direct_sum_records,
                                              fine_window_kernel,
                                              prolong_truth_to_fine)
from unfoldlarpix.fwk.component import ALGORITHMS

B = 3                       # coarse stride, fine ticks
NX, NY, M = 5, 4, 12        # block: pads and record windows
KR = 1                      # kernel half-width in pads (3 x 3)
NTAU = 7                    # response length in fine ticks


def toy_response(seed: int = 3) -> np.ndarray:
    """A (3, 3, NTAU) impact-averaged response with vanishing ring integrals."""
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(2 * KR + 1, 2 * KR + 1, NTAU)) * 0.1
    R[KR, KR] += np.exp(-((np.arange(NTAU) - 4.0) ** 2) / 2.0)
    # ring integrals exactly zero, own integral exactly one
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if i == KR and j == KR:
                R[i, j] /= R[i, j].sum()
            else:
                R[i, j] -= R[i, j].mean()
    return R


def make_op(dtype=torch.float64) -> FineOperator:
    return FineOperator(toy_response(), (NX, NY, M), B, device="cpu",
                        dtype=dtype)


# ---------------------------------------------------------------------------
def test_registration():
    for name in ("FineBasisInverse", "FineBasisPlots", "FineBasisProbe",
                 "FineBasisProbePlots"):
        assert name in ALGORITHMS


def test_h_is_the_moving_sum_of_Kbar():
    """``h_d(tau) = sum_{m=tau-B+1}^{tau} Kbar_d(m)``, support ``0..NTAU+B-1``."""
    R = toy_response()
    h = fine_window_kernel(R, B)
    assert h.shape == (3, 3, NTAU + B)
    for tau in range(NTAU + B):
        lo = max(0, tau - B + 1)
        hi = min(NTAU, tau + 1)
        want = R[KR, KR, lo:hi].sum() if hi > lo else 0.0
        assert h[KR, KR, tau] == pytest.approx(want, abs=1e-13)
    # total: each response tick is counted B times
    assert h.sum() == pytest.approx(B * R.sum(), rel=1e-13)


def test_forward_matches_the_direct_sum():
    """``y_p[w] = sum_{p',j} x_{p'}(j) h_{p-p'}(b + B(w+1) - j)``, term by term.

    The direct sum uses no FFT and no periodicity; the operator is circular, so
    this also asserts that a signal placed well inside the grid does not wrap.
    """
    F = make_op()
    rng = np.random.default_rng(0)
    ix = np.array([1, 2, 2, 3])
    iy = np.array([1, 1, 2, 0])
    tj = np.array([4, 5, 5, 9])          # b = 0, well inside the length-36 grid
    q = rng.uniform(0.5, 2.0, size=4)
    x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
    for a, bb, t, v in zip(ix, iy, tj, q):
        x[a, bb, t] += v
    y = F.forward(x).numpy()[:NX, :NY]
    y_ref = direct_sum_records(F.h_np, ix, iy, tj, q, 0, B, NX, NY, M,
                               "cpu", torch.float64)
    assert np.abs(y - y_ref).max() < 1e-12 * max(np.abs(y_ref).max(), 1.0)


def test_adjoint_dot_product():
    """``<A x, y> = <x, A^T y>``."""
    F = make_op()
    g = torch.Generator().manual_seed(5)
    for _ in range(3):
        x = torch.randn((F.nxp, F.nyp, F.N), generator=g, dtype=torch.float64)
        y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
        a = float((F.forward(x) * y).sum())
        b = float((x * F.adjoint(y)).sum())
        assert abs(a - b) <= 1e-10 * max(abs(a), 1.0)


def test_AAt_diagonal_formula():
    """``G(nu) = (1/B) sum_m |hhat(nu + m M)|^2`` reproduces ``A(A^T y)``."""
    F = make_op()
    g = torch.Generator().manual_seed(7)
    for _ in range(3):
        y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
        u = F.AAt_fft(y)
        v = F.forward(F.adjoint(y))
        assert float(torch.abs(u - v).max()) < 1e-10 * float(torch.abs(v).max())


def test_AAt_diagonal_against_explicit_matrix():
    """The same symbol against a DENSE ``A A^T`` built column by column."""
    F = make_op()
    n = F.nxp * F.nyp * F.M
    A_At = np.zeros((n, n))
    for c in range(n):
        e = torch.zeros(n, dtype=torch.float64)
        e[c] = 1.0
        e = e.reshape(F.nxp, F.nyp, F.M)
        A_At[:, c] = F.forward(F.adjoint(e)).reshape(-1).numpy()
    g = torch.Generator().manual_seed(9)
    y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    want = A_At @ y.reshape(-1).numpy()
    got = F.AAt_fft(y).reshape(-1).numpy()
    assert np.abs(want - got).max() < 1e-9 * max(np.abs(want).max(), 1.0)
    assert np.abs(A_At - A_At.T).max() < 1e-10 * np.abs(A_At).max()


def test_lambda_to_zero_reproduces_a_row_space_signal():
    """For ``x = A^T u`` the inverse returns ``x`` exactly as ``lambda -> 0``.

    ``A^T u`` lies in the row space of ``A``, which is exactly the subspace the
    minimum-norm inverse can reach, so the reconstruction is not merely close
    to a signal consistent with the data -- it is the signal itself.
    """
    F = make_op()
    g = torch.Generator().manual_seed(13)
    u = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    x = F.adjoint(u)
    y = F.forward(x)
    xh = F.solve(y, 0.0)
    assert float(torch.abs(xh - x).max()) < 1e-8 * float(torch.abs(x).max())


def test_conservation_identity():
    """``sum xhat = B sum y / sum h = sum y / sum Kbar`` at ``lambda = 0``.

    ``h`` is the ``B``-tick moving sum of ``Kbar``, so its DFT vanishes at every
    non-zero multiple of the record-grid Nyquist period; DC therefore has no
    aliases and ``G(0) = hhat(0)^2 / B`` exactly.
    """
    F = make_op()
    R = toy_response()
    assert F.G_dc == pytest.approx(F.h_np.sum() ** 2 / B, rel=1e-12)
    g = torch.Generator().manual_seed(17)
    y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    xh = F.solve(y, 0.0)
    assert float(xh.sum()) == pytest.approx(
        B * float(y.sum()) / F.h_np.sum(), rel=1e-10)
    assert float(xh.sum()) == pytest.approx(
        float(y.sum()) / R.sum(), rel=1e-10)


def test_solve_is_least_norm_and_consistent():
    """``A xhat = y`` on the reachable part, and ``xhat`` is orthogonal to null(A)."""
    F = make_op()
    g = torch.Generator().manual_seed(19)
    u = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    y = F.forward(F.adjoint(u))
    xh = F.solve(y, 0.0)
    r = F.forward(xh) - y
    assert float(torch.linalg.vector_norm(r)) < 1e-8 * float(
        torch.linalg.vector_norm(y))
    # any n with A n = 0 must be orthogonal to xhat
    z = torch.randn((F.nxp, F.nyp, F.N), generator=g, dtype=torch.float64)
    nvec = z - F.solve(F.forward(z), 0.0)
    assert float(torch.linalg.vector_norm(F.forward(nvec))) < 1e-8
    assert abs(float((nvec * xh).sum())) < 1e-8 * float(
        torch.linalg.vector_norm(nvec) * torch.linalg.vector_norm(xh))


def test_lambda_shrinks_the_total_by_the_predicted_factor():
    """``sum xhat(lambda) / sum xhat(0) = G(0) / (G(0) + lambda)``."""
    F = make_op()
    g = torch.Generator().manual_seed(23)
    y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    s0 = float(F.solve(y, 0.0).sum())
    for lr in (1e-1, 1e-2, 1e-4):
        lam = lr * F.G_max
        s = float(F.solve(y, lam).sum())
        assert s / s0 == pytest.approx(F.G_dc / (F.G_dc + lam), rel=1e-9)


def test_decimation_phase_is_the_window_upper_edge():
    """A delta at fine tick ``j`` first shows up in the window whose latch is ``j``.

    Window ``w`` covers ``(b + Bw, b + B(w+1)]``, so a delta at ``j = B(w+1)``
    with ``h(0) = Kbar(0)`` non-zero lands in window ``w`` and not ``w+1``.
    """
    F = make_op()
    w0 = 4
    j = B * (w0 + 1)
    x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
    x[2, 2, j] = 1.0
    y = F.forward(x).numpy()
    assert y[2, 2, w0] == pytest.approx(F.h_np[KR, KR, 0], abs=1e-13)
    x2 = torch.zeros_like(x)
    x2[2, 2, j + 1] = 1.0
    y2 = F.forward(x2).numpy()
    assert y2[2, 2, w0] == pytest.approx(0.0, abs=1e-13)


# ---------------------------------------------------------------------------
# the intermediate (cell) time basis
# ---------------------------------------------------------------------------
BC = 6                      # coarse stride for the cell tests: divisors 1,2,3,6
MC = 8                      # record windows
NTAUC = 11                  # response length in fine ticks


def toy_response_c(seed: int = 5) -> np.ndarray:
    """A (3, 3, NTAUC) impact-averaged response, same shape rules as above."""
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(2 * KR + 1, 2 * KR + 1, NTAUC)) * 0.1
    R[KR, KR] += np.exp(-((np.arange(NTAUC) - 5.0) ** 2) / 2.0)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if i == KR and j == KR:
                R[i, j] /= R[i, j].sum()
            else:
                R[i, j] -= R[i, j].mean()
    return R


def make_cell_op(c: int, model: str = "uniform", shift: int = 0):
    return FineOperator(toy_response_c(), (NX, NY, MC), BC, device="cpu",
                        dtype=torch.float64, cell_ticks=c, cell_model=model,
                        release_shift=shift)


def test_cell_ticks_one_is_the_fine_operator_bit_for_bit():
    """``cell_ticks = 1`` changes NOTHING: kernel, symbol and maps are equal.

    Element for element, for both charge models -- at one tick per cell the
    two models are the same statement -- against the operator built with the
    default arguments.
    """
    ref = FineOperator(toy_response(), (NX, NY, M), B, device="cpu",
                       dtype=torch.float64)
    for model in ("uniform", "delta"):
        F = FineOperator(toy_response(), (NX, NY, M), B, device="cpu",
                         dtype=torch.float64, cell_ticks=1, cell_model=model)
        assert F.D == B and F.N == B * M == ref.N
        assert np.array_equal(F.g_np, ref.h_np)
        assert np.array_equal(F.g_np, fine_window_kernel(toy_response(), B))
        assert torch.equal(F.Hr, ref.Hr)
        assert torch.equal(F.G, ref.G)
        g = torch.Generator().manual_seed(101)
        x = torch.randn((F.nxp, F.nyp, F.N), generator=g, dtype=torch.float64)
        y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
        assert torch.equal(F.forward(x), ref.forward(x))
        assert torch.equal(F.adjoint(y), ref.adjoint(y))
        assert torch.equal(F.solve(y, 0.03), ref.solve(y, 0.03))


def test_cell_window_kernel_is_its_definition():
    """``g[s] = h_c(c s - shift)`` with ``h_c`` the moving average (uniform)
    or ``h`` itself (delta)."""
    R = toy_response_c()
    h = fine_window_kernel(R, BC)
    L = h.shape[-1]
    for c in (2, 3, 6):
        gu = cell_window_kernel(h, c, "uniform")
        for s in range(gu.shape[-1]):
            want = sum(h[..., c * s - u] if 0 <= c * s - u < L else 0.0
                       for u in range(c)) / c
            assert np.abs(gu[..., s] - want).max() < 1e-15
        gd = cell_window_kernel(h, c, "delta")
        for s in range(gd.shape[-1]):
            want = h[..., c * s] if c * s < L else 0.0
            assert np.abs(gd[..., s] - want).max() < 1e-15
        # the release shift moves the sampling grid by whole fine ticks
        gs = cell_window_kernel(h, c, "delta", 1)
        assert np.abs(gs[..., 0]).max() == 0.0
        assert np.abs(gs[..., 1] - h[..., c - 1]).max() < 1e-15


def test_cell_forward_matches_the_fine_operator_on_the_prolonged_truth():
    """``A_c (R_c x) = A_fine (P_0 R_c x)`` for ``cell_model = uniform``.

    The right-hand side is evaluated twice: by the FINE operator, and by the
    direct sum of the defining formula (no FFT, no periodicity).
    """
    Ff = FineOperator(toy_response_c(), (NX, NY, MC), BC, device="cpu",
                      dtype=torch.float64)
    rng = np.random.default_rng(4)
    for c in (2, 3, 6):
        F = make_cell_op(c, "uniform")
        grid = CellGrid(0, c, F.N)
        Rx = np.zeros((NX, NY, F.N))
        for p, q, m in ((1, 1, 2), (2, 1, 2), (2, 2, 3), (3, 0, 5)):
            Rx[p, q, m] = rng.uniform(0.5, 2.0)
        xc = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
        xc[:NX, :NY] = torch.as_tensor(Rx)
        y_cell = F.forward(xc).numpy()[:NX, :NY]

        taps, w = cell_charge_model_taps(grid, "uniform")
        ix, iy, tt, qq = prolong_truth_to_fine(grid, Rx, taps, w)
        xf = torch.zeros((Ff.nxp, Ff.nyp, Ff.N), dtype=torch.float64)
        for a, bb, t, v in zip(ix, iy, tt, qq):
            xf[a, bb, t] += v
        y_fine = Ff.forward(xf).numpy()[:NX, :NY]
        y_dir = direct_sum_records(Ff.h_np, ix, iy, tt, qq, 0, BC, NX, NY, MC,
                                   "cpu", torch.float64)
        scale = max(np.abs(y_dir).max(), 1.0)
        assert np.abs(y_cell - y_fine).max() < 1e-12 * scale
        assert np.abs(y_cell - y_dir).max() < 1e-12 * scale


def test_cell_delta_model_is_a_fine_delta_at_the_cell_lower_edge():
    """``cell_model = delta`` releases cell ``m`` at fine tick ``b + c m``."""
    Ff = FineOperator(toy_response_c(), (NX, NY, MC), BC, device="cpu",
                      dtype=torch.float64)
    for c in (2, 3, 6):
        F = make_cell_op(c, "delta")
        m = 3
        xc = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
        xc[2, 2, m] = 1.0
        xf = torch.zeros((Ff.nxp, Ff.nyp, Ff.N), dtype=torch.float64)
        xf[2, 2, c * m] = 1.0
        assert float(torch.abs(F.forward(xc) - Ff.forward(xf)).max()) < 1e-14


def test_cell30_delta_with_shift_one_is_the_bin_integrated_operator():
    """``c = B``, ``delta``, ``release_shift = 1`` IS ``A_coarse``.

    ``A_coarse[., k]`` has row ``w`` equal to
    ``Kbar_bin[w-k] = sum_{tau in [B(w-k), B(w-k+1))} Kbar(tau)`` -- the
    production operator's column.  The identity is algebraic, so the tolerance
    here is the float64 floor, not the 4.5e-5 measured on the GPU in float32.
    """
    R = toy_response_c()
    nb = int(np.ceil(R.shape[-1] / BC))
    kbin = np.zeros(R.shape[:2] + (nb,))
    for j in range(nb):
        kbin[..., j] = R[..., BC * j:BC * (j + 1)].sum(axis=-1)
    F = make_cell_op(BC, "delta", 1)
    k = 2
    xc = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
    xc[2, 2, k] = 1.0
    y = F.forward(xc).numpy()[:NX, :NY]
    want = np.zeros((NX, NY, MC))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for w in range(MC):
                j = w - k
                if 0 <= j < nb and 0 <= 2 + dx < NX and 0 <= 2 + dy < NY:
                    want[2 + dx, 2 + dy, w] = kbin[KR + dx, KR + dy, j]
    assert np.abs(y - want).max() < 1e-14


def test_cell_AAt_symbol_and_conservation():
    """``G_c``, ``G_c(0) = ghat(0)^2/D``, ``sum g = D sum Kbar``, ``sum xhat``."""
    R = toy_response_c()
    for c in (2, 3, 6):
        for model in ("uniform", "delta"):
            F = make_cell_op(c, model)
            assert F.g_np.sum() == pytest.approx(F.D * R.sum(), rel=1e-12)
            assert F.G_dc == pytest.approx(F.g_np.sum() ** 2 / F.D, rel=1e-12)
            g = torch.Generator().manual_seed(53)
            y = torch.randn((F.nxp, F.nyp, F.M), generator=g,
                            dtype=torch.float64)
            u = F.AAt_fft(y)
            v = F.forward(F.adjoint(y))
            assert float(torch.abs(u - v).max()) < 1e-10 * float(
                torch.abs(v).max())
            xh = F.solve(y, 0.0)
            assert float(xh.sum()) == pytest.approx(
                float(y.sum()) / R.sum(), rel=1e-10)


def test_cell_grid_prolongations_are_right_inverses_of_Rc():
    """``R_c P_0 = I``, ``R_c P_1 = I``, ``1^T P = 1^T``; ``c = 1`` is ``I``."""
    for c in (1, 2, 3, 5, 6, 10, 30):
        grid = CellGrid(-7050, c, max(60 // c, 6) + 20)
        rep = grid.report(n_probe=2)
        assert rep["uniform_Rc_P_minus_I_max"] < 1e-13
        assert rep["corrected_hat_Rc_P_minus_I_max"] < 1e-13
        assert rep["uniform_colsum_minus_1_max"] < 1e-13
        assert rep["corrected_hat_colsum_minus_1_max"] < 1e-13
        assert rep["hat_tap_weight_sum"] == pytest.approx(1.0, abs=1e-14)
        assert rep["T_band_sum"] == pytest.approx(1.0, abs=1e-14)
    g1 = CellGrid(-7050, 1, 40)
    assert list(g1.hat_taps) == [0] and float(g1.hat_w[0]) == 1.0
    assert (g1.T_diag, g1.T_sub, g1.T_sup) == (1.0, 0.0, 0.0)
    x = np.arange(40.0).reshape(1, 40)
    assert np.array_equal(g1.to_fine(x, "uniform", 0, 40), x)
    assert np.array_equal(g1.to_fine(x, "corrected_hat", 0, 40), x)


def test_cell_grid_T_bands_against_an_explicit_hat_sum():
    """``T = R_c P_hat`` band by band, from the hat evaluated tick by tick."""
    for c in (2, 5, 10, 30):
        grid = CellGrid(0, c, 9)
        m = 4
        j = np.arange(c * (m - 2), c * (m + 3))
        v = np.maximum(0.0, 1.0 - np.abs(j - grid.cc[m]) / c) / c
        kk = grid.index(j)
        assert v.sum() == pytest.approx(1.0, abs=1e-14)
        assert v[kk == m].sum() == pytest.approx(grid.T_diag, abs=1e-14)
        assert v[kk == m + 1].sum() == pytest.approx(grid.T_sub, abs=1e-14)
        assert v[kk == m - 1].sum() == pytest.approx(grid.T_sup, abs=1e-14)


def test_to_fine_on_a_cropped_window_matches_the_full_grid():
    """A candidate stored only on the evaluation window prolongs identically.

    ``P_0`` is local, so this is exact; ``P_1`` couples the cells through
    ``T^{-1}``, whose off-diagonal decay per cell is
    ``T_sub / T_diag <= 0.17``, so a crop a few tens of cells away from the
    signal is below the float64 floor.  Measured here, not argued.
    """
    c, n = 5, 400
    grid = CellGrid(-7050, c, n)
    rng = np.random.default_rng(11)
    x = np.zeros((2, n))
    x[:, 180:220] = rng.uniform(0.0, 3.0, size=(2, 40))
    m_lo, m_hi = 120, 280
    full = grid.to_fine(x, "corrected_hat", m_lo, m_hi)
    crop = grid.to_fine(x[:, m_lo:m_hi], "corrected_hat", m_lo, m_hi,
                        x_lo=m_lo)
    assert np.abs(full - crop).max() < 1e-12 * np.abs(full).max()
    f0 = grid.to_fine(x, "uniform", m_lo, m_hi)
    c0 = grid.to_fine(x[:, m_lo:m_hi], "uniform", m_lo, m_hi, x_lo=m_lo)
    assert np.array_equal(f0, c0)


def test_cell_prolongation_conserves_charge_and_places_it_in_the_cell():
    """``P_0`` puts a cell's charge on exactly its own ``c`` ticks."""
    c, n = 5, 20
    grid = CellGrid(0, c, n)
    x = np.zeros((1, n))
    x[0, 7] = 3.0
    pf = grid.to_fine(x, "uniform", 0, n)[0]
    nz = np.nonzero(pf)[0]
    assert list(nz) == list(range(c * 7, c * 8))
    assert pf.sum() == pytest.approx(3.0, abs=1e-14)
    assert np.allclose(pf[nz], 3.0 / c)


def test_column_sum_over_records_is_sum_Kbar_at_every_cell_width():
    """One unit of charge is credited ``sum Kbar`` by the records, at every ``c``.

    ``sum_w A_c[w, v] = sum_{d,tau} Kbar_d(tau)`` for a column ``v`` well
    inside the grid, whatever the cell width and whatever the charge model.
    This is why an l1 penalty ``alpha sum_v x_v`` -- a price per unit of
    RECONSTRUCTED CHARGE -- carries the same meaning on every basis and does
    NOT scale with the cell width.
    """
    R = toy_response_c()
    for c in (1, 2, 3, 6):
        for model in ("uniform", "delta"):
            F = make_cell_op(c, model)
            x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
            x[2, 2, F.N // 2] = 1.0
            assert float(F.forward(x).sum()) == pytest.approx(R.sum(),
                                                              rel=1e-12)
