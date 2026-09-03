"""Tests for the fine-binned operator and its closed-form Tikhonov inverse.

CPU only, float64, small synthetic grids.  Every test states the identity it
checks; nothing here loads the response file or touches the GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from unfoldlarpix.algs.finebasis_algs import (FineOperator, direct_sum_records,
                                              fine_window_kernel)
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
