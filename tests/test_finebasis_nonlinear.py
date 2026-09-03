"""Tests for the ``ZSOperator`` wrapper around the fine operator and for the
nonlinear (positivity, positivity + l1) arms built on it.

CPU only, float64, small synthetic grids.  Every test states the identity it
checks; nothing here loads the response file or touches the GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from unfoldlarpix.algs.finebasis_algs import FineOperator
from unfoldlarpix.algs.finebasis_nonlinear_algs import (FineZSOperator,
                                                        solve_fine_arm,
                                                        upsample_support)
from unfoldlarpix.fwk.component import ALGORITHMS
from unfoldlarpix.solve.engine import Fista
from unfoldlarpix.terms.base import CoordProx
from unfoldlarpix.terms.data import DataFidelity

from test_finebasis import B, KR, M, NX, NY, toy_response


def make_pair(dtype=torch.float64):
    """The fine operator and its ``ZSOperator`` wrapper on the same kernel."""
    F = FineOperator(toy_response(), (NX, NY, M), B, device="cpu", dtype=dtype)
    d = torch.zeros((F.nxp, F.nyp, F.M), dtype=dtype)
    return F, FineZSOperator(F, d, NX, NY)


# ---------------------------------------------------------------------------
def test_registration():
    for name in ("FineNonlinearArms", "FineNonlinearPlots",
                 "FineNonlinearProbe", "FineNonlinearProbePlots"):
        assert name in ALGORITHMS


def test_wrapper_shapes_and_lipschitz_is_exactly_G_max():
    """``lipschitz`` is read off the closed-form symbol, not power-iterated."""
    F, zop = make_pair()
    assert zop.q_shape == (NX, NY, F.N)
    assert zop.n_data == F.nxp * F.nyp * F.M
    assert zop.lipschitz == F.G_max
    # and G_max IS ||A A^T||: the largest eigenvalue of the dense A A^T
    n = F.nxp * F.nyp * F.M
    AAt = np.zeros((n, n))
    for c in range(n):
        e = torch.zeros(n, dtype=torch.float64)
        e[c] = 1.0
        AAt[:, c] = F.forward(
            F.adjoint(e.reshape(F.nxp, F.nyp, F.M))).reshape(-1).numpy()
    assert float(np.linalg.eigvalsh(0.5 * (AAt + AAt.T)).max()) == pytest.approx(
        F.G_max, rel=1e-9)


def test_wrapper_forward_agrees_with_the_fine_operator():
    """``zop.forward(q)`` is ``F.forward`` of ``q`` embedded in the padded grid."""
    F, zop = make_pair()
    g = torch.Generator().manual_seed(31)
    q = torch.randn((NX, NY, F.N), generator=g, dtype=torch.float64)
    x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
    x[:NX, :NY] = q
    want = F.forward(x)
    got = zop.forward(q)
    assert float(torch.abs(want - got).max()) < 1e-12 * float(
        torch.abs(want).max())


def test_wrapper_adjoint_agrees_with_the_fine_operator():
    """``zop.adjoint(r)`` is ``F.adjoint(r)`` cropped to the real pads."""
    F, zop = make_pair()
    g = torch.Generator().manual_seed(37)
    r = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    want = F.adjoint(r)[:NX, :NY]
    got = zop.adjoint(r)
    assert float(torch.abs(want - got).max()) < 1e-12 * float(
        torch.abs(want).max())


def test_wrapper_adjoint_dot_product():
    """``<A q, r> = <q, A^T r>`` for the restricted operator itself."""
    F, zop = make_pair()
    g = torch.Generator().manual_seed(41)
    for _ in range(3):
        q = torch.randn((NX, NY, F.N), generator=g, dtype=torch.float64)
        r = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
        a = float((zop.forward(q) * r).sum())
        b = float((q * zop.adjoint(r)).sum())
        assert abs(a - b) <= 1e-10 * max(abs(a), 1.0)


def test_measurement_gain_is_A_transpose_of_the_row_mask():
    """``c_v = A^T 1_rec`` term by term for one voxel."""
    F, zop = make_pair()
    mask = torch.zeros((F.nxp, F.nyp, F.M), dtype=torch.float64)
    mask[:NX, :NY, 2:9] = 1.0
    c = zop.measurement_gain(mask)
    # one voxel, checked against the defining sum over the rows in the mask
    p, q, j = 2, 2, 11
    want = 0.0
    for w in range(2, 9):
        tau = B * (w + 1) - j
        if 0 <= tau < F.h_np.shape[-1]:
            want += F.h_np[KR, KR, tau]
    assert float(c[p, q, j]) == pytest.approx(want, abs=1e-12)


def test_nnls_recovers_a_nonnegative_row_space_truth():
    """Synthetic NNLS: a non-negative truth in the row space is recovered.

    ``x = (A^T u)_+`` is not in the row space in general, so the truth here is
    built the other way round: take a non-negative, band-limited ``x`` (a
    Gaussian bump on a few pads, so its fine-grid spectrum is concentrated well
    below the record Nyquist), form ``y = A x``, and check that FISTA with
    ``x >= 0`` on the full support returns ``x`` itself.  It can, because ``x``
    is the minimum-norm consistent solution up to a null-space component that
    positivity forbids: any admissible perturbation must keep ``x >= 0`` and
    ``A x`` fixed, and on this toy operator the only such perturbation that
    survives is numerically zero -- which is what the residual assertion below
    measures directly.
    """
    F, zop = make_pair()
    t = np.arange(F.N)
    prof = np.exp(-0.5 * ((t - 18.0) / 3.0) ** 2)
    x = np.zeros((NX, NY, F.N))
    for p, q, a in ((2, 1, 1.0), (2, 2, 0.6), (3, 2, 0.3)):
        x[p, q] = a * prof
    xt = torch.as_tensor(x, dtype=torch.float64)
    zop.d = zop.forward(xt)
    supp = torch.ones((NX, NY, F.N), dtype=torch.bool)
    xh, _hist, _w = solve_fine_arm(zop, supp, 0.0, 4000, log_every=0)
    r = zop.forward(xh) - zop.d
    assert float(torch.linalg.vector_norm(r)) < 1e-6 * float(
        torch.linalg.vector_norm(zop.d))
    assert float(xh.min()) >= 0.0
    assert float(xh.sum()) == pytest.approx(float(xt.sum()), rel=2e-3)
    assert float(torch.abs(xh - xt).max()) < 2e-2 * float(xt.max())


def test_l1_only_shrinks_and_positivity_is_enforced():
    """``alpha > 0`` lowers the total charge and never makes it negative."""
    F, zop = make_pair()
    t = np.arange(F.N)
    x = np.zeros((NX, NY, F.N))
    x[2, 2] = np.exp(-0.5 * ((t - 18.0) / 3.0) ** 2)
    zop.d = zop.forward(torch.as_tensor(x, dtype=torch.float64))
    supp = torch.ones((NX, NY, F.N), dtype=torch.bool)
    tot = []
    for alpha in (0.0, 1e-3, 1e-2):
        xh, _h, _w = solve_fine_arm(zop, supp, alpha, 1500, log_every=0)
        assert float(xh.min()) >= 0.0
        tot.append(float(xh.sum()))
    assert tot[0] > tot[1] > tot[2]


def test_the_shipped_prox_and_engine_are_what_runs():
    """``solve_fine_arm`` is exactly ``Fista + DataFidelity + CoordProx``."""
    F, zop = make_pair()
    g = torch.Generator().manual_seed(43)
    zop.d = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    supp = torch.ones((NX, NY, F.N), dtype=torch.bool)
    a, _h, _w = solve_fine_arm(zop, supp, 0.02, 60, log_every=0)
    b = Fista(n_iter=60).minimize(zop, [DataFidelity(zop)],
                                  CoordProx(0.02, supp), q0=None)
    assert float(torch.abs(a - b).max()) == 0.0


def test_alpha_zero_is_positively_homogeneous():
    """``xhat(c y) = c xhat(y)`` exactly for ``alpha = 0``, and NOT for l1.

    With ``q0 = 0``, a data-independent step and the prox
    ``max(v, 0) . supp``, every FISTA iterate is positively homogeneous in the
    data, so a positivity-only arm has no charge dependence at all -- its
    normalised impulse response is the same at every ``Q``.  ``alpha > 0``
    introduces an absolute charge scale and breaks it.
    """
    F, zop = make_pair()
    g = torch.Generator().manual_seed(47)
    y = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
    supp = torch.ones((NX, NY, F.N), dtype=torch.bool)
    zop.d = y
    a, _h, _w = solve_fine_arm(zop, supp, 0.0, 120, log_every=0)
    zop.d = 7.0 * y
    b, _h, _w = solve_fine_arm(zop, supp, 0.0, 120, log_every=0)
    scale = float(torch.abs(b - 7.0 * a).max()) / max(
        float(torch.abs(7.0 * a).max()), 1e-30)
    assert scale < 1e-12
    # with l1 the same test must FAIL, i.e. the two answers differ
    zop.d = y
    c1, _h, _w = solve_fine_arm(zop, supp, 0.05, 120, log_every=0)
    zop.d = 7.0 * y
    c7, _h, _w = solve_fine_arm(zop, supp, 0.05, 120, log_every=0)
    assert float(torch.abs(c7 - 7.0 * c1).max()) > 1e-6 * float(
        torch.abs(7.0 * c1).max())


def test_upsample_support_maps_every_fine_tick_to_its_cell():
    """``base_fine[p, j] = base[p, k(b + j)]``, ``False`` outside every cell."""
    n_coarse = 4
    c = np.array([1.5, 4.5, 7.5, 10.5])       # centres, B = 3
    base = np.zeros((2, 1, n_coarse), dtype=bool)
    base[0, 0, 1] = True                      # cell 1 covers ticks 3, 4, 5
    base[1, 0, 3] = True                      # cell 3 covers ticks 9, 10, 11
    out = upsample_support(base, c, 3.0, 0, 14)
    assert out.shape == (2, 1, 14)
    assert list(np.nonzero(out[0, 0])[0]) == [3, 4, 5]
    assert list(np.nonzero(out[1, 0])[0]) == [9, 10, 11]
    assert not out[:, :, 12:].any()           # beyond the last cell


# ---------------------------------------------------------------------------
# the intermediate (cell) time basis
# ---------------------------------------------------------------------------
def test_cell_registration():
    assert "CellNonlinearArms" in ALGORITHMS
    assert "CellBasisInverse" in ALGORITHMS


def test_wrapper_agrees_with_the_cell_operator():
    """The ``ZSOperator`` wrapper decimates by ``D``, not by ``B``.

    ``zop.forward`` must be ``F.forward`` of the embedded unknown and
    ``zop.adjoint`` its crop, at every cell width -- the wrapper carries its
    own copy of the transform, so this is a test and not a restatement.
    """
    from test_finebasis import BC, MC, toy_response_c
    for c in (1, 2, 3, 6):
        F = FineOperator(toy_response_c(), (NX, NY, MC), BC, device="cpu",
                         dtype=torch.float64, cell_ticks=c)
        zop = FineZSOperator(F, torch.zeros((F.nxp, F.nyp, F.M),
                                            dtype=torch.float64), NX, NY)
        assert zop.q_shape == (NX, NY, F.N)
        g = torch.Generator().manual_seed(61 + c)
        q = torch.randn((NX, NY, F.N), generator=g, dtype=torch.float64)
        r = torch.randn((F.nxp, F.nyp, F.M), generator=g, dtype=torch.float64)
        x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
        x[:NX, :NY] = q
        assert float(torch.abs(zop.forward(q) - F.forward(x)).max()) < 1e-12
        assert float(torch.abs(zop.adjoint(r)
                               - F.adjoint(r)[:NX, :NY]).max()) < 1e-12
        a = float((zop.forward(q) * r).sum())
        bb = float((q * zop.adjoint(r)).sum())
        assert abs(a - bb) <= 1e-10 * max(abs(a), 1.0)


def test_upsample_support_on_the_cell_basis():
    """``base_cell[p, m] = base[p, k(cc_m)]``: the cell's own centre decides."""
    c = np.array([1.5, 4.5, 7.5, 10.5])        # production centres, B = 3
    base = np.zeros((2, 1, 4), dtype=bool)
    base[0, 0, 1] = True                       # production cell 1: ticks 3,4,5
    out1 = upsample_support(base, c, 3.0, 0, 12, 1)
    assert list(np.nonzero(out1[0, 0])[0]) == [3, 4, 5]
    # cell_ticks = 3: cell m covers ticks 3m..3m+2, centre 3m+1, so exactly
    # cell m = 1 falls inside production cell 1
    out3 = upsample_support(base, c, 3.0, 0, 4, 3)
    assert list(np.nonzero(out3[0, 0])[0]) == [1]
