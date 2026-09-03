"""Tests for the t_0-truncated kernel and the depth-ladder lifetime fit.

CPU only, float64, small synthetic grids.  Nothing here loads the response
file or touches the GPU.  Every test states the identity it checks.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from unfoldlarpix.algs.depthladder_algs import (arrival_first_above,
                                                arrival_shape_fit,
                                                charge_weighted_median,
                                                kernel_offset_rings, ls_line,
                                                ring_masks)
from unfoldlarpix.algs.finebasis_algs import (FineOperator,
                                              cell_window_kernel,
                                              direct_sum_records,
                                              fine_window_kernel,
                                              truncated_response)
from unfoldlarpix.fwk.component import ALGORITHMS

B = 30                      # record stride, fine ticks
C = 5                       # the working cell basis
NX, NY, M = 5, 4, 8         # block: pads and record windows
KR = 1                      # kernel half-width in pads (3 x 3)
NTAU = 91                   # response length in fine ticks
ARRIVAL = 60                # where the toy response collects


def toy_response(seed: int = 7) -> np.ndarray:
    """A (3, 3, NTAU) impact-averaged response built from a weighting potential.

    Constructed so that it has the three properties of the real one that the
    truncation argument uses:

    * the PAD-SUMMED cumulative ``S(tau) = sum_d Kcum_d(tau)`` is monotone
      increasing from 0 to 1 (it is ``sum_k W_k`` along the drift path), hence
      the pad-summed current is non-negative and
      ``sum_{tau >= cut} sum_d Kbar_d = 1 - S(cut - 1)`` decreases with the cut;
    * every neighbour offset integrates to EXACTLY zero (its weighting
      potential returns to zero once the charge is collected);
    * the own pad integrates to exactly one, with a leading induced part and a
      sharp collection step at ``ARRIVAL``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(NTAU, dtype=float)
    S = 0.25 * np.clip(t / ARRIVAL, 0, 1) ** 2 \
        + 0.75 / (1.0 + np.exp(-(t - ARRIVAL) / 1.5))
    S = S - S[0]
    S = S / S[-1]                                   # monotone, S[0]=0, S[-1]=1
    w0 = np.exp(-((t - 38.0) / 13.0) ** 2)
    w = w0 - (w0[0] + (w0[-1] - w0[0]) * t / (NTAU - 1))   # w[0] = w[-1] = 0
    W = np.zeros((2 * KR + 1, 2 * KR + 1, NTAU))
    for i in range(3):
        for j in range(3):
            if i == KR and j == KR:
                continue
            W[i, j] = rng.uniform(0.02, 0.08) * w
    W[KR, KR] = S - W.sum(axis=(0, 1))
    K = np.diff(W, axis=-1, prepend=0.0)
    return K


def make_op(cut=None, cell_model="uniform", dtype=torch.float64):
    return FineOperator(toy_response(), (NX, NY, M), B, device="cpu",
                        dtype=dtype, cell_ticks=C, cell_model=cell_model,
                        kernel_cut_tick=cut)


# ---------------------------------------------------------------------------
def test_registration():
    for name in ("DepthLadderEvent", "DepthLadderFit", "DepthLadderFigures"):
        assert name in ALGORITHMS


def test_truncated_response_is_its_definition():
    """``Kbar . [tau >= cut]``; ``None`` and ``<= 0`` return the array itself."""
    R = toy_response()
    assert truncated_response(R, None) is R
    assert truncated_response(R, 0) is R
    for cut in (1, 17, NTAU - 1):
        T = truncated_response(R, cut)
        assert np.all(T[..., :cut] == 0.0)
        assert np.array_equal(T[..., cut:], R[..., cut:])
    with pytest.raises(ValueError):
        truncated_response(R, NTAU)


def test_default_is_bit_identical_to_the_untruncated_operator():
    """``kernel_cut_tick = None`` changes no array and no solve, bit for bit."""
    a = FineOperator(toy_response(), (NX, NY, M), B, device="cpu",
                     cell_ticks=C, cell_model="uniform")
    b = make_op(cut=None)
    assert a.kernel_cut_tick is None
    assert np.array_equal(a.h_np, b.h_np)
    assert np.array_equal(a.g_np, b.g_np)
    assert torch.equal(a.Hr, b.Hr)
    assert torch.equal(a.G, b.G)
    # and the same for the kernel builders used stand-alone
    h = fine_window_kernel(toy_response(), B)
    assert np.array_equal(a.h_np, h)
    assert np.array_equal(a.g_np, cell_window_kernel(h, C, "uniform"))
    g = torch.Generator(device="cpu").manual_seed(4)
    y = torch.randn((a.nxp, a.nyp, M), generator=g, dtype=torch.float64)
    assert torch.equal(a.solve(y, 1e-6 * a.G_max),
                       b.solve(y, 1e-6 * b.G_max))


def test_truncated_forward_is_the_tred_truncated_exact_functional():
    """``A_c`` with ``kernel_cut_tick`` = the tred ``current_zero_before_tick``
    functional, for a single fine tick of charge.

    tred (``graph_effq.py:148-159``) zeroes the current before ``t_z``, so the
    accumulator is ``A(t) = 0`` for ``t < t_z`` and ``A(t) = q [Kcum(t - j) -
    Kcum(t_z - 1 - j)]`` for ``t >= t_z``, and the record is
    ``y[w] = A(l_w) - A(l_{w-1})`` -- exactly what
    :class:`~unfoldlarpix.algs.exactrows_algs.FineTruthClosure` builds.  With
    all the charge at ONE fine tick ``j``, ``tau_cut = t_z - j`` is exact and
    the truncated operator must reproduce it to the float floor.
    """
    R = toy_response()
    Kc = np.cumsum(R, axis=-1)                       # Kcum, dt already in R
    b = 0
    m0 = 6                                           # a cell inside the grid
    j = b + C * m0                                   # the cell's lower edge
    q = 3.7
    for t_z in (0, 40, 100):
        cut = t_z - j
        if cut <= 0:
            continue
        F = make_op(cut=cut, cell_model="delta")
        x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
        x[2, 2, m0] = q
        y_op = F.forward(x).numpy()[:NX, :NY]

        def kcum(tau):
            tau = np.asarray(tau)
            return np.where(tau < 0, 0.0, Kc[..., np.clip(tau, 0, NTAU - 1)])

        latch = b + B * np.arange(M + 1)             # l_0 .. l_M
        ped = kcum(t_z - 1 - j)                      # (3, 3)
        A = kcum(latch - j) - ped[..., None]         # (3, 3, M+1)
        A[..., latch < t_z] = 0.0
        yhat = q * np.diff(A, axis=-1)               # (3, 3, M)
        y_exact = np.zeros((NX, NY, M))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                y_exact[2 + dx, 2 + dy] = yhat[KR + dx, KR + dy]
        assert np.abs(y_op - y_exact).max() < 1e-12 * max(
            float(np.abs(y_exact).max()), 1.0)


def test_truncated_c5_forward_matches_the_direct_sum_on_uniform_cells():
    """The ``c = 5`` truncated forward against the exact-functional direct sum.

    With ``cell_model = uniform`` the cell's fine representative is ``1/c`` on
    each of its ``c`` ticks; the direct sum evaluates
    ``y_p[w] = sum_j x(j) h_trunc(b + B(w+1) - j)`` term by term with no FFT
    and no periodicity.
    """
    cut = 23
    F = make_op(cut=cut)
    rng = np.random.default_rng(1)
    Rx = np.zeros((NX, NY, F.N))
    Rx[2, 2, 5:9] = rng.uniform(1.0, 4.0, 4)
    Rx[1, 2, 6] = 2.0
    x = torch.zeros((F.nxp, F.nyp, F.N), dtype=torch.float64)
    x[:NX, :NY] = torch.as_tensor(Rx)
    y_op = F.forward(x).numpy()[:NX, :NY]
    ix, iy, mm = np.nonzero(Rx)
    IX = np.repeat(ix, C)
    IY = np.repeat(iy, C)
    TT = C * np.repeat(mm, C) + np.tile(np.arange(C), len(mm))
    QQ = np.repeat(Rx[ix, iy, mm], C) / C
    y_dir = direct_sum_records(F.h_np, IX, IY, TT, QQ, 0, B, NX, NY, M,
                               "cpu", torch.float64)
    assert np.abs(y_op - y_dir).sum() / np.abs(y_dir).sum() < 1e-13


def test_sum_h_is_monotone_and_bounded_by_the_full_kernel():
    """``sum h_trunc <= sum h_full``, decreasing in ``tau_cut``, and
    ``sum_s g[s] = D sum K_trunc`` at every cut."""
    R = toy_response()
    full = fine_window_kernel(R, B).sum()
    prev = full
    for cut in (0, 5, 20, 40, 60, 80):
        F = make_op(cut=(None if cut == 0 else cut))
        s = F.h_np.sum()
        assert s <= full + 1e-12
        assert s <= prev + 1e-12
        prev = s
        assert abs(s / B - truncated_response(R, cut).sum()) < 1e-12
        assert abs(F.g_np.sum() / F.D
                   - truncated_response(R, cut).sum()) < 1e-12


def test_conservation_identity_uses_the_truncated_kernel_gain():
    """``sum xhat = sum y / sum K_trunc`` at ``lambda -> 0``."""
    cut = 30
    F = make_op(cut=cut)
    g = torch.Generator(device="cpu").manual_seed(9)
    y = torch.randn((F.nxp, F.nyp, M), generator=g, dtype=torch.float64)
    lam = 1e-10 * F.G_max
    xh = F.solve(y, lam)
    sumK = truncated_response(toy_response(), cut).sum()
    pred = float(y.sum()) / sumK * (F.G_dc / (F.G_dc + lam))
    assert abs(float(xh.sum()) - pred) < 1e-9 * abs(pred)


def test_ls_line_recovers_a_known_lambda_exactly():
    """Synthetic exponential totals: the fit returns ``lambda`` and zero rms."""
    t = np.array([4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5]) \
        / 0.159645 * 1e-3
    for lam in (1.0, 0.05, 0.2142):
        E = 4212.0 * np.exp(-lam * t)
        f = ls_line(t, np.log(E))
        assert abs(f["lambda_per_ms"] - lam) < 1e-12
        assert f["rms_resid_lnE"] < 1e-12
        assert f["lambda_err"] < 1e-12
        assert abs(f["q0_ke"] - 4212.0) < 1e-8
    # a known scatter gives the textbook standard error
    E = 4212.0 * np.exp(-1.0 * t)
    y = np.log(E)
    y[0] += 0.01
    f = ls_line(t, y)
    n = len(t)
    tb = t.mean()
    A = np.vstack([t, np.ones_like(t)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ coef
    se = np.sqrt((r ** 2).sum() / (n - 2) / ((t - tb) ** 2).sum())
    assert abs(f["lambda_err"] - se) < 1e-12


def test_ring_masks_partition_and_kernel_rings_are_chebyshev():
    ch = np.array([0, 1, 2, 3, 4, 5, 6, 9])
    m = ring_masks(ch)
    assert sum(int(v.sum()) for v in m.values()) == len(ch)
    kr = kernel_offset_rings(25, 25)
    assert int(kr["0"].sum()) == 1
    assert int(kr["1"].sum()) == 8
    assert int(kr["2"].sum()) == 16
    assert sum(int(v.sum()) for v in kr.values()) == 625


def test_charge_weighted_median():
    assert charge_weighted_median(np.array([-3, -2, -1]),
                                  np.array([1.0, 1.0, 1.0])) == -2.0
    assert charge_weighted_median(np.array([-3, -2, -1]),
                                  np.array([0.1, 0.1, 10.0])) == -1.0


def test_arrival_estimators_on_a_synthetic_record():
    """Both observed estimators recover the plane tick of a synthetic event.

    A single isochronous charge crossing the plane at ``j`` with the tred
    truncation at ``t_0 = 0``; the records are built from the toy kernel's own
    cumulative, so the two estimators are being asked to invert exactly the
    forward model they assume.
    """
    R = toy_response()
    j = -37                                   # the plane-crossing tick
    # a fine record grid (Bt = 5, 40 windows) so the cumulative record is
    # sampled often enough for the one-parameter shape fit to be determined;
    # at Bt = 30 only two windows carry any shape and the minimum is flat
    b, Bt, Mt = -60, 5, 40
    latch = b + Bt * (np.arange(Mt) + 1)
    Kc = np.concatenate([[0.0],
                         np.cumsum(truncated_response(R, -j),
                                   axis=-1).sum(axis=(0, 1))])

    def cum(tau):
        return Kc[int(np.clip(tau + 1, 0, len(Kc) - 1))]

    edges = np.concatenate([[b], latch])
    A = np.array([cum(l - j) for l in edges])
    A[edges < 0] = 0.0
    Y = np.diff(A)
    fa = arrival_first_above(Y, b, Bt, 0.5)
    # the arrival window is the one containing the collection tick j + ARRIVAL
    assert fa["latch_tick"] - Bt < j + ARRIVAL <= fa["latch_tick"]
    assert abs(fa["t_arrival_tick"] - (j + ARRIVAL)) <= Bt / 2.0
    # the shape fit is exact: it inverts the model that built Y.  ARRIVAL_TICK
    # is the real kernel's transit, so the scan is centred on the toy's own.
    sf = arrival_shape_fit(Y, R, b, Bt,
                           fa["t_arrival_tick"] - ARRIVAL, 40)
    assert sf["t_plane_tick"] == j
    assert sf["tau_cut"] == -j
