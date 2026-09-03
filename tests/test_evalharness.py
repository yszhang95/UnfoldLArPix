"""Tests for :mod:`unfoldlarpix.algs.evalharness_algs`.

Four groups, all on small synthetic grids and all CPU-only:

* the prolongations satisfy ``R P = I`` and ``1^T P = 1^T`` for the delta, the
  uniform and the corrected hat, at BOTH release offsets that occur in the
  campaign (0 for the shipped delta kernel, 14.5 for
  ``within_bin: uniform, subbin: 30``);
* ``H`` has unit sum and does not wrap;
* the fine passthrough control returns exactly zero;
* ``R x`` agrees with ``grid_truth(mode="round")`` -- with the one documented
  exception, a fine tick landing exactly on ``c_k - B/2``, where ``np.rint``
  breaks the tie to the nearest even cell index and the lower-closed cell
  ``C_k`` does not;
* the ``row_meta`` -> window mapping reproduces a synthetic ``op.d``;
* the three algorithms are registered with the declared reads/writes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unfoldlarpix.algs.evalharness_algs import (
    EvalHarness, LinearArms, ResolutionProbe, ResolutionScore, coarse_index,
    cell_fine_ticks, hat_column_sums, prolongation, records_from_impulse,
    restriction_of_hat, row_lookup, smooth_columns, time_kernel)
from unfoldlarpix.algs.fixedgrid_algs import grid_truth
from unfoldlarpix.fwk.component import ALGORITHMS
from unfoldlarpix.fwk.store import EventStore

B = 30.0
N_COARSE = 24
T0 = -300.0
OFFSETS = (0.0, 14.5)          # delta kernel; within_bin uniform subbin 30


def _grid(off):
    c = T0 + np.arange(N_COARSE) * B + off
    fine = np.arange(int(cell_fine_ticks(c[0], B)[0]),
                     int(cell_fine_ticks(c[-1], B)[-1]) + 1)
    return c, fine


# ---------------------------------------------------------------------------
# the store contract
# ---------------------------------------------------------------------------
def test_registered_and_reads():
    assert ALGORITHMS["LinearArms"] is LinearArms
    assert ALGORITHMS["ResolutionScore"] is ResolutionScore
    assert ALGORITHMS["ResolutionProbe"] is ResolutionProbe
    assert LinearArms.writes == ("arms.q",)
    assert ResolutionScore.reads[0] == "arms.q"
    assert ResolutionScore.writes == ("eval.score",)
    assert ResolutionProbe.writes == ("eval.resolution",)
    for key in ("op", "row_meta", "charge_model"):
        assert key in ResolutionProbe.reads


# ---------------------------------------------------------------------------
# the coarsening R
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("off", OFFSETS)
def test_cells_tile_the_fine_grid(off):
    """Every fine tick belongs to exactly one cell and every cell holds B."""
    c, fine = _grid(off)
    kk = coarse_index(fine, c[0], B)
    assert kk.min() == 0 and kk.max() == N_COARSE - 1
    assert np.all(np.bincount(kk) == int(B))


# ---------------------------------------------------------------------------
# the prolongations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("off", OFFSETS)
@pytest.mark.parametrize("name", ["delta", "uniform", "corrected_hat"])
def test_RP_is_identity_and_columns_sum_to_one(off, name):
    c, fine = _grid(off)
    Tinv = np.linalg.inv(restriction_of_hat(c, B))
    P = prolongation(name, fine, c, B, Tinv)
    kk = coarse_index(fine, c[0], B)
    RP = np.zeros((N_COARSE, N_COARSE))
    np.add.at(RP, kk, P)
    assert np.abs(RP - np.eye(N_COARSE)).max() < 1e-10
    assert np.abs(P.sum(axis=0) - 1.0).max() < 1e-10


@pytest.mark.parametrize("off", OFFSETS)
def test_R_Phat_is_tridiagonal(off):
    """``R P_hat`` is tridiagonal; the entries are MEASURED, not assumed.

    On the continuum the off-diagonals are 1/8.  On the integer tick grid the
    lower-closed cell ``[c_k - B/2, c_k + B/2)`` is symmetric about ``c_k``
    only when ``c_k`` is a half-integer, i.e. for the uniform charge model
    (release offset 14.5).  For the delta model (offset 0) the two
    off-diagonals are 7/60 and 2/15 instead.
    """
    c = T0 + np.arange(N_COARSE) * B + off
    T = restriction_of_hat(c, B)
    band = (np.diag(np.diag(T)) + np.diag(np.diag(T, 1), 1)
            + np.diag(np.diag(T, -1), -1))
    assert np.abs(T - band).max() < 1e-12
    k = N_COARSE // 2
    assert T[k, k] == pytest.approx(0.75)
    if off == 14.5:
        assert T[k - 1, k] == pytest.approx(0.125)
        assert T[k + 1, k] == pytest.approx(0.125)
    else:
        assert T[k - 1, k] == pytest.approx(7.0 / 60.0)
        assert T[k + 1, k] == pytest.approx(2.0 / 15.0)
    assert T[:, k].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("off", OFFSETS)
def test_corrected_hat_row_sums(off):
    """``1^T P_1 = 1^T`` holds at the grid boundary too.

    ``P_hat``'s boundary column loses the mass that falls off the end of the
    fine grid, and ``R P_hat``'s boundary column loses exactly the same mass,
    so ``1^T_coarse (R P_hat) = 1^T_fine P_hat`` and the identity survives.
    """
    c = T0 + np.arange(N_COARSE) * B + off
    Tinv = np.linalg.inv(restriction_of_hat(c, B))
    assert np.abs(hat_column_sums(c, B) @ Tinv - 1.0).max() < 1e-10


@pytest.mark.parametrize("off", OFFSETS)
def test_P1_is_local(off):
    """The corrected hat is local: measured absolute mass per cell.

    SIGNED mass off the own cell is zero -- that is ``R P_1 = I`` again -- so
    locality has to be read off the ABSOLUTE mass
    ``a_m = sum_{j in C_{k+m}} |P_1[j, k]|``.  Measured on this grid:
    ``1, 0.207, 0.0355, 0.0061, 0.0010`` at offset 14.5, i.e. a factor
    ``3 - 2*sqrt(2) = 0.1716`` per cell, the decay of the inverse of the
    tridiagonal ``(1/8, 3/4, 1/8)``.
    """
    c, fine = _grid(off)
    Tinv = np.linalg.inv(restriction_of_hat(c, B))
    P = prolongation("corrected_hat", fine, c, B, Tinv)
    kk = coarse_index(fine, c[0], B)
    k = N_COARSE // 2
    absm = np.array([np.abs(P[kk == k + m, k]).sum() for m in range(-4, 5)])
    sgn = np.array([P[kk == k + m, k].sum() for m in range(-4, 5)])
    assert np.abs(sgn - (np.arange(-4, 5) == 0)).max() < 1e-12
    assert absm[4] == pytest.approx(1.0, abs=1e-12)
    assert absm[3] < 0.25 and absm[5] < 0.25
    assert absm[2] < 0.05 and absm[6] < 0.05
    assert absm[1] < 0.01 and absm[7] < 0.01
    assert absm[0] < 0.002 and absm[8] < 0.002
    if off == 14.5:
        assert absm[5] == pytest.approx(0.206822, abs=1e-5)
        assert absm[6] / absm[5] == pytest.approx(3 - 2 * np.sqrt(2),
                                                  abs=1e-4)


# ---------------------------------------------------------------------------
# H
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sig", [0.0, 3.0, 10.0, 30.0])
def test_time_kernel_unit_sum(sig):
    g = time_kernel(sig)
    assert g.sum() == pytest.approx(1.0)
    assert len(g) % 2 == 1
    assert np.argmax(g) == (len(g) - 1) // 2


def test_smoothing_does_not_wrap():
    """A delta at the first row must not put mass at the last row."""
    g = time_kernel(4.0)
    m = np.zeros((80, 1))
    m[0, 0] = 1.0
    s = smooth_columns(m, g)
    assert np.abs(s[40:]).max() < 1e-15
    # and a delta in the middle keeps the whole unit mass
    m2 = np.zeros((80, 1))
    m2[40, 0] = 1.0
    assert smooth_columns(m2, g).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the harness on a synthetic store
# ---------------------------------------------------------------------------
def _store_and_op(off, ticks, charges, pads):
    """Minimal store carrying only what :class:`EvalHarness` reads."""
    nx, ny = 5, 5
    op = SimpleNamespace(q_shape=(nx, ny, N_COARSE))
    loc = np.array([[p[0], p[1], t] for (p, t) in zip(pads, ticks)],
                   dtype=np.int64)
    dat = np.array([[0.0, 0.0, 0.0, q] for q in charges])
    ev = SimpleNamespace(effq=SimpleNamespace(location=loc, data=dat))
    store = EventStore()
    store.put("job.config", {"_meta": {"git": "test"}}, by="test")
    store.put("event", ev, by="test")
    store.put("readout_config",
              SimpleNamespace(adc_hold_delay=int(B)), by="test")
    store.put("block_offset", np.array([10.0, 20.0, T0]), by="test")
    if off:
        store.put("charge_model", {"release_offset_ticks": off}, by="test")
    return store, op


def _harness(off, ticks, charges, pads, margin=8):
    """``margin`` cells each side.  At the default the nearest window edge is
    8 * B = 240 fine ticks from the truth, i.e. 8 sigma at the widest
    ``sigma_H`` used in these tests, so the smoothing loses no mass off the
    end of the window and ``sum H x = sum x`` exactly."""
    store, op = _store_and_op(off, ticks, charges, pads)
    H = EvalHarness(store, op, margin_windows=margin,
                    line_pixel_y_range=(-10 ** 6, 10 ** 6),
                    segment_pixels=1, segment_edge_exclude=0)
    return store, op, H


@pytest.mark.parametrize("off", OFFSETS)
def test_Rx_matches_grid_truth_round_away_from_the_tie(off):
    """``R x`` is ``grid_truth(round)`` on every tick except the exact tie."""
    c = T0 + np.arange(N_COARSE) * B + off
    tie = {int(np.floor(v - B / 2.0)) for v in c
           if float(v - B / 2.0).is_integer()}
    ticks = [t for t in range(int(c[10]) - 40, int(c[14]) + 40)
             if t not in tie]
    charges = [1.0 + 0.01 * i for i in range(len(ticks))]
    pads = [(12, 22)] * len(ticks)
    store, op, H = _harness(off, ticks, charges, pads)
    qg = grid_truth(store, op, mode="round")
    assert np.array_equal(H.Rx, qg)


def test_Rx_differs_from_grid_truth_only_on_the_tie():
    """At ``j = c_k - B/2`` exactly, ``np.rint`` goes to the even cell.

    This is the one documented disagreement between ``R`` as defined by the
    lower-closed cell and ``grid_truth(mode="round")``; it exists only when
    ``c_k - B/2`` is an integer, i.e. for the delta charge model with even
    ``B``, and it moves the charge by one cell.
    """
    c = T0 + np.arange(N_COARSE) * B          # offset 0
    k = 13
    j = int(c[k] - B / 2.0)
    store, op, H = _harness(0.0, [j], [5.0], [(12, 22)])
    qg = grid_truth(store, op, mode="round")
    assert H.Rx[2, 2, k] == pytest.approx(5.0)          # lower-closed cell
    assert qg[2, 2, k] == 0.0
    kg = int(np.rint((j - c[0]) / B))
    assert kg == k - 1 and kg % 2 == 0                  # np.rint: ties to even
    assert qg[2, 2, kg] == pytest.approx(5.0)
    assert np.abs(H.Rx - qg).sum() == pytest.approx(10.0)


@pytest.mark.parametrize("off", OFFSETS)
@pytest.mark.parametrize("sig", [0.0, 0.5, 1.5])
def test_fine_passthrough_control_is_exactly_zero(off, sig):
    c = T0 + np.arange(N_COARSE) * B + off
    ticks = list(range(int(c[12]) - 10, int(c[12]) + 10))
    charges = [2.0] * len(ticks)
    pads = [(12, 22)] * len(ticks)
    _, _, H = _harness(off, ticks, charges, pads)
    cp = H.control_passthrough(sig)
    assert cp["E_rel"] < 1e-12
    assert cp["sum_Hx_over_sum_x"] == pytest.approx(1.0, abs=1e-12)


def test_measure_representation_term_vanishes_for_a_fine_grid_truth():
    """``e = H P (xbar - R x) + H (P R - I) x``: the pieces add up.

    With ``xbar = R x`` the estimation term is identically zero, so the full
    error equals the representation term term-by-term; the signed sums are
    checked because they must agree exactly, not approximately.
    """
    c = T0 + np.arange(N_COARSE) * B + 14.5
    ticks = list(range(int(c[12] - 5), int(c[12] + 6)))
    _, _, H = _harness(14.5, ticks, [3.0] * len(ticks),
                       [(12, 22)] * len(ticks))
    full = H.measure(H.Rx, "corrected_hat", 1.5)
    est = H.measure(H.Rx - H.Rx, "corrected_hat", 1.5, with_truth=False)
    assert est["sum_abs_e_ke"] == 0.0
    # H P conserves charge exactly; what is left is the |P_1| tail beyond the
    # 8-cell window edge, measured at 7e-9 of the total on this grid.
    assert abs(full["conservation_rel"]) < 1e-7
    # and with a NON-trivial candidate the signed sums of the two terms add up
    rng = np.random.default_rng(3)
    xbar = H.Rx + rng.normal(size=H.Rx.shape)
    fu = H.measure(xbar, "corrected_hat", 1.5)
    es = H.measure(xbar - H.Rx, "corrected_hat", 1.5, with_truth=False)
    assert fu["sum_e_ke"] == pytest.approx(
        es["sum_e_ke"] + full["sum_e_ke"], rel=1e-10, abs=1e-9)


def test_measure_is_linear_in_xbar():
    """The metric's signed sums are linear, which is what makes the
    decomposition of the score exact rather than approximate."""
    c = T0 + np.arange(N_COARSE) * B
    ticks = list(range(int(c[12] - 3), int(c[12] + 4)))
    _, op, H = _harness(0.0, ticks, [1.0] * len(ticks),
                        [(12, 22)] * len(ticks))
    rng = np.random.default_rng(0)
    a = rng.normal(size=op.q_shape)
    b = rng.normal(size=op.q_shape)
    for pn in ("delta", "uniform", "corrected_hat"):
        sa = H.measure(a, pn, 1.0, with_truth=False)["sum_e_ke"]
        sb = H.measure(b, pn, 1.0, with_truth=False)["sum_e_ke"]
        sab = H.measure(a + b, pn, 1.0, with_truth=False)["sum_e_ke"]
        assert sab == pytest.approx(sa + sb, rel=1e-10, abs=1e-10)


# ---------------------------------------------------------------------------
# the row_meta -> window mapping
# ---------------------------------------------------------------------------
def test_row_lookup_reproduces_a_synthetic_operator_data_vector():
    """The mapping used by the probe must rebuild ``op.d`` exactly.

    A synthetic ``row_meta`` in the block frame, with the same
    ``(px, py, t_hi)`` key the probe uses, is filled from an independently
    built window array; the reconstructed vector must equal the data vector
    entry for entry.
    """
    Bi = int(B)
    trig, nwin = -1800, 5
    pads = [(0, 0), (0, 1), (2, 3)]
    t0 = -7050.0
    rows = {"px": [], "py": [], "t_lo": [], "t_hi": [],
            "kind": [], "post_reset": []}
    y = {}
    rng = np.random.default_rng(1)
    for (px, py) in pads:
        for k in range(1, nwin + 1):
            rows["px"].append(px)
            rows["py"].append(py)
            rows["t_lo"].append(float(trig + (k - 1) * Bi - t0))
            rows["t_hi"].append(float(trig + k * Bi - t0))
            rows["kind"].append("lumped" if k == 1 else "diff")
            rows["post_reset"].append(False)
            y[(px, py, k)] = float(rng.normal())
    d = np.array([y[(rows["px"][r], rows["py"][r],
                     int((rows["t_hi"][r] + t0 - trig) / Bi))]
                  for r in range(len(rows["px"]))])
    store = EventStore()
    store.put("row_meta", rows, by="test")
    op = SimpleNamespace(n_data=len(d))
    look = row_lookup(store, op)
    assert len(look) == len(d)
    rebuilt = np.zeros(len(d))
    for (px, py) in pads:
        for k in range(1, nwin + 1):
            rebuilt[look[(px, py, int(trig + k * Bi - t0))]] = y[(px, py, k)]
    assert np.array_equal(rebuilt, d)


def test_records_from_impulse_matches_an_explicit_accumulator():
    """``y[d, k]`` differences the cumulative kernel at the latch times.

    Checked against an explicit ``cumsum`` of a synthetic current, which is
    what ``tred.readout.fixed_interval_readout`` does, including the
    ``current_zero_before_tick`` deletion.
    """
    nt = 400
    rng = np.random.default_rng(2)
    cur = rng.normal(size=(25, 25, nt)) * 1e-3
    cur[12, 12] += np.exp(-((np.arange(nt) - 300.0) / 20.0) ** 2)
    kcum = np.cumsum(cur, axis=-1)
    offsets = np.array([[0, 0], [1, 0], [-1, 2]])
    latch = np.arange(-60, 400, 30)
    t_star, Q, tz = -40, 7.0, 0
    y = records_from_impulse(kcum, offsets, latch, t_star, Q, tz)
    for i, (dx, dy) in enumerate(offsets):
        acc = np.zeros(600)
        acc[t_star + 200: t_star + 200 + nt] = Q * kcum[12 + dx, 12 + dy]
        acc[t_star + 200 + nt:] = Q * kcum[12 + dx, 12 + dy, -1]
        ped = acc[tz - 1 + 200]
        a = np.where(latch < tz, 0.0, acc[latch + 200] - ped)
        assert np.allclose(y[i], np.diff(a), atol=1e-12)
