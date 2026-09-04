"""Tests for the zero-suppressed c-tick cell basis (``algs.zsbasis_algs``).

Everything here runs on the CPU with a synthetic kernel, so the exactness
claims are checked as algebra rather than against an archived number.
"""
import numpy as np
import pytest
import torch

from unfoldlarpix.algs import zsbasis_algs as Z
from unfoldlarpix.constrained_solver import LatchWindow, build_latch_rows
from unfoldlarpix.deconv_workflow import integrate_kernel_over_time
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.model.subbin_operator import ZSOperatorUniform

B = 30


def _hits():
    """One pixel, two trigger sequences; a second pixel, one sequence.

    Columns of ``location``: pixel_x, pixel_y, trigger, hold, next start.
    Columns of ``data``: x, y, z, then the cumulative burst charges.
    """
    loc = np.array([[5, 7, 100, 130, 156],
                    [5, 7, 170, 200, 226],
                    [6, 7, 110, 140, 166]], dtype=np.int64)
    dat = np.array([[0., 0., 0., 11.],
                    [0., 0., 0., 22.],
                    [0., 0., 0., 33.]], dtype=float)
    return loc, dat


def test_first_window_edge_is_the_convention():
    loc, dat = _hits()
    boff = np.array([0, 0, -60])
    w_edge, m_edge = build_latch_rows(loc, dat, B, boff, csa_reset_time=2,
                                      split_threshold=None, acq_start=None)
    w_t0, m_t0 = build_latch_rows(loc, dat, B, boff, csa_reset_time=2,
                                  split_threshold=None, acq_start=0)
    assert [w.value for w in w_edge] == [w.value for w in w_t0]
    # first window of each pixel
    assert w_edge[0].t_lo == -np.inf
    assert w_t0[0].t_lo == 0 - (-60)          # block-local ticks
    # the later window is (previous hold + csa_reset_time, hold] in both
    for w in (w_edge, w_t0):
        assert w[1].t_lo == pytest.approx(130 - (-60) + 2)
        assert w[1].t_hi == pytest.approx(170 - (-60) + B)
    # metadata: exactly one first window per pixel
    assert sum(not m.post_reset for m in m_edge) == 2
    assert [m.kind for m in m_edge] == ["lumped"] * 3


def test_window_upper_edges_are_not_multiples_of_the_fit_bin():
    """The exact-row condition: a ZS window edge is a general fine tick."""
    loc, dat = _hits()
    windows, _ = build_latch_rows(loc, dat, B, np.array([0, 0, -60]),
                                  csa_reset_time=2, split_threshold=None,
                                  acq_start=0)
    edges = [w.t_hi for w in windows] + [w.t_lo for w in windows]
    assert any(float(e) % B != 0 for e in edges)


def _impulse_kernel(kt=8):
    """A kernel that is a unit impulse at tick 0 on the centre pixel only.

    With it, ``conv`` is the identity on the time axis and a row is the plain
    sum of the charge over the window's ticks, so the sampling geometry can be
    read off the answer.
    """
    K = np.zeros((3, 3, kt))
    K[1, 1, 0] = 1.0
    return K


def test_fine_operator_row_is_the_sum_over_the_window_ticks():
    kt = 8
    K = _impulse_kernel(kt)
    nt_fine = 40
    windows = [LatchWindow(1, 1, 3.0, 9.0, 0.0),
               LatchWindow(1, 1, 12.0, 15.0, 0.0)]
    op = ZSOperator(K, (3, 3, nt_fine), windows, 1, device="cpu",
                    dtype=torch.float64)
    x = np.zeros(op.q_shape)
    x[1, 1, :] = np.arange(op.q_shape[2], dtype=float)
    y = op.forward(op.to_tensor(x)).numpy()
    # block bin t = fine tick t; the window (lo, hi] selects bins lo..hi-1
    assert y[0] == pytest.approx(x[1, 1, 3:9].sum())
    assert y[1] == pytest.approx(x[1, 1, 12:15].sum())


@pytest.mark.parametrize("c", [2, 5, 10])
def test_cell_operator_is_the_uniform_release_of_the_fine_operator(c):
    """``ZSOperatorUniform(K1, coarse, windows, c, c)`` samples at ONE fine
    tick and equals the fine operator applied to ``P_0 x``."""
    K = _impulse_kernel(10)
    nt_fine = 20 * c
    windows = [LatchWindow(1, 1, 3.0, 9.0, 0.0),
               LatchWindow(1, 1, 11.0, 4 * c + 7.0, 0.0)]
    fine = ZSOperator(K, (3, 3, nt_fine), windows, 1, device="cpu",
                      dtype=torch.float64)
    cell = ZSOperatorUniform(K, (3, 3, nt_fine // c), windows, c, c,
                             device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(1)
    xc = np.zeros(cell.q_shape)
    xc[1, 1, :] = rng.random(cell.q_shape[2])
    xf = np.zeros(fine.q_shape)
    spread = np.repeat(xc[1, 1, :], c) / c
    xf[1, 1, :len(spread)] = spread[:fine.q_shape[2]]
    yc = cell.forward(cell.to_tensor(xc)).numpy()
    yf = fine.forward(fine.to_tensor(xf)).numpy()
    assert yc == pytest.approx(yf, abs=1e-12)
    # and the sampling weights really are 1 per fine bin
    assert np.allclose(cell._weights.numpy(), 1.0)


def test_cell_operator_adjoint_is_the_adjoint():
    K = _impulse_kernel(10)
    c = 5
    windows = [LatchWindow(1, 1, 3.0, 9.0, 0.0),
               LatchWindow(1, 1, 11.0, 27.0, 0.0)]
    op = ZSOperatorUniform(K, (3, 3, 20), windows, c, c, device="cpu",
                           dtype=torch.float64)
    rng = np.random.default_rng(2)
    x = op.to_tensor(rng.random(op.q_shape))
    r = op.to_tensor(rng.random(op.n_data))
    lhs = float((op.forward(x) * r).sum())
    rhs = float((x * op.adjoint(r)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_integrate_kernel_over_one_tick_is_the_identity():
    rng = np.random.default_rng(3)
    K = rng.random((3, 3, 12))
    assert np.array_equal(integrate_kernel_over_time(K, 1), K)


def test_conventions_table_matches_build_latch_rows_arguments():
    assert Z.CONVENTIONS["acq_edge"] is None
    assert Z.CONVENTIONS["acq_t0"] == 0.0
    assert set(Z.CONV_LABEL) == set(Z.CONVENTIONS)
    assert "acquisition start" in Z.CONV_LABEL["acq_edge"]
    assert "event t0" in Z.CONV_LABEL["acq_t0"]


class _Harness:
    """Minimal stand-in exposing the two fields :func:`ring_sums` reads."""

    def __init__(self, chebyshev):
        self.chebyshev = np.asarray(chebyshev)


def test_ring_sums_group_by_chebyshev_distance():
    H = _Harness([0, 1, 2, 3, 4])
    x = np.array([[1.0, -0.5], [2.0, 0.0], [-1.0, 0.0],
                  [0.25, 0.0], [-0.25, 0.0]])
    out = Z.ring_sums(H, x)
    assert out["ionised"]["sum_ke"] == pytest.approx(0.5)
    assert out["plus1"]["sum_ke"] == pytest.approx(2.0)
    assert out["plus2"]["sum_ke"] == pytest.approx(-1.0)
    assert out["plus3_or_more"]["n_pixels"] == 2
    assert out["plus3_or_more"]["sum_positive_ke"] == pytest.approx(0.25)
    assert out["plus3_or_more"]["sum_negative_ke"] == pytest.approx(-0.25)


class _FakeJob:
    """``_BasisJob`` reduced to what the support lift and ``to_fine`` need."""

    B = 30

    def __init__(self, store):
        self.store = store

    support_on_basis = Z._BasisJob.support_on_basis
    to_fine = Z._BasisJob.to_fine


class _FakeStore:
    def __init__(self, support):
        self._s = support

    def get(self, key):
        assert key == "support"
        return self._s


class _FakeOp:
    def __init__(self, q_shape):
        self.q_shape = q_shape


def test_support_lift_maps_each_cell_to_its_coarse_bin():
    base = np.zeros((2, 2, 4), dtype=bool)
    base[:, :, 1] = True                      # coarse bin 1 = fine ticks 30..59
    job = _FakeJob(_FakeStore(base))
    for c, n in ((1, 120), (5, 24), (30, 4)):
        supp = job.support_on_basis(_FakeOp((2, 2, n)), c)
        assert supp.shape == (2, 2, n)
        on = np.nonzero(supp[0, 0])[0]
        # cells whose FIRST fine tick lies in [30, 60)
        want = np.array([m for m in range(n) if 30 <= m * c < 60])
        assert np.array_equal(on, want)


def test_to_fine_is_the_uniform_prolongation():
    job = _FakeJob(_FakeStore(None))
    x = np.zeros((2, 3, 4))
    x[1, 2, 3] = 6.0
    f = job.to_fine(x, 3)
    assert f.shape == (6, 12)
    row = f[1 * 3 + 2]
    assert row[9:12] == pytest.approx([2.0, 2.0, 2.0])
    assert row.sum() == pytest.approx(6.0)
    # c = 1 leaves the array alone
    assert np.array_equal(job.to_fine(x, 1), x.reshape(6, 4))


def test_trim_mask_excludes_n_pixels_at_each_end():
    pixel_y = np.arange(-2, 12)          # -2 .. 11
    m0 = Z.trim_mask(pixel_y, 0, 10, 0)
    assert pixel_y[m0].tolist() == list(range(0, 11))
    m3 = Z.trim_mask(pixel_y, 0, 10, 3)
    assert pixel_y[m3].tolist() == list(range(3, 8))
    # a trim that meets in the middle keeps nothing
    assert not Z.trim_mask(pixel_y, 0, 10, 6).any()


def test_trim_ratio_applies_the_same_trim_to_both_sides():
    created = np.zeros((2, 5))
    created[0, 1:4] = [10.0, 20.0, 30.0]
    reco = np.zeros((2, 5))
    reco[0, 1:4] = [5.0, 20.0, 60.0]
    reco[1, 0] = 100.0                   # outside the kept columns at n = 1
    pixel_y = np.arange(5)
    m0 = Z.trim_mask(pixel_y, 1, 3, 0)
    assert Z.trim_ratio(reco, created, m0) == pytest.approx(85.0 / 60.0)
    m1 = Z.trim_mask(pixel_y, 1, 3, 1)
    assert Z.trim_ratio(reco, created, m1) == pytest.approx(20.0 / 20.0)
    assert np.isnan(Z.trim_ratio(reco, created, np.zeros(5, bool)))


def _arms_doc(total, sum_y, arms):
    return {"result": {"truth_total_ke": total,
                       "data": {"acq_edge_c5": {"sum_y_ke": sum_y}},
                       "arms": [{"convention": "acq_edge", "cell_ticks": 5,
                                 "arm": k, "sum_xhat_ke": v,
                                 "wall_time_s": 1.0}
                                for k, v in arms.items()]}}


def _sample_doc(total, recorded):
    return {"result": {"n_records": 10, "n_pixels_with_records": 5,
                       "n_ionised_pixels": 4, "sum_effq_ke": total,
                       "sum_recorded_ke": recorded,
                       "recorded_over_created": recorded / total,
                       "by_distance": {
                           "ionised": {"n_pixels": 4, "recorded_ke": recorded,
                                       "created_ke": total},
                           "plus1": {"n_pixels": 1, "recorded_ke": 5.0}}}}


def test_ladder_fit_recovers_a_known_decay_rate(tmp_path):
    import json

    from unfoldlarpix.fwk.store import EventStore

    v = 0.159645
    inputs = []
    for tau, lam in ((1.0, 1.0), (20.0, 0.05)):
        for d in (4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5):
            t = d / v * 1e-3
            q = 4000.0 * np.exp(-lam * t)
            aj = tmp_path / f"arms_{d}_{tau}.json"
            sj = tmp_path / f"sample_{d}_{tau}.json"
            # every estimate is a fixed FRACTION of the truth, so every fitted
            # lambda must be the truth's lambda
            aj.write_text(json.dumps(_arms_doc(
                q, 1.02 * q, {"ls": 0.76 * q, "pos_a0": 0.93 * q,
                              "pos_l1": 0.92 * q})))
            sj.write_text(json.dumps(_sample_doc(q, 1.02 * q)))
            inputs.append({"depth_cm": d, "tau_ms": tau,
                           "arms_json": str(aj), "sample_json": str(sj)})
    alg = Z.ZSLadderFit(inputs=inputs, d_min_cm=[4.5, 16.5],
                        velocity_cm_per_us=v)
    alg.initialize({})
    store = EventStore()
    alg.execute(store)
    R = store.get("zs.ladder")
    for key, lam in (("1ms", 1.0), ("20ms", 0.05)):
        for name in ("sum_effq", "sum_y", "ls", "pos_a0", "pos_l1"):
            f = R["fits"][key][name]
            for dm in ("4.5", "16.5"):
                assert f["by_d_min"][dm]["lambda_per_ms"] == pytest.approx(
                    lam, abs=1e-9)
                assert f["by_d_min"][dm]["rms_resid_lnE"] < 1e-9
        # a pure fraction leaves the ratio flat with depth
        assert R["ratios"][key]["pos_l1"] == pytest.approx([0.92] * 9)
    d = R["lambda_difference_20ms_minus_1ms"]
    assert d["expected_per_ms"] == -0.95
    for name, per_dm in d["by_estimate"].items():
        for dm, e in per_dm.items():
            assert e["lambda_difference_per_ms"] == pytest.approx(-0.95,
                                                                  abs=1e-9)


def test_censor_violation_reads_the_terms_own_statistic():
    """``censor_violation`` must report max(0, C - threshold) over the armed
    bins, in ke, without the penalty's beta or norm."""
    from unfoldlarpix.terms.censor import CensorRunningMax

    K = _impulse_kernel(4)
    windows = [LatchWindow(1, 1, 0.0, 4.0, 0.0)]
    op = ZSOperator(K, (3, 3, 12), windows, 1, device="cpu",
                    dtype=torch.float64)
    nx, ny, nt = op.block_shape
    reset = np.zeros((nx, ny))
    arm = np.zeros((nx, ny))
    term = CensorRunningMax(op, reset, arm, censor_end=nt, threshold=2.0,
                            beta=3.0, norm="l2")
    x = np.zeros(op.q_shape)
    x[1, 1, :3] = 1.0                    # cumulative reaches 3 on pixel (1,1)
    out = Z.censor_violation(term, op, x)
    assert out["threshold_ke"] == pytest.approx(2.0)
    assert out["max_violation_ke"] == pytest.approx(1.0)
    assert out["n_violating_pixels"] == 1
    assert out["sum_violation_ke"] == pytest.approx(1.0)
    assert out["n_armed_pixels"] == nx * ny
    # below the threshold: no violation, and beta plays no part
    x2 = np.zeros(op.q_shape)
    x2[1, 1, 0] = 1.5
    assert Z.censor_violation(term, op, x2)["max_violation_ke"] == 0.0


def test_build_zs_operator_is_uncached_and_matches_the_cached_one():
    """A study that builds several row sets for one convention needs a fresh
    operator; ``_BasisJob.operator`` caches by (c, convention)."""

    class _J:
        pass

    J = _J()
    J.K1 = _impulse_kernel(6)
    J.nx, J.ny, J.nt_fine = 3, 3, 24

    class _C:
        device = "cpu"
        dtype = torch.float64

    J.comp = _C()
    w1 = [LatchWindow(1, 1, 0.0, 6.0, 1.0)]
    w2 = [LatchWindow(1, 1, 0.0, 6.0, 1.0), LatchWindow(1, 1, 7.0, 12.0, 2.0)]
    a = Z.build_zs_operator(J, 3, w1)
    b = Z.build_zs_operator(J, 3, w2)
    assert a.n_data == 1 and b.n_data == 2          # not the same object
    assert a.q_shape == b.q_shape
    fine = Z.build_zs_operator(J, 1, w1)
    assert fine.q_shape[2] == J.nt_fine - 6 + 1
    with pytest.raises(ValueError):
        Z.build_zs_operator(J, 5, w1)               # 24 is not a multiple of 5
