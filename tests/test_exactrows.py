"""Tests for :mod:`unfoldlarpix.algs.exactrows_algs`.

Three groups:

* the impact-resolved kernel is what the module docstring says it is (charge
  integrals per ring, the relation to ``FieldResponseProcessor``, and the
  SIGN of the impact index against the pixel offset);
* the window weights and the impact-averaging error obey the two identities
  the plan relies on (``sum_s w = 1``, ``sum_i d = 0``, ``sum_s d = 0``);
* :class:`FineTruthClosure` returns exactly zero residual on a synthetic
  event whose records were built independently from the same response array
  -- this pins the latch-time bookkeeping and the pixel-offset sign.

The kernel tests load the real response file (local disk) and are marked
``slow``: expanding it takes ~10 s and ~1 GB.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unfoldlarpix.algs.exactrows_algs import (
    ARRIVAL_TICK, COLLECTION_PIXEL, FineTruthClosure, KernelStructure,
    cumulative, gaussian_cell_weights, kcum_at, load_impact_response,
    window_weights)
from unfoldlarpix.data_containers import ReadoutConfig
from unfoldlarpix.fwk.component import ALGORITHMS
from unfoldlarpix.fwk.store import EventStore
from unfoldlarpix.io.hits import HitsView

RESPONSE = ("/srv/storage1/yousen/tred_workspace/"
            "response_44_v2a_full_25x25pixel_tred.npz")
DT = 0.05
B = 30
C = COLLECTION_PIXEL

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(RESPONSE).exists(),
    reason="response file not on this machine")


@pytest.fixture(scope="module")
def resp():
    R, meta = load_impact_response(RESPONSE)
    return R, meta


@pytest.fixture(scope="module")
def kcum_own(resp):
    R, _ = resp
    return cumulative(R[C, :, C, :, :], DT)          # (10, 10, Nt)


@pytest.fixture(scope="module")
def kcum_bar(resp):
    R, _ = resp
    return np.cumsum(R.mean(axis=(1, 3), dtype=np.float64), axis=-1) * DT


# ---------------------------------------------------------------------------
# the store contract
# ---------------------------------------------------------------------------
def test_registered_and_reads():
    assert ALGORITHMS["KernelStructure"] is KernelStructure
    assert ALGORITHMS["FineTruthClosure"] is FineTruthClosure
    assert KernelStructure.reads == ()
    assert FineTruthClosure.reads == ("event", "readout_config", "hits_view")


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_shape_and_metadata(resp):
    R, meta = resp
    assert R.shape == (25, 10, 25, 10, 3900)
    assert meta["time_tick_us"] == pytest.approx(DT)
    assert meta["npath"] == 10


@pytest.mark.slow
def test_collection_integral_is_one_for_every_impact(resp):
    R, _ = resp
    integ = R[C, :, C, :, :].sum(axis=-1, dtype=np.float64) * DT
    assert integ.shape == (10, 10)
    assert np.abs(integ - 1.0).max() < 1e-5


@pytest.mark.slow
def test_ring1_integral_vanishes_for_every_impact(resp):
    R, _ = resp
    worst = 0.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            integ = R[C + dx, :, C + dy, :, :].sum(axis=-1,
                                                   dtype=np.float64) * DT
            worst = max(worst, float(np.abs(integ).max()))
    assert worst < 1e-6


@pytest.mark.slow
def test_impact_mean_equals_process_response(resp, kcum_bar):
    """``ndlarsim`` and ``_flip_kernel_for_convolution`` apply the SAME flip.

    Both split the expanded plane into (pixel, impact) pairs and flip the
    PIXEL axes only, so the impact mean of the ndlarsim array equals
    ``process_response()`` elementwise with no further transformation.  The
    check is elementwise on the full (25, 25, 3900) array, and it is a real
    check of the flip only because it is done BEFORE any impact averaging
    would symmetrise it -- the impact-averaged array happens to be invariant
    under the flip, so the flip cannot be detected from it (asserted below).
    """
    from unfoldlarpix.field_response import FieldResponseProcessor
    R, _ = resp
    mean = R.mean(axis=(1, 3), dtype=np.float64) * DT
    proc = FieldResponseProcessor(RESPONSE).process_response()
    assert proc.shape == mean.shape
    assert np.allclose(mean, proc, atol=1e-6, rtol=0)
    # ... and the impact-averaged array is flip-invariant, which is why the
    # test above cannot by itself decide the flip:
    assert np.allclose(np.flip(mean, axis=(0, 1)), proc, atol=1e-6, rtol=0)


@pytest.mark.slow
def test_integrate_kernel_over_time_sum(resp):
    from unfoldlarpix.deconv_workflow import integrate_kernel_over_time
    R, _ = resp
    mean = R.mean(axis=(1, 3), dtype=np.float64) * DT
    assert integrate_kernel_over_time(mean, B).sum() == pytest.approx(
        1.000266, abs=1e-5)


@pytest.mark.slow
def test_impact_index_sign_convention(resp):
    """Impact ``ix = 0`` is at the LOW edge, so ``dx = -1`` sees the excursion.

    The docstring's sign convention -- impact cell ``ix`` centred at
    ``(ix - 4.5) * pitch/10`` along +pixel_x, pixel index ``C + dx`` the
    SENSING pad at offset ``d`` from the collecting pad -- has an observable
    consequence: an electron landing at the low edge of its pad induces the
    large transient on the pad below it, not above.
    """
    R, _ = resp
    kc = cumulative(R, DT)
    low = float(kc[C - 1, 0, C, 0].max())
    high = float(kc[C + 1, 0, C, 0].max())
    assert low > 5 * high
    assert low == pytest.approx(0.205, abs=0.005)
    # the mirror impact gives the mirror answer
    assert float(kc[C + 1, 9, C, 4].max()) > 5 * float(kc[C - 1, 9, C, 4].max())


@pytest.mark.slow
def test_own_kernel_timing_matches_forward_model(kcum_own, resp):
    R, _ = resp
    cc = kcum_own[4, 4]
    pts = [int(np.searchsorted(cc, f)) - ARRIVAL_TICK
           for f in (0.01, 0.10, 0.50, 0.90, 0.99)]
    assert pts == [-215, -63, -18, -4, -1]
    cur = R[C, 4, C, 4]
    assert float(cur.max()) == pytest.approx(0.755, abs=0.002)
    assert int(cur.argmax()) == 3810


# ---------------------------------------------------------------------------
# window weights and the impact-averaging error
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_window_weights_sum_to_one(kcum_own):
    """``sum_s w_{i,(0,0)}(s; phi) = 1`` for every impact and every phase."""
    n = np.arange(-140, 4)
    for phi in range(B):
        w = window_weights(kcum_own, phi, n, B)
        tot = w.sum(axis=-1)
        assert np.abs(tot - 1.0).max() < 1e-5


@pytest.mark.slow
def test_impact_averaging_error_identities(kcum_own, kcum_bar):
    """``sum_i d = 0`` and ``sum_s d = 0`` over the full support."""
    n = np.arange(-140, 4)
    for phi in (0, 7, 17, 29):
        w = window_weights(kcum_own, phi, n, B)
        wb = window_weights(kcum_bar[C, C], phi, n, B)
        d = w - wb
        assert np.abs(d.mean(axis=(0, 1))).max() < 1e-9
        # limited by the float32 storage of the ndlarsim expansion: the
        # collecting-pad time integral varies by ~1.6e-7 across impacts, and
        # sum_s d_i is exactly that variation.
        assert np.abs(d.sum(axis=-1)).max() < 2e-7


def test_gaussian_cell_weights_normalised():
    p, out = gaussian_cell_weights(5.789, 1.18, 10)
    assert p.shape == (10,)
    assert p.sum() == pytest.approx(1.0)
    assert 0.0 < out < 1e-3
    assert int(np.argmax(p)) == 5


def test_kcum_at_clamps():
    kc = np.arange(10.0)
    assert kcum_at(kc, -1) == 0.0
    assert kcum_at(kc, 0) == 0.0
    assert kcum_at(kc, 5) == 5.0
    assert kcum_at(kc, 999) == 9.0


# ---------------------------------------------------------------------------
# closure on synthetic events
# ---------------------------------------------------------------------------
def _readout_config():
    return ReadoutConfig(time_spacing=DT, adc_hold_delay=B, adc_down_time=24,
                         csa_reset_time=2, one_tick=2, nburst=1,
                         threshold=5.0, uncorr_noise=None, thres_noise=None,
                         reset_noise=None)


def _synthetic_store(cells, pads, trigger, nburst, Kbar_current):
    """Build a store whose records are the exact response to ``cells``.

    ``cells`` is a list of ``(pixel_x, pixel_y, tick, charge)``.  The records
    are made by an EXPLICIT accumulator: the per-tick current on each pad is
    summed over source cells, cumulatively summed and sampled at the latch
    times -- the same construction ``tred.readout.fixed_interval_readout``
    performs, and deliberately NOT the ``Kcum(l_k) - Kcum(l_{k-1})`` form the
    algorithm under test uses.
    """
    nt = Kbar_current.shape[-1]
    latch = trigger + np.arange(nburst + 1) * B
    t_lo = min(int(t) for _, _, t, _ in cells)
    t_hi = max(int(latch[-1]), t_lo + nt) + 2
    span = t_hi - t_lo + 1
    cur = {p: np.zeros(span) for p in pads}
    for (sx, sy, tj, q) in cells:
        for p in pads:
            dx, dy = p[0] - sx, p[1] - sy
            if abs(dx) > 12 or abs(dy) > 12:
                continue
            k = Kbar_current[C + dx, C + dy] * q
            o = int(tj) - t_lo
            cur[p][o:o + nt] += k[:min(nt, span - o)]
    acc = {p: np.cumsum(v) for p, v in cur.items()}

    def A(p, t):
        i = int(t) - t_lo
        if i < 0:
            return 0.0
        return float(acc[p][min(i, span - 1)])

    q = np.array([[A(p, latch[k]) - A(p, latch[0])
                   for k in range(1, nburst + 1)] for p in pads])
    loc = np.array([[p[0], p[1], trigger, trigger + B,
                     trigger + nburst * B + 1] for p in pads], dtype=np.int64)
    dat = np.concatenate([np.zeros((len(pads), 3)), q], axis=1)
    hv = HitsView(loc, dat, B)
    effq_loc = np.array([[sx, sy, tj] for (sx, sy, tj, _) in cells],
                        dtype=np.int64)
    effq_dat = np.array([[0.0, 0.0, 0.0, qq] for (_, _, _, qq) in cells])
    ev = SimpleNamespace(
        tpc_id=0, event_id=0,
        effq=SimpleNamespace(location=effq_loc, data=effq_dat))
    store = EventStore()
    store.put("job.config", {"_meta": {"git": "test"}}, by="test")
    store.put("event", ev, by="test")
    store.put("readout_config", _readout_config(), by="test")
    store.put("hits_view", hv, by="test")
    return store


def _run_closure(store, **props):
    alg = FineTruthClosure(response=RESPONSE, transverse_model="none",
                           current_zero_before_tick=None,
                           shift_scan=[0], group_sizes=[1, 2],
                           line_pixel_y_range=[-10 ** 6, 10 ** 6], **props)
    alg.initialize({})
    alg.execute(store)
    return store.get("closure.record")


@pytest.mark.slow
def test_closure_exact_single_cell(resp):
    """One unit charge, one pad, one tick: the residual must vanish."""
    R, _ = resp
    Kbar_cur = R.mean(axis=(1, 3), dtype=np.float64) * DT
    pads = [(50 + dx, 50 + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    store = _synthetic_store([(50, 50, 100, 1.0)], pads,
                             trigger=60, nburst=150,
                             Kbar_current=Kbar_cur)
    rec = _run_closure(store)
    r = rec["residual"]["bar"]
    assert r["max_abs_eps_ke"] < 1e-9
    assert abs(r["sum_eps_ke"]) < 1e-9


@pytest.mark.slow
def test_closure_exact_two_pads_ring1(resp):
    """Two source pads one apart: exercises the ring-1 columns and the sign."""
    R, _ = resp
    Kbar_cur = R.mean(axis=(1, 3), dtype=np.float64) * DT
    pads = [(50 + dx, 50 + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    cells = [(50, 50, 100, 2.0), (51, 50, 137, 3.0)]
    store = _synthetic_store(cells, pads, trigger=60, nburst=150,
                             Kbar_current=Kbar_cur)
    rec = _run_closure(store)
    r = rec["residual"]["bar"]
    assert r["max_abs_eps_ke"] < 1e-9
    # the ring-1 pads really do carry signal, so the test is not vacuous
    assert rec["monotonicity"]["ring1_side"]["n_pads"] > 0
    assert r["total_abs_y_ke"] > 1.0


@pytest.mark.slow
def test_closure_truncation_term(resp):
    """``current_zero_before_tick`` removes exactly the pre-cut accumulator."""
    R, _ = resp
    Kbar_cur = R.mean(axis=(1, 3), dtype=np.float64) * DT
    Kcum = np.cumsum(Kbar_cur, axis=-1)
    pads = [(50 + dx, 50 + dy) for dx in range(-1, 2) for dy in range(-1, 2)]
    store = _synthetic_store([(50, 50, 100, 1.0)], pads, trigger=60,
                             nburst=150, Kbar_current=Kbar_cur)
    alg = FineTruthClosure(response=RESPONSE, transverse_model="none",
                           current_zero_before_tick=1000, shift_scan=[0],
                           group_sizes=[1],
                           line_pixel_y_range=[-10 ** 6, 10 ** 6])
    alg.initialize({})
    alg.execute(store)
    rec = store.get("closure.record")
    # the untruncated model closes exactly; the truncated one is short by the
    # accumulator at t_z - 1 = 999, i.e. Kcum(999 - 100) on the own pad
    assert rec["residual"]["bar"]["max_abs_eps_ke"] < 1e-9
    lost = float(sum(Kcum[C + p[0] - 50, C + p[1] - 50, 999 - 100]
                     for p in pads))
    assert rec["residual"]["bar_trunc"]["sum_eps_ke"] == pytest.approx(
        lost, abs=1e-9)
