"""Killed truth split by whether it was ever recorded (``KilledTruthOrigin``).

Built on a hand-laid universal grid so each of the three cases is placed
deliberately: one killed voxel on a pixel that never fired, one on a firing
pixel after its last latch, one inside the recorded interval.  The point of the
algorithm is that these three are not interchangeable, so the test asserts the
split and not just the total.
"""
import numpy as np
import pytest
from unfoldlarpix.algs.eval_algs import KilledTruthOrigin
from unfoldlarpix.fwk.store import EventStore
from unfoldlarpix.io.hits import HitsView

B, CUT = 30, 0.5
# universal grid: 4 pixels x 1 x 6 bins, absolute pixel origin (10, 20),
# first bin index 0, edges at multiples of B
ORIGIN = {"u_min": 0, "p_min": [10, 20], "bin_ticks": B, "phi": 0.0,
          "b_off": [10.0, 20.0, 0.0]}


def _hits(seqs, nburst=1, q=20.0):
    loc, dat = [], []
    for px, py, trig in seqs:
        loc.append([px, py, trig, trig + B, trig + B * nburst + 26])
        dat.append([0.0, 0.0, 0.0] + list(np.cumsum([q] * nburst)))
    return HitsView(np.array(loc), np.array(dat, dtype=float), B)


def _run(T, R, hv, **props):
    store = EventStore()
    store.put("eval.truth", T)
    store.put("eval.reco", R)
    store.put("eval.origin", ORIGIN)
    store.put("eval.protocol", {"corr_threshold": CUT})
    store.put("hits_view", hv)
    alg = KilledTruthOrigin(**props)
    alg.initialize({})
    alg.execute(store)
    return store.get("killed.summary")


def test_three_origins_are_separated():
    T = np.zeros((4, 1, 6))
    R = np.zeros((4, 1, 6))
    # pixel index 0 -> absolute (10, 20): fires at tick 0, last latch = 30,
    # so bin 0 ([0,30)) is covered and bin 3 ([90,120)) is after the last latch
    T[0, 0, 0] = 2.0            # covered   -> killed (reco left at 0)
    T[0, 0, 3] = 5.0            # after last latch
    # pixel index 2 -> absolute (12, 20): never fires
    T[2, 0, 1] = 9.0            # no_hit
    # a matched voxel so the truth total is not all killed
    T[0, 0, 4] = 4.0
    R[0, 0, 4] = 4.0
    hv = _hits([(10, 20, 0)])
    r = _run(T, R, hv)

    assert r["Q_truth_ke"] == pytest.approx(20.0)
    assert r["killed_ke"] == pytest.approx(16.0)
    assert r["n_killed_voxels"] == 3
    o = r["by_origin"]
    assert o["covered"]["charge_ke"] == pytest.approx(2.0)
    assert o["after_last_latch"]["charge_ke"] == pytest.approx(5.0)
    assert o["no_hit"]["charge_ke"] == pytest.approx(9.0)
    # the two headline numbers
    assert r["killed_pct_recoverable"] == pytest.approx(100 * 2.0 / 20.0)
    assert r["killed_pct_never_recorded"] == pytest.approx(100 * 14.0 / 20.0)
    assert (r["killed_pct_recoverable"]
            + r["killed_pct_never_recorded"]) == pytest.approx(r["killed_pct"])


def test_cut_comes_from_the_evaluation_not_a_prop():
    """A second definition of the cut is how two tables came to disagree."""
    T = np.zeros((4, 1, 6))
    R = np.zeros((4, 1, 6))
    T[0, 0, 0] = 0.4                      # BELOW the 0.5 cut: not killed
    hv = _hits([(10, 20, 0)])
    store = EventStore()
    store.put("eval.truth", T)
    store.put("eval.reco", R)
    store.put("eval.origin", ORIGIN)
    store.put("eval.protocol", {"corr_threshold": 0.3})   # a different cut
    store.put("hits_view", hv)
    alg = KilledTruthOrigin(corr_threshold=0.9)           # ignored on purpose
    alg.initialize({})
    alg.execute(store)
    r = store.get("killed.summary")
    assert r["corr_threshold"] == 0.3
    assert r["n_killed_voxels"] == 1       # 0.4 > 0.3, so it IS killed here


def test_kill_rate_is_a_ratio_per_bin_not_a_distribution():
    T = np.zeros((4, 1, 6))
    R = np.zeros((4, 1, 6))
    T[0, 0, 0] = 1.0        # killed, lands in the bin containing 1.0
    T[0, 0, 1] = 1.0        # recovered, same bin
    R[0, 0, 1] = 1.0
    hv = _hits([(10, 20, 0)])
    r = _run(T, R, hv, bins=[0.5, 2.0, 50.0])
    sp = r["spectrum"]
    assert sp["truth_ke"][0] == pytest.approx(2.0)
    assert sp["killed_ke"][0] == pytest.approx(1.0)
    assert sp["kill_rate"][0] == pytest.approx(0.5)     # a ratio, bounded by 1
    assert sp["kill_rate"][1] is None                   # empty bin, not zero
    assert sp["n_voxels"][0] == 2


def test_never_firing_pixel_charge_is_reported_whole():
    """Charge on a non-firing pixel counts even where it is below the cut."""
    T = np.zeros((4, 1, 6))
    R = np.zeros((4, 1, 6))
    T[0, 0, 0] = 10.0                  # firing pixel
    R[0, 0, 0] = 10.0
    T[3, 0, 2] = 0.2                   # non-firing, BELOW the cut
    hv = _hits([(10, 20, 0)])
    r = _run(T, R, hv)
    assert r["killed_ke"] == pytest.approx(0.0)      # nothing above the cut
    assert r["Q_on_never_firing_pixels_ke"] == pytest.approx(0.2)
    assert r["pixels_fired"] == 1
    assert r["pixels_in_grid"] == 4
