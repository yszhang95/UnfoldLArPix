"""The immediate-re-trigger fraction (``ImmediateFraction``).

The word "immediate" has named three different populations in three places and
they differ by up to a factor of three, so what is pinned here is not one
number but the *distinction* between the readings -- see the algorithm's
docstring and docs/BURST_TAU.md.  The published fraction is
``frac_immediate``: ``gap < burst_tau``, strict, over ALL sequences.

The gate condition must stay identical to ``build_latch_rows``' own
``threshold_limited``; the last test asserts that against the row builder
rather than restating it.
"""
import numpy as np
import pytest
from unfoldlarpix.algs.readout_algs import ImmediateFraction
from unfoldlarpix.constrained_solver import build_latch_rows
from unfoldlarpix.fwk.store import EventStore
from unfoldlarpix.io.hits import HitsView

B, DOWN, TICK, THR = 30, 24, 2, 5.0
FLOOR = B + DOWN + TICK          # 56
EARLIEST = DOWN + TICK           # 26 -- the shortest gap the readout allows
BOFF = np.array([0, 0, 0])


class _RC:
    adc_hold_delay, adc_down_time, one_tick, csa_reset_time = B, DOWN, TICK, 2
    threshold, nburst = THR, 1


def _hits(seqs, nburst=1, q=20.0):
    """``seqs`` = [(px, py, trigger), ...].  q is the FIRST burst charge."""
    loc, dat = [], []
    for px, py, trig in seqs:
        loc.append([px, py, trig, trig + B, trig + B * nburst + EARLIEST])
        qs = [q] + [q * 0.1] * (nburst - 1)
        dat.append([0.0, 0.0, 0.0] + list(np.cumsum(qs)))
    return np.array(loc), np.array(dat, dtype=float)


def _run(loc, dat, nburst=1, **props):
    store = EventStore()
    store.put("hits_view", HitsView(loc, dat, B))
    store.put("readout_config", _RC())
    alg = ImmediateFraction(**props)
    alg.initialize({})
    alg.execute(store)
    return store.get("immediate.summary")


def test_first_sequence_of_a_pixel_is_never_immediate():
    """It has no previous latch, and it counts in the DENOMINATOR."""
    r = _run(*_hits([(0, 0, 100), (1, 1, 100), (2, 2, 100)]))
    assert r["n_sequences"] == 3
    assert r["n_retriggers"] == 0
    assert r["frac_immediate"] == 0.0
    assert r["frac_immediate_of_retriggers"] == 0.0


def test_the_four_readings_differ_on_one_hand_counted_event():
    """One pixel, four sequences: re-arm, mid-window, exactly at tau, beyond.

    gaps are measured from the previous LAST latch = trigger + nburst*B.
    """
    t0 = 100
    t1 = t0 + B + EARLIEST          # gap 26 -> at re-arm, immediate
    t2 = t1 + B + 40                # gap 40 -> immediate, not at re-arm
    t3 = t2 + B + FLOOR             # gap 56 -> NOT immediate (strict)
    r = _run(*_hits([(0, 0, t) for t in (t0, t1, t2, t3)]))
    assert r["n_sequences"] == 4 and r["n_retriggers"] == 3
    assert r["gap_ticks"]["min"] == EARLIEST
    assert r["gap_ticks"]["earliest_allowed"] == EARLIEST
    # published reading: 2 of 4 sequences
    assert r["frac_immediate"] == pytest.approx(0.5)
    # gap == tau counts only in the inclusive reading: 3 of 4
    assert r["frac_immediate_inclusive"] == pytest.approx(0.75)
    # of the re-triggers alone: 2 of 3
    assert r["frac_immediate_of_retriggers"] == pytest.approx(2 / 3)
    # literally immediate -- fired at the re-arm instant: 1 of 4
    assert r["frac_at_rearm"] == pytest.approx(0.25)


def test_lumped_in_B_is_not_the_immediate_fraction():
    """The split gate is ``immediate OR c0 < threshold``.

    Conflating the two is why ``ab_anatomy/ab_rows.py``'s ``imm_frac``
    disagrees with the published table on every cell: its second branch
    survives to nburst = 64, where the immediate population is exactly zero.
    """
    loc, dat = _hits([(0, 0, 100), (1, 1, 100)])
    dat[1, 3:] = THR / 2                      # pixel (1,1): sub-threshold
    r = _run(loc, dat)
    assert r["frac_immediate"] == 0.0         # neither is a re-trigger
    assert r["frac_subthreshold"] == pytest.approx(0.5)
    assert r["frac_lumped_in_B"] == pytest.approx(0.5)
    assert r["frac_gate_only"] == 0.0
    assert r["frac_subthreshold_only"] == pytest.approx(0.5)


def test_gap_uses_the_last_latch_not_the_first():
    """With nburst > 1 the previous sequence ends at trigger + nburst*B.

    Measuring from the FIRST latch instead would call this pair immediate.
    """
    nb = 4
    t0, gap = 100, EARLIEST
    t1 = t0 + B * nb + gap
    r = _run(*_hits([(0, 0, t0), (0, 0, t1)], nburst=nb), nburst=nb)
    assert r["nburst"] == nb
    assert r["gap_ticks"]["min"] == gap
    assert r["frac_immediate"] == pytest.approx(0.5)
    # the same triggers read against the first latch would give a gap of
    # t1 - (t0 + B) = 116, i.e. not immediate -- the wrong answer
    assert t1 - (t0 + B) > FLOOR


def test_gate_agrees_with_build_latch_rows():
    """``frac_gate_only`` must equal the rows the row builder refuses to split.

    build_latch_rows emits a ``pseudo`` row exactly when the trigger is
    threshold-limited AND c0 >= threshold, so 1 - (pseudo rows / sequences) is
    the lumped fraction the algorithm reports.
    """
    t0 = 100
    triggers = [t0, t0 + B + EARLIEST, t0 + 2 * B + EARLIEST + FLOOR + 10]
    loc, dat = _hits([(0, 0, t) for t in triggers])
    _, metas = build_latch_rows(loc, dat, B, BOFF, csa_reset_time=2,
                                split_threshold=THR, burst_tau=FLOOR)
    n_seq = sum(1 for m in metas if m.kind in ("lumped", "pseudo"))
    n_lumped = sum(1 for m in metas if m.kind == "lumped")
    r = _run(loc, dat)
    assert n_seq == r["n_sequences"]
    assert n_lumped / n_seq == pytest.approx(r["frac_lumped_in_B"])
