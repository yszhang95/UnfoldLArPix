"""Gate on the split-trigger pseudo-measurement (``burst_tau``).

The pseudo row asserts "the accumulator equalled the threshold at the
trigger", which holds only for a THRESHOLD-limited trigger.  A re-trigger
that fires the instant the discriminator re-arms is DEAD-TIME limited: its
pre-trigger window holds the whole dead-time pile-up instead.  ``burst_tau``
is the gap below which a re-trigger counts as immediate; such sequences are
emitted as one lumped window with no split.

See docs/BURST_TAU.md for the measured justification.
"""
import numpy as np
import pytest

from unfoldlarpix.constrained_solver import build_latch_windows
from unfoldlarpix.model.conventions import burst_tau_min, resolve_burst_tau

B, THR, RESET = 30, 5.0, 2
BOFF = np.array([0, 0, 0])
FLOOR = B + 24 + 2          # adc_hold_delay + adc_down_time + one_tick = 56


class _RC:
    adc_hold_delay, adc_down_time, one_tick = B, 24, 2


def _seq(trigger, nburst=1, q=20.0):
    """One trigger sequence on pixel (0,0): location row + cumulative data."""
    loc = [0, 0, trigger, trigger + B, trigger + B * nburst + 26]
    dat = [0.0, 0.0, 0.0] + [q * (k + 1) for k in range(nburst)]
    return loc, dat


def _build(triggers, burst_tau=None, nburst=1):
    loc, dat = zip(*(_seq(t, nburst) for t in triggers))
    return build_latch_windows(np.array(loc), np.array(dat, dtype=float), B,
                               BOFF, csa_reset_time=RESET,
                               split_threshold=THR, burst_tau=burst_tau)


def _pseudo(windows):
    return [w for w in windows if w.value == pytest.approx(THR)]


def test_floor_matches_readout():
    assert burst_tau_min(_RC()) == FLOOR
    assert resolve_burst_tau(_RC(), None) == FLOOR


def test_default_none_is_legacy():
    """Default must reproduce the pre-feature windows bit for bit."""
    triggers = [100, 100 + B + 10]          # second one is an immediate re-trigger
    legacy = _build(triggers, burst_tau=None)
    assert len(_pseudo(legacy)) == 2        # ungated: both sequences split


def test_immediate_retrigger_is_not_split():
    # gap = trigger2 - last_latch1 = (130+10) - 130 = 10 ticks << FLOOR
    wins = _build([100, 100 + B + 10], burst_tau=FLOOR)
    assert len(_pseudo(wins)) == 1          # only the first sequence splits


def test_threshold_limited_retrigger_is_split():
    # gap = FLOOR exactly -> still threshold-limited (>= is inclusive)
    wins = _build([100, 100 + B + FLOOR], burst_tau=FLOOR)
    assert len(_pseudo(wins)) == 2


def test_first_sequence_on_a_pixel_always_splits():
    """No previous latch -> nothing to be dead-time limited by."""
    wins = _build([100], burst_tau=FLOOR)
    assert len(_pseudo(wins)) == 1


def test_gate_conserves_charge():
    triggers = [100, 100 + B + 10, 100 + 2 * B + 20]
    ungated = _build(triggers, burst_tau=None)
    gated = _build(triggers, burst_tau=FLOOR)
    assert sum(w.value for w in gated) == pytest.approx(
        sum(w.value for w in ungated))
    assert len(gated) < len(ungated)        # rows removed, charge kept


def test_gate_applies_per_pixel():
    """A short gap across DIFFERENT pixels is not an immediate re-trigger."""
    loc = np.array([[0, 0, 100, 100 + B, 100 + B + 26],
                    [1, 0, 100 + B + 10, 100 + 2 * B + 10, 0]])
    dat = np.array([[0.0, 0.0, 0.0, 20.0], [0.0, 0.0, 0.0, 20.0]])
    wins = build_latch_windows(loc, dat, B, BOFF, csa_reset_time=RESET,
                               split_threshold=THR, burst_tau=FLOOR)
    assert len(_pseudo(wins)) == 2          # both are each pixel's first
