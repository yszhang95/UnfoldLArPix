"""Burst-merge gap ``tau``: physical floor, cap, and the length/shift it guards.

The template compensation loses ~threshold of charge on immediate re-triggers
routed to it (gap below the floor), and dead-time merge across a genuine
silence (gap above the cap) changes the merged-sequence length and displaces
it by ~one bin.  ``resolve_burst_tau`` bounds ``tau`` to [floor, cap].
"""
from types import SimpleNamespace

import numpy as np
import pytest

from unfoldlarpix.burst_processor import BurstSequence
from unfoldlarpix.burst_processor_v3 import BurstSequenceProcessorV3
from unfoldlarpix.model.conventions import burst_tau_min
from unfoldlarpix.model.warm_start import resolve_burst_tau


def rc(adc_hold_delay=30, adc_down_time=24, one_tick=2):
    return SimpleNamespace(adc_hold_delay=adc_hold_delay,
                           adc_down_time=adc_down_time, one_tick=one_tick)


class TestResolveBurstTau:
    def test_floor_value(self):
        assert burst_tau_min(rc()) == 30 + 24 + 2          # B + down + tick

    def test_default_is_floor(self):
        assert resolve_burst_tau(rc(), None) == 56

    def test_in_range_passthrough(self):
        assert resolve_burst_tau(rc(), 58) == 58           # floor < 58 < cap(60)

    def test_below_floor_warns_and_clamps_up(self):
        with pytest.warns(RuntimeWarning, match="below the physical floor"):
            assert resolve_burst_tau(rc(), 30) == 56

    def test_above_cap_warns_and_clamps_down(self):
        with pytest.warns(RuntimeWarning, match="exceeds the cap"):
            assert resolve_burst_tau(rc(), 100) == 60      # 2 * B

    def test_floor_above_cap_warns_keeps_floor(self):
        # adc_down_time too large vs adc_hold_delay: floor(41) > cap(20)
        with pytest.warns(RuntimeWarning, match="exceeds cap"):
            assert resolve_burst_tau(rc(10, 30, 1)) == 41


ADC = 10.0


def _proc(tau):
    tmpl = np.cumsum(np.ones(60))                          # monotone, long enough
    return BurstSequenceProcessorV3(
        adc_hold_delay=ADC, tau=tau, deadtime=1.0, threshold=30.0,
        template_coll=tmpl, template_indu=tmpl)


def _seq(trigger, charge, px=0, py=0):
    c = np.asarray([charge], dtype=float)
    return BurstSequence(pixel_x=px, pixel_y=py, trigger_time_idx=trigger,
                         t_first=trigger + ADC, t_last=trigger + ADC,
                         charges=c, last_adc_latch=0, next_integration_start=0)


class TestGapAboveCapChangesLength:
    """A gap > 2*ADC routed to dead-time merge (uncapped tau) yields a
    different merged length than template routing — the shift the cap guards.
    """

    def _two_seq_gap(self):
        # seqA t_last=ADC=10; seqB t_first=35 => gap=25 > 2*ADC=20
        return [_seq(0, 50.0), _seq(25, 50.0)]

    def test_deadtime_and_template_differ_in_length(self):
        seqs = self._two_seq_gap()
        gap = seqs[1].t_first - seqs[0].t_last
        assert gap > 2 * ADC
        m_template = _proc(gap - 1).process_pixel_sequences(list(seqs))   # gap>tau
        m_deadtime = _proc(gap + 1).process_pixel_sequences(list(seqs))   # gap<=tau
        assert len(m_template.times) != len(m_deadtime.times)

    def test_cap_forbids_deadtime_for_above_cap_gap(self):
        # no requested tau can route a >2*ADC gap to dead-time merge
        cap = 2 * int(ADC)
        assert resolve_burst_tau(rc(ADC, ADC - 2, 1), 9999) == cap
        # with tau at the cap, the gap-25 pair stays on the template path,
        # so length matches the below-cap (floor) routing exactly
        m_floor = _proc(resolve_burst_tau(rc(ADC, ADC - 2, 1), None)).\
            process_pixel_sequences(self._two_seq_gap())
        m_cap = _proc(cap).process_pixel_sequences(self._two_seq_gap())
        assert len(m_floor.times) == len(m_cap.times)
