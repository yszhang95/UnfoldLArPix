"""Tests for BurstSequenceProcessorV3 — two-pass dead-time then template flow."""

import numpy as np

from unfoldlarpix.burst_processor import BurstSequence
from unfoldlarpix.burst_processor_v3 import BurstSequenceProcessorV3


ADC = 10.0
TAU = 5.0
DEADTIME = 1.0


def make_processor(template=None, threshold=30.0, tau=TAU, deadtime=DEADTIME):
    tmpl = (
        template
        if template is not None
        else np.array([1, 2, 3, 4, 6, 8, 16, 36], dtype=float)
    )
    return BurstSequenceProcessorV3(
        adc_hold_delay=ADC,
        tau=tau,
        deadtime=deadtime,
        template=tmpl,
        threshold=threshold,
    )


def make_seq(trigger_time_idx, charges, pixel_x=0, pixel_y=0):
    charges = np.asarray(charges, dtype=float)
    return BurstSequence(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        trigger_time_idx=trigger_time_idx,
        t_first=trigger_time_idx + ADC,
        t_last=trigger_time_idx + ADC * len(charges),
        charges=charges,
        last_adc_latch=0,
        next_integration_start=0,
    )


class TestFirstPassGrouping:
    def test_merges_only_close_sequences_in_first_pass(self):
        proc = make_processor()
        seq_a = make_seq(0, [90.0, 100.0])
        seq_b = make_seq(13, [130.0, 10.0])  # t_first=23, gap=3 < tau
        seq_c = make_seq(60, [50.0, 20.0])   # t_first=70, far from merged AB

        groups = proc._merge_close_sequences_first_pass([seq_a, seq_b, seq_c])

        assert len(groups) == 2
        np.testing.assert_allclose(groups[0].times, np.array([10.0, 20.0, 23.0, 33.0]))
        np.testing.assert_allclose(groups[0].charges, np.array([90.0, 100.0, 195.0, 10.0]))
        np.testing.assert_allclose(groups[1].times, np.array([70.0, 80.0]))
        np.testing.assert_allclose(groups[1].charges, np.array([50.0, 20.0]))


class TestSecondPassTemplateMerging:
    LONG_TEMPLATE = np.cumsum(np.ones(200, dtype=float))

    def test_template_connects_groups_but_preserves_internal_deadtime_merges(self):
        proc = make_processor(template=self.LONG_TEMPLATE)
        seq_a = make_seq(0, [90.0, 100.0])
        seq_b = make_seq(13, [130.0, 10.0])   # close to A
        seq_c = make_seq(50, [40.0, 10.0])
        seq_d = make_seq(63, [80.0, 20.0])    # close to C

        merged = proc.process_pixel_sequences([seq_a, seq_b, seq_c, seq_d])

        # Internal first-pass merge times must survive unchanged in the final output.
        for expected_time in [10.0, 20.0, 23.0, 33.0, 60.0, 70.0, 73.0, 83.0]:
            assert np.any(np.isclose(merged.times, expected_time)), merged.times

        # Pass 2 should insert template points only between the merged groups.
        bridge_times = merged.times[(merged.times > 33.0) & (merged.times < 60.0)]
        assert len(bridge_times) > 0

        # No template points should appear inside the close gaps already merged in pass 1.
        gap_ab = merged.times[(merged.times > 20.0) & (merged.times < 23.0)]
        gap_cd = merged.times[(merged.times > 70.0) & (merged.times < 73.0)]
        assert len(gap_ab) == 0
        assert len(gap_cd) == 0
        assert np.all(np.diff(merged.times) > 0)

    def test_template_anchors_follow_merged_groups(self):
        proc = make_processor(template=self.LONG_TEMPLATE)
        seq_a = make_seq(0, [90.0, 100.0])
        seq_b = make_seq(13, [130.0, 10.0])   # group 1 peak at t=23 after compensation
        seq_c = make_seq(50, [40.0, 10.0])
        seq_d = make_seq(63, [80.0, 20.0])    # group 2 peak at t=73 after compensation

        proc.process_pixel_sequences([seq_a, seq_b, seq_c, seq_d])

        assert len(proc.template_compensation_anchors) == 2
        assert proc.template_compensation_anchors[0].is_bootstrap is True
        assert proc.template_compensation_anchors[1].is_bootstrap is False
        np.testing.assert_allclose(
            proc.template_compensation_anchors[0].sequence_peak_time, 23.0
        )
        np.testing.assert_allclose(
            proc.template_compensation_anchors[1].sequence_peak_time, 73.0
        )
