"""Tests for template-compensation anchor diagnostics."""

import numpy as np

from unfoldlarpix.burst_processor import BurstSequence, BurstSequenceProcessor
from unfoldlarpix.data_containers import EffectiveCharge, EventData
from unfoldlarpix.deconv_workflow import build_template_compensation_diagnostics


def make_seq(
    *,
    trigger_time_idx: int,
    charges: list[float],
    pixel_x: int = 0,
    pixel_y: int = 0,
    adc_hold_delay: int = 10,
) -> BurstSequence:
    """Build a burst sequence with consistent time metadata."""
    return BurstSequence(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        trigger_time_idx=trigger_time_idx,
        t_first=trigger_time_idx + adc_hold_delay,
        t_last=trigger_time_idx + adc_hold_delay * len(charges),
        charges=np.asarray(charges, dtype=float),
        last_adc_latch=0,
        next_integration_start=0,
    )


def test_processor_records_template_compensation_anchors() -> None:
    """Bootstrap and non-close template steps should both emit anchors."""
    processor = BurstSequenceProcessor(
        adc_hold_delay=10,
        tau=15,
        deadtime=1,
        template=np.cumsum(np.ones(200, dtype=float)),
        threshold=5.0,
    )
    seq_a = make_seq(trigger_time_idx=20, charges=[1.0, 7.0, 3.0])
    seq_b = make_seq(trigger_time_idx=80, charges=[2.0, 9.0, 4.0])

    processor.process_pixel_sequences([seq_a, seq_b])

    anchors = processor.template_compensation_anchors
    assert len(anchors) == 2

    first_anchor = anchors[0]
    assert first_anchor.is_bootstrap is True
    assert first_anchor.sequence_peak_index == 1
    assert first_anchor.sequence_peak_time == 40.0
    assert first_anchor.sequence_peak_charge == 7.0

    second_anchor = anchors[1]
    assert second_anchor.is_bootstrap is False
    assert second_anchor.sequence_peak_index == 1
    assert second_anchor.sequence_peak_time == 100.0
    assert second_anchor.sequence_peak_charge == 9.0


def test_build_template_compensation_diagnostics_uses_channel_effq_peak() -> None:
    """Each anchor should be compared against the peak effq sample in its channel."""
    processor = BurstSequenceProcessor(
        adc_hold_delay=10,
        tau=15,
        deadtime=1,
        template=np.cumsum(np.ones(200, dtype=float)),
        threshold=5.0,
    )
    seq_a = make_seq(trigger_time_idx=20, charges=[1.0, 7.0, 3.0], pixel_x=4, pixel_y=9)
    seq_b = make_seq(trigger_time_idx=80, charges=[2.0, 9.0, 4.0], pixel_x=4, pixel_y=9)
    processor.process_pixel_sequences([seq_a, seq_b])

    effq = EffectiveCharge(
        data=np.array(
            [
                [0.0, 0.0, 0.0, 2.5],
                [0.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=float,
        ),
        location=np.array(
            [
                [4, 9, 18],
                [4, 9, 66],
                [4, 9, 80],
            ],
            dtype=int,
        ),
        tpc_id=0,
        event_id=0,
    )
    event = EventData(tpc_id=0, event_id=0, effq=effq)

    diagnostics = build_template_compensation_diagnostics(
        event,
        tuple(processor.template_compensation_anchors),
    )

    np.testing.assert_array_equal(
        diagnostics["template_comp_peak_locations"],
        np.array([[4.0, 9.0, 40.0], [4.0, 9.0, 100.0]], dtype=float),
    )
    np.testing.assert_array_equal(
        diagnostics["template_comp_peak_indices"],
        np.array([1, 1], dtype=int),
    )
    np.testing.assert_array_equal(
        diagnostics["template_comp_effq_peak_time"],
        np.array([66.0, 66.0], dtype=float),
    )
    np.testing.assert_array_equal(
        diagnostics["template_comp_effq_peak_value"],
        np.array([6.0, 6.0], dtype=float),
    )
    np.testing.assert_array_equal(
        diagnostics["template_comp_effq_peak_distance"],
        np.array([26.0, -34.0], dtype=float),
    )
    np.testing.assert_array_equal(
        diagnostics["template_comp_effq_peak_distance_abs"],
        np.array([26.0, 34.0], dtype=float),
    )
