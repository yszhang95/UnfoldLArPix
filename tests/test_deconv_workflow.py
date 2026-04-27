"""Tests for shared deconvolution workflow helpers."""

from pathlib import Path

import numpy as np
import pytest

from unfoldlarpix.burst_processor import BurstSequenceProcessor
from unfoldlarpix.burst_processor_v2 import BurstSequenceProcessorV2
from unfoldlarpix.data_containers import EffectiveCharge, EventData, Geometry, Hits, ReadoutConfig
from unfoldlarpix.deconv_workflow import (
    EventDeconvolutionResult,
    build_event_output_payload,
    create_burst_processor,
    integrate_kernel_over_time,
    prepare_field_response,
    process_event_deconvolution,
    shift_time_offset,
)


@pytest.fixture
def readout_config() -> ReadoutConfig:
    return ReadoutConfig(
        time_spacing=0.1,
        adc_hold_delay=2,
        adc_down_time=2,
        csa_reset_time=1,
        one_tick=1,
        nburst=2,
        threshold=5.0,
        uncorr_noise=None,
        thres_noise=None,
        reset_noise=None,
    )


@pytest.fixture
def sample_event() -> EventData:
    hits = Hits(
        data=np.array([[0.0, 0.0, 0.0, 2.0, 5.0]], dtype=float),
        location=np.array([[3, 4, 10, 12, 14]], dtype=int),
        tpc_id=0,
        event_id=11,
    )
    effq = EffectiveCharge(
        data=np.array([[0.0, 0.0, 0.0, 7.0]], dtype=float),
        location=np.array([[3, 4, 10]], dtype=int),
        tpc_id=0,
        event_id=11,
    )
    return EventData(
        tpc_id=0,
        event_id=11,
        hits=hits,
        effq=effq,
        global_tref=np.array([123.0]),
    )


class TestIntegrateKernelOverTime:
    def test_integrates_last_axis_in_fixed_bins(self):
        kernel = np.arange(16, dtype=float).reshape(2, 2, 4)
        integrated = integrate_kernel_over_time(kernel, 2)
        expected = np.array(
            [
                [[1.0, 5.0], [9.0, 13.0]],
                [[17.0, 21.0], [25.0, 29.0]],
            ]
        )
        np.testing.assert_allclose(integrated, expected)

    def test_rejects_nonzero_tail(self):
        kernel = np.ones((1, 1, 5), dtype=float)
        with pytest.raises(ValueError, match="zero-padded"):
            integrate_kernel_over_time(kernel, 2)


class TestPrepareFieldResponse:
    def test_prepare_field_response_returns_response_products_and_integrated_kernel(
        self, tmp_path: Path, readout_config: ReadoutConfig
    ):
        raw_response = np.zeros((2, 2, 4), dtype=float)
        raw_response[0, 0, :] = 2.5
        raw_response[0, 1, :] = 1.0
        raw_response[1, 0, :] = 0.5
        raw_response[1, 1, :] = 0.25
        npz_path = tmp_path / "field_response.npz"
        np.savez(
            npz_path,
            response=raw_response,
            npath=np.array(1),
            drift_length=np.array(42.0),
            bin_size=np.array(0.1),
            time_tick=np.array(0.1),
        )

        prepared = prepare_field_response(npz_path, readout_config.adc_hold_delay)

        assert prepared.full_response.shape == (4, 4, 4)
        assert prepared.integrated_response.shape == (4, 4, 2)
        assert prepared.center_response.shape == (4,)
        assert prepared.collection_response.shape == (4,)
        assert prepared.collection_plus_neighbors_response.shape == (4,)
        assert prepared.selected_response.shape == (4,)
        assert prepared.selected_response_mode == "center"
        assert prepared.template_search_mode == "monotonic"
        assert prepared.drift_length == 42.0

    def test_prepare_field_response_selects_collection_plus_neighbors_template(
        self, tmp_path: Path, readout_config: ReadoutConfig
    ):
        raw_response = np.zeros((2, 2, 4), dtype=float)
        raw_response[0, 0, :] = 2.5
        raw_response[0, 1, :] = 1.0
        raw_response[1, 0, :] = 0.5
        raw_response[1, 1, :] = 0.25
        npz_path = tmp_path / "field_response.npz"
        np.savez(
            npz_path,
            response=raw_response,
            npath=np.array(1),
            drift_length=np.array(42.0),
            bin_size=np.array(0.1),
            time_tick=np.array(0.1),
        )

        prepared = prepare_field_response(
            npz_path,
            readout_config.adc_hold_delay,
            response_template="collection_plus_neighbors",
        )

        np.testing.assert_allclose(
            prepared.selected_response,
            prepared.collection_plus_neighbors_response,
        )
        assert prepared.selected_response_mode == "collection_plus_neighbors"
        assert prepared.template_search_mode == "positive_cumulative"
        np.testing.assert_allclose(
            prepared.selected_response,
            prepared.full_response[1:4, 1:4, :].mean(axis=(0, 1)),
        )


class TestCreateBurstProcessor:
    def test_builds_v1_processor_from_template_response(self, readout_config: ReadoutConfig):
        template_response = np.array([1.0, 2.0, 3.0], dtype=float)
        processor = create_burst_processor(readout_config, template_response)

        assert isinstance(processor, BurstSequenceProcessor)
        np.testing.assert_allclose(processor.template, np.array([1.0, 3.0, 6.0]))
        assert processor.tau == readout_config.adc_hold_delay

    def test_builds_v2_processor_with_custom_tau(self, readout_config: ReadoutConfig):
        template_response = np.array([1.0, 2.0, 3.0], dtype=float)
        processor = create_burst_processor(
            readout_config,
            template_response,
            processor_cls=BurstSequenceProcessorV2,
            tau=7,
        )

        assert isinstance(processor, BurstSequenceProcessorV2)
        assert processor.tau == 7


class TestProcessEventDeconvolution:
    def test_runs_shared_pipeline(self, monkeypatch, sample_event: EventData, readout_config):
        prepared = prepare_stub_response()

        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.hits_to_merged_block",
            lambda *args, **kwargs: (
                np.array([3, 4, 20]),
                np.ones((2, 2, 3), dtype=float),
                12.5,
                (),
            ),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.build_gaussian_deconv_kernel",
            lambda *args, **kwargs: np.full((3, 3, 2), 2.0),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.deconv_fft",
            lambda measurement, kernel, filter_fft: (
                measurement + kernel[0, 0, 0] + filter_fft[0, 0, 0],
                (0, 0, 0),
            ),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.smear_effective_charge",
            lambda *args, **kwargs: (np.array([1, 2, 3]), np.full((2, 2, 2), 9.0)),
        )

        result = process_event_deconvolution(
            sample_event,
            readout_config,
            prepared,
            sigma_time=0.005,
            sigma_pixel=0.2,
        )

        assert result.compensated_charge == 12.5
        np.testing.assert_array_equal(result.hwf_block_offset, np.array([3, 4, 20]))
        np.testing.assert_allclose(result.deconv_q, np.full((2, 2, 3), 4.0))
        np.testing.assert_array_equal(result.smear_offset, np.array([1, 2, 3]))
        np.testing.assert_allclose(result.smeared_true, np.full((2, 2, 2), 9.0))
        assert result.local_offset == (0, 0, 0)
        assert result.response_template_mode == "collection_plus_neighbors"
        assert result.template_search_mode == "positive_cumulative"
        assert result.burst_compensation_mode == "v1"
        assert result.tau == readout_config.adc_hold_delay

    def test_requires_zero_local_offset_when_requested(
        self, monkeypatch, sample_event: EventData, readout_config
    ):
        prepared = prepare_stub_response()

        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.hits_to_merged_block",
            lambda *args, **kwargs: (
                np.array([0, 0, 0]),
                np.ones((1, 1, 2), dtype=float),
                1.0,
                (),
            ),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.build_gaussian_deconv_kernel",
            lambda *args, **kwargs: np.ones((1, 1, 2), dtype=float),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.deconv_fft",
            lambda *args, **kwargs: (np.ones((1, 1, 2), dtype=float), (0, 1, 0)),
        )
        monkeypatch.setattr(
            "unfoldlarpix.deconv_workflow.smear_effective_charge",
            lambda *args, **kwargs: (np.array([0, 0, 0]), np.ones((1, 1, 2), dtype=float)),
        )

        with pytest.raises(ValueError, match="Expected zero local offset"):
            process_event_deconvolution(
                sample_event,
                readout_config,
                prepared,
                sigma_time=0.005,
                sigma_pixel=0.2,
                require_zero_local_offset=True,
            )


class TestBuildEventOutputPayload:
    def test_builds_payload_and_optionally_includes_block(
        self, sample_event: EventData, readout_config: ReadoutConfig
    ):
        geometry = Geometry(
            tpc_id=0,
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
            drift_direction=1,
            anode_position=10.0,
            cathode_position=0.0,
            pixel_pitch=0.4,
        )
        result = EventDeconvolutionResult(
            compensated_charge=1.0,
            hwf_block=np.ones((2, 2, 2), dtype=float),
            hwf_block_offset=np.array([3, 4, 20]),
            deconv_q=np.full((2, 2, 2), 7.0),
            local_offset=(0, 0, 0),
            smeared_true=np.full((2, 2, 2), 8.0),
            smear_offset=np.array([1, 2, 3]),
            template_compensation_diagnostics={},
        )

        payload = build_event_output_payload(
            sample_event,
            geometry,
            readout_config,
            result,
            drift_length=12.0,
            boffset_time_shift=-2,
            include_hwf_block=True,
        )

        np.testing.assert_array_equal(payload["boffset"], np.array([3, 4, 18]))
        np.testing.assert_array_equal(payload["hwf_block_offset"], np.array([3, 4, 20]))
        np.testing.assert_allclose(payload["hwf_block"], np.ones((2, 2, 2)))
        assert payload["drtoa"] == 12.0
        assert payload["response_template_mode"] == "center"
        assert payload["template_search_mode"] == "monotonic"
        assert payload["burst_compensation_mode"] == "v1"
        assert payload["tau"] is None

    def test_shift_time_offset_returns_shifted_copy(self):
        offset = np.array([1, 2, 3])
        shifted = shift_time_offset(offset, 5)

        np.testing.assert_array_equal(shifted, np.array([1, 2, 8]))
        np.testing.assert_array_equal(offset, np.array([1, 2, 3]))


def prepare_stub_response():
    return type(
        "StubPreparedResponse",
        (),
        {
            "center_response": np.array([1.0, 2.0, 3.0], dtype=float),
            "selected_response": np.array([4.0, 5.0, 6.0], dtype=float),
            "selected_response_mode": "collection_plus_neighbors",
            "template_search_mode": "positive_cumulative",
            "integrated_response": np.ones((2, 2, 3), dtype=float),
        },
    )()
