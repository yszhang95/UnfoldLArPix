"""Test suite for data loader functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from unfoldlarpix.data_containers import (
    Current,
    EffectiveCharge,
    EventData,
    Geometry,
    Hits,
    ReadoutConfig,
)
from unfoldlarpix.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def mock_npz_data(self) -> dict:
        """Create mock NPZ data structure."""
        data = {}

        # Geometry data for TPC 0 and 1
        data["tpc_lower_left_tpc0"] = np.array([0.0, 0.0, 0.0])
        data["tpc_upper_tpc0"] = np.array([10.0, 10.0, 10.0])
        data["drift_direction_tpc0"] = np.array(1)
        data["tpc_anode_tpc0"] = np.array(5.0)
        data["tpc_cathode_tpc0"] = np.array(0.0)
        data["pixel_pitch_tpc0"] = np.array(0.1)

        data["tpc_lower_left_tpc1"] = np.array([0.0, 0.0, 0.0])
        data["tpc_upper_tpc1"] = np.array([10.0, 10.0, 10.0])
        data["drift_direction_tpc1"] = np.array(-1)
        data["tpc_anode_tpc1"] = np.array(5.0)
        data["tpc_cathode_tpc1"] = np.array(0.0)
        data["pixel_pitch_tpc1"] = np.array(0.1)

        # Readout configuration
        data["time_spacing"] = np.array(0.1)
        data["adc_hold_delay"] = np.array(10)
        data["adc_down_time"] = np.array(5)
        data["csa_reset_time"] = np.array(100)
        data["one_tick"] = np.array(1)
        data["threshold"] = np.array(1.5)
        data["uncorr_noise"] = np.array(0.1)
        data["thres_noise"] = np.array(0.05)
        data["reset_noise"] = np.array(0.02)

        # Event data - TPC 0, Event 42 (spans 2 batches)
        data["event_id_tpc0_batch0"] = np.array([42])
        data["global_tref_tpc0_batch0"] = np.array([1000.0, 0.0])
        data["effq_tpc0_batch0"] = np.random.rand(5, 4)
        data["effq_tpc0_batch0_location"] = np.random.randint(0, 10, (5, 3))
        data["current_tpc0_batch0"] = np.random.rand(100)
        data["current_tpc0_batch0_location"] = np.random.randint(0, 10, (1, 3))
        data["hits_tpc0_batch0"] = np.random.rand(3, 4)
        data["hits_tpc0_batch0_location"] = np.random.randint(0, 10, (3, 5))

        data["event_id_tpc0_batch1"] = np.array([42])
        data["global_tref_tpc0_batch1"] = np.array([1000.0, 0.0])
        data["effq_tpc0_batch1"] = np.random.rand(3, 4)
        data["effq_tpc0_batch1_location"] = np.random.randint(0, 10, (3, 3))
        data["current_tpc0_batch1"] = np.random.rand(50)
        data["current_tpc0_batch1_location"] = np.random.randint(0, 10, (1, 3))
        data["hits_tpc0_batch1"] = np.random.rand(2, 4)
        data["hits_tpc0_batch1_location"] = np.random.randint(0, 10, (2, 5))

        # Event data - TPC 1, Event 43 (single batch)
        data["event_id_tpc1_batch0"] = np.array([43])
        data["global_tref_tpc1_batch0"] = np.array([2000.0, 0.0])
        data["effq_tpc1_batch0"] = np.random.rand(2, 4)
        data["effq_tpc1_batch0_location"] = np.random.randint(0, 10, (2, 3))
        data["hits_tpc1_batch0"] = np.random.rand(1, 4)
        data["hits_tpc1_batch0_location"] = np.random.randint(0, 10, (1, 5))

        return data

    @pytest.fixture
    def temp_npz_file(self, tmp_path, mock_npz_data) -> Path:
        """Create a temporary NPZ file with mock data."""
        npz_path = tmp_path / "test_data.npz"
        np.savez(npz_path, **mock_npz_data)
        return npz_path

    def test_init_with_valid_file(self, temp_npz_file):
        """Test initialization with valid NPZ file."""
        loader = DataLoader(temp_npz_file)
        assert loader.npz_path == temp_npz_file

    def test_init_with_nonexistent_file(self):
        """Test initialization with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent.npz")

    def test_parse_geometry(self, temp_npz_file):
        """Test geometry parsing."""
        loader = DataLoader(temp_npz_file)
        geometry = loader.get_all_geometry()

        assert len(geometry) == 2
        assert 0 in geometry
        assert 1 in geometry

        geom0 = geometry[0]
        assert geom0.tpc_id == 0
        assert geom0.drift_direction == 1
        assert isinstance(geom0.lower, np.ndarray)
        assert isinstance(geom0.upper, np.ndarray)

        geom1 = geometry[1]
        assert geom1.tpc_id == 1
        assert geom1.drift_direction == -1

    def test_parse_readout_config(self, temp_npz_file):
        """Test readout configuration parsing."""
        loader = DataLoader(temp_npz_file)
        config = loader.get_readout_config()

        assert isinstance(config, ReadoutConfig)
        assert config.time_spacing == 0.1
        assert config.adc_hold_delay == 10
        assert config.threshold == 1.5

    def test_list_events(self, temp_npz_file):
        """Test event listing."""
        loader = DataLoader(temp_npz_file)
        events = loader.list_events()

        assert len(events) == 2
        assert (0, 42) in events
        assert (1, 43) in events

    def test_iter_events(self, temp_npz_file):
        """Test event iteration."""
        loader = DataLoader(temp_npz_file)
        events = list(loader.iter_events())

        assert len(events) == 2

        # Find event (0, 42)
        event_0_42 = next(e for e in events if e.tpc_id == 0 and e.event_id == 42)
        assert event_0_42.effq is not None
        assert event_0_42.current is not None
        assert event_0_42.hits is not None
        assert event_0_42.global_tref is not None

        # Check that data from multiple batches is merged
        assert event_0_42.effq.data.shape[0] == 8  # 5 + 3 from two batches
        assert event_0_42.effq.location.shape[0] == 8

        # Find event (1, 43)
        event_1_43 = next(e for e in events if e.tpc_id == 1 and e.event_id == 43)
        assert event_1_43.effq is not None
        assert event_1_43.hits is not None
        assert event_1_43.current is None  # No current data for this event

    def test_get_geometry_specific_tpc(self, temp_npz_file):
        """Test getting geometry for specific TPC."""
        loader = DataLoader(temp_npz_file)
        geometry = loader.get_geometry(0)

        assert geometry.tpc_id == 0
        assert geometry.drift_direction == 1

    def test_get_geometry_nonexistent_tpc(self, temp_npz_file):
        """Test getting geometry for nonexistent TPC."""
        loader = DataLoader(temp_npz_file)
        with pytest.raises(ValueError):
            loader.get_geometry(999)


class TestDataContainers:
    """Test cases for data container classes."""

    def test_effective_charge_container(self):
        """Test EffectiveCharge container."""
        data = np.random.rand(10, 4)
        location = np.random.randint(0, 10, (10, 3))

        effq = EffectiveCharge(data=data, location=location, tpc_id=0, event_id=42)

        assert effq.tpc_id == 0
        assert effq.event_id == 42
        assert np.array_equal(effq.data, data)
        assert np.array_equal(effq.location, location)

    def test_effective_charge_invalid_shape(self):
        """Test EffectiveCharge with invalid data shape."""
        data = np.random.rand(10, 3)  # Wrong shape
        location = np.random.randint(0, 10, (10, 3))

        with pytest.raises(ValueError):
            EffectiveCharge(data=data, location=location, tpc_id=0, event_id=42)

    def test_current_container(self):
        """Test Current container."""
        data = np.random.rand(100)
        location = np.random.randint(0, 10, (1, 3))

        current = Current(data=data, location=location, tpc_id=0, event_id=42)

        assert current.tpc_id == 0
        assert current.event_id == 42
        assert np.array_equal(current.data, data)
        assert np.array_equal(current.location, location)

    def test_hits_container(self):
        """Test Hits container."""
        data = np.random.rand(5, 4)
        location = np.random.randint(0, 10, (5, 5))

        hits = Hits(data=data, location=location, tpc_id=0, event_id=42)

        assert hits.tpc_id == 0
        assert hits.event_id == 42
        assert np.array_equal(hits.data, data)
        assert np.array_equal(hits.location, location)

    def test_event_data_container(self):
        """Test EventData container."""
        event_data = EventData(tpc_id=0, event_id=42)

        assert event_data.tpc_id == 0
        assert event_data.event_id == 42
        assert not event_data.has_data()
        assert event_data.get_data_types() == []

        # Add some data
        effq_data = np.random.rand(5, 4)
        effq_location = np.random.randint(0, 10, (5, 3))
        event_data.effq = EffectiveCharge(
            data=effq_data, location=effq_location, tpc_id=0, event_id=42
        )

        assert event_data.has_data()
        assert event_data.get_data_types() == ["effq"]

    def test_geometry_container(self):
        """Test Geometry container."""
        geometry = Geometry(
            tpc_id=0,
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([10.0, 10.0, 10.0]),
            drift_direction=1,
            anode_position=5.0,
            cathode_position=0.0,
            pixel_pitch=0.1,
        )

        assert geometry.tpc_id == 0
        assert geometry.drift_direction == 1

    def test_geometry_invalid_drift_direction(self):
        """Test Geometry with invalid drift direction."""
        with pytest.raises(ValueError):
            Geometry(
                tpc_id=0,
                lower=np.array([0.0, 0.0, 0.0]),
                upper=np.array([10.0, 10.0, 10.0]),
                drift_direction=2,  # Invalid
                anode_position=5.0,
                cathode_position=0.0,
                pixel_pitch=0.1,
            )

    def test_readout_config_container(self):
        """Test ReadoutConfig container."""
        config = ReadoutConfig(
            time_spacing=0.1,
            adc_hold_delay=10,
            adc_down_time=5,
            csa_reset_time=100,
            one_tick=1,
            threshold=1.5,
            uncorr_noise=0.1,
            thres_noise=0.05,
            reset_noise=0.02,
        )

        assert config.time_spacing == 0.1
        assert config.adc_hold_delay == 10
        assert config.threshold == 1.5
