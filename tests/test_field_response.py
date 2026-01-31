"""Test suite for field response processor."""

import pytest
import numpy as np
from pathlib import Path

from unfoldlarpix.field_response import FieldResponseProcessor


class TestFieldResponseProcessor:
    """Test cases for FieldResponseProcessor class."""

    @pytest.fixture
    def mock_field_response_data(self) -> dict:
        """Create mock field response data structure."""
        # Create a simple 5x5 quadrant response with 10 time steps
        raw_response = np.random.rand(5, 5, 10).astype(np.float32)

        # Normalize origin point to sum to 20 (as specified in requirements)
        raw_response[0, 0, :] = 2.0  # Sum over 10 time steps = 20

        return {
            "response": raw_response,
            "npath": np.array(2),
            "drift_length": np.array(50.0),
            "bin_size": np.array(0.1),
            "time_tick": np.array(0.5),
        }

    @pytest.fixture
    def temp_field_response_file(self, tmp_path, mock_field_response_data) -> Path:
        """Create a temporary NPZ file with mock field response data."""
        npz_path = tmp_path / "field_response.npz"
        np.savez(npz_path, **mock_field_response_data)
        return npz_path

    def test_init_with_valid_file(self, temp_field_response_file):
        """Test initialization with valid NPZ file."""
        processor = FieldResponseProcessor(temp_field_response_file)
        assert processor.npz_filepath == temp_field_response_file
        assert processor._processed_response is None

    def test_init_with_nonexistent_file(self):
        """Test initialization with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            FieldResponseProcessor("nonexistent.npz")

    def test_load_data(self, temp_field_response_file):
        """Test data loading."""
        processor = FieldResponseProcessor(temp_field_response_file)
        data = processor._load_data()

        assert "response" in data
        assert "npath" in data
        assert isinstance(data["response"], np.ndarray)
        assert data["response"].shape == (5, 5, 10)

    def test_quadrant_copy_3d_array(self):
        """Test quadrant expansion for 3D array."""
        half_plane_size = 2.5
        npath = 2
        nt = 3
        raw = np.arange(int(round(half_plane_size * npath
                                  * half_plane_size * npath * nt)))\
                .reshape(int(round(half_plane_size * npath)),
                         int(round(half_plane_size * npath)), nt)

        result = FieldResponseProcessor._quadrant_copy(None, raw)

        # Check shape: should be doubled in first two dimensions, same in third
        expected_shape = (2 * half_plane_size * npath,
                          2 * half_plane_size * npath, nt)
        expected_shape = tuple(int(round(dim)) for dim in expected_shape)
        assert result.shape == expected_shape

    def test_quadrant_copy_wrong_dimensions(self):
        """Test quadrant expansion with wrong input dimensions."""
        raw = np.array([1, 2, 3])  # 1D array

        with pytest.raises(ValueError):
            FieldResponseProcessor._quadrant_copy(None, raw)

    def test_quadrant_copy_custom_axis(self):
        """Test quadrant expansion with custom time axis."""
        raw = np.random.rand(5, 3*2, 3*2)

        # Test with time axis at position 0
        result_axis0 = FieldResponseProcessor._quadrant_copy(None, raw, axis=0)
        assert result_axis0.shape == (12, 12, 5)

    def test_downsample_by_averaging(self):
        """Test downsampling by averaging intra-pixel positions."""
        # Create expanded response: 2x3x2x3 spatial, 4 time steps, npath=2
        expanded = np.random.rand(6, 6, 4)
        npath = 2

        result = FieldResponseProcessor._downsample_by_averaging(None, expanded, npath)

        # Expected result shape: 3x3x3 (6/2, 6/2, 4)
        assert result.shape == (3, 3, 4)

        # Verify averaging logic manually
        # For pixel (0,0), we need to average expanded[0:2, 0:2, :]
        expected_00 = np.mean([
            expanded[0, 0, :], expanded[0, 1, :],
            expanded[1, 0, :], expanded[1, 1, :]
        ], axis=0)
        np.testing.assert_allclose(result[0, 0, :], expected_00)

    def test_downsample_unequal_dimensions(self):
        """Test downsampling with unequal spatial dimensions."""
        expanded = np.random.rand(6, 4, 3)  # Unequal spatial dimensions
        npath = 2

        with pytest.raises(ValueError, match="Spatial dimensions must be equal"):
            FieldResponseProcessor._downsample_by_averaging(None, expanded, npath)

    def test_downsample_not_divisible(self):
        """Test downsampling with dimensions not divisible by npath."""
        expanded = np.random.rand(5, 5, 3)  # Not divisible by npath=2
        npath = 2

        with pytest.raises(ValueError, match="not divisible by npath"):
            FieldResponseProcessor._downsample_by_averaging(None, expanded, npath)

    def test_process_response_complete_workflow(self, temp_field_response_file):
        """Test complete processing workflow."""
        processor = FieldResponseProcessor(temp_field_response_file)

        # Process response
        result = processor.process_response()

        # Check that result is cached
        assert processor._processed_response is result

        # Verify final shape: (2*5, 2*5, 10) -> downsampled to (5, 5, 10)
        expected_shape = (5, 5, 10)
        assert result.shape == expected_shape

    def test_process_response_missing_keys(self, tmp_path):
        """Test processing with missing required keys."""
        # Create NPZ file with missing 'response' key
        incomplete_data = {"npath": np.array(2)}
        npz_path = tmp_path / "incomplete.npz"
        np.savez(npz_path, **incomplete_data)

        processor = FieldResponseProcessor(npz_path)

        with pytest.raises(ValueError, match="Missing required keys"):
            processor.process_response()

    def test_get_metadata(self, temp_field_response_file):
        """Test metadata extraction."""
        processor = FieldResponseProcessor(temp_field_response_file)
        metadata = processor.get_metadata()

        assert "drift_length" in metadata
        assert "bin_size" in metadata
        assert "time_tick" in metadata
        assert "npath" in metadata

        assert metadata["drift_length"] == 50.0
        assert metadata["npath"] == 2

    def test_response_property(self, temp_field_response_file):
        """Test response property access."""
        processor = FieldResponseProcessor(temp_field_response_file)

        # Should trigger processing on first access
        result = processor.response
        assert result is not None
        assert result.shape == (5, 5, 10)

        # Should be cached after first access
        assert processor._processed_response is result

    def test_validate_response_normalization_correct(self, mock_field_response_data, tmp_path):
        """Test normalization validation with correct data."""
        # Create temporary NPZ file
        npz_path = tmp_path / "correct_normalization.npz"
        np.savez(npz_path, **mock_field_response_data)

        processor = FieldResponseProcessor(npz_path)
        processor.process_response()

        assert processor.validate_response_normalization(expected_sum=20.0)

    def test_validate_response_normalization_incorrect(self, mock_field_response_data, tmp_path):
        """Test normalization validation with incorrect data."""
        # Modify data to have incorrect sum
        mock_field_response_data["response"][0, 0, :] = 1.0  # Sum to 10 instead of 20

        # Create temporary NPZ file
        npz_path = tmp_path / "incorrect_normalization.npz"
        np.savez(npz_path, **mock_field_response_data)

        processor = FieldResponseProcessor(npz_path)
        processor.process_response()

        assert not processor.validate_response_normalization(expected_sum=20.0)

    def test_validate_response_normalization_tolerance(self, mock_field_response_data, tmp_path):
        """Test normalization validation with tolerance."""
        # Set sum close but not exactly 20
        mock_field_response_data["response"][0, 0, :] = 2.0005  # Sum to 20.005

        # Create temporary NPZ file
        npz_path = tmp_path / "tolerance_test.npz"
        np.savez(npz_path, **mock_field_response_data)

        processor = FieldResponseProcessor(npz_path)
        processor.process_response()

        # Should pass with larger tolerance
        assert processor.validate_response_normalization(
            expected_sum=20.0, tolerance=0.01
        )

        # Should fail with stricter tolerance
        assert not processor.validate_response_normalization(
            expected_sum=20.0, tolerance=0.001
        )

    def test_quadrant_copy_symmetry(self):
        """Test that quadrant copy creates symmetric result."""
        # Create a simple quadrant with 3D structure
        raw = np.array([
            [[1, 2],
            [3, 4]],
            [[5, 6],
            [7, 8]]
        ])

        result = FieldResponseProcessor._quadrant_copy(None, raw)

        # Check symmetry properties
        # Center quadrants should match original
        np.testing.assert_array_equal(result[2:, 2:, :], raw)

        # Check that quadrants are correctly flipped
        np.testing.assert_array_equal(
            result[:2, 2:, :],  # negative-positive quadrant
            np.flip(raw, axis=0)
        )
        np.testing.assert_array_equal(
            result[2:, :2, :],  # positive-negative quadrant
            np.flip(raw, axis=1)
        )
        np.testing.assert_array_equal(
            result[:2, :2, :],  # negative-negative quadrant
            np.flip(raw, axis=(0, 1))
        )

    def test_downsampling_preserves_time_axis(self):
        """Test that downsampling preserves time axis dimension."""
        expanded = np.random.rand(6, 6, 8)  # 6x6 spatial, 8 time steps
        npath = 3

        result = FieldResponseProcessor._downsample_by_averaging(None, expanded, npath)

        # Time axis should be preserved
        assert result.shape[2] == expanded.shape[2] == 8
        assert result.shape == (2, 2, 8)  # (6/3, 6/3, 8)

    def test_integration_with_realistic_data(self, tmp_path):
        """Test integration with realistic field response parameters."""
        # Simulate realistic field response data
        npath = 10
        pixel_count = 9  # 4.5 pixels on each side of origin
        time_steps = 6400

        # Create quadrant response
        quadrant_size = pixel_count * npath
        raw_response = np.random.rand(quadrant_size, quadrant_size, time_steps).astype(
            np.float32
        )
        raw_response = raw_response[quadrant_size//2:, quadrant_size//2:, :]

        # Normalize origin point
        raw_response[0, 0, :] = 20.0 / time_steps

        data = {
            "response": raw_response,
            "npath": np.array(npath),
            "drift_length": np.array(100.0),
            "bin_size": np.array(0.05),
            "time_tick": np.array(0.025),
        }

        # Create temporary NPZ file
        npz_path = tmp_path / "realistic_data.npz"
        np.savez(npz_path, **data)

        processor = FieldResponseProcessor(npz_path)
        result = processor.process_response()

        # Verify final shape
        assert result.shape == (pixel_count, pixel_count, time_steps)

        # Verify metadata
        metadata = processor.get_metadata()
        assert metadata["npath"] == npath
        assert metadata["drift_length"] == 100.0
