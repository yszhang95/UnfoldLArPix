"""Field response processor for kernel loading and preprocessing."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class FieldResponseProcessor:
    """Processor for field response data from particle detector simulation.

    This class loads and preprocesses field response arrays from NPZ files,
    expanding quadrant data to full detector plane and downsampling by averaging
    intra-pixel impact positions.
    """

    def __init__(self, npz_filepath: Union[str, Path]) -> None:
        """Initialize field response processor.

        Args:
            npz_filepath: Path to NPZ file containing field response data.
        """
        self.npz_filepath = Path(npz_filepath)
        if not self.npz_filepath.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_filepath}")

        self._data: Optional[dict] = None
        self._processed_response: Optional[np.ndarray] = None

    def _load_data(self) -> dict:
        """Load data from NPZ file."""
        if self._data is None:
            self._data = dict(np.load(self.npz_filepath, allow_pickle=True))
        return self._data

    def _quadrant_copy(self, raw: np.ndarray, axis: int = -1) -> np.ndarray:
        """Expand quadrant to full plane by copying and flipping.

        The raw array represents one quadrant of a symmetric 2D plane.
        This function expands it to cover all four quadrants.

        Args:
            raw: 3D array representing one quadrant of response.
            axis: Time axis index (default: -1).

        Returns:
            Full response plane with all quadrants filled.
        """
        if len(raw.shape) != 3:
            raise ValueError(f"quadrant_copy operates only on 3D array, got {len(raw.shape)}")

        # Always work with time axis at last position for simplicity
        if axis != -1:
            raw = np.moveaxis(raw, axis, -1)

        # Create full array: double spatial dimensions, keep time dimension
        shape = np.array(raw.shape)
        if shape[0] != shape[1]:
            raise ValueError("Input array must have equal spatial dimensions for quadrant copy.")
        shape[:2] = shape[:2] * 2  # Double spatial dimensions
        full = np.zeros(shape.astype(int), dtype=raw.dtype)

        h0 = raw.shape[0]
        h1 = raw.shape[1]

        # The input `raw` tensor represents a quadrant located in the positive-positive corner.
        # Shifting the origin to the lower corner of the pixel means the new indices of input
        # must cover a larger positive index range.

        # positive-positive quadrant (original): no flip needed
        full[h0:  , h1:  , :] = raw
        # negative-positive quadrant: flip vertically (axis 0)
        full[  :h0, h1:  , :] = np.flip(raw, axis=0)
        # positive-negative quadrant: flip horizontally (axis 1)
        full[h0:  ,   :h1, :] = np.flip(raw, axis=1)
        # negative-negative quadrant: flip both axes
        full[  :h0,   :h1, :] = np.flip(raw, axis=(0, 1))

        return full

    def _downsample_by_averaging(
        self, expanded_response: np.ndarray, npath: int
    ) -> np.ndarray:
        """Downsample expanded response by averaging intra-pixel positions.

        Args:
            expanded_response: Full response plane with shape ((2r+1)*npath, (2r+1)*npath, Nt).
            npath: Number of simulated impact positions per pixel pitch.

        Returns:
            Downsampled response array with shape (2r+1, 2r+1, Nt).
        """
        # Calculate number of pixels along each dimension
        spatial_shape = expanded_response.shape[:2]
        n_pixels = spatial_shape[0] // npath  # Should be equal for both dimensions

        if spatial_shape[0] != spatial_shape[1]:
            raise ValueError(
                f"Spatial dimensions must be equal, got {spatial_shape}"
            )
        if spatial_shape[0] % npath != 0:
            raise ValueError(
                f"Spatial dimensions {spatial_shape} not divisible by npath={npath}"
            )

        # Reshape for averaging: (n_pixels, npath, n_pixels, npath, Nt)
        reshaped = expanded_response.reshape(
            n_pixels, npath, n_pixels, npath, expanded_response.shape[2]
        )

        # Average over npath axes (axes 1 and 3)
        downsampled = np.mean(reshaped, axis=(1, 3))

        return downsampled

    def _flip_per_pixel(
            self, expanded_response: np.ndarray, npath: int
    ) -> np.ndarray:
        """Flip expanded response per pixel.

        Returns:
            Flipped response array with shape ((2r+1)*npath, (2r+1)*npath, Nt).
        """
        # Calculate number of pixels along each dimension
        spatial_shape = expanded_response.shape[:2]
        n_pixels = spatial_shape[0] // npath  # Should be equal for both dimensions

        if spatial_shape[0] != spatial_shape[1]:
            raise ValueError(
                f"Spatial dimensions must be equal, got {spatial_shape}"
            )
        if spatial_shape[0] % npath != 0:
            raise ValueError(
                f"Spatial dimensions {spatial_shape} not divisible by npath={npath}"
            )

        # Reshape for averaging: (n_pixels, npath, n_pixels, npath, Nt)
        reshaped = expanded_response.reshape(
            n_pixels, npath, n_pixels, npath, expanded_response.shape[2]
        )
        reshaped = np.flip(reshaped, axis=(0, 2)).reshape(
            n_pixels * npath,
            n_pixels * npath,
            expanded_response.shape[2]
        )
        return reshaped

    def process_response(self) -> np.ndarray:
        """Process field response data from NPZ file.

        Loads raw response, expands quadrant to full plane,
        and downsamples by averaging intra-pixel positions.

        Returns:
            Processed response array with shape (2r+1, 2r+1, Nt).
        """
        if self._processed_response is not None:
            return self._processed_response

        data = self._load_data()

        # Validate required keys
        required_keys = ["response", "npath"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys in NPZ file: {missing_keys}")

        # Extract data
        raw_response = data["response"]
        npath = int(data["npath"])

        # Step 1: Expand quadrant to full plane
        expanded_response = self._quadrant_copy(raw_response)

        # Step 2: Flip per pixel
        expanded_response = self._flip_per_pixel(expanded_response, npath)

        # Step 3: Downsample by averaging intra-pixel positions
        self._processed_response = self._downsample_by_averaging(
            expanded_response, npath
        )

        return self._processed_response

    def get_metadata(self) -> dict:
        """Get metadata from NPZ file.

        Returns:
            Dictionary containing metadata like drift_length, bin_size, time_tick, npath.
        """
        data = self._load_data()

        metadata_keys = [
            "drift_length",
            "bin_size",
            "time_tick",
            "npath",
        ]

        metadata = {}
        for key in metadata_keys:
            if key in data:
                metadata[key] = data[key]

        return metadata

    @property
    def response(self) -> Optional[np.ndarray]:
        """Get processed response array."""
        if self._processed_response is None:
            self.process_response()
        return self._processed_response

    def validate_response_normalization(
        self, expected_sum: float = 20.0, tolerance: float = 1e-6
    ) -> bool:
        """Validate response normalization.

        The response should be normalized such that sum over time axis
        for an impact at origin equals expected_sum.

        Args:
            expected_sum: Expected sum value (default: 20.0).
            tolerance: Tolerance for comparison (default: 1e-6).

        Returns:
            True if normalized correctly, False otherwise.
        """
        if self._processed_response is None:
            self.process_response()

        # Get center pixel response (impact at origin)
        center_response = self._processed_response[
            self._processed_response.shape[0] // 2,
            self._processed_response.shape[1] // 2,
            :,
        ]

        time_sum = np.sum(center_response)
        return abs(time_sum - expected_sum) <= tolerance
