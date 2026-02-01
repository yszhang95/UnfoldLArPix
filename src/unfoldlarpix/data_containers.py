"""Data container classes for UnfoldLArPix."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DataContainer:
    """Base container for data with associated location information."""

    data: np.ndarray
    location: np.ndarray
    tpc_id: int
    event_id: Union[int, str]

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if not isinstance(self.location, np.ndarray):
            raise TypeError("location must be a numpy array")
        if len(self.data) != len(self.location):
            raise ValueError(
                "data and location must have the same number of entries"
            )
    def __len__(self) -> int:
        """Return number of entries in the data container."""
        return len(self.data)

    @property
    def nbatches(self) -> int:
        return len(self)

@dataclass
class EffectiveCharge(DataContainer):
    """Container for effective charge (EffQ) data."""

    def __post_init__(self) -> None:
        """Validate EffQ-specific constraints."""
        super().__post_init__()
        if self.data.shape[1] != 4:
            raise ValueError(f"EffQ data must have shape (N, 4), got {self.data.shape}")
        if self.location.shape[1] != 3:
            raise ValueError(
                f"EffQ location must have shape (N, 3), got {self.location.shape}"
            )


@dataclass
class Current(DataContainer):
    """Container for current/waveform data."""

    def __post_init__(self) -> None:
        """Validate current-specific constraints."""
        super().__post_init__()
        if self.location.shape[1] != 3:
            raise ValueError(
                f"Current location must have shape (N, 3), got {self.location.shape}"
            )


@dataclass
class Hits(DataContainer):
    """Container for hit data."""

    def __post_init__(self) -> None:
        """Validate hit-specific constraints."""
        super().__post_init__()
        if self.data.shape[1] < 4:
            raise ValueError(
                f"Hits data must have at least 4 columns, got {self.data.shape[1]}"
            )
        if self.location.shape[1] != 5:
            raise ValueError(
                f"Hits location must have shape (N, 5), got {self.location.shape}"
            )


@dataclass
class EventData:
    """Container for all data types associated with a single (event_id, tpc_id)."""

    tpc_id: int
    event_id: Union[int, str]
    effq: Optional[EffectiveCharge] = None
    current: Optional[Current] = None
    hits: Optional[Hits] = None
    global_tref: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def has_data(self) -> bool:
        """Check if any data containers are present."""
        return any(
            container is not None for container in [self.effq, self.current, self.hits]
        )

    def get_data_types(self) -> List[str]:
        """Get list of available data types."""
        types = []
        if self.effq is not None:
            types.append("effq")
        if self.current is not None:
            types.append("current")
        if self.hits is not None:
            types.append("hits")
        return types


@dataclass
class Geometry:
    """Container for TPC geometry information."""

    tpc_id: int
    lower: np.ndarray
    upper: np.ndarray
    drift_direction: int
    anode_position: float
    cathode_position: float
    pixel_pitch: float

    def __post_init__(self) -> None:
        """Validate geometry data."""
        if self.drift_direction not in [1, -1]:
            raise ValueError("drift_direction must be 1 or -1")


@dataclass
class ReadoutConfig:
    """Container for readout electronics configuration."""

    time_spacing: float
    adc_hold_delay: int
    adc_down_time: int
    csa_reset_time: int
    one_tick: int
    nburst: int
    threshold: float
    uncorr_noise: float | None
    thres_noise: float | None
    reset_noise: float | None
