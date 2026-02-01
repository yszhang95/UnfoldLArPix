"""Data loader for UnfoldLArPix NPZ files."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .data_containers import (
    Current,
    EffectiveCharge,
    EventData,
    Geometry,
    Hits,
    ReadoutConfig,
)


class DataLoader:
    """Loader for NPZ files produced by tred package."""

    def __init__(self, npz_path: Union[str, Path]) -> None:
        """Initialize the DataLoader.

        Args:
            npz_path: Path to the NPZ file.
        """
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        self._data: Optional[Dict[str, np.ndarray]] = None
        self._geometry: Dict[int, Geometry] = {}
        self._readout_config: Optional[ReadoutConfig] = None

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load data from NPZ file."""
        if self._data is None:
            self._data = dict(np.load(self.npz_path, allow_pickle=True))
        return self._data

    def _extract_tpc_id(self, key: str) -> Optional[int]:
        """Extract TPC ID from key name."""
        import re

        match = re.search(r"tpc(\d+)", key)
        return int(match.group(1)) if match else None

    def _extract_batch_id(self, key: str) -> Optional[int]:
        """Extract batch ID from key name."""
        import re

        match = re.search(r"batch(\d+)", key)
        return int(match.group(1)) if match else None

    def _get_data_type(self, key: str) -> Optional[str]:
        """Extract data type from key name."""
        # Skip location arrays
        if key.endswith("_location"):
            return None

        for data_type in ["effq", "current", "hits", "event_id", "global_tref"]:
            if key.startswith(f'{data_type}_tpc'):
                return data_type
        return None

    def _parse_geometry(self) -> Dict[int, Geometry]:
        """Parse TPC geometry from data."""
        if self._geometry:
            return self._geometry

        data = self._load_data()
        tpc_ids = set()

        # Find all TPC IDs
        for key in data.keys():
            tpc_id = self._extract_tpc_id(key)
            if tpc_id is not None:
                tpc_ids.add(tpc_id)

        for tpc_id in tpc_ids:
            try:
                lower = data[f"tpc_lower_left_tpc{tpc_id}"]
                upper = data[f"tpc_upper_tpc{tpc_id}"]
                drift_direction = int(data[f"drift_direction_tpc{tpc_id}"])
                anode_position = float(data[f"tpc_anode_tpc{tpc_id}"])
                cathode_position = float(data[f"tpc_cathode_tpc{tpc_id}"])
                pixel_pitch = float(data[f"pixel_pitch_tpc{tpc_id}"])

                self._geometry[tpc_id] = Geometry(
                    tpc_id=tpc_id,
                    lower=lower,
                    upper=upper,
                    drift_direction=drift_direction,
                    anode_position=anode_position,
                    cathode_position=cathode_position,
                    pixel_pitch=pixel_pitch,
                )
            except KeyError as e:
                raise ValueError(f"Missing geometry data for TPC {tpc_id}: {e}")

        return self._geometry

    def _parse_readout_config(self) -> ReadoutConfig:
        """Parse readout configuration from data."""
        if self._readout_config:
            return self._readout_config

        data = self._load_data()
        required_keys = [
            "time_spacing",
            "adc_hold_delay",
            "adc_down_time",
            "csa_reset_time",
            "one_tick",
            "nburst",
            "threshold",
            "uncorr_noise",
            "thres_noise",
            "reset_noise",
        ]

        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing readout configuration keys: {missing_keys}")

        self._readout_config = ReadoutConfig(
            time_spacing=float(data["time_spacing"]),
            adc_hold_delay=int(data["adc_hold_delay"]),
            adc_down_time=int(data["adc_down_time"]),
            csa_reset_time=int(data["csa_reset_time"]),
            one_tick=int(data["one_tick"]),
            nburst = int(data["nburst"]),
            threshold=float(data["threshold"]),
            uncorr_noise=float(data["uncorr_noise"]) if data["uncorr_noise"] else None,
            thres_noise=float(data["thres_noise"]) if data["thres_noise"] else None,
            reset_noise=float(data["reset_noise"]) if data["reset_noise"] else None,
        )

        return self._readout_config

    def _group_by_event_tpc(
        self,
    ) -> Dict[Tuple[int, Union[int, str]], Dict[str, List[np.ndarray]]]:
        """Group all data by (event_id, tpc_id) pairs,
        assuming single event per batch.
        """
        data = self._load_data()
        grouped: Dict[Tuple[int, Union[int, str]], Dict[str, List[np.ndarray]]] = {}

        for key, array in data.items():
            tpc_id = self._extract_tpc_id(key)
            batch_id = self._extract_batch_id(key)
            data_type = self._get_data_type(key)

            if tpc_id is None or batch_id is None or data_type is None:
                continue

            # Get event ID for this batch
            event_key = f"event_id_tpc{tpc_id}_batch{batch_id}"
            if event_key not in data:
                continue

            event_id = data[event_key]  # Assuming single event per batch
            group_key = (tpc_id, int(event_id))

            if group_key not in grouped:
                grouped[group_key] = {
                    "effq": [],
                    "current": [],
                    "hits": [],
                    "global_tref": [],
                }

            if data_type in ["effq", "current", "hits"]:
                grouped[group_key][data_type].append(array)
            elif data_type == "global_tref":
                grouped[group_key]["global_tref"].append(array)

        return grouped

    def _create_data_containers(
        self,
        group_key: Tuple[int, Union[int, str]],
        group_data: Dict[str, List[np.ndarray]],
    ) -> EventData:
        """Create data containers for a grouped (event_id, tpc_id) pair."""
        tpc_id, event_id = group_key
        event_data = EventData(tpc_id=tpc_id, event_id=event_id)

        # Create EffectiveCharge container
        if group_data["effq"]:
            effq_arrays = group_data["effq"]
            effq_location_arrays = []

            # Find corresponding location arrays
            data = self._load_data()
            for key in data.keys():
                if key.startswith(f"effq_tpc{tpc_id}") and key.endswith("_location"):
                    batch_id = self._extract_batch_id(key)
                    if batch_id is not None:
                        event_key = f"event_id_tpc{tpc_id}_batch{batch_id}"
                        if event_key in data and data[event_key] == event_id:
                            effq_location_arrays.append(data[key])

            if effq_arrays and len(effq_location_arrays) == len(effq_arrays):
                merged_data = np.vstack(effq_arrays)
                merged_location = np.vstack(effq_location_arrays)
                event_data.effq = EffectiveCharge(
                    data=merged_data,
                    location=merged_location,
                    tpc_id=tpc_id,
                    event_id=event_id,
                )

        # Create Current container
        if group_data["current"]:
            current_arrays = group_data["current"]
            current_location_arrays = []

            data = self._load_data()
            for key in data.keys():
                if key.startswith(f"current_tpc{tpc_id}") and key.endswith("_location"):
                    batch_id = self._extract_batch_id(key)
                    if batch_id is not None:
                        event_key = f"event_id_tpc{tpc_id}_batch{batch_id}"
                        if event_key in data and data[event_key] == event_id:
                            current_location_arrays.append(data[key])

            if current_arrays and len(current_location_arrays) == len(current_arrays):
                merged_data = np.concatenate(current_arrays)
                merged_location = np.vstack(current_location_arrays)
                event_data.current = Current(
                    data=merged_data,
                    location=merged_location,
                    tpc_id=tpc_id,
                    event_id=event_id,
                )

        # Create Hits container
        if group_data["hits"]:
            hits_arrays = group_data["hits"]
            hits_location_arrays = []

            data = self._load_data()
            for key in data.keys():
                if key.startswith(f"hits_tpc{tpc_id}") and key.endswith("_location"):
                    batch_id = self._extract_batch_id(key)
                    if batch_id is not None:
                        event_key = f"event_id_tpc{tpc_id}_batch{batch_id}"
                        if event_key in data and data[event_key] == event_id:
                            hits_location_arrays.append(data[key])

            if hits_arrays and len(hits_location_arrays) == len(hits_arrays):
                merged_data = np.vstack(hits_arrays)
                merged_location = np.vstack(hits_location_arrays)
                event_data.hits = Hits(
                    data=merged_data,
                    location=merged_location,
                    tpc_id=tpc_id,
                    event_id=event_id,
                )

        # Set global_tref if available
        if group_data["global_tref"]:
            event_data.global_tref = group_data["global_tref"][
                0
            ]  # Take first reference

        return event_data

    def iter_events(self) -> Iterator[EventData]:
        """Iterate over all (event_id, tpc_id) groups."""
        grouped = self._group_by_event_tpc()

        for group_key, group_data in grouped.items():
            event_data = self._create_data_containers(group_key, group_data)
            if event_data.has_data():
                yield event_data

    def get_geometry(self, tpc_id: int) -> Geometry:
        """Get geometry information for a specific TPC."""
        geometry = self._parse_geometry()
        if tpc_id not in geometry:
            raise ValueError(f"No geometry found for TPC {tpc_id}")
        return geometry[tpc_id]

    def get_readout_config(self) -> ReadoutConfig:
        """Get readout configuration."""
        return self._parse_readout_config()

    def get_all_geometry(self) -> Dict[int, Geometry]:
        """Get geometry information for all TPCs."""
        return self._parse_geometry()

    def list_events(self) -> List[Tuple[int, Union[int, str]]]:
        """List all (tpc_id, event_id) pairs in the file."""
        grouped = self._group_by_event_tpc()
        return list(grouped.keys())
