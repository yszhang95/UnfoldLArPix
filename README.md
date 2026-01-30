# UnfoldLArPix

Signal processing package for LArPix data unfolding.

## Installation

```bash
# TO BE DONE
```

## Development Installation

```bash
git clone <repository-url>
cd UnfoldLArPix
uv venv
uv sync
```

## Usage

### Data Loading

```python
from unfoldlarpix import DataLoader

# Load NPZ file produced by tred
loader = DataLoader("path/to/data.npz")

# Iterate over events grouped by (event_id, tpc_id)
for event in loader.iter_events():
    print(f"TPC {event.tpc_id}, Event {event.event_id}")

    # Access effective charge data
    if event.effq:
        print(f"  EffQ shape: {event.effq.data.shape}")
        print(f"  EffQ location shape: {event.effq.location.shape}")

    # Access current/waveform data
    if event.current:
        print(f"  Current shape: {event.current.data.shape}")
        print(f"  Current location shape: {event.current.location.shape}")

    # Access hit data
    if event.hits:
        print(f"  Hits shape: {event.hits.data.shape}")
        print(f"  Hits location shape: {event.hits.location.shape}")

# Get geometry information
geometry = loader.get_geometry(0)
print(f"TPC 0 geometry: {geometry.lower} to {geometry.upper}")

# Get readout configuration
config = loader.get_readout_config()
print(f"Time spacing: {config.time_spacing} μs")
```

### Signal Processing

```python
```

## Development

This project follows strict coding standards using Black, isort, Ruff, and mypy.

### Format code
```bash
isort .
black .
```

### Lint and fix
```bash
ruff check . --fix
```

### Type check
```bash
mypy .
```

### Run tests
```bash
pytest
```

## License

