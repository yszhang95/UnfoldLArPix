# UnfoldLArPix

Signal processing package for LArPix data unfolding.

## Installation


## Development Installation

```bash
git clone <repository-url>
cd UnfoldLArPix
uv venv
uv sync
```

## Usage

```bash
uv run unfoldlarpix
# or try the following
uv run python -m unfoldlarpix
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

