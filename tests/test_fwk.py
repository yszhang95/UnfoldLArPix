"""Phase-3 framework: store semantics, sequence validation, registry."""
import pytest

from unfoldlarpix.fwk.component import ALGORITHMS, Algorithm, algorithm
from unfoldlarpix.fwk.runner import validate_sequence
from unfoldlarpix.fwk.store import EventStore


class TestEventStore:
    def test_write_once(self):
        s = EventStore()
        s.put("x", 1, by="A")
        with pytest.raises(KeyError, match="write-once"):
            s.put("x", 2, by="B")

    def test_missing_key_lists_available(self):
        s = EventStore()
        s.put("a", 1)
        with pytest.raises(KeyError, match="available"):
            s.get("b")

    def test_provenance(self):
        s = EventStore()
        s.put("x", 1, by="Producer")
        assert s.provenance()["x"] == "Producer"


class TestSequenceValidation:
    def _alg(self, name, reads=(), writes=()):
        a = Algorithm()
        a.name, a.reads, a.writes = name, reads, writes
        return a

    def test_valid_chain_passes(self):
        validate_sequence([
            self._alg("A", writes=("x",)),
            self._alg("B", reads=("x",), writes=("y",)),
            self._alg("C", reads=("x", "y")),
        ])

    def test_unsatisfied_read_rejected(self):
        with pytest.raises(ValueError, match="reads .* not produced"):
            validate_sequence([
                self._alg("A", writes=("x",)),
                self._alg("B", reads=("z",)),
            ])


class TestRegistry:
    def test_production_algorithms_registered(self):
        import unfoldlarpix.algs  # noqa: F401
        for name in ("LoadEvent", "FFTWarmStart", "BuildMeasurement",
                     "BuildSupport", "Solve", "CentroidPositions",
                     "WriteCharges"):
            assert name in ALGORITHMS, name

    def test_decorator_registers(self):
        @algorithm("_TestAlg")
        class _TestAlg(Algorithm):
            pass
        assert ALGORITHMS["_TestAlg"] is _TestAlg
        del ALGORITHMS["_TestAlg"]
