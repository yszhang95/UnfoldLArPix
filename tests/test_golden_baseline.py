"""Phase-0 golden baseline: JSON sanity + comparator behavior."""
import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent / "golden"))
from golden_gate import GOLDEN_PATH, TOL, compare_to_golden, load_golden


def test_golden_file_exists_and_complete():
    g = load_golden()
    assert set(g) == {"nb4_adopted_centroid_w1", "nb1_censorL2_600_centroid_w2"}
    for tag, entry in g.items():
        assert set(TOL) <= set(entry["metrics"]), tag
        assert {"q_sharp_sum", "q_sharp_nnz", "q_sharp_max"} <= set(
            entry["signatures"]), tag


def test_comparator_passes_on_golden_itself():
    g = load_golden()
    for tag, entry in g.items():
        assert compare_to_golden(tag, entry["metrics"],
                                 entry["signatures"]) == []


def test_comparator_catches_regression():
    g = load_golden()
    tag = "nb4_adopted_centroid_w1"
    bad = dict(g[tag]["metrics"])
    bad["pearson_r"] = bad["pearson_r"] - 0.05   # way outside tolerance
    fails = compare_to_golden(tag, bad)
    assert any("pearson_r" in f for f in fails)
