"""Golden regression gate for the refactor (Phase 0).

`golden_metrics.json` pins the headline metrics and output-array
signatures of the two canonical configurations:

- nb4_adopted_centroid_w1: the adopted burst config (nburst >= 2).
- nb1_censorL2_600_centroid_w2: the self-trigger config (nburst = 1;
  censor L2, 600 iters, centroid w2).  Both are zero-suppressed.

`compare_to_golden` checks a fresh result against the pinned values
within tolerances.  Intended use: every refactor phase re-runs the two
configs (any implementation path) and must pass this gate.  INTENDED
behavior changes rebase the JSON in the same commit with the deltas
documented in the commit message.
"""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden_metrics.json"

# metric -> absolute tolerance
TOL = {
    "integral_pct": 0.30,
    "pearson_r": 0.004,
    "slope": 0.010,
    "ghost_frac": 0.006,
    "ghost_adj_frac": 0.006,
    "ghost_iso_frac": 0.002,
    "ghost_iso_charge": 15.0,
    "true_killed": 40.0,
    "n_voxels_gt_thr": 300,
}
SIG_REL_TOL = {"q_sharp_sum": 0.01, "q_sharp_nnz": 0.05, "q_sharp_max": 0.05}


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def compare_to_golden(tag: str, metrics: dict,
                      signatures: dict | None = None) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    g = load_golden()[tag]
    fails = []
    for k, tol in TOL.items():
        got, want = float(metrics[k]), float(g["metrics"][k])
        if abs(got - want) > tol:
            fails.append(f"{tag}.{k}: got {got:.5g}, golden {want:.5g} "
                         f"(tol {tol})")
    if signatures is not None:
        for k, rtol in SIG_REL_TOL.items():
            got, want = float(signatures[k]), float(g["signatures"][k])
            if abs(got - want) > rtol * max(abs(want), 1e-9):
                fails.append(f"{tag}.sig.{k}: got {got:.6g}, golden "
                             f"{want:.6g} (rtol {rtol})")
    return fails
