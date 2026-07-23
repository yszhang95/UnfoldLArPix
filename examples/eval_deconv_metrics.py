#!/usr/bin/env python3
"""Compact scalar metrics for one or more deconvolution output NPZ files.

For each file, aligns ``smeared_true`` onto the ``deconv_q`` voxel grid
(same convention as ``muon_filter_report.py``) and reports:

- ``sum_deconv_q``, ``sum_truth``: total charge of reconstruction and of the
  aligned/voxel-summed smeared truth [ke-].
- ``integral_pct``: 100 * (sum_deconv_q / sum_truth - 1).
- ``pearson_r``, ``slope``: 2-D correlation stats on voxels with
  ``deconv_q > corr_threshold`` (default 0.5 ke-), as in Fig 3 of the muon
  filter report.
- ``spec_dev``: mean |P_deconv/P_truth - 1| over active pixels (Fig 2 scalar).
- ``ghost_frac``: fraction of voxels with ``deconv_q > corr_threshold`` whose
  aligned truth is below ``corr_threshold``.
- ``true_killed``: total truth charge in voxels where truth > threshold but
  deconv_q <= threshold (missed charge).

Usage::

    python examples/eval_deconv_metrics.py out1.npz [out2.npz ...] \
        [--labels a b ...] [--json metrics.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muon_filter_report import align_voxel_blocks  # noqa: E402
from unfoldlarpix.eval import (metrics_from_blocks,  # noqa: E402,F401
                               pool_block, universal_rebin)




def evaluate(npz_path: Path, corr_threshold: float = 0.5,
             active_threshold_frac: float = 0.10,
             key: str = "deconv_q",
             truth_npz: Path | None = None,
             group_pixels: int = 1,
             group_time: int = 1,
             universal: bool = False,
             content_offset_ticks: float = 0.0,
             deposit_shape: str = "linear",
             use_fitted_offsets: bool = False) -> dict:
    if universal:
        time_offsets = None
        if use_fitted_offsets:
            f0 = np.load(npz_path, allow_pickle=True)
            if "deconv_q_offsets" not in f0.files:
                raise SystemExit(
                    f"{npz_path}: no deconv_q_offsets key "
                    "(run the solver with --subbin-rounds)."
                )
            time_offsets = np.asarray(
                f0["deconv_q_offsets"], dtype=np.float64)
        smear_summed, aligned_dq = universal_rebin(
            npz_path, truth_npz=truth_npz,
            content_offset_ticks=content_offset_ticks,
            deposit_shape=deposit_shape,
            time_offsets=time_offsets,
        )
    else:
        f = np.load(npz_path, allow_pickle=True)
        t = (np.load(truth_npz, allow_pickle=True)
             if truth_npz is not None else f)
        smeared_true = np.asarray(t["smeared_true"], dtype=np.float64)
        smear_offset = t["smear_offset"]
        deconv_q = np.asarray(f[key], dtype=np.float64)
        _, aligned_dq, smear_summed, _ = align_voxel_blocks(
            fine_lower_corner=smear_offset,
            coarse_lower_corner=f["boffset"],
            fine_voxels=smeared_true,
            coarse_voxels=deconv_q,
            bin_size=f["adc_hold_delay"],
        )

    aligned_dq = pool_block(aligned_dq, group_pixels, group_time)
    smear_summed = pool_block(smear_summed, group_pixels, group_time)

    out = metrics_from_blocks(smear_summed, aligned_dq,
                              corr_threshold=corr_threshold,
                              active_threshold_frac=active_threshold_frac)
    out["file"] = str(npz_path)
    return out



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--corr-threshold", type=float, default=0.5)
    p.add_argument("--key", default="deconv_q",
                   help="NPZ key to evaluate (e.g. deconv_q_roi).")
    p.add_argument("--truth-npz", default=None,
                   help="Reference NPZ providing smeared_true/smear_offset "
                        "(for --lean-output runs that omit them).")
    p.add_argument("--group-pixels", type=int, default=1,
                   help="Sum-pool NxN pixel groups before all metrics "
                        "(local charge fidelity at group scale).")
    p.add_argument("--group-time", type=int, default=1,
                   help="Sum-pool N time bins before all metrics.")
    p.add_argument("--universal-grid", action="store_true",
                   help="Reconstruction-INDEPENDENT evaluation: truth and "
                        "reco are each rebinned onto the universal grid "
                        "(edges at global multiples of adc_hold_delay).")
    p.add_argument("--content-offset-ticks", type=float, default=0.0,
                   help="Diagnostic shift of the reco physical centers "
                        "[ticks] in universal mode (declaration scans).")
    p.add_argument("--deposit-shape", choices=("linear", "gaussian"),
                   default="linear",
                   help="Universal-mode reco deposit: linear split of "
                        "coarse content, or Gaussian shapes around the "
                        "regressed means (filter width) from the sharp "
                        "charges — removes the rebinning artifact.")
    p.add_argument("--use-fitted-offsets", action="store_true",
                   help="Universal gaussian mode: deposit each charge at "
                        "its FITTED sub-bin position (deconv_q_offsets "
                        "from --subbin-rounds) instead of the bin center.")
    p.add_argument("--json", default=None, help="Optional output JSON path.")
    args = p.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("--labels count must match number of files")

    results = {}
    truth = Path(args.truth_npz) if args.truth_npz else None
    for label, path in zip(labels, args.files):
        results[label] = evaluate(Path(path), corr_threshold=args.corr_threshold,
                                  key=args.key, truth_npz=truth,
                                  group_pixels=args.group_pixels,
                                  group_time=args.group_time,
                                  universal=args.universal_grid,
                                  content_offset_ticks=args.content_offset_ticks,
                                  deposit_shape=args.deposit_shape,
                                  use_fitted_offsets=args.use_fitted_offsets)

    header = (f"{'label':<28} {'int%':>7} {'r':>8} {'slope':>7} "
              f"{'specdev':>8} {'ghost%':>7} {'gAdj%':>6} {'gIso%':>6} "
              f"{'gIsoQ':>7} {'killed':>8} {'nvox':>7}")
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        print(f"{label:<28} {m['integral_pct']:>7.2f} {m['pearson_r']:>8.4f} "
              f"{m['slope']:>7.3f} {m['spec_dev']:>8.3f} "
              f"{100 * m['ghost_frac']:>7.2f} "
              f"{100 * m['ghost_adj_frac']:>6.2f} "
              f"{100 * m['ghost_iso_frac']:>6.2f} "
              f"{m['ghost_iso_charge']:>7.1f} {m['true_killed']:>8.1f} "
              f"{m['n_voxels_gt_thr']:>7d}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
