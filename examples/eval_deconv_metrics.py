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


def pool_block(block: np.ndarray, group_pixels: int = 1,
               group_time: int = 1) -> np.ndarray:
    """Sum-pool a (nx, ny, nt) block over non-overlapping voxel groups."""
    gp, gt = int(group_pixels), int(group_time)
    if gp == 1 and gt == 1:
        return block
    nx, ny, nt = block.shape
    px, py, pt = (-nx) % gp, (-ny) % gp, (-nt) % gt
    b = np.pad(block, ((0, px), (0, py), (0, pt)))
    return b.reshape((nx + px) // gp, gp, (ny + py) // gp, gp,
                     (nt + pt) // gt, gt).sum(axis=(1, 3, 5))


def evaluate(npz_path: Path, corr_threshold: float = 0.5,
             active_threshold_frac: float = 0.10,
             key: str = "deconv_q",
             truth_npz: Path | None = None,
             group_pixels: int = 1,
             group_time: int = 1) -> dict:
    f = np.load(npz_path, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True) if truth_npz is not None else f
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

    sum_dq = float(aligned_dq.sum())
    sum_truth = float(smear_summed.sum())

    # Fig-3 style correlation on voxels above threshold.
    mask = aligned_dq > corr_threshold
    x = smear_summed[mask]
    y = aligned_dq[mask]
    if x.size > 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        pearson_r, slope = float("nan"), float("nan")

    ghost_frac = float((x < corr_threshold).sum() / max(x.size, 1))
    killed_mask = (smear_summed > corr_threshold) & ~mask
    true_killed = float(smear_summed[killed_mask].sum())

    # Ghost decomposition: a ghost voxel adjacent (Chebyshev distance 1) to
    # truth above the cut is a one-voxel offset of the smeared truth
    # (tolerable); an isolated ghost injects unphysical charge (disapproved).
    truth_near = smear_summed > corr_threshold
    for ax in range(truth_near.ndim):
        truth_near = (truth_near
                      | np.roll(truth_near, 1, axis=ax)
                      | np.roll(truth_near, -1, axis=ax))
    ghost_mask = mask & (smear_summed < corr_threshold)
    ghost_adj = ghost_mask & truth_near
    ghost_iso = ghost_mask & ~truth_near
    n_sel = max(int(mask.sum()), 1)
    ghost_adj_frac = float(ghost_adj.sum() / n_sel)
    ghost_iso_frac = float(ghost_iso.sum() / n_sel)
    ghost_iso_charge = float(aligned_dq[ghost_iso].sum())

    # Fig-2 style spectral deviation over active pixels.
    charge = smear_summed.sum(axis=2)
    cmax = float(charge.max())
    xs, ys = np.where(charge > active_threshold_frac * cmax)
    P_true = (np.abs(np.fft.rfft(smear_summed[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    P_dec = (np.abs(np.fft.rfft(aligned_dq[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    safe = P_true > 0
    ratio = P_dec[safe] / P_true[safe]
    spec_dev = float(np.mean(np.abs(ratio - 1.0)))

    return {
        "file": str(npz_path),
        "sum_deconv_q": round(sum_dq, 2),
        "sum_truth": round(sum_truth, 2),
        "integral_pct": round(100.0 * (sum_dq / sum_truth - 1.0), 3),
        "pearson_r": round(pearson_r, 5),
        "slope": round(slope, 5),
        "spec_dev": round(spec_dev, 4),
        "ghost_frac": round(ghost_frac, 5),
        "ghost_adj_frac": round(ghost_adj_frac, 5),
        "ghost_iso_frac": round(ghost_iso_frac, 5),
        "ghost_iso_charge": round(ghost_iso_charge, 2),
        "true_killed": round(true_killed, 2),
        "n_voxels_gt_thr": int(x.size),
        "corr_threshold": corr_threshold,
    }


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
                                  group_time=args.group_time)

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
