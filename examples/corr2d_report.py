#!/usr/bin/env python3
"""N-panel 2-D correlation (smeared truth vs deconvolved charge) figure.

Each panel marks the reco selection cut (horizontal dashed line) and is
annotated with the truth smearing (sigma_t in us, sigma_pxl in pixel pitch).

Usage::

    python examples/corr2d_report.py out1.npz out2.npz ... \
        --labels a b ... --out report/corr2d.png \
        [--truth-npz reference.npz] [--ncols 4] \
        [--sigma-time 0.005 --sigma-pxl 0.2 --tick-us 0.05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muon_filter_report import align_voxel_blocks  # noqa: E402


def correlation_arrays(path: Path, key: str = "deconv_q",
                       truth_npz: Path | None = None):
    f = np.load(path, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True) if truth_npz is not None else f
    smeared_true = np.asarray(t["smeared_true"], dtype=np.float64)
    deconv_q = np.asarray(f[key], dtype=np.float64)
    _, aligned_dq, smear_summed, _ = align_voxel_blocks(
        fine_lower_corner=t["smear_offset"],
        coarse_lower_corner=f["boffset"],
        fine_voxels=smeared_true,
        coarse_voxels=deconv_q,
        bin_size=f["adc_hold_delay"],
    )
    return smear_summed, aligned_dq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default="deconv_q")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Selection cut applied on the RECO axis [ke-].")
    p.add_argument("--truth-npz", default=None,
                   help="Reference NPZ providing smeared_true/smear_offset "
                        "(for lean outputs).")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--sigma-time", type=float, default=0.005,
                   help="Truth temporal smearing sigma [cycles/tick].")
    p.add_argument("--sigma-pxl", type=float, default=0.2,
                   help="Truth pixel smearing sigma [cycles/pixel].")
    p.add_argument("--tick-us", type=float, default=0.05,
                   help="Fine tick length [us].")
    p.add_argument("--hist-max", type=float, default=None,
                   help="Fixed axis maximum [ke-] (e.g. 3.0 to zoom into "
                        "the faint-charge region). Default: 99th percentile.")
    p.add_argument("--nburst", type=int, default=None,
                   help="Readout nburst to display (auto from NPZ if saved).")
    p.add_argument("--threshold-ke", type=float, default=None,
                   help="Trigger threshold [ke-] to display (auto from NPZ).")
    p.add_argument("--group-pixels", type=int, default=1,
                   help="Sum-pool NxN pixel groups before correlating.")
    p.add_argument("--group-time", type=int, default=1,
                   help="Sum-pool N time bins before correlating.")
    args = p.parse_args()
    if len(args.labels) != len(args.files):
        raise SystemExit("--labels count must match files")

    sigma_t_us = 1.0 / (2.0 * np.pi * args.sigma_time) * args.tick_us
    sigma_p_pitch = 1.0 / (2.0 * np.pi * args.sigma_pxl)
    smear_note = (f"truth smearing: $\\sigma_t$ = {sigma_t_us:.2f} $\\mu$s, "
                  f"$\\sigma_{{pxl}}$ = {sigma_p_pitch:.2f} pitch")

    # readout annotation: auto-read from the first NPZ, CLI overrides
    f0 = np.load(args.files[0], allow_pickle=True)
    adc = int(f0["adc_hold_delay"]) if "adc_hold_delay" in f0.files else None
    nburst = args.nburst
    if nburst is None and "readout_nburst" in f0.files:
        nburst = int(f0["readout_nburst"])
    thr = args.threshold_ke
    if thr is None and "readout_threshold" in f0.files:
        thr = float(f0["readout_threshold"])
    ro_parts = []
    if nburst is not None:
        ro_parts.append(f"nburst = {nburst}")
    if adc is not None:
        ro_parts.append(f"adc_hold_delay = {adc} ticks ({adc * args.tick_us:.2f} $\\mu$s)")
    if thr is not None:
        ro_parts.append(f"trigger thr = {thr:g} ke-")
    readout_note = ("  |  " + ", ".join(ro_parts)) if ro_parts else ""

    n = len(args.files)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.4 * ncols, 6.0 * nrows),
                             squeeze=False)
    truth = Path(args.truth_npz) if args.truth_npz else None
    for i, (label, path) in enumerate(zip(args.labels, args.files)):
        ax = axes[i // ncols][i % ncols]
        smear_summed, aligned_dq = correlation_arrays(
            Path(path), args.key, truth_npz=truth)
        if args.group_pixels > 1 or args.group_time > 1:
            from eval_deconv_metrics import pool_block

            smear_summed = pool_block(smear_summed, args.group_pixels,
                                      args.group_time)
            aligned_dq = pool_block(aligned_dq, args.group_pixels,
                                    args.group_time)
        mask = aligned_dq > args.threshold
        x = smear_summed[mask]
        y = aligned_dq[mask]
        if args.hist_max is not None:
            hi = args.hist_max
        else:
            hi = max(1.0, float(np.percentile(np.concatenate([x, y]), 99))) \
                if x.size else 10.0
        _, _, _, img = ax.hist2d(x, y, bins=50, range=[[0, hi], [0, hi]],
                                 norm=LogNorm())
        fig.colorbar(img, ax=ax, label="voxels")
        ax.plot([0, hi], [0, hi], color="white", linestyle="--",
                linewidth=1.0)
        ax.axhline(args.threshold, color="red", linestyle=":", linewidth=1.6)
        ax.text(0.98, args.threshold + 0.02 * hi,
                f"reco cut {args.threshold} ke-",
                color="red", fontsize=9, ha="right", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel("smeared truth [ke-/voxel]")
        ax.set_ylabel("deconvolved charge [ke-/voxel]")
        ax.set_title(f"{label}  (n = {x.size})", fontsize=11)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    group_note = ""
    if args.group_pixels > 1 or args.group_time > 1:
        group_note = (f" — pooled {args.group_pixels}x{args.group_pixels} "
                      f"pixels x {args.group_time} time bin(s)")
    fig.suptitle(
        f"truth vs reco correlation — {smear_note}{group_note}{readout_note}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
