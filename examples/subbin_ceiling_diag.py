#!/usr/bin/env python3
"""Ceiling diagnostic for a sub-bin position stage (truth-informed).

The solver declares each fitted charge at its bin center; the remaining
"ghosts" are dominated by one-voxel offsets, i.e. real charge rounded
into the neighbouring bin.  Before implementing a per-charge sub-bin
offset fit, this diagnostic measures the CEILING: for every active
charge in ``deconv_q_sharp`` it computes the truth-optimal time offset
(charge-weighted centroid of the same-pixel smeared truth in a local
window, clipped to +-B/2 — the bound the real stage would use), then
re-evaluates the universal-grid Gaussian-deposit metrics with those
offsets.  The gap between the two rows is the maximum a sub-bin stage
could ever buy; the fit itself can only do worse (it has no truth).

Usage::

    python subbin_ceiling_diag.py <solver.npz> --truth-npz <ref.npz> \
        [--window-bins 2] [--json out.json] [--plot out.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deconv_metrics import (  # noqa: E402
    metrics_from_blocks,
    universal_rebin,
)


def truth_optimal_offsets(npz_path: Path, truth_npz: Path,
                          window_bins: float = 2.0,
                          min_truth: float = 0.05):
    """Per-voxel truth-centroid time offsets [ticks], clipped to +-B/2.

    For each active sharp charge, the offset is the charge-weighted
    centroid of the SAME-PIXEL smeared truth within +-window_bins*B of
    the declared center, minus the declared center.  Voxels with no
    same-pixel truth in the window (spatial ghosts) keep offset 0.
    """
    f = np.load(npz_path, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True)
    B = int(f["adc_hold_delay"])
    q_sharp = np.asarray(f["deconv_q_sharp"], dtype=np.float64)
    b_off = np.asarray(f["boffset"], dtype=np.float64)
    smeared = np.asarray(t["smeared_true"], dtype=np.float64)
    s_off = np.asarray(t["smear_offset"], dtype=np.int64)

    offsets = np.zeros_like(q_sharp)
    half_win = window_bins * B
    xs, ys, ks = np.nonzero(q_sharp > 1e-6)
    n_moved = 0
    for x, y, k in zip(xs, ys, ks):
        tx = int(b_off[0]) + x - int(s_off[0])
        ty = int(b_off[1]) + y - int(s_off[1])
        if not (0 <= tx < smeared.shape[0] and 0 <= ty < smeared.shape[1]):
            continue
        c = b_off[2] + (k + 0.5) * B          # declared center [ticks]
        lo = int(np.floor(c - half_win - s_off[2]))
        hi = int(np.ceil(c + half_win - s_off[2])) + 1
        lo, hi = max(lo, 0), min(hi, smeared.shape[2])
        if hi <= lo:
            continue
        w = smeared[tx, ty, lo:hi]
        wsum = float(w.sum())
        if wsum < min_truth:
            continue
        ticks = s_off[2] + np.arange(lo, hi, dtype=np.float64)
        centroid = float((w * ticks).sum() / wsum)
        offsets[x, y, k] = np.clip(centroid - c, -B / 2.0, B / 2.0)
        n_moved += 1
    return offsets, q_sharp, B, n_moved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("solver_npz")
    p.add_argument("--truth-npz", required=True)
    p.add_argument("--window-bins", type=float, default=2.0,
                   help="Half-width [bins] of the truth-centroid window.")
    p.add_argument("--corr-threshold", type=float, default=0.5)
    p.add_argument("--label", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--plot", default=None)
    p.add_argument("--hist-max", type=float, default=15.0)
    args = p.parse_args()

    npz, ref = Path(args.solver_npz), Path(args.truth_npz)
    label = args.label or npz.stem

    offsets, q_sharp, B, n_moved = truth_optimal_offsets(
        npz, ref, window_bins=args.window_bins)
    active = q_sharp > 1e-6
    off_act = offsets[active]
    print(f"{label}: {int(active.sum())} active charges, "
          f"{n_moved} with truth in window; "
          f"|offset| mean {np.abs(off_act).mean():.2f} ticks, "
          f"at bound (+-{B // 2}) {(np.abs(off_act) >= B / 2 - 1e-9).mean() * 100:.1f}%")

    results = {}
    blocks = {}
    for tag, off in (("declared", None), ("truth-optimal", offsets)):
        truth, reco = universal_rebin(
            npz, truth_npz=ref, deposit_shape="gaussian", time_offsets=off)
        m = metrics_from_blocks(truth, reco,
                                corr_threshold=args.corr_threshold)
        results[tag] = m
        blocks[tag] = (truth, reco)
        print(f"  [{tag:>14s}] int {m['integral_pct']:+6.2f}%  "
              f"r {m['pearson_r']:.4f}  slope {m['slope']:.3f}  "
              f"ghost {100 * m['ghost_frac']:5.2f}%  "
              f"(adj {100 * m['ghost_adj_frac']:.2f} / "
              f"iso {100 * m['ghost_iso_frac']:.3f})  "
              f"killed {m['true_killed']:.0f}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({label: results}, fh, indent=2)
        print(f"Wrote {args.json}")

    if args.plot:
        f = np.load(npz, allow_pickle=True)
        nburst = int(f["readout_nburst"]) if "readout_nburst" in f.files else None
        thr = float(f["readout_threshold"]) if "readout_threshold" in f.files else None
        tick_us = 0.05
        sig_t_us = 1.0 / (2.0 * np.pi * 0.005) * tick_us
        note = (f"truth smearing: $\\sigma_t$ = {sig_t_us:.2f} $\\mu$s, "
                f"$\\sigma_{{pxl}}$ = 0.80 pitch")
        if nburst is not None:
            note += (f"  |  nburst = {nburst}, adc_hold_delay = {B} ticks "
                     f"({B * tick_us:.2f} $\\mu$s)")
        if thr is not None:
            note += f", trigger thr = {thr:g} ke-"

        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
        vmax = args.hist_max
        bins = np.linspace(0, vmax, 46)
        for ax, tag in zip(axes[:2], ("declared", "truth-optimal")):
            truth, reco = blocks[tag]
            mask = reco > args.corr_threshold
            h = ax.hist2d(truth[mask], reco[mask], bins=(bins, bins),
                          norm=LogNorm(), cmap="viridis")
            ax.plot([0, vmax], [0, vmax], "w--", lw=0.8, alpha=0.7)
            ax.axhline(args.corr_threshold, color="red", ls=":", lw=1.2)
            ax.text(0.97 * vmax, args.corr_threshold + 0.15,
                    f"reco cut {args.corr_threshold:g} ke-",
                    color="red", ha="right", fontsize=9)
            m = results[tag]
            ax.set_title(f"positions: {tag}  (r {m['pearson_r']:.4f}, "
                         f"ghost {100 * m['ghost_frac']:.2f}%)")
            ax.set_xlabel("smeared truth [ke-/voxel]")
            ax.set_ylabel("deconvolved charge [ke-/voxel]")
            fig.colorbar(h[3], ax=ax, label="voxels")

        ax = axes[2]
        ax.hist(off_act, bins=61, range=(-B / 2, B / 2), color="tab:blue")
        ax.set_xlabel("truth-optimal offset [ticks]")
        ax.set_ylabel("charges")
        ax.set_title(f"per-charge offsets (bound $\\pm${B // 2} ticks = "
                     f"$\\pm$0.5 bin)")
        ax.axvline(0, color="k", lw=0.8)

        fig.suptitle(f"sub-bin position ceiling (truth-informed) — {label}\n"
                     + note, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=140)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
