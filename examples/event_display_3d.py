#!/usr/bin/env python3
"""Standalone 3D event display: smeared truth vs deconvolved charge.

Produces a self-contained interactive HTML (plotly; traces toggleable via
the legend) and a static projections PNG.  Voxels above the selection cut
are drawn at (pixel_x, pixel_y, time [us]) with charge as color; for each
reconstruction a 'ghosts' overlay (reco above cut, truth below) can be
switched on in the legend.

Usage::

    python examples/event_display_3d.py fit1.npz [fit2.npz ...] \
        --labels a [b ...] --truth-npz reference.npz \
        --out display.html [--threshold 0.5] [--tick-us 0.05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muon_filter_report import align_voxel_blocks  # noqa: E402


def load_grid(path: Path, truth_npz: Path | None):
    f = np.load(path, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True) if truth_npz is not None else f
    smeared_true = np.asarray(t["smeared_true"], dtype=np.float64)
    deconv_q = np.asarray(f["deconv_q"], dtype=np.float64)
    _, aligned_dq, smear_summed, target_lower = align_voxel_blocks(
        fine_lower_corner=t["smear_offset"],
        coarse_lower_corner=f["boffset"],
        fine_voxels=smeared_true,
        coarse_voxels=deconv_q,
        bin_size=f["adc_hold_delay"],
    )
    return smear_summed, aligned_dq, np.asarray(target_lower), int(f["adc_hold_delay"])


def voxels(block, lower, bin_size, tick_us, cut):
    xs, ys, ts = np.where(block > cut)
    q = block[xs, ys, ts]
    x = lower[0] + xs
    y = lower[1] + ys
    t_us = (lower[2] + ts * bin_size) * tick_us
    return x, y, t_us, q


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--truth-npz", default=None)
    p.add_argument("--out", required=True, help="Output HTML path.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tick-us", type=float, default=0.05)
    args = p.parse_args()
    if len(args.labels) != len(args.files):
        raise SystemExit("--labels count must match files")

    truth = Path(args.truth_npz) if args.truth_npz else Path(args.files[0])
    cut = args.threshold

    fig = go.Figure()
    truth_added = False
    colors = ["dodgerblue", "mediumseagreen", "orange", "violet"]
    proj_data = []

    for i, (label, path) in enumerate(zip(args.labels, args.files)):
        smear_summed, aligned_dq, lower, B = load_grid(Path(path), truth)
        if not truth_added:
            x, y, t, q = voxels(smear_summed, lower, B, args.tick_us, cut)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=t, mode="markers", name="smeared truth",
                marker=dict(size=2.2, color=q, colorscale="Greys",
                            cmin=0, cmax=float(np.percentile(q, 98)),
                            showscale=False),
                hovertemplate="truth %{marker.color:.2f} ke-<extra></extra>",
            ))
            truth_added = True
            proj_data.append(("smeared truth", smear_summed, lower, B))
        x, y, t, q = voxels(aligned_dq, lower, B, args.tick_us, cut)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=t, mode="markers", name=label,
            visible=True if i == 0 else "legendonly",
            marker=dict(size=2.2, color=q, colorscale="Viridis",
                        cmin=0, cmax=float(np.percentile(q, 98)),
                        showscale=False),
            hovertemplate=label + " %{marker.color:.2f} ke-<extra></extra>",
        ))
        ghost_mask = (aligned_dq > cut) & (smear_summed < cut)
        gx, gy, gt = np.where(ghost_mask)
        fig.add_trace(go.Scatter3d(
            x=lower[0] + gx, y=lower[1] + gy,
            z=(lower[2] + gt * B) * args.tick_us,
            mode="markers", name=f"{label} ghosts",
            visible="legendonly",
            marker=dict(size=2.6, color="red"),
            hovertemplate="ghost<extra></extra>",
        ))
        proj_data.append((label, aligned_dq, lower, B))

    fig.update_layout(
        scene=dict(xaxis_title="pixel x", yaxis_title="pixel y",
                   zaxis_title="time [us]", aspectmode="data"),
        legend=dict(itemsizing="constant"),
        title=(f"event display — voxels > {cut} ke-  "
               "(click legend entries to toggle; red = reco w/o truth)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True)
    print(f"Saved {out}")

    # ---- static projections PNG (truth row + one row per reco) ----
    n_rows = len(proj_data)
    fig2, axes = plt.subplots(n_rows, 3, figsize=(16, 4.4 * n_rows),
                              squeeze=False)
    proj_specs = [((2,), "pixel x", "pixel y", 0, 1),
                  ((1,), "pixel x", "time bin", 0, 2),
                  ((0,), "pixel y", "time bin", 1, 2)]
    for r, (label, block, lower, B) in enumerate(proj_data):
        masked = np.where(block > cut, block, 0.0)
        for c, (sum_ax, xl, yl, ax_i, ax_j) in enumerate(proj_specs):
            ax = axes[r][c]
            img = masked.sum(axis=sum_ax[0])
            img = np.where(img > 0, img, np.nan)
            extent = None
            ax.imshow(img.T, origin="lower", aspect="auto",
                      interpolation="nearest", cmap="viridis")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{label} — {xl} vs {yl}", fontsize=10)
    fig2.suptitle(f"projections of voxels > {cut} ke-", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.98))
    png = out.with_suffix(".png")
    fig2.savefig(png, dpi=130, bbox_inches="tight")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
