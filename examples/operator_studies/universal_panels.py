"""2D correlation panels straight from the universal-grid voxel files.

Reads the self-describing exports written by universal_export.py and
draws, for each configuration, reco vs truth per universal voxel -- the
same quantity the note's appendix-B panels show, but built from the
final deliverable rather than from a solver-internal representation.
Rows: mu / positron at 0 and 75 degrees; columns: nb1 and nb4; each
panel overlays the stock and the band-limited (A G) solve.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
AO = f"{ROOT}/examples/analysis_output"
VOX = f"{AO}/universal_voxels"
CUT = 0.5
TAGS = [("mu_a00", "muon $0^\\circ$"), ("mu_a75", "muon $75^\\circ$"),
        ("pos_a00", "positron $0^\\circ$"), ("pos_a75", "positron $75^\\circ$")]
NBS = ["nb1", "nb4"]
VARIANTS = [("stock", "#2e6fb7"), ("smeared", "#c2410c")]


def load(tag, variant):
    p = f"{VOX}/{tag}_{variant}_universal.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return (np.asarray(z["charge_truth"], float),
            np.asarray(z["charge_reco"], float),
            json.loads(str(z["meta"])))


def main(out=f"{VOX}/universal_panels"):
    fig, axes = plt.subplots(len(TAGS), len(NBS) * 2,
                             figsize=(19, 4.1 * len(TAGS)))
    for i, (base, lab) in enumerate(TAGS):
        for j, nb in enumerate(NBS):
            for k, (var, col) in enumerate(VARIANTS):
                ax = axes[i, 2 * j + k]
                d = load(f"{base}_{nb}", var)
                if d is None:
                    ax.axis("off")
                    continue
                T, R, meta = d
                m = (R > CUT) | (T > CUT)
                x, y = T[m], R[m]
                hi = float(np.percentile(np.concatenate([x, y]), 99.8))
                ax.hist2d(x, y, bins=70, range=[[0, hi], [0, hi]],
                          cmap="viridis", norm=LogNorm())
                ax.plot([0, hi], [0, hi], "r--", lw=0.8)
                mm = meta["metrics"]
                ax.set_title(f"{lab} {nb} --- {var}", fontsize=10)
                ax.text(0.04, 0.96,
                        f"r = {mm['pearson_r']:.4f}\n"
                        f"slope = {mm['slope']:.3f}\n"
                        f"int = {mm['integral_pct']:+.2f}%\n"
                        f"iso-ghost = {mm['ghost_iso_charge']:.1f} ke\n"
                        f"killed = {mm['true_killed']:.0f} ke",
                        transform=ax.transAxes, va="top", fontsize=7.5,
                        family="monospace",
                        bbox=dict(fc="w", alpha=0.75, ec="none"))
                ax.set_xlabel("smeared truth [ke/voxel]", fontsize=8)
                if 2 * j + k == 0:
                    ax.set_ylabel("reco [ke/voxel]", fontsize=8)
    fig.suptitle("Universal-grid voxel charge: stock vs band-limited "
                 "(A G, 1.6 us / 0.318 px), truth smeared once", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=115, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"-> {out}.png/.pdf", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
