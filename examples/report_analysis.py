#!/usr/bin/env python3
"""Per-dataset analysis for the reset-fix technical report.

For one solved (deconv) NPZ, produce (all with the PHYSICAL reco>cut selection):
  - <tag>_corr2d.png   : reco-vs-truth 2D correlation (reco cut line + smearing)
  - <tag>_event.png    : 3-projection event display (truth grey, reco colored)
  - stats dict (JSON)  : integral bias, pearson r, slope, ghost fractions,
                         and pointwise / 2x2x2 relative RMS of (reco-truth)/reco.
Usage: report_analysis.py SOLVED.npz TAG OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = Path("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix")
sys.path.insert(0, str(REPO / "src"))
from unfoldlarpix.eval.universal import universal_rebin   # noqa

CUT = 0.5


def boxsum(a, size):
    return uniform_filter(a, size=size, mode="constant") * int(np.prod(size))


def analyze(npz, tag, outdir):
    npz = Path(npz); outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    f = np.load(npz, allow_pickle=True)
    B = int(f["adc_hold_delay"]) if "adc_hold_delay" in f.files else 30
    offs = np.asarray(f["deconv_q_offsets"], np.float64) if "deconv_q_offsets" in f.files else None
    truth, reco = universal_rebin(npz, deposit_shape="gaussian", sigma_time=0.005,
                                  sigma_pxl=0.2, time_offsets=offs)
    rc = reco > CUT
    tr = truth > CUT
    both = rc | tr

    # ---- scalar metrics ----
    s_reco, s_truth = float(reco.sum()), float(truth.sum())
    int_pct = 100 * (s_reco - s_truth) / s_truth
    x, y = truth[both], reco[both]
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    # ghosts (reco>cut & truth<cut), split iso/adjacent
    near = np.zeros_like(tr)
    for ax in range(3):
        for sft in (-1, 1):
            near |= np.roll(tr, sft, axis=ax)
    ghost = rc & ~tr
    g_iso = ghost & ~near
    ghost_frac = float(ghost.sum()) / max(rc.sum(), 1)
    giso_q = float(reco[g_iso].sum())
    killed = int((tr & ~rc).sum())
    # pointwise & 2x2x2 relative RMS of (reco-truth)/reco on reco>cut, high-reco
    def relrms(size, himask):
        rs = boxsum(reco, size)[rc]; ts = boxsum(truth, size)[rc]
        fr = (rs - ts) / rs
        return float(fr[himask].std())
    r_orig = reco[rc]
    hi = r_orig >= 8
    rms_pt_all = relrms((1, 1, 1), np.ones_like(r_orig, bool))
    rms_pt_hi = relrms((1, 1, 1), hi)
    rms_2_all = relrms((2, 2, 2), np.ones_like(r_orig, bool))
    rms_2_hi = relrms((2, 2, 2), hi)

    stats = dict(tag=tag, n_reco=int(rc.sum()), sum_reco=s_reco, sum_truth=s_truth,
                 int_pct=int_pct, pearson_r=r, slope=slope, ghost_frac=ghost_frac,
                 ghost_iso_charge=giso_q, true_killed=killed,
                 relrms_pointwise_all=100 * rms_pt_all, relrms_pointwise_hi=100 * rms_pt_hi,
                 relrms_2x2x2_all=100 * rms_2_all, relrms_2x2x2_hi=100 * rms_2_hi)

    # ---- corr2d plot (reco selection) ----
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    top = max(np.percentile(y, 99.8), 5)
    h = ax.hist2d(x, y, bins=80, range=[[0, top], [0, top]], norm=LogNorm(), cmap="viridis")
    ax.plot([0, top], [0, top], "w--", lw=1)
    ax.axhline(CUT, color="red", ls=":", lw=1); ax.text(top * 0.55, CUT + top*0.01, "reco cut 0.5 ke-", color="red", fontsize=8)
    ax.set_xlabel("smeared truth [ke-/voxel]"); ax.set_ylabel("deconv reco [ke-/voxel]")
    sig_t = 0.005 * B * 0.05 * 1e3 / 1e3  # placeholder; use title from data below
    ax.set_title(f"{tag}   r={r:.4f}  slope={slope:.3f}  int={int_pct:+.2f}%")
    ax.set_aspect("equal"); fig.colorbar(h[3], ax=ax, label="voxels", shrink=0.85)
    fig.tight_layout(); fig.savefig(outdir / f"{tag}_corr2d.png", dpi=130); plt.close(fig)

    # ---- event display: 3 projections ----
    # categories: truth underlay (grey), matched reco coloured by charge,
    # ghosts (reco>cut & truth<cut) in a dedicated colour.
    proj = [((0, 1), "pixel_x", "pixel_y"), ((0, 2), "pixel_x", "time bin"),
            ((1, 2), "pixel_y", "time bin")]
    near_t = near | tr      # truth + its neighbourhood
    col = rc & near_t       # reco on/adjacent to truth -> coloured by charge
    iso = rc & ~near_t      # isolated ghost -> dedicated colour
    interest = tr | rc
    xs, ys, ts = np.where(interest)
    lim = [(xs.min(), xs.max()), (ys.min(), ys.max()), (ts.min(), ts.max())]
    VMIN, VMAX = 1.0, 10.0          # cap; <VMIN overflows (under), >VMAX (over)
    cmap = plt.get_cmap("plasma").copy(); cmap.set_under("#2b2b40")
    norm = LogNorm(vmin=VMIN, vmax=VMAX)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ti, tj, tk = np.where(tr)
    ci, cj, ck = np.where(col); cq = reco[col]
    gi, gj, gk = np.where(iso)
    for axp, (pair, xl, yl) in zip(axes, proj):
        a, b = pair
        axp.scatter([ti, tj, tk][a], [ti, tj, tk][b], s=7, c="0.4", alpha=0.6,
                    edgecolors="none", label="truth")
        sc = axp.scatter([ci, cj, ck][a], [ci, cj, ck][b], s=9, c=cq, cmap=cmap,
                         norm=norm, alpha=0.9)
        if gi.size:
            axp.scatter([gi, gj, gk][a], [gi, gj, gk][b], s=44, c="#39ff14",
                        marker="x", linewidths=1.5,
                        label=f"isolated ghost ({iso.sum()})")
        axp.set_xlabel(xl); axp.set_ylabel(yl)
        axp.set_xlim(lim[a][0]-2, lim[a][1]+2); axp.set_ylim(lim[b][0]-2, lim[b][1]+2)
        axp.set_title(f"{xl} vs {yl}")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axes[-1], label="reco charge [ke-]", shrink=0.8, extend="both")
    fig.suptitle(f"{tag} — event display (truth grey, reco coloured, isolated "
                 f"ghost green ×; charge {VMIN:g}-{VMAX:g} ke- capped)", fontsize=12)
    fig.tight_layout(); fig.savefig(outdir / f"{tag}_event.png", dpi=120); plt.close(fig)

    (outdir / f"{tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2], sys.argv[3])
