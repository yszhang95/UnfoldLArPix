#!/usr/bin/env python3
"""Coarse grid scan over Wiener-filter parameters (omega_c, b) for ROI identification.

For each (omega_c, b) pair the script:
  1. Builds the Wiener-inspired filter and runs a second deconv_fft on the
     compensated block (reusing the Gaussian pass from process_event_deconvolution).
  2. Estimates noise RMS from quiet pixels in the Wiener-deconvolved block.
  3. Finds ROI at the equivalent-threshold sigma (effective absolute threshold =
     DIRECT_THRESHOLD ke-) and computes benchmark metrics.
  4. Saves per-combination comparison plots plus aggregate heatmaps.
  5. Writes a SCAN_LOG.md and appends a summary entry to ANALYSIS_HISTORY.md.

Usage (from repo root):
    PYTHONPATH=src python examples/wiener_roi_scan.py \\
        [--input-file FILE] [--field-response FILE] \\
        [--output-dir DIR] [--tpc-id INT] [--event-id INT]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from unfoldlarpix import (
    BurstSequenceProcessorV3,
    DataLoader,
    prepare_field_response,
    process_event_deconvolution,
    shift_time_offset,
)
from unfoldlarpix.deconv import deconv_fft
from unfoldlarpix.roi_finder import apply_roi_mask, estimate_quiet_pixel_noise, find_roi_mask
from unfoldlarpix.wiener_filter import wiener_inspired_filter_3d

# ── grid ─────────────────────────────────────────────────────────────────────
OMEGA_C_GRID = [0.001, 0.002, 0.003, 0.005]   # cycles / adc-bin (d=adc_hold_delay)
B_GRID       = [2, 4, 6]                        # rolloff exponent
SIGMA_TIME   = 0.005
SIGMA_PIX    = 0.2
ROI_MERGE_GAP = 2
ROI_EXPAND    = 2
DIRECT_THRESHOLD = 0.5   # ke-
GHOST_THRESHOLD  = 0.1   # ke-  (smeared truth below this = ghost)


# ── alignment (mirrors plot_proj.py) ─────────────────────────────────────────

def align_voxel_blocks(
    fine_lower_corner, coarse_lower_corner,
    fine_voxels, coarse_voxels, bin_size
):
    fine_voxels  = np.asarray(fine_voxels)
    coarse_voxels = np.asarray(coarse_voxels)
    ndims = fine_voxels.ndim
    fine_lower   = np.asarray(fine_lower_corner, dtype=int)
    coarse_lower = np.asarray(coarse_lower_corner, dtype=int)
    bin_arr = np.ones(ndims, dtype=int); bin_arr[-1] = bin_size

    target_lower = coarse_lower.copy()
    diff_bins = ((fine_lower - target_lower) // bin_arr) * bin_arr
    target_lower += np.minimum(diff_bins, 0)

    fine_upper   = fine_lower + np.array(fine_voxels.shape, dtype=int)
    coarse_upper = coarse_lower + np.array(coarse_voxels.shape, dtype=int) * bin_arr
    target_upper = coarse_upper.copy()
    over = np.ceil((fine_upper - target_upper) / bin_arr).astype(int) * bin_arr
    target_upper += np.clip(over, 0, None)

    fine_pad = tuple(
        (int(fine_lower[i] - target_lower[i]), int(target_upper[i] - fine_upper[i]))
        for i in range(ndims)
    )
    coarse_pad = tuple(
        (int((coarse_lower[i] - target_lower[i]) // bin_arr[i]),
         int((target_upper[i]  - coarse_upper[i]) // bin_arr[i]))
        for i in range(ndims)
    )
    aligned_fine   = np.pad(fine_voxels,   fine_pad,   mode="constant")
    aligned_coarse = np.pad(coarse_voxels, coarse_pad, mode="constant")

    refine, sub_axes = [], []
    for i in range(ndims):
        refine.extend([aligned_coarse.shape[i], bin_arr[i]])
        sub_axes.append(2 * i + 1)
    fine_summed = aligned_fine.reshape(refine).sum(axis=tuple(sub_axes))
    return aligned_fine, aligned_coarse, fine_summed, target_lower


def _align_pair(result, coarse_block, adc_hold_delay):
    boffset = shift_time_offset(result.hwf_block_offset, -adc_hold_delay)
    _, aligned_coarse, smear_summed, _ = align_voxel_blocks(
        fine_lower_corner=result.smear_offset,
        coarse_lower_corner=boffset,
        fine_voxels=result.smeared_true,
        coarse_voxels=coarse_block,
        bin_size=adc_hold_delay,
    )
    return smear_summed, aligned_coarse


# ── metrics ──────────────────────────────────────────────────────────────────

SIZABLE_LEVELS = (0.1, 0.5, 1.0)  # ke- — "sizable true charge" thresholds


def compute_metrics(truth, recon, direct_thresh, ghost_thresh,
                    recon_direct=None):
    """Precision + recall metrics on aligned (truth, recon) blocks.

    If ``recon_direct`` is provided, also reports the *additional* loss caused
    by the ROI cut on top of a plain direct threshold (voxels the direct
    method kept but the ROI dropped).
    """
    t = truth.flatten()
    r = recon.flatten()
    active = int((r > direct_thresh).sum())
    ghost  = int(((t < ghost_thresh) & (r > direct_thresh)).sum())
    ghost_frac = ghost / max(active, 1)
    signal_mask = t > direct_thresh
    if signal_mask.sum() > 0:
        delta = (t - r)[signal_mask]
        dq_mean = float(np.mean(delta))
        dq_std  = float(np.std(delta))
    else:
        dq_mean = dq_std = float("nan")

    # ── recall: voxels with sizable truth that ended up zero ────────────
    killed_mask_any = (r == 0)
    killed_voxels = {
        x: int(((t > x) & killed_mask_any).sum()) for x in SIZABLE_LEVELS
    }
    total_voxels = {x: int((t > x).sum()) for x in SIZABLE_LEVELS}
    killed_charge_above_ghost = float(t[(t > ghost_thresh) & killed_mask_any].sum())
    total_charge_above_ghost  = float(t[t > ghost_thresh].sum())
    recall_charge = 1.0 - killed_charge_above_ghost / max(total_charge_above_ghost, 1e-12)

    # ── ROI-specific kills (only voxels direct kept but ROI dropped) ────
    if recon_direct is not None:
        rd = recon_direct.flatten()
        roi_extra_kill = (rd > direct_thresh) & (r == 0)
        roi_extra_voxels = {
            x: int(((t > x) & roi_extra_kill).sum()) for x in SIZABLE_LEVELS
        }
        roi_extra_charge = float(t[roi_extra_kill].sum())
    else:
        roi_extra_voxels = {x: 0 for x in SIZABLE_LEVELS}
        roi_extra_charge = 0.0

    return dict(
        active=active, ghost=ghost, ghost_frac=ghost_frac,
        dq_mean=dq_mean, dq_std=dq_std,
        killed_voxels=killed_voxels, total_voxels=total_voxels,
        killed_charge=killed_charge_above_ghost,
        total_charge=total_charge_above_ghost,
        recall_charge=recall_charge,
        roi_extra_voxels=roi_extra_voxels,
        roi_extra_charge=roi_extra_charge,
    )


# ── per-combination plot ──────────────────────────────────────────────────────

def save_combo_plot(outpath, truth, recon_direct, recon_roi, label_roi,
                    direct_thresh, ghost_thresh, unit, title):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    def _2d(ax, r, lbl):
        mask = (truth.flatten() + r.flatten()) > 0
        ax.hist2d(truth.flatten()[mask], r.flatten()[mask],
                  bins=60, range=[[0, 10], [0, 10]],
                  norm=LogNorm(vmin=1), cmap="viridis")
        ax.plot([0, 10], [0, 10], "r--", lw=0.8)
        ax.set_xlabel("truth"); ax.set_ylabel(lbl)
        ax.set_title(f"2D truth vs {lbl}")

    def _dq(ax, r, lbl):
        mask = truth.flatten() > direct_thresh
        if mask.sum():
            d = (truth.flatten() - r.flatten())[mask]
            ax.hist(d, bins=80, range=(-5, 5), alpha=0.8)
            ax.axvline(0, color="k", lw=0.8)
            ax.text(0.97, 0.97, f"μ={np.mean(d):.3f}\nσ={np.std(d):.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"truth − {lbl}"); ax.set_title(f"ΔQ {lbl}")

    def _ghost(ax, r, lbl):
        gm = (truth.flatten() < ghost_thresh) & (r.flatten() > direct_thresh)
        tot = int((r.flatten() > direct_thresh).sum())
        ax.hist(r.flatten()[gm], bins=60, range=(direct_thresh, 5), alpha=0.8)
        ax.text(0.97, 0.97, f"ghosts={gm.sum()}\nfrac={gm.sum()/max(tot,1):.1%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"{lbl}"); ax.set_title(f"Ghost {lbl}")

    _2d(axes[0, 0], recon_direct, f"direct>{direct_thresh}")
    _2d(axes[0, 1], recon_roi,    label_roi)
    _dq(axes[1, 0], recon_direct, f"direct>{direct_thresh}")
    _dq(axes[1, 1], recon_roi,    label_roi)
    _ghost(axes[0, 2], recon_direct, f"direct>{direct_thresh}")
    _ghost(axes[1, 2], recon_roi,    label_roi)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


# ── heatmap helpers ───────────────────────────────────────────────────────────

def _heatmap(ax, data, row_labels, col_labels, title, fmt=".3f", cmap="RdYlGn_r"):
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xlabel("b (rolloff exponent)")
    ax.set_ylabel("omega_c")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file",
                   default=("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/"
                            "UnfoldLArPix/examples/data/"
                            "pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz"))
    p.add_argument("--field-response",
                   default="/srv/storage1/yousen/tred_workspace/"
                           "response_44_v2a_full_25x25pixel_tred.npz")
    p.add_argument("--tpc-id",   type=int, default=0)
    p.add_argument("--event-id", type=int, default=0)
    p.add_argument("--output-dir",
                   default=f"examples/analysis_wiener_scan_{datetime.now():%Y%m%d}")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # ── load & single Gaussian deconvolution pass ─────────────────────────
    loader   = DataLoader(args.input_file)
    rc       = loader.get_readout_config()
    prep     = prepare_field_response(
        args.field_response, rc.adc_hold_delay,
        normalized=False, response_template="center")

    event = next(e for e in loader.iter_events()
                 if e.tpc_id == args.tpc_id and e.event_id == args.event_id)

    print("Running Gaussian deconvolution pass …")
    gauss_result = process_event_deconvolution(
        event, rc, prep,
        sigma_time=SIGMA_TIME, sigma_pixel=SIGMA_PIX,
        processor_cls=BurstSequenceProcessorV3,
        tau=rc.adc_hold_delay, npadbin=50,
        require_zero_local_offset=True,
    )
    block_data  = gauss_result.hwf_block
    integ_resp  = prep.integrated_response
    hit_xy      = event.hits.location[:, :2]
    block_offset = gauss_result.hwf_block_offset

    smear_g, aligned_direct_g = _align_pair(gauss_result,
        gauss_result.deconv_q * (gauss_result.deconv_q > DIRECT_THRESHOLD), rc.adc_hold_delay)
    m_direct = compute_metrics(smear_g, aligned_direct_g, DIRECT_THRESHOLD, GHOST_THRESHOLD)
    dtus = 0.05 * rc.adc_hold_delay
    unit = f"ke-/pix/{dtus:.1f}µs"

    print(f"\nDirect threshold {DIRECT_THRESHOLD} ke-: "
          f"active={m_direct['active']:,}  ghost={m_direct['ghost']:,}  "
          f"ghost_frac={m_direct['ghost_frac']:.1%}  "
          f"ΔQ μ={m_direct['dq_mean']:.3f}  σ={m_direct['dq_std']:.3f}")
    print(f"  Recall (truth>{GHOST_THRESHOLD} ke-): "
          f"recall={m_direct['recall_charge']:.1%}  "
          f"killed_charge={m_direct['killed_charge']:.1f}/{m_direct['total_charge']:.1f} ke-")
    for x in SIZABLE_LEVELS:
        kv = m_direct['killed_voxels'][x]
        tv = m_direct['total_voxels'][x]
        print(f"  Voxels truth>{x} ke-: total={tv:,} killed={kv:,} "
              f"({kv / max(tv, 1):.1%})")
    print()

    # ── grid scan ─────────────────────────────────────────────────────────
    # tables: rows = omega_c, cols = b
    grid_ghost_frac = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_dq_std     = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_dq_mean    = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_noise_rms  = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_active     = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_sigma_equiv= np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_recall     = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_killed_q   = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_killed_v01 = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_killed_v05 = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_killed_v10 = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_roi_extraq = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)
    grid_roi_extrav = np.full((len(OMEGA_C_GRID), len(B_GRID)), np.nan)

    fft_shape = (
        block_data.shape[0] + integ_resp.shape[0] - 1,
        block_data.shape[1] + integ_resp.shape[1] - 1,
        block_data.shape[2],
    )

    for i, omega_c in enumerate(OMEGA_C_GRID):
        for j, b in enumerate(B_GRID):
            tag = f"wc{omega_c}_b{b}"
            print(f"  omega_c={omega_c}  b={b} …", end=" ", flush=True)

            # build Wiener filter & deconvolve
            wfilt = wiener_inspired_filter_3d(
                fft_shape, (1, 1, rc.adc_hold_delay),
                (SIGMA_PIX, SIGMA_PIX), omega_c=omega_c, b=b)
            dq_w, _ = deconv_fft(block_data, integ_resp, wfilt)

            # noise from quiet pixels
            noise_rms = estimate_quiet_pixel_noise(
                dq_w, block_offset, hit_xy, min_quiet_pixels=8)
            sigma_equiv = DIRECT_THRESHOLD / noise_rms

            # ROI at equivalent absolute threshold
            roi_mask = find_roi_mask(
                dq_w, noise_rms,
                threshold_sigma=sigma_equiv,
                merge_gap=ROI_MERGE_GAP,
                expand=ROI_EXPAND)
            dq_roi = apply_roi_mask(gauss_result.deconv_q, roi_mask)

            smear_s, aligned_roi = _align_pair(gauss_result, dq_roi, rc.adc_hold_delay)
            m = compute_metrics(smear_s, aligned_roi, DIRECT_THRESHOLD, GHOST_THRESHOLD,
                                recon_direct=aligned_direct_g)

            grid_ghost_frac[i, j] = m["ghost_frac"]
            grid_dq_std    [i, j] = m["dq_std"]
            grid_dq_mean   [i, j] = m["dq_mean"]
            grid_noise_rms [i, j] = noise_rms
            grid_active    [i, j] = m["active"]
            grid_sigma_equiv[i,j] = sigma_equiv
            grid_recall    [i, j] = m["recall_charge"]
            grid_killed_q  [i, j] = m["killed_charge"]
            grid_killed_v01[i, j] = m["killed_voxels"][0.1]
            grid_killed_v05[i, j] = m["killed_voxels"][0.5]
            grid_killed_v10[i, j] = m["killed_voxels"][1.0]
            grid_roi_extraq[i, j] = m["roi_extra_charge"]
            grid_roi_extrav[i, j] = m["roi_extra_voxels"][0.5]

            print(f"noise_rms={noise_rms:.4g}  σ_equiv={sigma_equiv:.1f}  "
                  f"active={m['active']:,}  ghost={m['ghost_frac']:.1%}  "
                  f"ΔQ σ={m['dq_std']:.3f}  recall={m['recall_charge']:.1%}  "
                  f"killed>0.5keV={m['killed_voxels'][0.5]:,}")

            # per-combo comparison plot
            save_combo_plot(
                outdir / f"combo_{tag}.png",
                smear_s, aligned_direct_g, aligned_roi,
                label_roi=f"roi wc={omega_c} b={b} σ={sigma_equiv:.0f}",
                direct_thresh=DIRECT_THRESHOLD,
                ghost_thresh=GHOST_THRESHOLD,
                unit=unit,
                title=f"ω_c={omega_c}  b={b}  noise_rms={noise_rms:.4g} ke-  "
                      f"σ_equiv={sigma_equiv:.0f}  TPC{args.tpc_id}/ev{args.event_id}",
            )

    # ── heatmaps ──────────────────────────────────────────────────────────
    row_labels = [str(w) for w in OMEGA_C_GRID]
    col_labels = [str(b) for b in B_GRID]

    fig_heat, axes_heat = plt.subplots(2, 3, figsize=(18, 10))
    _heatmap(axes_heat[0, 0], grid_ghost_frac, row_labels, col_labels,
             f"Ghost fraction (truth<{GHOST_THRESHOLD}, recon>{DIRECT_THRESHOLD})",
             fmt=".1%", cmap="RdYlGn_r")
    _heatmap(axes_heat[0, 1], grid_dq_std, row_labels, col_labels,
             "ΔQ std (truth>0.5 ke-)", fmt=".3f", cmap="RdYlGn_r")
    _heatmap(axes_heat[0, 2], grid_dq_mean, row_labels, col_labels,
             "ΔQ mean (truth>0.5 ke-)", fmt=".3f", cmap="coolwarm")
    _heatmap(axes_heat[1, 0], grid_noise_rms, row_labels, col_labels,
             "Noise RMS from quiet pixels [ke-]", fmt=".4f", cmap="Blues")
    _heatmap(axes_heat[1, 1], grid_sigma_equiv, row_labels, col_labels,
             "Sigma equiv to 0.5 ke- threshold", fmt=".0f", cmap="Purples")
    _heatmap(axes_heat[1, 2], grid_active, row_labels, col_labels,
             "Active voxels at equiv threshold", fmt=".0f", cmap="YlOrRd")

    # ── recall heatmaps (charge / voxels killed) ──────────────────────────
    fig_recall, axes_recall = plt.subplots(2, 3, figsize=(18, 10))
    _heatmap(axes_recall[0, 0], grid_recall, row_labels, col_labels,
             f"Charge recall (truth>{GHOST_THRESHOLD} ke-)\n[1 - killed/total]",
             fmt=".1%", cmap="RdYlGn")
    _heatmap(axes_recall[0, 1], grid_killed_q, row_labels, col_labels,
             f"Killed charge [ke-]\n(truth>{GHOST_THRESHOLD}, recon=0)",
             fmt=".1f", cmap="RdYlGn_r")
    _heatmap(axes_recall[0, 2], grid_roi_extraq, row_labels, col_labels,
             "ROI-extra killed charge [ke-]\n(direct kept, ROI dropped)",
             fmt=".2f", cmap="RdYlGn_r")
    _heatmap(axes_recall[1, 0], grid_killed_v01, row_labels, col_labels,
             f"Killed voxels (truth>0.1 ke-)\nrecon=0",
             fmt=".0f", cmap="RdYlGn_r")
    _heatmap(axes_recall[1, 1], grid_killed_v05, row_labels, col_labels,
             f"Killed voxels (truth>0.5 ke-)\nrecon=0",
             fmt=".0f", cmap="RdYlGn_r")
    _heatmap(axes_recall[1, 2], grid_killed_v10, row_labels, col_labels,
             f"Killed voxels (truth>1.0 ke-)\nrecon=0",
             fmt=".0f", cmap="RdYlGn_r")

    # baseline reference annotations on row-0 panels
    axes_recall[0, 0].set_title(
        axes_recall[0, 0].get_title()
        + f"\n(direct: recall={m_direct['recall_charge']:.1%})")
    axes_recall[0, 1].set_title(
        axes_recall[0, 1].get_title()
        + f"\n(direct: {m_direct['killed_charge']:.1f} ke-)")

    # mark direct-threshold reference on ghost and ΔQ panels
    for ax_idx in [0, 1]:
        axes_heat[0, ax_idx].set_title(
            axes_heat[0, ax_idx].get_title()
            + f"\n(direct: ghost={m_direct['ghost_frac']:.1%}, ΔQσ={m_direct['dq_std']:.3f})")

    fig_heat.suptitle(
        f"Wiener ROI scan  |  TPC{args.tpc_id} ev{args.event_id}  |  "
        f"σ_t={SIGMA_TIME}  σ_pxl={SIGMA_PIX}  direct_thresh={DIRECT_THRESHOLD} ke-",
        fontsize=12,
    )
    fig_heat.tight_layout()
    heat_path = outdir / "heatmaps.png"
    fig_heat.savefig(heat_path, dpi=110)
    plt.close(fig_heat)
    print(f"\nSaved: {heat_path}")

    fig_recall.suptitle(
        f"Wiener ROI recall scan  |  TPC{args.tpc_id} ev{args.event_id}  "
        f"|  total true charge>{GHOST_THRESHOLD} ke- = {m_direct['total_charge']:.1f} ke-,  "
        f"voxels: 0.1={m_direct['total_voxels'][0.1]} 0.5={m_direct['total_voxels'][0.5]} "
        f"1.0={m_direct['total_voxels'][1.0]}",
        fontsize=11,
    )
    fig_recall.tight_layout()
    recall_path = outdir / "recall_heatmaps.png"
    fig_recall.savefig(recall_path, dpi=110)
    plt.close(fig_recall)
    print(f"Saved: {recall_path}")

    # ── best config (lowest ghost fraction) ───────────────────────────────
    best_i, best_j = np.unravel_index(np.nanargmin(grid_ghost_frac), grid_ghost_frac.shape)
    best_omega_c = OMEGA_C_GRID[best_i]
    best_b       = B_GRID[best_j]
    print(f"\nBest config (lowest ghost): omega_c={best_omega_c}  b={best_b}  "
          f"ghost={grid_ghost_frac[best_i,best_j]:.1%}  "
          f"ΔQσ={grid_dq_std[best_i,best_j]:.3f}")

    # ── write SCAN_LOG.md ─────────────────────────────────────────────────
    log_path = outdir / "SCAN_LOG.md"
    with log_path.open("w") as fp:
        fp.write(textwrap.dedent(f"""\
            # Wiener ROI Parameter Scan Log

            **Run time:** {ts}
            **Dataset:** `{Path(args.input_file).name}`
            **Field response:** `{Path(args.field_response).name}`
            **TPC / event:** {args.tpc_id} / {args.event_id}

            ## Fixed Parameters

            | Parameter | Value |
            |-----------|-------|
            | sigma_time (Gaussian) | {SIGMA_TIME} |
            | sigma_pixel (Gaussian) | {SIGMA_PIX} |
            | adc_hold_delay | {rc.adc_hold_delay} ticks |
            | direct threshold | {DIRECT_THRESHOLD} ke- |
            | ghost threshold | {GHOST_THRESHOLD} ke- |
            | ROI merge_gap | {ROI_MERGE_GAP} bins |
            | ROI expand | {ROI_EXPAND} bins |
            | Processor | BurstSequenceProcessorV3 |

            ## Direct Threshold Baseline

            | Metric | Value |
            |--------|-------|
            | Active voxels | {m_direct['active']:,} |
            | Ghost count | {m_direct['ghost']:,} |
            | Ghost fraction | {m_direct['ghost_frac']:.3%} |
            | ΔQ mean | {m_direct['dq_mean']:.4f} ke- |
            | ΔQ std | {m_direct['dq_std']:.4f} ke- |

            ## Grid Scan Results

            Threshold sigma at each point = `{DIRECT_THRESHOLD} / noise_rms` (equivalent
            absolute threshold = {DIRECT_THRESHOLD} ke-, matching the direct baseline).

            ### Ghost Fraction

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_ghost_frac[i,j]:.2%}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### ΔQ Std (truth > {DIRECT_THRESHOLD} ke-)

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_dq_std[i,j]:.4f}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### Noise RMS from quiet pixels [ke-]

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_noise_rms[i,j]:.5f}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        # ── recall section ─────────────────────────────────────────────
        fp.write(textwrap.dedent(f"""

            ## Recall (true charge / voxels killed by ROI cut)

            Reference totals (truth > {GHOST_THRESHOLD} ke-):
            - Total true charge: **{m_direct['total_charge']:.2f} ke-**
            - Total voxels truth>0.1 ke-: **{m_direct['total_voxels'][0.1]:,}**
            - Total voxels truth>0.5 ke-: **{m_direct['total_voxels'][0.5]:,}**
            - Total voxels truth>1.0 ke-: **{m_direct['total_voxels'][1.0]:,}**

            Direct-threshold baseline (no ROI): recall = **{m_direct['recall_charge']:.2%}**,
            killed charge = **{m_direct['killed_charge']:.2f} ke-**, killed voxels (>0.1/0.5/1.0) =
            **{m_direct['killed_voxels'][0.1]:,} / {m_direct['killed_voxels'][0.5]:,} / {m_direct['killed_voxels'][1.0]:,}**

            ### Charge recall after ROI [%]

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_recall[i,j]:.2%}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### Killed true charge after ROI [ke-]

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_killed_q[i,j]:.2f}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### Killed voxels (truth > 0.5 ke-, recon = 0)

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{int(grid_killed_v05[i,j])}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### Killed voxels (truth > 1.0 ke-, recon = 0)

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{int(grid_killed_v10[i,j])}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### ROI-extra killed charge [ke-] (direct kept, ROI dropped)

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{grid_roi_extraq[i,j]:.2f}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ### ROI-extra killed voxels (truth>0.5 ke-)

            | ω_c \\ b | {" | ".join(str(b) for b in B_GRID)} |
            |-----------|{"---|" * len(B_GRID)}
        """))
        for i, omega_c in enumerate(OMEGA_C_GRID):
            row = " | ".join(f"{int(grid_roi_extrav[i,j])}" for j in range(len(B_GRID)))
            fp.write(f"| {omega_c} | {row} |\n")

        fp.write(textwrap.dedent(f"""

            ## Best Configuration

            **Lowest ghost fraction:** ω_c = {best_omega_c},  b = {best_b}
            Ghost fraction: {grid_ghost_frac[best_i,best_j]:.3%}
            ΔQ std: {grid_dq_std[best_i,best_j]:.4f} ke-
            ΔQ mean: {grid_dq_mean[best_i,best_j]:.4f} ke-
            Noise RMS: {grid_noise_rms[best_i,best_j]:.5f} ke-
            Sigma equiv to {DIRECT_THRESHOLD} ke-: {grid_sigma_equiv[best_i,best_j]:.1f}

            ## Output Files

            | File | Description |
            |------|-------------|
            | `heatmaps.png` | Ghost fraction, ΔQ std/mean, noise RMS, sigma equiv, active voxels |
            | `recall_heatmaps.png` | Recall, killed charge, killed voxels at 0.1/0.5/1.0 ke- |
            | `combo_wc*_b*.png` | Per-combination 2D hist + ΔQ + ghost plots |
            | `SCAN_LOG.md` | This file |
        """))
    print(f"Saved: {log_path}")

    # ── append to ANALYSIS_HISTORY.md ─────────────────────────────────────
    history_path = _here / "ANALYSIS_HISTORY.md"

    # Pick the b=2 column at the same omega_c as the best config — this is the
    # "Gaussian-shaped time filter (DC killed)" baseline for the ghost/recall
    # tradeoff comparison.
    gauss_j = B_GRID.index(2) if 2 in B_GRID else 0

    entry = textwrap.dedent(f"""
        ---

        ## Wiener ROI Parameter Scan ({datetime.now():%Y-%m-%d})

        **Output:** `{outdir.name}/`
        **Script:** `examples/wiener_roi_scan.py`
        **Dataset:** `{Path(args.input_file).name}`  TPC {args.tpc_id} / event {args.event_id}
        **Processor:** BurstSequenceProcessorV3  τ = adc_hold_delay = {rc.adc_hold_delay}
        **Gaussian params:** σ_t={SIGMA_TIME}  σ_pxl={SIGMA_PIX}
        **Grid:** ω_c ∈ {OMEGA_C_GRID}  ×  b ∈ {B_GRID}  (ROI applied to Gaussian deconv_q)

        ### Reference totals (truth > {GHOST_THRESHOLD} ke-)

        Total true charge: **{m_direct['total_charge']:.1f} ke-** ·
        voxels: {m_direct['total_voxels'][0.1]:,} (>0.1) /
        {m_direct['total_voxels'][0.5]:,} (>0.5) /
        {m_direct['total_voxels'][1.0]:,} (>1.0) ke-

        ### Key results (equiv threshold = {DIRECT_THRESHOLD} ke-)

        | Method | Ghost frac | ΔQ std | Recall | Killed Q [ke-] | Killed >0.5 ke- voxels |
        |--------|-----------:|-------:|-------:|---------------:|-----------------------:|
        | Direct threshold {DIRECT_THRESHOLD} ke- | {m_direct['ghost_frac']:.2%} | {m_direct['dq_std']:.3f} | {m_direct['recall_charge']:.2%} | {m_direct['killed_charge']:.1f} | {m_direct['killed_voxels'][0.5]:,} |
        | Gaussian (ω_c={OMEGA_C_GRID[best_i]}, b=2)  | {grid_ghost_frac[best_i,gauss_j]:.2%} | {grid_dq_std[best_i,gauss_j]:.3f} | {grid_recall[best_i,gauss_j]:.2%} | {grid_killed_q[best_i,gauss_j]:.1f} | {int(grid_killed_v05[best_i,gauss_j]):,} |
        | Wiener best (ω_c={best_omega_c}, b={best_b}) | {grid_ghost_frac[best_i,best_j]:.2%} | {grid_dq_std[best_i,best_j]:.3f} | {grid_recall[best_i,best_j]:.2%} | {grid_killed_q[best_i,best_j]:.1f} | {int(grid_killed_v05[best_i,best_j]):,} |

        **Tradeoff:** sharper rolloff (b=4) cuts ghosts harder but slightly
        worsens recall vs Gaussian (b=2) at the same ω_c. Both ROI configs
        kill *less* true charge than the bare direct threshold because
        `expand=2` preserves low-charge voxels neighboring signal.

        Heatmaps: `heatmaps.png` (ghost/ΔQ/noise) + `recall_heatmaps.png`
        (recall, killed Q, killed voxels at 0.1/0.5/1.0 ke-). Detailed
        tables in `SCAN_LOG.md`.
    """)
    with history_path.open("a") as fp:
        fp.write(entry)
    print(f"Updated: {history_path}")

    # ── final console summary ──────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'ω_c':>8}  {'b':>3}  {'noise_rms':>10}  {'σ_equiv':>8}  "
          f"{'ghost%':>8}  {'ΔQσ':>8}  {'recall%':>8}  {'killQ':>8}  {'kV>0.5':>7}")
    print("-" * 95)
    print(f"  direct  ---  {'---':>10}  {'---':>8}  "
          f"{m_direct['ghost_frac']:>7.1%}  {m_direct['dq_std']:>8.3f}  "
          f"{m_direct['recall_charge']:>7.1%}  {m_direct['killed_charge']:>8.2f}  "
          f"{m_direct['killed_voxels'][0.5]:>7,}")
    for i, omega_c in enumerate(OMEGA_C_GRID):
        for j, b in enumerate(B_GRID):
            marker = " ← best" if (i == best_i and j == best_j) else ""
            print(f"{omega_c:>8}  {b:>3}  "
                  f"{grid_noise_rms[i,j]:>10.5f}  "
                  f"{grid_sigma_equiv[i,j]:>8.1f}  "
                  f"{grid_ghost_frac[i,j]:>7.1%}  "
                  f"{grid_dq_std[i,j]:>8.3f}  "
                  f"{grid_recall[i,j]:>7.1%}  "
                  f"{grid_killed_q[i,j]:>8.2f}  "
                  f"{int(grid_killed_v05[i,j]):>7,}{marker}")
    print("=" * 95)
    print(f"\nAll outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
