#!/usr/bin/env python3
"""Compare direct threshold vs Wiener-ROI on Gaussian-deconvolved charge.

Three filtering methods are compared on the same Gaussian-deconvolved block:

  1. direct_hard  : deconv_q * (deconv_q > DIRECT_THRESHOLD)
  2. wiener_500   : ROI from Wiener deconv (threshold_sigma=500), applied to deconv_q
  3. wiener_5     : ROI from Wiener deconv (threshold_sigma=5), applied to deconv_q

For each method, the filtered deconv block is aligned with smeared_true and the
following benchmark plots are produced:

  • 2D histogram  : smeared_true (x) vs filtered_deconv_q (y)
  • 1D delta-Q    : smeared_true - filtered_deconv_q for truth-selected voxels
  • Ghost         : filtered_deconv_q for voxels where smeared_true < ghost_thresh
  • Ghost scatter : 2D: smeared_true (x, near-zero) vs filtered_deconv_q (y, non-zero)

Run from the repo root or examples/ directory:

    PYTHONPATH=src python examples/compare_wiener_roi.py \\
        --input-file examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz \\
        --field-response /srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz \\
        --tpc-id 0 --event-id 0 --output-dir /tmp/roi_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

# ── path bootstrap (allow running without pip install) ──────────────────────
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
from unfoldlarpix.roi_finder import apply_roi_mask, estimate_quiet_pixel_noise, find_roi_mask

# ── constants ───────────────────────────────────────────────────────────────
SIGMA_TIME = 0.005
SIGMA_PIX = 0.2
WIENER_OMEGA_C = 0.005   # cycles / adc-bin  (matches Gaussian sigma_time for comparable cutoff)
WIENER_B = 4.0            # sharper rolloff than b=2 Gaussian
DIRECT_THRESHOLD = 0.5    # ke-
GHOST_THRESHOLD = 0.1     # ke-  (smeared truth below this = "ghost" region)
HIST2D_RANGE = [[0, 10], [0, 10]]  # ke-


# ── alignment helper (mirrors plot_proj.py) ──────────────────────────────────

def align_voxel_blocks(
    fine_lower_corner: np.ndarray,
    coarse_lower_corner: np.ndarray,
    fine_voxels: np.ndarray,
    coarse_voxels: np.ndarray,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fine_voxels = np.asarray(fine_voxels)
    coarse_voxels = np.asarray(coarse_voxels)
    ndims = fine_voxels.ndim
    fine_lower = np.asarray(fine_lower_corner, dtype=int)
    coarse_lower = np.asarray(coarse_lower_corner, dtype=int)
    bin_arr = np.ones(ndims, dtype=int)
    bin_arr[-1] = bin_size

    target_lower = coarse_lower.copy()
    diff_bins = ((fine_lower - target_lower) // bin_arr) * bin_arr
    target_lower += np.minimum(diff_bins, 0)

    fine_upper = fine_lower + np.array(fine_voxels.shape, dtype=int)
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
         int((target_upper[i] - coarse_upper[i]) // bin_arr[i]))
        for i in range(ndims)
    )
    aligned_fine = np.pad(fine_voxels, fine_pad, mode="constant")
    aligned_coarse = np.pad(coarse_voxels, coarse_pad, mode="constant")

    refine = []
    sub_axes = []
    for i in range(ndims):
        refine.extend([aligned_coarse.shape[i], bin_arr[i]])
        sub_axes.append(2 * i + 1)
    fine_summed = aligned_fine.reshape(refine).sum(axis=tuple(sub_axes))
    return aligned_fine, aligned_coarse, fine_summed, target_lower


# ── plotting helpers ─────────────────────────────────────────────────────────

def _setup_ax(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_2d_hist(
    ax: plt.Axes,
    truth: np.ndarray,
    recon: np.ndarray,
    label: str,
    unit: str,
) -> None:
    mask = truth.flatten() + recon.flatten() > 0
    t = truth.flatten()[mask]
    r = recon.flatten()[mask]
    h, xe, ye, img = ax.hist2d(
        t, r,
        bins=60,
        range=HIST2D_RANGE,
        norm=LogNorm(vmin=1),
        cmap="viridis",
    )
    plt.colorbar(img, ax=ax)
    ax.plot(HIST2D_RANGE[0], HIST2D_RANGE[1], "r--", lw=0.8, alpha=0.7)
    _setup_ax(ax, f"Smeared truth [{unit}]", f"{label} [{unit}]",
              f"2D: truth vs {label}")


def plot_delta_q(
    ax: plt.Axes,
    truth: np.ndarray,
    recon: np.ndarray,
    label: str,
    unit: str,
    threshold: float,
) -> None:
    mask = truth.flatten() > threshold
    delta = (truth.flatten() - recon.flatten())[mask]
    ax.hist(delta, bins=80, range=(-5, 5), alpha=0.7)
    mean, std = float(np.mean(delta)), float(np.std(delta))
    ax.axvline(0, color="k", lw=0.8)
    ax.text(
        0.97, 0.97,
        f"μ={mean:.3f}\nσ={std:.3f}\nn={mask.sum()}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _setup_ax(ax,
              f"Smeared truth − {label} [{unit}]",
              "Count",
              f"ΔQ (truth−recon) | truth > {threshold} {unit}")


def plot_ghost(
    ax: plt.Axes,
    truth: np.ndarray,
    recon: np.ndarray,
    label: str,
    unit: str,
    ghost_thresh: float,
    signal_thresh: float,
) -> None:
    ghost_mask = (truth.flatten() < ghost_thresh) & (recon.flatten() > signal_thresh)
    values = recon.flatten()[ghost_mask]
    total_recon_active = int((recon.flatten() > signal_thresh).sum())
    ghost_frac = float(ghost_mask.sum()) / max(total_recon_active, 1)
    ax.hist(values, bins=60, range=(signal_thresh, 10), alpha=0.7)
    ax.text(
        0.97, 0.97,
        f"ghosts={ghost_mask.sum()}\ntotal recon>{signal_thresh}={total_recon_active}\nfrac={ghost_frac:.3%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _setup_ax(ax,
              f"{label} [{unit}]",
              "Count",
              f"Ghost: truth<{ghost_thresh} & recon>{signal_thresh}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input-file",
        default=(
            "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/"
            "examples/data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz"
        ),
    )
    p.add_argument(
        "--field-response",
        default="/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz",
    )
    p.add_argument("--tpc-id", type=int, default=0)
    p.add_argument("--event-id", type=int, default=0)
    p.add_argument("--output-dir", default="/tmp/roi_compare")
    p.add_argument("--sigma-time", type=float, default=SIGMA_TIME)
    p.add_argument("--sigma-pxl", type=float, default=SIGMA_PIX)
    p.add_argument("--wiener-omega-c", type=float, default=WIENER_OMEGA_C)
    p.add_argument("--wiener-b", type=float, default=WIENER_B)
    p.add_argument("--direct-threshold", type=float, default=DIRECT_THRESHOLD,
                   help="Hard threshold on deconv_q in ke- (default: 0.5)")
    p.add_argument("--ghost-threshold", type=float, default=GHOST_THRESHOLD,
                   help="Smeared truth below this is 'ghost' region (ke-)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(outdir / "roi_compare")

    # ── load ──────────────────────────────────────────────────────────────
    loader = DataLoader(args.input_file)
    rc = loader.get_readout_config()
    prep = prepare_field_response(
        args.field_response, rc.adc_hold_delay,
        normalized=False, response_template="center",
    )
    print(f"adc_hold_delay={rc.adc_hold_delay}  threshold={rc.threshold} ke-")
    print(f"Wiener omega_c={args.wiener_omega_c}  b={args.wiener_b}")

    event = None
    for ev in loader.iter_events():
        if ev.tpc_id == args.tpc_id and ev.event_id == args.event_id:
            event = ev
            break
    if event is None:
        raise SystemExit(f"Event tpc={args.tpc_id} event={args.event_id} not found.")

    # ── deconvolution (single pass gives both Gaussian and Wiener) ─────────
    result = process_event_deconvolution(
        event, rc, prep,
        sigma_time=args.sigma_time,
        sigma_pixel=args.sigma_pxl,
        processor_cls=BurstSequenceProcessorV3,
        tau=rc.adc_hold_delay,
        npadbin=50,
        require_zero_local_offset=True,
        enable_wiener_roi=True,
        wiener_omega_c=args.wiener_omega_c,
        wiener_b=args.wiener_b,
        roi_threshold_sigma=500.0,
        roi_merge_gap=2,
        roi_expand=2,
    )
    print(f"noise_rms (quiet pixels) = {result.roi_noise_rms:.6g} ke-")
    print(f"noise_rms * 500 = {500 * result.roi_noise_rms:.6g} ke-")
    print(f"noise_rms *   5 = {5 * result.roi_noise_rms:.6g} ke-")

    noise_rms = result.roi_noise_rms
    # Sigma that gives effective absolute threshold matching the direct threshold.
    sigma_equiv = args.direct_threshold / noise_rms
    print(f"Sigma equiv to {args.direct_threshold} ke- = {sigma_equiv:.1f}σ")

    # Method 1: direct hard threshold on Gaussian deconv
    deconv_direct = result.deconv_q * (result.deconv_q > args.direct_threshold)

    # Method 2: ROI with threshold_sigma=500 (conservative start)
    roi_mask_500 = result.roi_mask
    deconv_roi_500 = result.deconv_q_roi

    # Method 3: ROI with threshold_sigma=sigma_equiv (same effective absolute threshold as direct)
    roi_mask_equiv = find_roi_mask(
        result.deconv_q_wiener,
        noise_rms,
        threshold_sigma=sigma_equiv,
        merge_gap=2,
        expand=2,
    )
    deconv_roi_equiv = apply_roi_mask(result.deconv_q, roi_mask_equiv)

    # Method 4: ROI with threshold_sigma=5 (low threshold for sensitivity study)
    roi_mask_5 = find_roi_mask(
        result.deconv_q_wiener,
        noise_rms,
        threshold_sigma=5.0,
        merge_gap=2,
        expand=2,
    )
    deconv_roi_5 = apply_roi_mask(result.deconv_q, roi_mask_5)

    print(f"Active voxels — direct: {int((deconv_direct > 0).sum()):,}")
    print(f"Active voxels — roi_500: {int((deconv_roi_500 > 0).sum()):,}")
    print(f"Active voxels — roi_equiv ({sigma_equiv:.0f}σ): {int((deconv_roi_equiv > 0).sum()):,}")
    print(f"Active voxels — roi_5:   {int((deconv_roi_5 > 0).sum()):,}")

    # ── alignment with smeared_true ────────────────────────────────────────
    boffset = shift_time_offset(result.hwf_block_offset, -rc.adc_hold_delay)
    dtus = 0.05 * rc.adc_hold_delay
    unit = f"ke-/pix/{dtus:.1f}µs"

    def _align(coarse_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, aligned_coarse, smear_summed, _ = align_voxel_blocks(
            fine_lower_corner=result.smear_offset,
            coarse_lower_corner=boffset,
            fine_voxels=result.smeared_true,
            coarse_voxels=coarse_block,
            bin_size=rc.adc_hold_delay,
        )
        return smear_summed, aligned_coarse

    smear_summed_d, aligned_direct = _align(deconv_direct)
    smear_summed_500, aligned_roi_500 = _align(deconv_roi_500)
    smear_summed_equiv, aligned_roi_equiv = _align(deconv_roi_equiv)
    smear_summed_5, aligned_roi_5 = _align(deconv_roi_5)

    print(f"Aligned shape: {aligned_direct.shape}")

    # ── figures ────────────────────────────────────────────────────────────
    methods = [
        ("direct_0.5ke",             aligned_direct,     smear_summed_d,     f"direct>{args.direct_threshold}ke-"),
        ("roi_500sig",               aligned_roi_500,    smear_summed_500,   "roi σ=500"),
        (f"roi_{sigma_equiv:.0f}sig", aligned_roi_equiv, smear_summed_equiv, f"roi σ={sigma_equiv:.0f}"),
        ("roi_5sig",                 aligned_roi_5,      smear_summed_5,     "roi σ=5"),
    ]

    n_methods = len(methods)

    # 1) Combined 2D histogram — one column per method
    fig_2d, axes_2d = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    for ax, (tag, recon, truth, lbl) in zip(axes_2d, methods):
        plot_2d_hist(ax, truth, recon, lbl, unit)
    fig_2d.suptitle(f"2D: smeared truth vs filtered deconv  [TPC{args.tpc_id} ev{args.event_id}]")
    fig_2d.tight_layout()
    fig_2d.savefig(f"{prefix}_2dhist.png", dpi=120)
    plt.close(fig_2d)
    print(f"Saved: {prefix}_2dhist.png")

    # 2) ΔQ 1D histograms
    fig_dq, axes_dq = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    for ax, (tag, recon, truth, lbl) in zip(axes_dq, methods):
        plot_delta_q(ax, truth, recon, lbl, unit, threshold=args.direct_threshold)
    fig_dq.suptitle(f"ΔQ = smeared truth − filtered deconv  [TPC{args.tpc_id} ev{args.event_id}]")
    fig_dq.tight_layout()
    fig_dq.savefig(f"{prefix}_deltaQ.png", dpi=120)
    plt.close(fig_dq)
    print(f"Saved: {prefix}_deltaQ.png")

    # 3) Ghost histograms
    fig_ghost, axes_ghost = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    for ax, (tag, recon, truth, lbl) in zip(axes_ghost, methods):
        plot_ghost(ax, truth, recon, lbl, unit,
                   ghost_thresh=args.ghost_threshold,
                   signal_thresh=args.direct_threshold)
    fig_ghost.suptitle(
        f"Ghost: truth<{args.ghost_threshold}ke- & recon>{args.direct_threshold}ke-  "
        f"[TPC{args.tpc_id} ev{args.event_id}]"
    )
    fig_ghost.tight_layout()
    fig_ghost.savefig(f"{prefix}_ghost.png", dpi=120)
    plt.close(fig_ghost)
    print(f"Saved: {prefix}_ghost.png")

    # 4) ΔQ for ghost-adjacent region: 0 < smear < ghost_threshold
    fig_nearghost, axes_ng = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    for ax, (tag, recon, truth, lbl) in zip(axes_ng, methods):
        mask = (truth.flatten() > 0) & (truth.flatten() < args.ghost_threshold)
        delta = (truth.flatten() - recon.flatten())[mask]
        ax.hist(delta, bins=80, range=(-5, 5), alpha=0.7)
        ax.axvline(0, color="k", lw=0.8)
        ax.text(
            0.97, 0.97,
            f"n={mask.sum()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        _setup_ax(ax,
                  f"truth − {lbl} [{unit}]",
                  "Count",
                  f"ΔQ near ghost: 0<truth<{args.ghost_threshold}")
    fig_nearghost.suptitle(f"ΔQ near ghost region  [TPC{args.tpc_id} ev{args.event_id}]")
    fig_nearghost.tight_layout()
    fig_nearghost.savefig(f"{prefix}_nearghost_deltaQ.png", dpi=120)
    plt.close(fig_nearghost)
    print(f"Saved: {prefix}_nearghost_deltaQ.png")

    # 5) Ghost 2D: smeared_true (near-zero) vs deconv for all three methods
    fig_g2d, axes_g2d = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    g_range = [[0, args.ghost_threshold], [0, 5]]
    for ax, (tag, recon, truth, lbl) in zip(axes_g2d, methods):
        t = truth.flatten()
        r = recon.flatten()
        mask = (t >= 0) & (t < args.ghost_threshold)
        if mask.sum() > 0:
            h, xe, ye, img = ax.hist2d(
                t[mask], r[mask], bins=50,
                range=g_range, norm=LogNorm(vmin=1),
                cmap="plasma",
            )
            plt.colorbar(img, ax=ax)
        _setup_ax(ax,
                  f"Smeared truth [{unit}]",
                  f"{lbl} [{unit}]",
                  f"Ghost region: truth<{args.ghost_threshold}")
    fig_g2d.suptitle(f"Ghost 2D  [TPC{args.tpc_id} ev{args.event_id}]")
    fig_g2d.tight_layout()
    fig_g2d.savefig(f"{prefix}_ghost_2d.png", dpi=120)
    plt.close(fig_g2d)
    print(f"Saved: {prefix}_ghost_2d.png")

    # ── text summary ───────────────────────────────────────────────────────
    summary_path = outdir / "summary.txt"
    with summary_path.open("w") as fp:
        fp.write(f"TPC {args.tpc_id}  event {args.event_id}\n")
        fp.write(f"adc_hold_delay = {rc.adc_hold_delay}  threshold = {rc.threshold} ke-\n")
        fp.write(f"Wiener omega_c = {args.wiener_omega_c}  b = {args.wiener_b}\n")
        fp.write(f"noise_rms (quiet pixels) = {result.roi_noise_rms:.6g} ke-\n\n")
        fp.write(f"{'Method':<20} {'active voxels':>15} {'ghost count':>12} {'ghost frac':>12}\n")
        fp.write("-" * 65 + "\n")
        for tag, recon, truth, lbl in methods:
            active = int((recon.flatten() > args.direct_threshold).sum())
            ghost = int(
                ((truth.flatten() < args.ghost_threshold) &
                 (recon.flatten() > args.direct_threshold)).sum()
            )
            frac = ghost / max(active, 1)
            fp.write(f"{lbl:<20} {active:>15,} {ghost:>12,} {frac:>12.3%}\n")
    print(f"Saved: {summary_path}")
    summary_path.read_text() and print(summary_path.read_text())


if __name__ == "__main__":
    main()
