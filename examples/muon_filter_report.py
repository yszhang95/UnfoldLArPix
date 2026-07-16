#!/usr/bin/env python3
"""Report on the muon-derived time-axis correction filter.

Produces three figures:

  1. Power-spectra ratio for the MUON (validation / self-consistency).
     Temporal power ratio compensated/continuous (v3_burst / v3) before
     correction, and after applying |H|² to the compensated spectrum.  By
     construction the corrected ratio should sit near 1 in the high-SNR band.

  2. Power-spectra ratio for the POSITRON after correction (the real test).
     Temporal power ratio of the deconvolved charge deconv_q:
     v3_burst/v3 uncorrected vs v3_burst(+muon filter)/v3.  The corrected
     curve should be flatter / closer to 1 — the muon filter transferred.

  3. 2-D correlation of smeared truth vs reconstruction (deconv_q) for the
     corrected positron.  NOTE: the Gaussian filter is applied to the smeared
     truth; the spectral |H| correction is applied ONLY to the reconstruction,
     never to the truth.

Usage::

    python examples/muon_filter_report.py \\
        --muon-v3 muon_out/deconv_positron_v3_event_0_0.npz \\
        --muon-v3-burst muon_out/deconv_positron_v3_burst_*_event_0_0.npz \\
        --filter-npz muon_time_filter.npz \\
        --pos-v3 pos_out/deconv_positron_v3_event_0_0.npz \\
        --pos-v3burst-uncorr pos_out/deconv_positron_v3_burst_<sfx>_event_0_0.npz \\
        --pos-v3burst-corr   pos_out/deconv_positron_v3_burst_<sfx>_muonfilt_event_0_0.npz \\
        --out-prefix report/muonfilt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Spectra helpers (per-pixel temporal power, averaged over active pixels)
# ---------------------------------------------------------------------------

def active_pixel_traces(block: np.ndarray, threshold_frac: float = 0.10) -> np.ndarray:
    """Return active-pixel time traces of a (nx, ny, nt) block, shape (n, nt)."""
    charge = block.sum(axis=2)
    cmax = float(charge.max())
    if cmax <= 0.0:
        return np.zeros((0, block.shape[2]), dtype=np.float64)
    xs, ys = np.where(charge > threshold_frac * cmax)
    return block[xs, ys, :]


def mean_power(block: np.ndarray, threshold_frac: float = 0.10):
    """Mean temporal power spectrum over active pixels.  Returns (freqs, P)."""
    traces = active_pixel_traces(block, threshold_frac)
    nt = block.shape[2]
    freqs = np.fft.rfftfreq(nt)
    if traces.shape[0] == 0:
        return freqs, np.zeros(len(freqs))
    powers = np.abs(np.fft.rfft(traces, axis=-1)) ** 2
    return freqs, powers.mean(axis=0)


def ratio_on_grid(num_freqs, num_P, den_freqs, den_P):
    """Interpolate numerator onto denominator grid and return safe ratio."""
    num_i = np.interp(den_freqs, num_freqs, num_P)
    safe = den_P > 0
    r = np.full_like(den_P, np.nan)
    r[safe] = num_i[safe] / den_P[safe]
    return den_freqs, r


# ---------------------------------------------------------------------------
# Voxel alignment (copied from plot_proj.py:align_voxel_blocks, self-contained)
# ---------------------------------------------------------------------------

def align_voxel_blocks(fine_lower_corner, coarse_lower_corner, fine_voxels,
                       coarse_voxels, bin_size):
    """Pad/align fine & coarse blocks to a shared lower corner and sum the fine
    block within each coarse voxel.  Returns (aligned_fine, aligned_coarse,
    fine_summed, output_offset)."""
    fine_voxels = np.asarray(fine_voxels)
    coarse_voxels = np.asarray(coarse_voxels)
    ndims = fine_voxels.ndim
    fine_lower = np.asarray(fine_lower_corner, dtype=int)
    coarse_lower = np.asarray(coarse_lower_corner, dtype=int)
    bin_size = np.asarray(bin_size, dtype=int)
    if bin_size.ndim == 0:
        bin_size = np.full((ndims,), bin_size, dtype=int)
        bin_size[:-1] = 1
    fine_shape = np.array(fine_voxels.shape, dtype=int)
    coarse_shape = np.array(coarse_voxels.shape, dtype=int)

    target_lower = coarse_lower.copy()
    diff_bins = ((fine_lower - target_lower) // bin_size) * bin_size
    target_lower += np.minimum(diff_bins, 0)
    fine_upper = fine_lower + fine_shape
    coarse_upper = coarse_lower + coarse_shape * bin_size
    target_upper = coarse_upper.copy()
    target_upper += np.clip(np.ceil((fine_upper - target_upper) / bin_size) * bin_size,
                            0, None).astype(int)
    fine_padding_lower = fine_lower - target_lower
    coarse_padding_lower = (coarse_lower - target_lower) // bin_size
    fine_padding_upper = target_upper - fine_upper
    coarse_padding_upper = (target_upper - coarse_upper) // bin_size
    aligned_fine = np.pad(fine_voxels, pad_width=tuple(
        (int(p), int(q)) for p, q in zip(fine_padding_lower, fine_padding_upper)),
        mode="constant")
    aligned_coarse = np.pad(coarse_voxels, pad_width=tuple(
        (int(p), int(q)) for p, q in zip(coarse_padding_lower, coarse_padding_upper)),
        mode="constant")

    refine_factor, sub_axes = [], []
    for i in range(ndims):
        refine_factor.append(aligned_coarse.shape[i])
        refine_factor.append(bin_size[i])
        sub_axes.append(2 * i + 1)
    fine_summed = aligned_fine.reshape(refine_factor).sum(axis=tuple(sub_axes))
    return aligned_fine, aligned_coarse, fine_summed, target_lower


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report on the muon-derived correction filter.")
    p.add_argument("--muon-v3", nargs="+", required=True)
    p.add_argument("--muon-v3-burst", nargs="+", required=True)
    p.add_argument("--filter-npz", required=True)
    p.add_argument("--pos-v3", default=None,
                   help="(Unused) continuous-positron reference. Fig 2 uses the "
                        "smeared truth of the same event as the reference instead.")
    p.add_argument("--pos-continuous", default=None,
                   help="Continuous-readout positron NPZ (v3, same event/FR as the "
                        "compensated positron). Enables Fig 4: readout-level "
                        "compensated/continuous deficit, muon vs positron.")
    p.add_argument("--pos-v3burst-uncorr", required=True)
    p.add_argument("--pos-v3burst-corr", required=True)
    p.add_argument("--out-prefix", default="report/muonfilt")
    p.add_argument("--threshold-frac", type=float, default=0.10,
                   help="Active-pixel fraction for spectra (default 0.10).")
    p.add_argument("--corr-threshold", type=float, default=0.5,
                   help="Charge threshold (ke-) for the 2-D correlation mask "
                        "(default 0.5).")
    return p.parse_args()


def pooled_mean_power(paths, threshold_frac):
    """Trace-count-weighted mean power over multiple NPZ hwf_blocks."""
    results = []
    for path in paths:
        block = np.asarray(np.load(path, allow_pickle=True)["hwf_block"], dtype=np.float64)
        traces = active_pixel_traces(block, threshold_frac)
        if traces.shape[0] == 0:
            continue
        freqs = np.fft.rfftfreq(block.shape[2])
        P = (np.abs(np.fft.rfft(traces, axis=-1)) ** 2).mean(axis=0)
        results.append((freqs, P, traces.shape[0]))
    if not results:
        raise RuntimeError(f"No active traces found in {paths}")
    max_nf = max(len(f) for f, _, _ in results)
    common = np.fft.rfftfreq((max_nf - 1) * 2)
    tot = np.zeros(max_nf)
    w = 0
    for f, P, n in results:
        tot += np.interp(common, f, P) * n
        w += n
    return common, tot / w


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    filt = np.load(args.filter_npz)
    H_freqs = np.asarray(filt["freqs_cycles_per_sample"], dtype=np.float64)
    H_mag = np.asarray(filt["H_mag"], dtype=np.float64)

    # ===== Figure 1: muon power-spectra ratio (hwf_block) =====
    mf_v3, mP_v3 = pooled_mean_power([Path(p) for p in args.muon_v3], args.threshold_frac)
    mf_b, mP_b = pooled_mean_power([Path(p) for p in args.muon_v3_burst], args.threshold_frac)

    f1, ratio_raw = ratio_on_grid(mf_b, mP_b, mf_v3, mP_v3)  # v3_burst / v3
    # Apply |H|^2 to compensated power (power-domain), recompute ratio.
    H_on_v3 = np.interp(mf_v3, H_freqs, H_mag, left=1.0, right=1.0)
    mP_b_on_v3 = np.interp(mf_v3, mf_b, mP_b)
    mP_b_corr = (H_on_v3 ** 2) * mP_b_on_v3
    safe = mP_v3 > 0
    ratio_corr = np.full_like(mP_v3, np.nan)
    ratio_corr[safe] = mP_b_corr[safe] / mP_v3[safe]

    fig1, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(f1, ratio_raw, color="tab:green", linewidth=1.4,
            label="v3_burst / v3  (uncorrected)")
    ax.plot(mf_v3, ratio_corr, color="tab:red", linewidth=1.4,
            label="|H|²·v3_burst / v3  (corrected)")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("frequency [cycles / ADC-sample]")
    ax.set_ylabel("temporal power ratio")
    ax.set_title("Fig 1 — MUON temporal power ratio (compensated / continuous)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    p1 = out_prefix.with_name(out_prefix.name + "_fig1_muon_ratio.png")
    fig1.savefig(p1, dpi=140, bbox_inches="tight")
    print(f"Saved {p1}")
    plt.close(fig1)

    # ===== Figure 2: positron deconv_q / smeared-truth temporal power ratio =====
    # Reference is the smeared truth of the SAME event (aligned to the deconv_q
    # grid).  A perfect reconstruction gives ratio == 1 at every frequency; the
    # muon correction should pull the compensated reconstruction toward 1.
    # (The v3 continuous positron deconv_q is NOT used as the reference: it is a
    #  different readout flavour processed with a mismatched field response, so
    #  its absolute spectrum is not comparable.)
    def deconv_over_truth_ratio(path, threshold_frac):
        f = np.load(path, allow_pickle=True)
        smeared_true = np.asarray(f["smeared_true"], dtype=np.float64)
        deconv_q = np.asarray(f["deconv_q"], dtype=np.float64)
        _, aligned_dq, smear_summed, _ = align_voxel_blocks(
            fine_lower_corner=f["smear_offset"],
            coarse_lower_corner=f["boffset"],
            fine_voxels=smeared_true,
            coarse_voxels=deconv_q,
            bin_size=f["adc_hold_delay"],
        )
        charge = smear_summed.sum(axis=2)
        cmax = float(charge.max())
        xs, ys = np.where(charge > threshold_frac * cmax)
        nt = smear_summed.shape[2]
        freqs = np.fft.rfftfreq(nt)
        P_true = (np.abs(np.fft.rfft(smear_summed[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
        P_dec = (np.abs(np.fft.rfft(aligned_dq[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
        safe = P_true > 0
        r = np.full_like(P_true, np.nan)
        r[safe] = P_dec[safe] / P_true[safe]
        return freqs, r

    f2f_un, ratio_un = deconv_over_truth_ratio(args.pos_v3burst_uncorr, args.threshold_frac)
    f2f_co, ratio_co = deconv_over_truth_ratio(args.pos_v3burst_corr, args.threshold_frac)

    fig2, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(f2f_un, ratio_un, color="tab:green", linewidth=1.4,
            label="v3_burst / truth  (uncorrected)")
    ax.plot(f2f_co, ratio_co, color="tab:red", linewidth=1.4,
            label="v3_burst + muon filter / truth  (corrected)")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("frequency [cycles / ADC-sample]")
    ax.set_ylabel("deconv_q / smeared-truth temporal power ratio")
    ax.set_title("Fig 2 — POSITRON reco/truth power ratio (target = 1)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    p2 = out_prefix.with_name(out_prefix.name + "_fig2_positron_ratio.png")
    fig2.savefig(p2, dpi=140, bbox_inches="tight")
    print(f"Saved {p2}")
    plt.close(fig2)

    # Band-integrated mean |1 - ratio| as a scalar "distance from truth".
    def band_dev(freqs, ratio):
        m = np.isfinite(ratio)
        return float(np.mean(np.abs(ratio[m] - 1.0)))
    dev_un = band_dev(f2f_un, ratio_un)
    dev_co = band_dev(f2f_co, ratio_co)
    print(f"Fig 2 mean |ratio-1|: uncorrected={dev_un:.4f}  corrected={dev_co:.4f}")

    # ===== Figure 3: 2-D correlation truth vs reco (corrected positron) =====
    def correlation_arrays(path):
        f = np.load(path, allow_pickle=True)
        smeared_true = np.asarray(f["smeared_true"], dtype=np.float64)
        deconv_q = np.asarray(f["deconv_q"], dtype=np.float64)
        _, aligned_coarse, smear_summed, _ = align_voxel_blocks(
            fine_lower_corner=f["smear_offset"],
            coarse_lower_corner=f["boffset"],
            fine_voxels=smeared_true,
            coarse_voxels=deconv_q,
            bin_size=f["adc_hold_delay"],
        )
        return smear_summed, aligned_coarse

    thr = args.corr_threshold

    def corr_stats(x, y):
        if x.size > 2 and np.std(x) > 0 and np.std(y) > 0:
            return float(np.corrcoef(x, y)[0, 1]), float(np.polyfit(x, y, 1)[0])
        return float("nan"), float("nan")

    panels = [
        ("uncorrected", args.pos_v3burst_uncorr),
        ("muon-corrected", args.pos_v3burst_corr),
    ]
    fig3, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    stats = {}
    for ax, (label, path) in zip(axes, panels):
        smear_summed, aligned_deconv_q = correlation_arrays(path)
        mask = aligned_deconv_q > thr
        x = smear_summed[mask]
        y = aligned_deconv_q[mask]
        corr, slope = corr_stats(x, y)
        stats[label] = (corr, slope, x.size)
        hi = max(1.0, float(np.percentile(np.concatenate([x, y]), 99))) if x.size else 10.0
        h, xe, ye, img = ax.hist2d(x, y, bins=50, range=[[0, hi], [0, hi]], norm=LogNorm())
        fig3.colorbar(img, ax=ax, label="voxels")
        ax.plot([0, hi], [0, hi], color="white", linestyle="--", linewidth=1.0, label="y = x")
        ax.set_xlabel("smeared truth (Gauss-filtered, summed) [ke-/voxel]")
        ax.set_ylabel(f"reconstruction deconv_q ({label}) [ke-/voxel]")
        ax.set_title(f"{label}:  Pearson r = {corr:.4f},  slope = {slope:.4f},  n = {x.size}")
        ax.legend(loc="upper left")
    fig3.suptitle(f"Fig 3 — POSITRON truth vs reco  (deconv_q > {thr})", fontsize=13)
    fig3.tight_layout()
    p3 = out_prefix.with_name(out_prefix.name + "_fig3_corr2d.png")
    fig3.savefig(p3, dpi=140, bbox_inches="tight")
    print(f"Saved {p3}")
    plt.close(fig3)

    # ===== Figure 4: readout-level deficit (compensated/continuous), muon vs positron =====
    # Apples-to-apples with the quantity the filter is built from (hwf_block power).
    # Shows WHY the muon needs a large boost but the positron does not.
    if args.pos_continuous is not None:
        muf_c, muP_c = pooled_mean_power([Path(p) for p in args.muon_v3], args.threshold_frac)
        muf_b, muP_b = pooled_mean_power([Path(p) for p in args.muon_v3_burst], args.threshold_frac)
        _, mu_ratio = ratio_on_grid(muf_b, muP_b, muf_c, muP_c)  # comp/cont, muon

        pcf_c, pcP_c = pooled_mean_power([Path(args.pos_continuous)], args.threshold_frac)
        pcf_b, pcP_b = pooled_mean_power([Path(args.pos_v3burst_uncorr)], args.threshold_frac)
        _, pos_ratio = ratio_on_grid(pcf_b, pcP_b, pcf_c, pcP_c)  # comp/cont, positron

        fig4, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(muf_c, mu_ratio, color="tab:blue", linewidth=1.5,
                label="muon (nburst4): compensated / continuous")
        ax.plot(pcf_c, pos_ratio, color="tab:red", linewidth=1.5,
                label="positron (nburst256): compensated / continuous")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("frequency [cycles / ADC-sample]")
        ax.set_ylabel("readout power ratio (compensated / continuous)")
        ax.set_title("Fig 4 — readout-level deficit: muon needs the boost, positron does not")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig4.tight_layout()
        p4 = out_prefix.with_name(out_prefix.name + "_fig4_deficit.png")
        fig4.savefig(p4, dpi=140, bbox_inches="tight")
        print(f"Saved {p4}")
        plt.close(fig4)
        mu_dev = float(np.nanmean(np.abs(mu_ratio - 1.0)))
        pos_dev = float(np.nanmean(np.abs(pos_ratio - 1.0)))
        print(f"Fig 4 mean |comp/cont - 1|: muon={mu_dev:.4f}  positron={pos_dev:.4f}")

    print("\n=== Summary ===")
    for label in ("uncorrected", "muon-corrected"):
        c, s, n = stats[label]
        print(f"Fig 3 {label:>15}: Pearson r = {c:.4f}, slope = {s:.4f}, n_voxels = {n}")


if __name__ == "__main__":
    main()
