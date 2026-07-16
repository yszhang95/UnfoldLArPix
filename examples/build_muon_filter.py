#!/usr/bin/env python3
"""Estimate the muon-derived time-axis correction filter for imperfect
template compensation.

The idea: a long muon track activates many pixels, so the per-pixel temporal
power spectrum can be measured with good statistics.  Comparing continuous
readout (v3, no template injection) against the template-compensated version
(v3_burst) gives a frequency-domain transfer function that encodes the
spectral distortion introduced by template injection:

    |H(f)| = sqrt( <|S_v3(f)|²> / <|S_v3burst(f)|²> )

When this filter is applied during positron deconvolution (via
``deconv_positron_v3_burst.py --time-filter-npz``), the compensated-positron
spectrum is corrected towards the continuous-sampling reference.  Positrons
are too sparse to self-calibrate; the muon provides the transfer function.

Usage::

    python examples/build_muon_filter.py \\
        --muon-v3      muon_out/deconv_positron_v3_event_*.npz \\
        --muon-v3-burst muon_out/deconv_positron_v3_burst_*_event_*.npz \\
        --out muon_time_filter.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers (mirrors load_source / find_active_region from spectra_diagnose.py)
# ---------------------------------------------------------------------------

def load_source(path: Path) -> dict:
    """Load hwf_block, hwf_block_offset, adc_hold_delay from an NPZ file."""
    data = np.load(path, allow_pickle=True)
    return {
        "block": np.asarray(data["hwf_block"], dtype=np.float64),
        "offset": np.asarray(data["hwf_block_offset"], dtype=np.float64),
        "adc_hold_delay": int(data["adc_hold_delay"]) if "adc_hold_delay" in data.files else None,
        "path": str(path),
    }


def align_voxel_blocks(fine_lower_corner, coarse_lower_corner, fine_voxels,
                       coarse_voxels, bin_size):
    """Pad/align fine (truth) and coarse (deconv) blocks to a shared lower
    corner and sum the fine block within each coarse voxel.  Returns
    (aligned_fine, aligned_coarse, fine_summed, output_offset).
    Mirrors plot_proj.align_voxel_blocks."""
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
    target_lower += np.minimum(((fine_lower - target_lower) // bin_size) * bin_size, 0)
    fine_upper = fine_lower + fine_shape
    coarse_upper = coarse_lower + coarse_shape * bin_size
    target_upper = coarse_upper.copy()
    target_upper += np.clip(np.ceil((fine_upper - target_upper) / bin_size) * bin_size,
                            0, None).astype(int)
    fpl = fine_lower - target_lower
    cpl = (coarse_lower - target_lower) // bin_size
    fpu = target_upper - fine_upper
    cpu = (target_upper - coarse_upper) // bin_size
    aligned_fine = np.pad(fine_voxels, tuple((int(a), int(b)) for a, b in zip(fpl, fpu)))
    aligned_coarse = np.pad(coarse_voxels, tuple((int(a), int(b)) for a, b in zip(cpl, cpu)))
    refine, sub_axes = [], []
    for i in range(ndims):
        refine += [aligned_coarse.shape[i], bin_size[i]]
        sub_axes.append(2 * i + 1)
    fine_summed = aligned_fine.reshape(refine).sum(axis=tuple(sub_axes))
    return aligned_fine, aligned_coarse, fine_summed, target_lower


def truth_deconv_power(npz_path: Path, active_threshold: float):
    """Per-pixel temporal power of (smeared truth, deconv_q) for one muon NPZ.

    The smeared truth is aligned onto the deconv_q voxel grid, active pixels are
    selected by truth charge, and mean |rFFT|^2 along time is returned for both.

    Returns ``(freqs, P_truth, P_deconv, n_pixels)`` with freqs in cycles/sample.
    """
    f = np.load(npz_path, allow_pickle=True)
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
    if cmax <= 0.0:
        raise RuntimeError(f"No truth charge in {npz_path}")
    xs, ys = np.where(charge > active_threshold * cmax)
    nt = smear_summed.shape[2]
    freqs = np.fft.rfftfreq(nt)
    P_truth = (np.abs(np.fft.rfft(smear_summed[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    P_dec = (np.abs(np.fft.rfft(aligned_dq[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    return freqs, P_truth, P_dec, int(xs.size)


def smooth_spectrum(h: np.ndarray, bins: int) -> np.ndarray:
    """Boxcar-smooth a 1-D spectrum over ``bins`` frequency points (odd window).
    ``bins <= 1`` returns the input unchanged.  Edges use reflection padding."""
    if bins <= 1:
        return h
    if bins % 2 == 0:
        bins += 1
    pad = bins // 2
    padded = np.pad(h, pad, mode="reflect")
    kernel = np.ones(bins) / bins
    return np.convolve(padded, kernel, mode="valid")


def split_pixel_traces(
    block: np.ndarray,
    active_threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 3-D charge block into active-signal and noise pixel traces.

    A pixel (x, y) is **active** (signal) when its integrated charge exceeds
    ``active_threshold * max_charge``.  A pixel is a **noise** trace when it
    carries non-zero content (i.e. it is not padding) but stays below that
    activity threshold.  Exactly-zero padding pixels are excluded from both.

    Args:
        block: Charge block of shape ``(nx, ny, nt)``.
        active_threshold: Fraction of peak pixel charge separating signal
            from noise pixels (default 0.10).

    Returns:
        ``(signal_traces, noise_traces)``, each of shape ``(n, nt)`` (possibly
        ``n == 0``).
    """
    nt = block.shape[2]
    charge_per_pixel = block.sum(axis=2)  # (nx, ny)
    max_charge = float(charge_per_pixel.max())
    if max_charge <= 0.0:
        empty = np.zeros((0, nt), dtype=np.float64)
        return empty, empty
    has_content = np.any(block != 0.0, axis=2)        # excludes padding
    active = charge_per_pixel > active_threshold * max_charge
    noise = has_content & ~active
    xs_a, ys_a = np.where(active)
    xs_n, ys_n = np.where(noise)
    return block[xs_a, ys_a, :], block[xs_n, ys_n, :]


def accumulate_power(
    npz_paths: list[Path],
    active_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Pool mean temporal power spectra (signal and noise) across NPZ files.

    Each file may have a different number of time bins.  Per-file mean powers
    are computed on the file's own ``rfftfreq(nt, d=1)`` grid (cycles/sample),
    then interpolated onto the longest common grid and trace-count weighted.

    Returns:
        ``(freqs, signal_power, noise_power, n_signal, n_noise)`` with ``freqs``
        in cycles/ADC-sample.  ``noise_power`` is all-zero when no noise pixels
        were found (the caller falls back to a high-frequency-plateau estimate).
    """
    sig_results: list[tuple[np.ndarray, np.ndarray, int]] = []
    noise_results: list[tuple[np.ndarray, np.ndarray, int]] = []

    for path in npz_paths:
        try:
            source = load_source(path)
        except Exception as exc:
            print(f"  Warning: could not load {path}: {exc}")
            continue

        block = source["block"]
        sig_traces, noise_traces = split_pixel_traces(block, active_threshold)
        if sig_traces.shape[0] == 0:
            print(f"  Warning: no active traces in {path}, skipping.")
            continue

        nt = sig_traces.shape[1]
        freqs = np.fft.rfftfreq(nt)  # cycles/ADC-sample, d=1
        sig_power = (np.abs(np.fft.rfft(sig_traces, axis=-1)) ** 2).mean(axis=0)
        sig_results.append((freqs, sig_power, sig_traces.shape[0]))
        if noise_traces.shape[0] > 0:
            noise_power = (np.abs(np.fft.rfft(noise_traces, axis=-1)) ** 2).mean(axis=0)
            noise_results.append((freqs, noise_power, noise_traces.shape[0]))
        print(f"  {path.name}: {sig_traces.shape[0]} signal / "
              f"{noise_traces.shape[0]} noise traces, nt={nt}")

    if not sig_results:
        return np.array([0.0, 0.5]), np.zeros(2), np.zeros(2), 0, 0

    max_nf = max(len(fr) for fr, _, _ in sig_results)
    common_nt = (max_nf - 1) * 2
    common_freqs = np.fft.rfftfreq(common_nt)

    def _pool(results: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, int]:
        total = np.zeros(max_nf, dtype=np.float64)
        weight = 0
        for freqs, power, n in results:
            total += np.interp(common_freqs, freqs, power) * n
            weight += n
        if weight == 0:
            return np.zeros(max_nf, dtype=np.float64), 0
        return total / weight, weight

    sig_power, n_sig = _pool(sig_results)
    noise_power, n_noise = _pool(noise_results)
    return common_freqs, sig_power, noise_power, n_sig, n_noise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a muon-derived time-axis magnitude filter |H(f)| that "
            "corrects for spectral distortion introduced by template "
            "compensation in burst-sequence processing."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("readout", "truth"),
        default="truth",
        help="readout: |H| = sqrt(P_continuous_readout / P_compensated_readout), a "
             "pure boost correcting the compensation power deficit. "
             "truth:   |H| = sqrt(P_smeared_truth / P_deconv_q), the "
             "deconvolution->truth transfer function (can attenuate where "
             "deconv_q over-shoots truth). Default: truth.",
    )
    parser.add_argument(
        "--muon-v3",
        nargs="+",
        default=None,
        metavar="NPZ",
        help="NPZ file(s) from the v3 (continuous sampling) muon run. "
             "Required for --mode readout; unused for --mode truth.",
    )
    parser.add_argument(
        "--muon-v3-burst",
        nargs="+",
        required=True,
        metavar="NPZ",
        help="NPZ file(s) from the v3_burst (template-compensated) muon run. "
             "In --mode truth this supplies both smeared_true and deconv_q.",
    )
    parser.add_argument(
        "--smooth-bins",
        type=int,
        default=9,
        help="Boxcar smoothing window (in frequency bins) applied to |H(f)| to "
             "suppress statistical oscillation (default: 9; use 0/1 to disable).",
    )
    parser.add_argument(
        "--out",
        default="muon_time_filter.npz",
        help="Output NPZ path (default: muon_time_filter.npz).",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.10,
        help="Pixel-activity threshold as a fraction of peak pixel charge "
             "(default: 0.10).  Pixels below this are excluded from the "
             "power-spectrum estimate.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-12,
        help="Tiny numerical guard added to denominators to avoid division by "
             "zero (default: 1e-12).  The Wiener gain W(f) handles the "
             "physical regularisation, so this only needs to be a float epsilon.",
    )
    parser.add_argument(
        "--noise-floor-frac",
        type=float,
        default=1e-2,
        help="Fallback noise floor as a fraction of peak signal power, used "
             "only when no below-threshold noise pixels are available to "
             "estimate N(f) directly (default: 1e-2).",
    )
    return parser.parse_args()


def _pool_truth_deconv(paths, active_threshold):
    """Trace-count-weighted mean (P_truth, P_deconv) across muon NPZ files."""
    results = []
    for path in paths:
        freqs, P_t, P_d, n = truth_deconv_power(path, active_threshold)
        results.append((freqs, P_t, P_d, n))
        print(f"  {path.name}: {n} active pixels, nt={(len(freqs)-1)*2}")
    max_nf = max(len(f) for f, _, _, _ in results)
    common = np.fft.rfftfreq((max_nf - 1) * 2)
    tot_t = np.zeros(max_nf); tot_d = np.zeros(max_nf); w = 0
    for f, P_t, P_d, n in results:
        tot_t += np.interp(common, f, P_t) * n
        tot_d += np.interp(common, f, P_d) * n
        w += n
    return common, tot_t / w, tot_d / w, w


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    paths_v3burst = [Path(p) for p in args.muon_v3_burst]
    eps = args.reg

    if args.mode == "truth":
        # |H(f)| = sqrt(P_smeared_truth / P_deconv_q), measured on the muon.
        # Naturally attenuates (|H|<1) where deconv_q over-shoots the (smoothed)
        # truth, and boosts where it under-shoots.  No W gating: the ratio is
        # self-limiting because P_truth -> 0 in the smoothed high-freq tail.
        print("Mode: truth  (|H| = sqrt(P_truth / P_deconv_q))")
        print(f"muon v3_burst files: {len(paths_v3burst)}")
        common_freqs, P_truth, P_dec, n_pix = _pool_truth_deconv(paths_v3burst, args.active_threshold)
        raw_H = np.sqrt((P_truth + eps) / (P_dec + eps))
        H_mag = smooth_spectrum(raw_H, args.smooth_bins)
        H_mag = np.nan_to_num(H_mag, nan=1.0, posinf=1.0, neginf=1.0)
        print(f"\nactive pixels pooled: {n_pix}")
        print(f"raw |H| range: [{raw_H.min():.3f}, {raw_H.max():.3f}]  "
              f"smoothed |H| range: [{H_mag.min():.3f}, {H_mag.max():.3f}]")

        np.savez(
            out_path, freqs_cycles_per_sample=common_freqs, H_mag=H_mag,
            raw_H=raw_H, P_truth=P_truth, P_deconv=P_dec, n_pixels=n_pix,
            mode="truth", smooth_bins=args.smooth_bins, reg=args.reg,
            active_threshold=args.active_threshold, paths_v3burst=str(paths_v3burst),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saved filter to: {out_path}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(common_freqs, P_truth, color="tab:red", linewidth=1.3,
                     label=f"P_smeared_truth ({n_pix} pixels)")
        axes[0].plot(common_freqs, P_dec, color="tab:green", linewidth=1.3,
                     label="P_deconv_q")
        axes[0].set_ylabel("mean power"); axes[0].set_yscale("log")
        axes[0].set_title("Muon temporal power: smeared truth vs deconv_q")
        axes[0].legend(fontsize=9); axes[0].grid(True, which="both", alpha=0.3)
        axes[1].plot(common_freqs, raw_H, color="grey", linewidth=0.8, alpha=0.6,
                     label="raw sqrt(P_truth/P_deconv)")
        axes[1].plot(common_freqs, H_mag, color="tab:red", linewidth=1.6,
                     label=f"|H(f)| smoothed ({args.smooth_bins} bins)")
        axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        axes[1].set_ylabel("|H(f)|"); axes[1].set_yscale("log")
        axes[1].set_xlabel("frequency [cycles / ADC-sample]")
        axes[1].set_title("truth/deconv_q correction filter |H(f)|  (can attenuate)")
        axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        print(f"Saved diagnostic plot to: {png_path}")
        plt.close(fig)
        return

    # ---- mode == "readout" (original): sqrt(P_continuous / P_compensated) ----
    if args.muon_v3 is None:
        raise SystemExit("--muon-v3 is required for --mode readout")
    paths_v3 = [Path(p) for p in args.muon_v3]
    print("Mode: readout  (|H| = sqrt(P_continuous / P_compensated))")
    print(f"v3 (continuous) files: {len(paths_v3)}  v3_burst files: {len(paths_v3burst)}")

    print("\nProcessing v3 (continuous) ...")
    freqs_v3, P_v3, N_v3, n_v3, n_noise_v3 = accumulate_power(paths_v3, args.active_threshold)
    print("\nProcessing v3_burst (compensated) ...")
    freqs_burst, P_burst, _, n_burst, _ = accumulate_power(paths_v3burst, args.active_threshold)
    if n_v3 == 0 or n_burst == 0:
        raise RuntimeError("No usable traces found; check the NPZ files contain 'hwf_block'.")

    if len(freqs_v3) >= len(freqs_burst):
        common_freqs, P_v3_c, N_c = freqs_v3, P_v3, N_v3
        P_burst_c = np.interp(common_freqs, freqs_burst, P_burst)
    else:
        common_freqs, P_burst_c = freqs_burst, P_burst
        P_v3_c = np.interp(common_freqs, freqs_v3, P_v3)
        N_c = np.interp(common_freqs, freqs_v3, N_v3)

    if n_noise_v3 > 0 and np.any(N_c > 0):
        noise_src = "below-threshold pixels"
    else:
        N_c = np.full_like(P_v3_c, args.noise_floor_frac * float(P_v3_c.max()))
        noise_src = f"fallback flat floor ({args.noise_floor_frac} * peak)"
    print(f"Noise floor source: {noise_src}")

    S = np.maximum(P_v3_c - N_c, 0.0)
    W = S / (S + N_c + eps)
    raw_ratio = np.sqrt((P_v3_c + eps) / (P_burst_c + eps))
    H_mag = 1.0 + W * (raw_ratio - 1.0)
    H_mag = smooth_spectrum(H_mag, args.smooth_bins)
    H_mag = np.nan_to_num(H_mag, nan=1.0, posinf=1.0, neginf=1.0)
    print(f"\n|H| range: [{H_mag.min():.4f}, {H_mag.max():.4f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, freqs_cycles_per_sample=common_freqs, H_mag=H_mag, W=W,
        raw_ratio=raw_ratio, P_v3=P_v3_c, P_v3burst=P_burst_c, N_noise=N_c,
        n_traces_v3=n_v3, n_traces_v3burst=n_burst, mode="readout",
        smooth_bins=args.smooth_bins, reg=args.reg,
        active_threshold=args.active_threshold,
        paths_v3=str(paths_v3), paths_v3burst=str(paths_v3burst),
    )
    print(f"Saved filter to: {out_path}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(common_freqs, P_v3_c, color="black", linewidth=1.2,
                 label=f"P_continuous ({n_v3} traces)")
    axes[0].plot(common_freqs, P_burst_c, color="tab:green", linewidth=1.2,
                 label=f"P_compensated ({n_burst} traces)")
    axes[0].set_ylabel("mean power"); axes[0].set_yscale("log")
    axes[0].set_title("Muon readout power: continuous vs compensated")
    axes[0].legend(fontsize=9); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(common_freqs, H_mag, color="tab:red", linewidth=1.6,
                 label=f"|H(f)| smoothed ({args.smooth_bins} bins)")
    axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("|H(f)|"); axes[1].set_xlabel("frequency [cycles / ADC-sample]")
    axes[1].set_title("readout continuous/compensated filter |H(f)|  (boost only)")
    axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    print(f"Saved diagnostic plot to: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
