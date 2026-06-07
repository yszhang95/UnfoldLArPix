#!/usr/bin/env python3
"""Diagnose multi-band structure and medium-frequency excess in per-pixel
temporal spectra of the four compensation/sampling outputs.

Three diagnostic figures are produced:

    1. Per-pixel decomposition  -- each perpendicular pixel's spectrum
       shown individually, ranked by integrated charge. Reveals which
       pixels (high-signal vs noise-dominated) contribute the bright
       "bands" seen in the 2-D histogram of spectra_compare.py.

    2. Residual spectrum        -- for each compensation pipeline we
       circular-shift the trace into temporal alignment with v3 (sampling
       baseline), then plot |FFT(comp - v3)|^2 to expose periodic
       structure introduced by template injection. Peaks identify the
       characteristic period of the artefact.

    3. Wiener-style correction  -- a frequency-domain amplitude filter
       H(f) = sqrt( <|V3|^2> / <|V_comp|^2> ) is calibrated per pipeline
       (averaged across pixels), applied to the compensation traces, and
       the corrected ratio (comp_corr / v3) is plotted. By construction
       the mean ratio is flat after correction; deviations of individual
       pixels reveal residual coherence loss that an amplitude filter
       alone cannot fix.

Usage::

    python examples/spectra_diagnose.py \
        --npz-v3       <path> \
        --npz-v1       <path> \
        --npz-v2       <path> \
        --npz-v3-burst <path> \
        --out-prefix output_spectra_compare/diag
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCES = ["v3", "v1", "v2", "v3_burst"]
COLORS = {
    "v3": "black",
    "v1": "tab:blue",
    "v2": "tab:orange",
    "v3_burst": "tab:green",
}
LABELS = {
    "v3": "v3 (sampling)",
    "v1": "v1 (dead-time)",
    "v2": "v2 (template)",
    "v3_burst": "v3_burst (selective)",
}


def find_active_region(block: np.ndarray, threshold: float = 0.10):
    if block.ndim != 3:
        raise ValueError("block must be 3-D")
    regions = []
    for axis in range(3):
        sum_axes = tuple(i for i in range(3) if i != axis)
        proj = block.sum(axis=sum_axes)
        proj_max = float(proj.max()) if proj.size else 0.0
        if proj_max <= 0.0:
            regions.append((0, proj.size))
            continue
        cutoff = threshold * proj_max
        above = np.where(proj > cutoff)[0]
        if above.size == 0:
            regions.append((0, proj.size))
        else:
            regions.append((int(above[0]), int(above[-1]) + 1))
    return regions


def load_source(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "block": np.asarray(data["hwf_block"], dtype=np.float64),
        "offset": np.asarray(data["hwf_block_offset"], dtype=np.float64),
        "adc_hold_delay": int(data["adc_hold_delay"]) if "adc_hold_delay" in data.files else None,
        "path": str(path),
    }


def align_spatial(sources: dict) -> tuple[dict, np.ndarray]:
    offsets = np.array([s["offset"] for s in sources.values()])
    mins_xy = offsets[:, :2].min(axis=0).astype(int)
    maxs_xy = np.array(
        [s["offset"][:2].astype(int) + np.array(s["block"].shape[:2], dtype=int)
         for s in sources.values()]
    ).max(axis=0)
    nx, ny = (maxs_xy - mins_xy).astype(int)
    aligned = {}
    for name, s in sources.items():
        rel = (s["offset"][:2].astype(int) - mins_xy)
        b = s["block"]
        canvas = np.zeros((int(nx), int(ny), b.shape[2]), dtype=np.float64)
        canvas[rel[0]:rel[0] + b.shape[0],
               rel[1]:rel[1] + b.shape[1], :] = b
        aligned[name] = canvas
    return aligned, mins_xy


def collapse_propagation(block, prop_axis, spatial_slices):
    sub = block[spatial_slices[0], spatial_slices[1], :]
    sub = np.moveaxis(sub, prop_axis, 0)  # (N_prop, N_perp, N_t)
    return sub.mean(axis=0)


def time_align_to_v3(traces: dict, offsets_t: dict, adc_hold_delay: int,
                    n_t_common: int) -> dict:
    """Zero-pad each source to n_t_common and circular-shift so that
    absolute time = offset[2] + i*adc_hold_delay aligns across sources.

    Sub-bin offset (e.g. 28.73 bins) is rounded to nearest integer; the
    residual fractional shift (<1 bin) is acceptable for amplitude-based
    diagnostics.
    """
    aligned: dict = {}
    ref_offset = offsets_t["v3"]
    for name, t in traces.items():
        n_t = t.shape[-1]
        if n_t < n_t_common:
            pad = np.zeros((t.shape[0], n_t_common - n_t), dtype=t.dtype)
            t = np.concatenate([t, pad], axis=-1)
        # bins to shift so that this source's i=0 lines up with v3's i=0
        delta_bins = (offsets_t[name] - ref_offset) / adc_hold_delay
        shift = int(np.round(delta_bins))
        # rolling forward by `shift` puts content originally at index 0
        # at index `shift`, matching v3 absolute time.
        aligned[name] = np.roll(t, shift, axis=-1)
    return aligned


def power_spectra(traces: np.ndarray):
    freqs = np.fft.rfftfreq(traces.shape[-1])
    spec = np.fft.rfft(traces, axis=-1)
    powers = np.abs(spec) ** 2
    return freqs, spec, powers


def pixel_power_spectra(
    block: np.ndarray,
    pixel_axis: int,
    spatial_slices: tuple[slice, slice],
    time_region: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trace one-sided FFT along a spatial pixel axis.

    Slices the active region, then for every (other_spatial × time) trace
    computes rfft along ``pixel_axis``.

    Returns (freqs in cycles/pixel, spec shape (N_traces, Nf), powers shape (N_traces, Nf)).
    """
    t_sl = slice(time_region[0], time_region[1])
    sub = block[spatial_slices[0], spatial_slices[1], t_sl]
    sub = np.moveaxis(sub, pixel_axis, -1)
    n_pix = sub.shape[-1]
    flat = sub.reshape(-1, n_pix)
    freqs = np.fft.rfftfreq(n_pix)
    spec = np.fft.rfft(flat, axis=-1)
    powers = np.abs(spec) ** 2
    return freqs, spec, powers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-v3", required=True, type=Path)
    p.add_argument("--npz-v1", required=True, type=Path)
    p.add_argument("--npz-v2", required=True, type=Path)
    p.add_argument("--npz-v3-burst", required=True, type=Path)
    p.add_argument("--out-prefix", required=True, type=Path,
                   help="Path prefix for output PNGs (e.g. .../diag)")
    p.add_argument("--prop-axis", type=int, choices=[0, 1], default=None)
    p.add_argument("--active-threshold", type=float, default=0.10)
    args = p.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "v3": args.npz_v3, "v1": args.npz_v1,
        "v2": args.npz_v2, "v3_burst": args.npz_v3_burst,
    }
    sources = {n: load_source(p) for n, p in paths.items()}

    holds = {n: s["adc_hold_delay"] for n, s in sources.items() if s["adc_hold_delay"]}
    if len(set(holds.values())) > 1:
        print(f"WARNING: adc_hold_delay differs: {holds}")
    adc_hold_delay = next(iter(holds.values())) if holds else 1

    aligned, _ = align_spatial(sources)
    base = aligned["v3"]
    regions = find_active_region(base, args.active_threshold)
    print(f"Active region (v3): x={regions[0]}, y={regions[1]}, t={regions[2]}")
    extents = {0: regions[0][1] - regions[0][0],
               1: regions[1][1] - regions[1][0]}
    prop_axis = args.prop_axis if args.prop_axis is not None else (
        0 if extents[0] >= extents[1] else 1)
    perp_axis = 1 - prop_axis
    spatial_slices = (slice(regions[0][0], regions[0][1]),
                      slice(regions[1][0], regions[1][1]))
    print(f"Propagation axis = {prop_axis}, perpendicular axis = {perp_axis}")

    traces = {n: collapse_propagation(b, prop_axis, spatial_slices)
              for n, b in aligned.items()}
    n_perp = traces["v3"].shape[0]
    print(f"N_perp = {n_perp}")
    for n in SOURCES:
        print(f"  {n:9s} (N_perp, N_t) = {traces[n].shape}")

    # Pad all to common N_t and time-align (circular shift)
    n_t_common = max(t.shape[-1] for t in traces.values())
    offsets_t = {n: float(s["offset"][2]) for n, s in sources.items()}
    traces = time_align_to_v3(traces, offsets_t, adc_hold_delay, n_t_common)
    print(f"Time-aligned to v3, common N_t = {n_t_common}")

    # Pixel-axis (propagation direction) spectra from original aligned blocks.
    time_region = (regions[2][0], regions[2][1])
    pix_freqs = None
    pix_spec_px: dict = {}
    pix_powers_px: dict = {}
    pix_mean_pow_px: dict = {}
    for name, blk in aligned.items():
        f, sp, pw = pixel_power_spectra(blk, prop_axis, spatial_slices, time_region)
        if pix_freqs is None:
            pix_freqs = f
        pix_spec_px[name] = sp
        pix_powers_px[name] = pw
        pix_mean_pow_px[name] = pw.mean(axis=0)
    n_pix_traces = pix_powers_px["v3"].shape[0]
    print(f"Pixel spectra (prop_axis={prop_axis}): N_traces={n_pix_traces}, Nf={len(pix_freqs)}")
    pix_freq_label = "frequency [cycles / pixel]"

    # Spatial residual: pixel-axis FFT of (comp - v3) for each compensation pipeline.
    # Clip to common time length when sources have different temporal extents.
    pix_res_powers: dict = {}
    nt_min = min(aligned[n].shape[2] for n in ["v1", "v2", "v3_burst", "v3"])
    v3_clip = aligned["v3"][..., :nt_min].astype(np.float64)
    clipped_time_region = (
        min(time_region[0], nt_min),
        min(time_region[1], nt_min),
    )
    for name in ["v1", "v2", "v3_burst"]:
        diff_blk = aligned[name][..., :nt_min].astype(np.float64) - v3_clip
        _, _, pw = pixel_power_spectra(diff_blk, prop_axis, spatial_slices, clipped_time_region)
        pix_res_powers[name] = pw

    # Per-pixel integrated charge from v3 (truth-most reference)
    pixel_charge = traces["v3"].sum(axis=-1)  # (N_perp,)
    rank = np.argsort(-pixel_charge)  # descending
    print("Per-pixel integrated charge (v3):",
          [(int(i), float(pixel_charge[i])) for i in rank])

    # Per-pixel spectra
    freqs = None
    spec: dict = {}
    powers: dict = {}
    for n in SOURCES:
        f, sp, pw = power_spectra(traces[n])
        freqs = f
        spec[n] = sp
        powers[n] = pw

    freq_label = f"frequency [cycles / ({adc_hold_delay} ticks)]"

    # ----- Figure 1: per-pixel spectra (N_perp panels) -----
    ncols = 4
    nrows = int(np.ceil(n_perp / ncols))
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                               squeeze=False)
    for plot_i, pix in enumerate(rank):
        r, c = plot_i // ncols, plot_i % ncols
        ax = axes1[r][c]
        for n in SOURCES:
            ax.plot(freqs, powers[n][pix] + 1e-30, color=COLORS[n],
                    label=LABELS[n], linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel(freq_label, fontsize=8)
        ax.set_ylabel("power", fontsize=8)
        ax.set_title(f"perp-pixel {pix}  (charge={pixel_charge[pix]:.1f} ke)",
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        if plot_i == 0:
            ax.legend(fontsize=7)
    for plot_i in range(n_perp, nrows * ncols):
        axes1[plot_i // ncols][plot_i % ncols].axis("off")
    fig1.suptitle("Per-pixel temporal power spectra (ranked by v3 integrated charge)",
                  fontsize=12, y=0.995)
    fig1.tight_layout()
    fig1_path = args.out_prefix.with_name(args.out_prefix.name + "_per_pixel_spectra.png")
    fig1.savefig(fig1_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig1_path}")

    # ----- Figure 2: per-pixel ratio (compensation / v3) -----
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                               squeeze=False)
    for plot_i, pix in enumerate(rank):
        r, c = plot_i // ncols, plot_i % ncols
        ax = axes2[r][c]
        denom = powers["v3"][pix].copy()
        safe = denom > 0
        for n in ["v1", "v2", "v3_burst"]:
            ratio = np.full_like(denom, np.nan)
            ratio[safe] = powers[n][pix][safe] / denom[safe]
            ax.plot(freqs, ratio, color=COLORS[n], label=LABELS[n],
                    linewidth=1.0)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_xlabel(freq_label, fontsize=8)
        ax.set_ylabel("ratio (comp / v3)", fontsize=8)
        ax.set_title(f"perp-pixel {pix}  (charge={pixel_charge[pix]:.1f} ke)",
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        if plot_i == 0:
            ax.legend(fontsize=7)
    for plot_i in range(n_perp, nrows * ncols):
        axes2[plot_i // ncols][plot_i % ncols].axis("off")
    fig2.suptitle("Per-pixel compensation/sampling ratio (ranked by v3 charge)",
                  fontsize=12, y=0.995)
    fig2.tight_layout()
    fig2_path = args.out_prefix.with_name(args.out_prefix.name + "_per_pixel_ratio.png")
    fig2.savefig(fig2_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig2_path}")

    # ----- Figure 3: residual analysis (top-charge pixel) -----
    top_pix = int(rank[0])
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 8))
    fig3.suptitle(f"Residual analysis — perpendicular pixel {top_pix} "
                  f"(highest v3 charge = {pixel_charge[top_pix]:.1f} ke)",
                  fontsize=12)

    t_idx = np.arange(n_t_common)
    for col, comp_name in enumerate(["v1", "v2", "v3_burst"]):
        # Top row: time domain (zoomed to active region)
        ax_t = axes3[0, col]
        v3_trace = traces["v3"][top_pix]
        comp_trace = traces[comp_name][top_pix]
        residual = comp_trace - v3_trace
        # find active region in time
        absmax = np.max(np.abs(v3_trace))
        if absmax > 0:
            active = np.where(np.abs(v3_trace) > 0.05 * absmax)[0]
            if active.size:
                t0 = max(0, int(active[0]) - 30)
                t1 = min(n_t_common, int(active[-1]) + 30)
            else:
                t0, t1 = 0, n_t_common
        else:
            t0, t1 = 0, n_t_common
        ax_t.plot(t_idx[t0:t1], v3_trace[t0:t1], color=COLORS["v3"],
                  label=LABELS["v3"], linewidth=1.0)
        ax_t.plot(t_idx[t0:t1], comp_trace[t0:t1], color=COLORS[comp_name],
                  label=LABELS[comp_name], linewidth=1.0, alpha=0.85)
        ax_t.plot(t_idx[t0:t1], residual[t0:t1], color="red",
                  label="residual", linewidth=0.8, alpha=0.7)
        ax_t.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax_t.set_xlabel(f"time bin (1 bin = {adc_hold_delay} ticks)", fontsize=9)
        ax_t.set_ylabel("charge")
        ax_t.set_title(f"{LABELS[comp_name]}: time domain", fontsize=10)
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)

        # Bottom row: residual power spectrum
        ax_f = axes3[1, col]
        res_power = np.abs(np.fft.rfft(residual)) ** 2
        ax_f.plot(freqs, res_power + 1e-30, color="red", linewidth=1.0,
                  label="|FFT(residual)|^2")
        ax_f.plot(freqs, powers[comp_name][top_pix] + 1e-30,
                  color=COLORS[comp_name], linewidth=1.0, alpha=0.5,
                  label=f"|FFT({comp_name})|^2")
        # Mark dominant peaks of the residual spectrum
        if res_power.size > 5:
            # ignore DC and near-DC
            search = res_power.copy()
            search[:3] = 0
            top_freq_idx = np.argsort(-search)[:3]
            for k, fi in enumerate(top_freq_idx):
                ax_f.axvline(freqs[fi], color="purple", linestyle=":",
                             linewidth=0.7, alpha=0.7)
                period = (1.0 / freqs[fi]) if freqs[fi] > 0 else np.inf
                ax_f.text(freqs[fi], res_power.max() * (0.5 ** k),
                          f"{freqs[fi]:.3f}\n(T≈{period:.1f} bins\n={period * adc_hold_delay:.0f} tk)",
                          fontsize=7, color="purple", ha="left")
        ax_f.set_yscale("log")
        ax_f.set_xlabel(freq_label, fontsize=9)
        ax_f.set_ylabel("power")
        ax_f.set_title(f"{LABELS[comp_name]}: residual spectrum", fontsize=10)
        ax_f.legend(fontsize=8)
        ax_f.grid(True, which="both", alpha=0.3)

    fig3.tight_layout()
    fig3_path = args.out_prefix.with_name(args.out_prefix.name + "_residual.png")
    fig3.savefig(fig3_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig3_path}")

    # ----- Figure 4: Wiener-style amplitude correction -----
    # Calibrate H(f) per source from cross-power (per-pixel mean):
    #   H(f) = mean( V3 conj(V_comp) ) / mean( |V_comp|^2 )
    # This is the optimal Wiener filter assuming linear comp -> v3 model.
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 9))

    H = {}
    for n in ["v1", "v2", "v3_burst"]:
        cross = (spec["v3"] * np.conj(spec[n])).mean(axis=0)
        denom = (np.abs(spec[n]) ** 2).mean(axis=0)
        H_f = np.zeros_like(cross)
        ok = denom > 0
        H_f[ok] = cross[ok] / denom[ok]
        H[n] = H_f

    ax = axes4[0]
    for n in ["v1", "v2", "v3_burst"]:
        ax.plot(freqs, np.abs(H[n]), color=COLORS[n],
                label=f"|H_{n}(f)|", linewidth=1.2)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel(freq_label)
    ax.set_ylabel("|H(f)|")
    ax.set_yscale("log")
    ax.set_title("Wiener filter magnitude — H(f) = <V3 V_comp*> / <|V_comp|²>")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Apply H(f) to compensation traces, recompute mean ratio
    ax = axes4[1]
    mean_pow_v3 = powers["v3"].mean(axis=0)
    for n in ["v1", "v2", "v3_burst"]:
        spec_corr = spec[n] * H[n][None, :]
        pow_corr = np.abs(spec_corr) ** 2
        mean_pow_corr = pow_corr.mean(axis=0)
        # Original ratio
        denom = mean_pow_v3.copy()
        safe = denom > 0
        ratio_orig = np.full_like(denom, np.nan)
        ratio_orig[safe] = powers[n].mean(axis=0)[safe] / denom[safe]
        # Corrected ratio
        ratio_corr = np.full_like(denom, np.nan)
        ratio_corr[safe] = mean_pow_corr[safe] / denom[safe]
        ax.plot(freqs, ratio_orig, color=COLORS[n], linestyle="--", alpha=0.5,
                label=f"{LABELS[n]} / v3 (uncorrected)", linewidth=1.0)
        ax.plot(freqs, ratio_corr, color=COLORS[n], linestyle="-",
                label=f"{LABELS[n]} / v3 (Wiener-corrected)", linewidth=1.4)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel(freq_label)
    ax.set_ylabel("mean power ratio")
    ax.set_yscale("log")
    ax.set_title("Mean ratio before vs after Wiener correction")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    fig4.tight_layout()
    fig4_path = args.out_prefix.with_name(args.out_prefix.name + "_wiener.png")
    fig4.savefig(fig4_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig4_path}")

    # ----- Figure 5: pixel-axis 2-D power spectrum histograms (one per source) -----
    fig5, axes5 = plt.subplots(1, 4, figsize=(18, 5))
    for i, name in enumerate(SOURCES):
        ax5 = axes5[i]
        log_p5 = np.log10(pix_powers_px[name] + 1e-30)
        f_rep5 = np.tile(pix_freqs, log_p5.shape[0])
        f_edges5 = np.linspace(pix_freqs[0], pix_freqs[-1], 61)
        p_edges5 = np.linspace(float(log_p5.min()), float(log_p5.max()), 61)
        H5, _, _ = np.histogram2d(f_rep5, log_p5.ravel(), bins=[f_edges5, p_edges5])
        ax5.imshow(H5.T, origin="lower", aspect="auto",
                   extent=[f_edges5[0], f_edges5[-1], p_edges5[0], p_edges5[-1]],
                   cmap="viridis")
        ax5.plot(pix_freqs, np.log10(pix_mean_pow_px[name] + 1e-30),
                 color=COLORS[name], linewidth=1.5, label="mean")
        ax5.set_title(f"{LABELS[name]}\n({pix_powers_px[name].shape[0]} traces)", fontsize=10)
        ax5.set_xlabel(pix_freq_label, fontsize=9)
        ax5.set_ylabel("log₁₀(power)", fontsize=9)
        ax5.legend(fontsize=8)
    fig5.suptitle(
        f"Pixel-domain spatial power spectra along propagation axis {prop_axis}  "
        f"({n_pix_traces} perp×time traces)",
        fontsize=12, y=1.01,
    )
    fig5.tight_layout()
    fig5_path = args.out_prefix.with_name(args.out_prefix.name + "_pixel_spectra.png")
    fig5.savefig(fig5_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig5_path}")

    # ----- Figure 6: pixel-axis ratio + spatial residual spectra -----
    fig6, axes6 = plt.subplots(2, 3, figsize=(15, 9))
    fig6.suptitle(
        f"Pixel-axis (prop axis {prop_axis}) ratio and spatial residual spectra",
        fontsize=12,
    )
    denom_pix = pix_mean_pow_px["v3"].copy()
    safe_pix = denom_pix > 0
    for col, comp_name in enumerate(["v1", "v2", "v3_burst"]):
        # Top row: mean pixel-axis power ratio (comp / v3)
        ax_r = axes6[0, col]
        ratio_pix = np.full_like(denom_pix, np.nan)
        ratio_pix[safe_pix] = pix_mean_pow_px[comp_name][safe_pix] / denom_pix[safe_pix]
        ax_r.plot(pix_freqs, ratio_pix, color=COLORS[comp_name], linewidth=1.2)
        ax_r.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        ax_r.set_yscale("log")
        ax_r.set_xlabel(pix_freq_label, fontsize=9)
        ax_r.set_ylabel("mean power ratio", fontsize=9)
        ax_r.set_title(f"{LABELS[comp_name]} / v3  (mean ratio)", fontsize=10)
        ax_r.grid(True, which="both", alpha=0.3)

        # Bottom row: spatial residual |FFT_x(comp - v3)|^2
        ax_s = axes6[1, col]
        res_mean = pix_res_powers[comp_name].mean(axis=0)
        ax_s.plot(pix_freqs, res_mean + 1e-30, color="red",
                  linewidth=1.0, label="|FFT_x(comp−v3)|²")
        ax_s.plot(pix_freqs, pix_mean_pow_px[comp_name] + 1e-30,
                  color=COLORS[comp_name], linewidth=1.0, alpha=0.5,
                  label=f"|FFT_x({comp_name})|²")
        ax_s.plot(pix_freqs, denom_pix + 1e-30, color=COLORS["v3"],
                  linewidth=1.0, alpha=0.5, label="|FFT_x(v3)|²")
        # Mark top-3 residual peaks (skip near-DC)
        if res_mean.size > 5:
            search = res_mean.copy()
            search[:2] = 0
            top_idx = np.argsort(-search)[:3]
            for fi in top_idx:
                if pix_freqs[fi] > 0:
                    period = 1.0 / pix_freqs[fi]
                    ax_s.axvline(pix_freqs[fi], color="purple", linestyle=":",
                                 linewidth=0.7, alpha=0.7)
                    ax_s.text(pix_freqs[fi], res_mean.max() * 0.5,
                              f"{pix_freqs[fi]:.3f}\n(T≈{period:.1f} px)",
                              fontsize=7, color="purple", ha="left")
        ax_s.set_yscale("log")
        ax_s.set_xlabel(pix_freq_label, fontsize=9)
        ax_s.set_ylabel("mean power", fontsize=9)
        ax_s.set_title(f"{LABELS[comp_name]}: spatial residual spectrum", fontsize=10)
        ax_s.legend(fontsize=8)
        ax_s.grid(True, which="both", alpha=0.3)

    fig6.tight_layout()
    fig6_path = args.out_prefix.with_name(args.out_prefix.name + "_pixel_ratio.png")
    fig6.savefig(fig6_path, dpi=140, bbox_inches="tight")
    print(f"Saved {fig6_path}")

    # Save filters and inputs for downstream use
    npz_path = args.out_prefix.with_suffix(".npz")
    np.savez(
        npz_path,
        freqs_cycles_per_sample=freqs,
        adc_hold_delay=adc_hold_delay,
        prop_axis=prop_axis,
        n_perp=n_perp,
        pixel_charge=pixel_charge,
        rank_by_charge=rank,
        **{f"power_{n}": powers[n] for n in SOURCES},
        **{f"H_{n}": H[n] for n in ["v1", "v2", "v3_burst"]},
        pix_freqs_cycles_per_pixel=pix_freqs,
        n_pixel_traces=n_pix_traces,
        **{f"pix_power_{n}": pix_powers_px[n] for n in SOURCES},
        **{f"pix_res_power_{n}": pix_res_powers[n] for n in ["v1", "v2", "v3_burst"]},
    )
    print(f"Saved companion data: {npz_path}")


if __name__ == "__main__":
    main()
