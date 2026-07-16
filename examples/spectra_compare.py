#!/usr/bin/env python3
"""Compare per-pixel temporal power spectra of the pre-deconvolution
``hwf_block`` produced by four pipelines:

    v3        - sampling readout baseline (no compensation)
    v1        - dead-time merge compensation
    v2        - template-insertion compensation
    v3_burst  - two-pass dead-time + selective template

The positron is shot isochronous to the anode, so each pixel along the
track sees a nearly time-aligned pulse. We average along the propagation
pixel axis to suppress per-pixel noise, then compute a per-perpendicular-
pixel temporal rFFT to expose how each compensation reshapes the
time-domain trace in frequency.

Usage::

    python examples/spectra_compare.py \
        --npz-v3       <path> \
        --npz-v1       <path> \
        --npz-v2       <path> \
        --npz-v3-burst <path> \
        --out spectra_compare.png \
        [--prop-axis 0|1] \
        [--active-threshold 0.10] \
        [--normalize-dc]
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
    """Per-axis (start, stop) bounding region above ``threshold * proj.max()``.

    Adapted from ``examples/snr2.py:147``.
    """
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
    if "hwf_block" not in data.files:
        raise KeyError(f"{path} does not contain 'hwf_block'")
    if "hwf_block_offset" not in data.files:
        raise KeyError(f"{path} does not contain 'hwf_block_offset'")
    return {
        "block": np.asarray(data["hwf_block"], dtype=np.float64),
        "offset": np.asarray(data["hwf_block_offset"], dtype=np.float64),
        "adc_hold_delay": int(data["adc_hold_delay"]) if "adc_hold_delay" in data.files else None,
        "path": str(path),
    }


def align_spatial(sources: dict) -> tuple[dict, np.ndarray]:
    """Pad each block's pixel axes (0, 1) onto a common canvas via offsets.

    Time axis (axis 2) is left as-is per source. Returns ``aligned`` (dict
    of name -> 3-D block on common spatial grid) and the spatial origin
    ``mins[:2]`` of the canvas in global pixel indices.
    """
    offsets = np.array([s["offset"] for s in sources.values()])
    mins_xy = offsets[:, :2].min(axis=0).astype(int)
    maxs_xy = np.array(
        [s["offset"][:2].astype(int) + np.array(s["block"].shape[:2], dtype=int)
         for s in sources.values()]
    ).max(axis=0)
    nx, ny = (maxs_xy - mins_xy).astype(int)

    aligned: dict = {}
    for name, s in sources.items():
        rel = (s["offset"][:2].astype(int) - mins_xy)
        b = s["block"]
        canvas = np.zeros((int(nx), int(ny), b.shape[2]), dtype=np.float64)
        canvas[rel[0]:rel[0] + b.shape[0],
               rel[1]:rel[1] + b.shape[1], :] = b
        aligned[name] = canvas
    return aligned, mins_xy


def detect_propagation_axis(block: np.ndarray, threshold: float) -> int:
    regions = find_active_region(block, threshold=threshold)
    ext0 = regions[0][1] - regions[0][0]
    ext1 = regions[1][1] - regions[1][0]
    return 0 if ext0 >= ext1 else 1


def collapse_propagation(
    block: np.ndarray, prop_axis: int, perp_axis: int,
    spatial_slices: tuple[slice, slice],
) -> np.ndarray:
    """Slice spatial axes to ``spatial_slices`` (in (axis0, axis1) order),
    average along the propagation axis. Returns 2-D ``(N_perp, N_t)``.
    """
    sub = block[spatial_slices[0], spatial_slices[1], :]
    sub = np.moveaxis(sub, prop_axis, 0)  # (N_prop, N_perp, N_t)
    return sub.mean(axis=0)


def power_spectra(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row one-sided power spectrum along last axis.

    Returns (freqs in cycles/sample, powers shape (N, Nf)).
    """
    freqs = np.fft.rfftfreq(traces.shape[-1])
    powers = np.abs(np.fft.rfft(traces, axis=-1)) ** 2
    return freqs, powers


def pixel_power_spectra(
    block: np.ndarray,
    pixel_axis: int,
    spatial_slices: tuple[slice, slice],
    time_region: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace one-sided power spectrum along a spatial pixel axis.

    Slices the active spatial + temporal region, then for every
    (other_spatial × time) trace computes rfft along ``pixel_axis``.

    Returns (freqs in cycles/pixel, powers shape (N_traces, Nf)).
    """
    t_sl = slice(time_region[0], time_region[1])
    sub = block[spatial_slices[0], spatial_slices[1], t_sl]
    sub = np.moveaxis(sub, pixel_axis, -1)
    n_pix = sub.shape[-1]
    flat = sub.reshape(-1, n_pix)
    freqs = np.fft.rfftfreq(n_pix)
    powers = np.abs(np.fft.rfft(flat, axis=-1)) ** 2
    return freqs, powers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-v3", required=True, type=Path)
    p.add_argument("--npz-v1", required=True, type=Path)
    p.add_argument("--npz-v2", required=True, type=Path)
    p.add_argument("--npz-v3-burst", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Output PNG path; companion NPZ written next to it.")
    p.add_argument("--prop-axis", type=int, choices=[0, 1], default=None,
                   help="Override auto-detection of the propagation pixel axis.")
    p.add_argument("--active-threshold", type=float, default=0.10,
                   help="Threshold (fraction of projection max) for active region.")
    p.add_argument("--normalize-dc", action="store_true",
                   help="Normalize each per-pixel trace to unit DC before FFT.")
    args = p.parse_args()

    paths = {
        "v3": args.npz_v3,
        "v1": args.npz_v1,
        "v2": args.npz_v2,
        "v3_burst": args.npz_v3_burst,
    }
    sources = {name: load_source(path) for name, path in paths.items()}

    holds = {n: s["adc_hold_delay"] for n, s in sources.items() if s["adc_hold_delay"]}
    if len(set(holds.values())) > 1:
        print(f"WARNING: adc_hold_delay differs across sources: {holds}")
    adc_hold_delay = next(iter(holds.values())) if holds else 1
    print(f"adc_hold_delay = {adc_hold_delay} ticks per time bin")
    for name, s in sources.items():
        print(f"  {name:9s} block shape {s['block'].shape}  offset {s['offset'].tolist()}")

    # Spatial alignment onto a common canvas (pixel axes only).
    aligned, mins_xy = align_spatial(sources)
    nx, ny = aligned["v3"].shape[:2]
    print(f"Common spatial canvas: ({nx}, {ny}) at global pixel origin {mins_xy.tolist()}")

    # Use v3 (sampling baseline) to choose active region & propagation axis.
    base = aligned["v3"]
    regions = find_active_region(base, threshold=args.active_threshold)
    print(f"Active region (v3): x={regions[0]}, y={regions[1]}, t={regions[2]}")
    extents = {0: regions[0][1] - regions[0][0],
               1: regions[1][1] - regions[1][0]}
    prop_axis = args.prop_axis if args.prop_axis is not None else (
        0 if extents[0] >= extents[1] else 1)
    perp_axis = 1 - prop_axis
    print(f"Propagation axis = {prop_axis} (extent {extents[prop_axis]} pixels);"
          f" perpendicular axis = {perp_axis} (extent {extents[perp_axis]} pixels)")

    spatial_slices = (slice(regions[0][0], regions[0][1]),
                      slice(regions[1][0], regions[1][1]))

    traces: dict = {}
    for name, blk in aligned.items():
        traces[name] = collapse_propagation(blk, prop_axis, perp_axis, spatial_slices)
    n_perp = traces["v3"].shape[0]
    print(f"Per-source averaged shape (N_perp, N_t): "
          + ", ".join(f"{n}={traces[n].shape}" for n in SOURCES))

    if args.normalize_dc:
        for name in traces:
            dc = traces[name].sum(axis=-1, keepdims=True)
            dc[dc == 0] = 1.0
            traces[name] = traces[name] / dc
        print("Applied unit-DC normalization per pixel trace.")

    # Zero-pad temporal axis to a common length so FFT bins align.
    n_t_common = max(t.shape[-1] for t in traces.values())
    print(f"Padding temporal length to N_t = {n_t_common} for common FFT bins.")
    for name in traces:
        n_t = traces[name].shape[-1]
        if n_t < n_t_common:
            pad = np.zeros((traces[name].shape[0], n_t_common - n_t),
                           dtype=traces[name].dtype)
            traces[name] = np.concatenate([traces[name], pad], axis=-1)

    freqs = None
    powers: dict = {}
    mean_pow: dict = {}
    for name, t in traces.items():
        f, pwr = power_spectra(t)
        if freqs is None:
            freqs = f
        powers[name] = pwr
        mean_pow[name] = pwr.mean(axis=0)

    freq_label = f"frequency [cycles / ({adc_hold_delay} ticks)]"
    pix_freq_label = "frequency [cycles / pixel]"

    # Pixel-axis (propagation direction) power spectra.
    time_region = (regions[2][0], regions[2][1])
    pix_freqs = None
    pix_powers: dict = {}
    pix_mean_pow: dict = {}
    for name, blk in aligned.items():
        f, pwr = pixel_power_spectra(blk, prop_axis, spatial_slices, time_region)
        if pix_freqs is None:
            pix_freqs = f
        pix_powers[name] = pwr
        pix_mean_pow[name] = pwr.mean(axis=0)
    n_pix_traces = pix_powers["v3"].shape[0]
    print(f"Pixel spectra along prop_axis={prop_axis}: "
          f"N_traces={n_pix_traces}, Nf={len(pix_freqs)}")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.4)

    # Panel A: overlay of mean spectra (full-width top row).
    axA = fig.add_subplot(gs[0, :])
    for name in SOURCES:
        axA.plot(freqs, mean_pow[name], color=COLORS[name],
                 label=LABELS[name], linewidth=1.5)
    axA.set_yscale("log")
    axA.set_xlabel(freq_label)
    axA.set_ylabel("mean power")
    axA.set_title(
        f"Mean per-pixel temporal power spectrum  "
        f"(propagation axis {prop_axis} averaged, {n_perp} perpendicular pixels)"
    )
    axA.grid(True, which="both", alpha=0.3)
    axA.legend(loc="best")

    # Panel B: 2x2 grid of per-source 2-D histograms (freq vs log10 power).
    n_freq_bins, n_power_bins = 80, 80
    for i, name in enumerate(SOURCES):
        r, c = 1 + i // 2, i % 2
        ax = fig.add_subplot(gs[r, (c * 2):(c * 2) + 2])
        log_p = np.log10(powers[name] + 1e-30)
        f_rep = np.tile(freqs, log_p.shape[0])
        f_edges = np.linspace(freqs[0], freqs[-1], n_freq_bins + 1)
        p_edges = np.linspace(float(log_p.min()), float(log_p.max()),
                              n_power_bins + 1)
        H, _, _ = np.histogram2d(f_rep, log_p.ravel(),
                                 bins=[f_edges, p_edges])
        ax.imshow(H.T, origin="lower", aspect="auto",
                  extent=[f_edges[0], f_edges[-1], p_edges[0], p_edges[-1]],
                  cmap="viridis")
        ax.plot(freqs, np.log10(mean_pow[name] + 1e-30),
                color=COLORS[name], linewidth=1.5, label="mean")
        ax.set_title(f"{LABELS[name]}  ({powers[name].shape[0]} traces)")
        ax.set_xlabel(freq_label)
        ax.set_ylabel("log₁₀(power)")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Pixel-domain temporal frequency spectra: compensation vs sampling",
        fontsize=13, y=0.995,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {args.out}")

    out_npz = args.out.with_suffix(".npz")
    np.savez(
        out_npz,
        freqs_cycles_per_sample=freqs,
        adc_hold_delay=adc_hold_delay,
        prop_axis=prop_axis,
        perp_axis=perp_axis,
        active_region_x=np.array(regions[0]),
        active_region_y=np.array(regions[1]),
        active_region_t=np.array(regions[2]),
        n_perpendicular_pixels=n_perp,
        **{f"mean_power_{name}": mean_pow[name] for name in SOURCES},
        pix_freqs_cycles_per_pixel=pix_freqs,
        n_pixel_traces=n_pix_traces,
        **{f"pix_mean_power_{name}": pix_mean_pow[name] for name in SOURCES},
    )
    print(f"Saved companion data: {out_npz}")

    # Ratio plot (compensation / sampling baseline) as a separate figure.
    fig2, ax = plt.subplots(figsize=(9, 5))
    denom = mean_pow["v3"].copy()
    safe = denom > 0
    for name in ["v1", "v2", "v3_burst"]:
        ratio = np.full_like(denom, np.nan)
        ratio[safe] = mean_pow[name][safe] / denom[safe]
        ax.plot(freqs, ratio, color=COLORS[name],
                label=f"{LABELS[name]} / {LABELS['v3']}", linewidth=1.5)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(freq_label)
    ax.set_ylabel("mean power ratio")
    ax.set_title("Mean power ratio: compensation pipelines vs sampling baseline")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ratio_out = args.out.with_name(args.out.stem + "_ratio.png")
    fig2.savefig(ratio_out, dpi=150, bbox_inches="tight")
    print(f"Saved ratio figure: {ratio_out}")

    # Figure 3: pixel-axis (propagation direction) spectral comparison.
    fig3 = plt.figure(figsize=(16, 12))
    gs3 = fig3.add_gridspec(3, 4, hspace=0.45, wspace=0.4)

    axP = fig3.add_subplot(gs3[0, :])
    for name in SOURCES:
        axP.plot(pix_freqs, pix_mean_pow[name], color=COLORS[name],
                 label=LABELS[name], linewidth=1.5)
    axP.set_yscale("log")
    axP.set_xlabel(pix_freq_label)
    axP.set_ylabel("mean power")
    axP.set_title(
        f"Mean per-trace spatial power spectrum along propagation axis {prop_axis}  "
        f"({n_pix_traces} perp×time traces)"
    )
    axP.grid(True, which="both", alpha=0.3)
    axP.legend(loc="best")

    for i, name in enumerate(SOURCES):
        r, c = 1 + i // 2, i % 2
        ax3 = fig3.add_subplot(gs3[r, (c * 2):(c * 2) + 2])
        log_p = np.log10(pix_powers[name] + 1e-30)
        f_rep = np.tile(pix_freqs, log_p.shape[0])
        f_edges = np.linspace(pix_freqs[0], pix_freqs[-1], n_freq_bins + 1)
        p_edges = np.linspace(float(log_p.min()), float(log_p.max()),
                              n_power_bins + 1)
        H3, _, _ = np.histogram2d(f_rep, log_p.ravel(), bins=[f_edges, p_edges])
        ax3.imshow(H3.T, origin="lower", aspect="auto",
                   extent=[f_edges[0], f_edges[-1], p_edges[0], p_edges[-1]],
                   cmap="viridis")
        ax3.plot(pix_freqs, np.log10(pix_mean_pow[name] + 1e-30),
                 color=COLORS[name], linewidth=1.5, label="mean")
        ax3.set_title(f"{LABELS[name]}  ({pix_powers[name].shape[0]} traces)")
        ax3.set_xlabel(pix_freq_label)
        ax3.set_ylabel("log₁₀(power)")
        ax3.legend(loc="upper right", fontsize=8)

    fig3.suptitle(
        "Pixel-domain spatial frequency spectra along propagation axis: compensation vs sampling",
        fontsize=13, y=0.995,
    )
    pix_out = args.out.with_name(args.out.stem + "_pixel_spectra.png")
    fig3.savefig(pix_out, dpi=150, bbox_inches="tight")
    print(f"Saved pixel spectra figure: {pix_out}")

    # Figure 4: pixel-axis ratio plot.
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    denom_pix = pix_mean_pow["v3"].copy()
    safe_pix = denom_pix > 0
    for name in ["v1", "v2", "v3_burst"]:
        ratio_pix = np.full_like(denom_pix, np.nan)
        ratio_pix[safe_pix] = pix_mean_pow[name][safe_pix] / denom_pix[safe_pix]
        ax4.plot(pix_freqs, ratio_pix, color=COLORS[name],
                 label=f"{LABELS[name]} / {LABELS['v3']}", linewidth=1.5)
    ax4.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax4.set_yscale("log")
    ax4.set_xlabel(pix_freq_label)
    ax4.set_ylabel("mean power ratio")
    ax4.set_title(
        f"Mean pixel-axis power ratio (prop axis {prop_axis}): compensation vs sampling baseline"
    )
    ax4.legend()
    ax4.grid(True, which="both", alpha=0.3)
    pix_ratio_out = args.out.with_name(args.out.stem + "_pixel_ratio.png")
    fig4.savefig(pix_ratio_out, dpi=150, bbox_inches="tight")
    print(f"Saved pixel ratio figure: {pix_ratio_out}")


if __name__ == "__main__":
    main()
