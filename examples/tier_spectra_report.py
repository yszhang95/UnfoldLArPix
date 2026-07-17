#!/usr/bin/env python3
"""Fourier-space and temporal-space diagnostics for deconvolution outputs.

For each labeled NPZ (deconv output with ``deconv_q``/``smeared_true``):

- Fourier space: mean per-pixel temporal power of reco divided by truth
  (target = 1 at every frequency), on active pixels.
- Temporal space: time projection (sum over pixels) of reco overlaid on the
  voxel-summed truth, plus the residual (reco - truth) trace.

Usage::

    python examples/tier_spectra_report.py out1.npz out2.npz \
        --labels baseline fixed --out report/tier1a_spectra.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muon_filter_report import align_voxel_blocks  # noqa: E402


def load_aligned(npz_path: Path):
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
    return smear_summed, aligned_dq


def power_ratio(smear_summed, aligned_dq, active_threshold_frac=0.10):
    charge = smear_summed.sum(axis=2)
    cmax = float(charge.max())
    xs, ys = np.where(charge > active_threshold_frac * cmax)
    freqs = np.fft.rfftfreq(smear_summed.shape[2])
    P_true = (np.abs(np.fft.rfft(smear_summed[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    P_dec = (np.abs(np.fft.rfft(aligned_dq[xs, ys, :], axis=-1)) ** 2).mean(axis=0)
    safe = P_true > 0
    r = np.full_like(P_true, np.nan)
    r[safe] = P_dec[safe] / P_true[safe]
    return freqs, r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if len(args.labels) != len(args.files):
        raise SystemExit("--labels count must match files")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    colors = plt.cm.tab10.colors

    truth_proj = None
    for i, (label, path) in enumerate(zip(args.labels, args.files)):
        smear_summed, aligned_dq = load_aligned(Path(path))
        freqs, ratio = power_ratio(smear_summed, aligned_dq)
        axes[0].plot(freqs, ratio, linewidth=1.3, color=colors[i % 10], label=label)

        proj_reco = aligned_dq.sum(axis=(0, 1))
        proj_true = smear_summed.sum(axis=(0, 1))
        t = np.arange(len(proj_reco))
        if truth_proj is None:
            truth_proj = proj_true
            axes[1].plot(t, proj_true, color="black", linewidth=1.8,
                         label="smeared truth (voxel-summed)")
        axes[1].plot(t, proj_reco, linewidth=1.1, color=colors[i % 10],
                     label=label, alpha=0.85)
        axes[2].plot(t, proj_reco - proj_true, linewidth=1.1,
                     color=colors[i % 10], label=f"{label} - truth", alpha=0.85)

    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("frequency [cycles / ADC-sample]")
    axes[0].set_ylabel("P_reco / P_truth")
    axes[0].set_title("Fourier space: reco/truth temporal power ratio (target = 1)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("time bin [ADC samples]")
    axes[1].set_ylabel("charge [ke-/bin]")
    axes[1].set_title("Temporal space: time projection (sum over pixels)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("time bin [ADC samples]")
    axes[2].set_ylabel("residual [ke-/bin]")
    axes[2].set_title("Temporal space: reco - truth residual")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
