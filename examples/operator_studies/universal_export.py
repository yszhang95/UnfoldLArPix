"""Final deliverable: the estimate as VOXELS on the universal grid.

Whatever parameterisation a solve uses internally (per-bin amplitudes,
band-limited coefficients, pulses), the product of the unfolding is a
charge per universal voxel: one pixel by one adc_hold_delay bin, pixels
in absolute hardware indices, bins anchored at global multiples of the
bin width. This module renders any solved NPZ onto that grid, next to
the truth smeared exactly once, and writes a self-describing file:

    charge_reco   (nx, ny, nt)  ke per voxel
    charge_truth  (nx, ny, nt)  smeared truth, same grid
    pixel_origin  (2,)          absolute pixel index of [0, 0]
    bin_origin    ()            absolute universal bin index of [.., 0]
    bin_ticks     ()            fine ticks per bin
    meta          json          provenance and the resolution convention

Each side is smeared once and with the same kernel: the truth by
smear_effective_charge, the reconstruction by the gaussian deposit of
its stored coefficients. Nothing is smeared twice.

Usage: universal_export.py <solved.npz> <tag> [outdir]
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/examples/analysis_output/_drivers")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"

from eval_alpha_beta import SIG_T, SP  # noqa: E402
from smeared_solve import truth_npz  # noqa: E402
from unfoldlarpix.eval.universal import (metrics_from_blocks,  # noqa: E402
                                         universal_rebin)


def export(solved, tag, outdir=f"{AO}/universal_voxels", label=None,
           use_offsets=True):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(solved, allow_pickle=True)
    to = np.asarray(z["deconv_q_offsets"], float) if use_offsets else None
    T, R = universal_rebin(solved, truth_npz=truth_npz(tag),
                           deposit_shape="gaussian", sigma_time=SIG_T,
                           sigma_pxl=SP, time_offsets=to)
    B = int(np.asarray(z["adc_hold_delay"]).ravel()[0])
    boff = np.asarray(z["boffset"], float)
    # universal_rebin's frame: pixels start at min(block, truth) origin and
    # bins at the corresponding global multiple of B. Recover both from the
    # truth file's smear_offset, which is in absolute fine ticks.
    t = np.load(truth_npz(tag), allow_pickle=True)
    s_off = np.asarray(t["smear_offset"], np.int64)
    p0 = (min(int(boff[0]), int(s_off[0])), min(int(boff[1]), int(s_off[1])))
    b0 = min(int(np.floor(boff[2] / B)), int(s_off[2] // B))
    m = metrics_from_blocks(T, R, corr_threshold=0.5)
    meta = {"tag": tag, "label": label or os.path.basename(
        os.path.dirname(solved)), "source": solved,
        "sigma_time": SIG_T, "sigma_pixel": SP,
        "resolution": "each side smeared exactly once with the same "
                      "Gaussian: 1/(2 pi sigma) = 32 fine ticks (1.6 us) "
                      "in time, 0.318 pixels transversally",
        "metrics": {k: float(m[k]) for k in
                    ("pearson_r", "slope", "integral_pct", "ghost_frac",
                     "ghost_iso_frac", "ghost_iso_charge", "true_killed")}}
    name = f"{tag}_{meta['label']}"
    out = f"{outdir}/{name}_universal.npz"
    np.savez_compressed(out, charge_reco=R.astype(np.float32),
                        charge_truth=T.astype(np.float32),
                        pixel_origin=np.array(p0, np.int64),
                        bin_origin=np.array(b0, np.int64),
                        bin_ticks=np.array(B, np.int64),
                        meta=json.dumps(meta))
    print(f"{name}: grid {R.shape}, reco {R.sum():.1f} ke, "
          f"truth {T.sum():.1f} ke, r {m['pearson_r']:.4f}, "
          f"slope {m['slope']:.3f} -> {os.path.basename(out)}", flush=True)
    del T, R
    gc.collect()
    return out, meta


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        export(sys.argv[1], sys.argv[2],
               *(sys.argv[3:4] or [f"{AO}/universal_voxels"]))
    else:
        # default: every configuration of the comparison, both arms
        TAGS = ["mu_a00_nb1", "mu_a75_nb1", "pos_a00_nb1", "pos_a75_nb1",
                "mu_a00_nb4", "mu_a75_nb4", "pos_a00_nb4", "pos_a75_nb4"]
        made = []
        for tag in TAGS:
            nb = tag.split("_")[2]
            base = (f"{AO}/nb1_fraccensor/B/{tag}/{tag}_event_0_0.npz"
                    if nb == "nb1" else
                    f"{AO}/angscan_tau/{tag}/{tag}_event_0_0.npz")
            if os.path.exists(base):
                made.append(export(base, tag, label="stock"))
            sm = f"{AO}/smeared_solve/a3/{tag}_event_0_0.npz"
            if os.path.exists(sm):
                made.append(export(sm, tag, label="smeared"))
        print(f"\n{len(made)} files -> {AO}/universal_voxels/", flush=True)
