"""Is the reconstruction spikier along time than across pixels?

The soft-seed alpha field measures distance in GRID INDICES, so a time
step (adc_hold_delay * time_spacing = 1.5 us = 2.395 mm) costs the same
as a pixel step (4.434 mm).  Per millimetre the time direction is
penalised 1.85x harder, which should push the solution towards
time-localised spikes.  Before attributing anything to that, the effect
has to be shown to exist.

Metric: normalised total variation along each axis,

    TV_ax(q) = sum |q_i+1 - q_i| along ax  /  sum q

evaluated on the universal grid for truth and reco alike, plus the
anisotropy ratio TV_t / TV_xy and the per-pixel time-profile occupancy
(how many time bins hold 90% of a pixel's charge).  A reco that is
spikier in time than the truth shows TV_t/TV_xy above the truth's value
and a smaller 90%-occupancy.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
AO = f"{ROOT}/examples/analysis_output"

from unfoldlarpix.eval.universal import universal_rebin  # noqa: E402

# grid geometry (from the sample metadata and the technote conventions)
PITCH_MM = 4.434
VDRIFT = 1.59645                 # mm/us
TBIN_US = 30 * 0.05              # adc_hold_delay * time_spacing
TBIN_MM = TBIN_US * VDRIFT


def tv(a, axis, denom):
    return float(np.abs(np.diff(a, axis=axis)).sum() / max(denom, 1e-12))


def occupancy(a, frac=0.90):
    """Median number of time bins holding `frac` of a live pixel's charge."""
    tot = a.sum(axis=2)
    live = tot > 1.0
    if not live.any():
        return np.nan
    prof = a[live]
    s = np.sort(prof, axis=1)[:, ::-1]
    c = np.cumsum(s, axis=1) / tot[live][:, None]
    return float(np.median((c < frac).sum(axis=1) + 1))


def one(tag, jobdir=f"{AO}/nb1_fraccensor/B"):
    p = f"{jobdir}/{tag}/{tag}_event_0_0.npz"
    if not os.path.exists(p):
        print(f"{tag}: {p} missing -- skip", flush=True)
        return None
    truth, reco = universal_rebin(p)
    st, sr = truth.sum(), reco.sum()
    row = {"tag": tag, "sum_truth": float(st), "sum_reco": float(sr)}
    for lab, a, s in (("truth", truth, st), ("reco", reco, sr)):
        tx, ty, tt = (tv(a, 0, s), tv(a, 1, s), tv(a, 2, s))
        txy = 0.5 * (tx + ty)
        row[lab] = {"TV_x": tx, "TV_y": ty, "TV_t": tt,
                    "TV_t_over_xy": tt / max(txy, 1e-12),
                    # the same ratio after converting index steps to mm,
                    # i.e. what an isotropic prior in PHYSICAL space sees
                    "TV_t_over_xy_mm": ((tt / TBIN_MM)
                                        / max(txy / PITCH_MM, 1e-12)),
                    "occ90": occupancy(a),
                    "nnz": int((a > 0.01).sum())}
    print(f"\n{tag}:  sum truth {st:.0f} ke, reco {sr:.0f} ke", flush=True)
    print("  %-6s %9s %9s %9s %11s %11s %8s %8s" %
          ("", "TV_x", "TV_y", "TV_t", "TV_t/TV_xy", "same, in mm",
           "occ90", "nnz"))
    for lab in ("truth", "reco"):
        r = row[lab]
        print("  %-6s %9.3f %9.3f %9.3f %11.3f %11.3f %8.1f %8d" %
              (lab, r["TV_x"], r["TV_y"], r["TV_t"], r["TV_t_over_xy"],
               r["TV_t_over_xy_mm"], r["occ90"], r["nnz"]), flush=True)
    a = row["reco"]["TV_t_over_xy"] / max(row["truth"]["TV_t_over_xy"], 1e-12)
    row["excess_time_roughness"] = float(a)
    print(f"  reco/truth anisotropy of roughness = {a:.3f}   "
          f"(>1 means the reco is spikier in TIME than the truth is)",
          flush=True)
    return row


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a25_nb1", "mu_a50_nb1",
                            "mu_a75_nb1", "pos_a00_nb1", "pos_a50_nb1",
                            "pos_a75_nb1"]
    res = [r for r in (one(t) for t in tags) if r]
    json.dump(res, open(f"{AO}/channel_coupling/roughness.json", "w"),
              indent=1)
    print(f"\ngrid: 1 pixel = {PITCH_MM} mm, 1 time bin = {TBIN_US} us "
          f"= {TBIN_MM:.3f} mm, ratio {PITCH_MM/TBIN_MM:.2f}", flush=True)
    print("-> channel_coupling/roughness.json", flush=True)
