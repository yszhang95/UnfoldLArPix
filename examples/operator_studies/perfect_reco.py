"""Is slope > 1 a property of the reconstruction, or of the metric?

Decisive control: feed a PERFECT reconstruction -- the exact bin
integrals of the truth on the fit grid, with sub-bin offsets taken from
the truth's own within-bin centroid -- through the SAME evaluation chain
(universal_rebin, gaussian deposit, sigma_pxl = 0.5) and regress it
against the smeared truth.

If the perfect reconstruction also reads slope > 1, the deviation is a
convention artefact: the reco side deposits each bin's charge as one
Gaussian (a point mass, smeared), while the truth side smears a
continuum, so wherever the truth has structure INSIDE a bin the reco
profile is necessarily peakier at equal integral. Two variants of the
perfect reco isolate that:

  point : offsets = 0, all of a bin's charge at the bin centre
  cen   : offsets = the truth's charge-weighted within-bin centroid
          (what CentroidPositions estimates)
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
AO = f"{ROOT}/examples/analysis_output"
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")
OUT = f"{AO}/slope_probe"
CUT = 0.5
PART = {"mu": "mu", "pos": "positron"}

from eval_alpha_beta import SIG_T, SP  # noqa: E402
from slope_origin import reg, truth_file  # noqa: E402
from unfoldlarpix.eval.universal import universal_rebin  # noqa: E402


def perfect(tag, mode="cen"):
    """Write a fake solved NPZ whose q is the truth's bin integrals."""
    ref = f"{AO}/nb1_fraccensor/B/{tag}/{tag}_event_0_0.npz"
    z = np.load(ref, allow_pickle=True)
    B = float(np.asarray(z["adc_hold_delay"]).ravel()[0])
    boff = np.asarray(z["boffset"], float)
    boff_raw = np.asarray(z["boffset_raw"], float)
    shape = np.asarray(z["deconv_q_sharp"]).shape
    part, ang, nb = (tag.split("_")[0], tag.split("_")[1][1:],
                     tag.split("_")[2][2:])
    f = np.load(f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz",
                allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    nx, ny, nt = shape
    ix = el[:, 0].astype(int) - int(boff_raw[0])
    iy = el[:, 1].astype(int) - int(boff_raw[1])
    tt = (el[:, 2] - boff_raw[2]) / B          # fractional bin position
    it = np.floor(tt).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    q = np.zeros(shape)
    np.add.at(q, (ix[ok], iy[ok], it[ok]), eq[ok])
    if mode == "cen":
        # charge-weighted mean position inside the bin, in fine ticks,
        # measured from the bin CENTRE (the convention of time_offsets)
        num = np.zeros(shape)
        np.add.at(num, (ix[ok], iy[ok], it[ok]),
                  eq[ok] * ((tt[ok] - it[ok]) - 0.5) * B)
        off = np.where(q > 0, num / np.maximum(q, 1e-12), 0.0)
    else:
        off = np.zeros(shape)
    p = f"{OUT}/perfect_{tag}_{mode}.npz"
    np.savez(p, deconv_q_sharp=q.astype(np.float32),
             deconv_q=q.astype(np.float32),
             deconv_q_offsets=off.astype(np.float32),
             boffset=boff, boffset_raw=boff_raw,
             adc_hold_delay=np.array(B))
    return p, float(q.sum())


def score(path, tag, use_offsets=True):
    z = np.load(path, allow_pickle=True)
    to = (np.asarray(z["deconv_q_offsets"], float) if use_offsets else None)
    T, R = universal_rebin(path, truth_npz=truth_file(tag),
                           deposit_shape="gaussian", sigma_time=SIG_T,
                           sigma_pxl=SP, time_offsets=to)
    m = R > CUT
    s, s0, r = reg(T[m], R[m])
    Tp, Rp = T.sum(axis=2), R.sum(axis=2)
    mp = Rp > CUT
    sp, _, rp = reg(Tp[mp], Rp[mp])
    out = {"slope": s, "slope0": s0, "r": r, "n": int(m.sum()),
           "sum_ratio": float(R.sum() / T.sum()),
           "slope_time_integrated": sp, "r_time_integrated": rp}
    del T, R
    gc.collect()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
                            "pos_a50_nb1", "pos_a75_nb1"]
    res = {}
    print("%-12s %-6s %8s %8s %8s %10s %9s" %
          ("sample", "mode", "slope", "slope0", "r", "sumR/T", "slope_tint"))
    for tag in tags:
        res[tag] = {}
        for mode in ("point", "cen"):
            p, sq = perfect(tag, mode)
            a = score(p, tag)
            res[tag][mode] = a
            print("%-12s %-6s %8.3f %8.3f %8.4f %10.3f %9.3f" %
                  (tag, mode, a["slope"], a["slope0"], a["r"],
                   a["sum_ratio"], a["slope_time_integrated"]), flush=True)
    json.dump(res, open(f"{OUT}/perfect_reco.json", "w"), indent=1)
    print(f"-> {OUT}/perfect_reco.json", flush=True)
