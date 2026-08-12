"""Where does slope > 1 come from: amplitude, selection, or time?

The reported slope regresses reco on smeared truth over voxels selected
by reco > 0.5 ke. Three candidate causes are separated on the SAME
aligned (truth, reco) universal-grid pair:

  selection : redo the regression selecting on truth instead of reco, and
              on the union; a selection-induced slope moves a lot.
  time      : integrate the time axis per pixel (and 3-bin smooth) before
              regressing. If charge is merely misplaced in time within
              the right envelope, the slope collapses to ~1.
  amplitude : the pixel-integrated ratio and the through-origin slope,
              which are insensitive to placement.

Also reports the per-pixel time shift (cross-correlation lag) and the
peak-to-peak ratio, to characterise the misplacement.
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

from eval_alpha_beta import SIG_T, SP  # noqa: E402
from unfoldlarpix.data_loader import DataLoader  # noqa: E402
from unfoldlarpix.deconv_workflow import smear_effective_charge  # noqa: E402
from unfoldlarpix.eval.universal import universal_rebin  # noqa: E402

CUT = 0.5
PART = {"mu": "mu", "pos": "positron"}
OUT = f"{AO}/slope_probe"


def truth_file(tag):
    part, ang, nb = (tag.split("_")[0], tag.split("_")[1][1:],
                     tag.split("_")[2][2:])
    p = f"{OUT}/truth_{tag}_sp{SP}.npz"
    if not os.path.exists(p):
        os.makedirs(OUT, exist_ok=True)
        ev = [e for e in DataLoader(
            f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz"
        ).iter_events() if e.hits and e.tpc_id == 0][0]
        off, smt = smear_effective_charge(ev, sigma_time=SIG_T, sigma_pixel=SP)
        np.savez(p, smeared_true=smt, smear_offset=np.array(off))
        del smt, ev
        gc.collect()
    return p


def reg(x, y):
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    s = float(np.polyfit(x, y, 1)[0])
    s0 = float((x * y).sum() / max((x * x).sum(), 1e-30))
    r = float(np.corrcoef(x, y)[0, 1])
    return s, s0, r


def analyse(tag, arm="B"):
    p = f"{AO}/nb1_fraccensor/{arm}/{tag}/{tag}_event_0_0.npz"
    z = np.load(p, allow_pickle=True)
    to = np.asarray(z["deconv_q_offsets"], dtype=np.float64)
    T, R = universal_rebin(p, truth_npz=truth_file(tag),
                           deposit_shape="gaussian", sigma_time=SIG_T,
                           sigma_pxl=SP, time_offsets=to)
    out = {"tag": tag, "arm": arm}
    for name, m in (("sel_reco", R > CUT), ("sel_truth", T > CUT),
                    ("sel_union", (R > CUT) | (T > CUT))):
        s, s0, r = reg(T[m], R[m])
        out[name] = {"slope": s, "slope0": s0, "r": r, "n": int(m.sum())}
    Tp, Rp = T.sum(axis=2), R.sum(axis=2)
    m = Rp > CUT
    s, s0, r = reg(Tp[m], Rp[m])
    out["time_integrated"] = {"slope": s, "slope0": s0, "r": r,
                              "n": int(m.sum())}
    k = np.ones(3) / 3.0
    Ts = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 2, T)
    Rs = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 2, R)
    m = Rs > CUT
    s, s0, r = reg(Ts[m], Rs[m])
    out["time_smoothed3"] = {"slope": s, "slope0": s0, "r": r,
                             "n": int(m.sum())}
    out["sum_ratio"] = float(R.sum() / T.sum())
    lags, prat = [], []
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            t, r_ = T[i, j], R[i, j]
            if t.sum() < 5 or r_.sum() < 5:
                continue
            c = np.correlate(r_ - r_.mean(), t - t.mean(), "full")
            lags.append(int(np.argmax(c)) - (len(t) - 1))
            prat.append(float(r_.max() / max(t.max(), 1e-9)))
    out["lag_median"] = float(np.median(lags)) if lags else float("nan")
    out["lag_rms"] = float(np.std(lags)) if lags else float("nan")
    out["lag_frac_nonzero"] = (float(np.mean(np.array(lags) != 0))
                               if lags else float("nan"))
    out["peak_ratio_median"] = (float(np.median(prat)) if prat
                                else float("nan"))
    out["n_pixels"] = len(lags)
    del T, R, Ts, Rs
    gc.collect()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
                            "pos_a50_nb1", "pos_a75_nb1"]
    res = []
    print("%-12s %9s %9s %9s %10s %10s %8s %7s %7s %7s" %
          ("sample", "sel_reco", "sel_truth", "sel_union", "time_int",
           "time_sm3", "sumR/T", "lagmed", "lagrms", "peakR"))
    for t in tags:
        a = analyse(t)
        res.append(a)
        print("%-12s %9.3f %9.3f %9.3f %10.3f %10.3f %8.3f %7.1f %7.2f %7.2f"
              % (t, a["sel_reco"]["slope"], a["sel_truth"]["slope"],
                 a["sel_union"]["slope"], a["time_integrated"]["slope"],
                 a["time_smoothed3"]["slope"], a["sum_ratio"],
                 a["lag_median"], a["lag_rms"], a["peak_ratio_median"]),
              flush=True)
    json.dump(res, open(f"{OUT}/slope_origin.json", "w"), indent=1)
    print(f"-> {OUT}/slope_origin.json", flush=True)
