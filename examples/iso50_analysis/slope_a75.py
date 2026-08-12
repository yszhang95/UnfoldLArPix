"""Why does mu_a75 regress with slope > 1, and which term decides it?

Hypothesis under test: at 75 deg the track plunges through the drift, so
one pixel collects charge over MANY fit bins while nburst = 1 gives it a
single trigger, i.e. three measurement rows. The within-pixel time
profile is then almost unconstrained (the coupling study measures
rank(99%)/active = 0.66, and rows/active = 0.80 -- fewer rows than
unknowns), and the sparsity prior resolves the ambiguity by concentrating
the charge into few time bins. Against a truth that is spread in time,
the voxel-wise regression then reads slope > 1 with a negative integral.

Two measurements:
  (1) ablations: re-solve mu_a75_nb1 with the l1 ladder rescaled, with
      the censor removed, and with the refit, then score each the way
      section 5 does (sigma_pxl = 0.5, threshold 0.5 ke). If the l1 is
      responsible, the slope must fall towards 1 as alpha falls.
  (2) time profiles: per pixel, the RMS time spread and the number of
      occupied bins in truth versus reconstruction, for 0/50/75 deg. The
      prediction is that only 75 deg shows reco under-spreading.

Usage: slope_a75.py [tag]        (default mu_a75_nb1)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/examples/analysis_output/_drivers")
AO = f"{ROOT}/examples/analysis_output"
NFS = "/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield"

from eval_alpha_beta import SIG_T, SP, metrics  # noqa: E402
from unfoldlarpix.data_loader import DataLoader  # noqa: E402
from unfoldlarpix.deconv_workflow import smear_effective_charge  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "mu_a75_nb1"
OUT = f"{AO}/slope_probe"
os.makedirs(OUT, exist_ok=True)
PART = {"mu": "mu", "pos": "positron"}
part, ang, nb = TAG.split("_")[0], TAG.split("_")[1][1:], TAG.split("_")[2][2:]
SAMPLE = f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz"
# the evaluator needs smeared_true at the ADOPTED width (sigma_pxl = 0.5),
# recomputed from the event exactly as the re-scoring convention requires
TRUTH = f"{OUT}/truth_{TAG}_sp{SP}.npz"


def ensure_truth():
    if os.path.exists(TRUTH):
        return
    os.makedirs(OUT, exist_ok=True)
    ev = [e for e in DataLoader(SAMPLE).iter_events()
          if e.hits and e.tpc_id == 0][0]
    off, smt = smear_effective_charge(ev, sigma_time=SIG_T, sigma_pixel=SP)
    np.savez(TRUTH, smeared_true=smt, smear_offset=np.array(off))
    del smt, ev
    gc.collect()

VARIANTS = {
    "baseline": {},
    "alpha0.5": {"ascale": 0.5},
    "alpha0.25": {"ascale": 0.25},
    "alpha0.1": {"ascale": 0.1},
    "nocensor": {"nocensor": True},
    "refit": {"refit": True},
    "alpha0.25+refit": {"ascale": 0.25, "refit": True},
    # the near-null subspace is decided by the initial point, not by any
    # term: vary the warm start's time regularisation and watch the slope
    "warm_st0.010": {"sigma_time": 0.010},
    "warm_st0.002": {"sigma_time": 0.002},
    "warm_st0.010+a0.25": {"sigma_time": 0.010, "ascale": 0.25},
}


def run(name, opts):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/B/job_{TAG}.yaml"))
    sc = [e for e in cfg["sequence"] if "Solve" in e][0]["Solve"]
    if opts.get("ascale"):
        sc["strategy"]["alphas"] = [a * opts["ascale"]
                                    for a in sc["strategy"]["alphas"]]
    if opts.get("nocensor"):
        sc["terms"] = []
    if opts.get("refit"):
        sc["refit"] = {"eps": 0.5, "alpha": 0.0}
    if opts.get("sigma_time"):
        ws = [e for e in cfg["sequence"] if "FFTWarmStart" in e][0]
        ws["FFTWarmStart"]["sigma_time"] = opts["sigma_time"]
    wc = [e for e in cfg["sequence"] if "WriteCharges" in e][0]["WriteCharges"]
    wc["out_dir"] = f"{OUT}/{TAG}_{name}"
    wc["prefix"] = TAG
    wc["embed_truth"] = True
    p = f"{OUT}/{TAG}_{name}/{TAG}_event_0_0.npz"
    if not os.path.exists(p):
        os.makedirs(f"{OUT}/{TAG}_{name}", exist_ok=True)
        cp = f"{OUT}/{TAG}_{name}/job.yaml"
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        os.system(f"PYTHONPATH={ROOT}/src CUDA_VISIBLE_DEVICES=0 "
                  f"{sys.executable} -m unfoldlarpix.fwk.runner {cp} "
                  f"> {OUT}/{TAG}_{name}/run.log 2>&1")
    m = metrics(p, TRUTH)
    z = np.load(p, allow_pickle=True)
    q = np.asarray(z["deconv_q_sharp"], float)
    m["sum_q"] = float(q.sum())
    m["nnz"] = int((q > 0.01).sum())
    # time concentration of the reconstruction: bins above 0.5 ke per pixel
    occ = (q > 0.5).sum(axis=2)
    m["bins_per_pixel"] = float(occ[occ > 0].mean()) if (occ > 0).any() else 0.0
    gc.collect()
    return m


def time_profiles(tag):
    """Per-pixel time spread, truth vs reco, at the adopted setting."""
    p = f"{AO}/nb1_fraccensor/B/{tag}/{tag}_event_0_0.npz"
    z = np.load(p, allow_pickle=True)
    q = np.asarray(z["deconv_q_sharp"], float)
    boff = np.asarray(z["boffset_raw" if "boffset_raw" in z.files
                      else "boffset"], float)
    B = float(np.asarray(z["adc_hold_delay"]).ravel()[0])
    part, ang, nb = tag.split("_")[0], tag.split("_")[1][1:], tag.split("_")[2][2:]
    f = np.load(f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz",
                allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    nx, ny, nt = q.shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    t = np.zeros_like(q)
    np.add.at(t, (ix[ok], iy[ok], it[ok]), eq[ok])

    def spread(a, cut=0.5):
        out = []
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                v = a[i, j]
                s = v.sum()
                if s < 5.0:
                    continue
                k = np.arange(len(v))
                mu = (v * k).sum() / s
                rms = float(np.sqrt(max((v * (k - mu) ** 2).sum() / s, 0)))
                out.append((s, rms, int((v > cut).sum()), float(v.max())))
        return np.array(out) if out else np.zeros((0, 4))

    st, sq = spread(t), spread(q)
    return {"tag": tag,
            "truth_pixels": int(len(st)), "reco_pixels": int(len(sq)),
            "truth_rms_bins": float(np.median(st[:, 1])) if len(st) else 0,
            "reco_rms_bins": float(np.median(sq[:, 1])) if len(sq) else 0,
            "truth_occ_bins": float(np.median(st[:, 2])) if len(st) else 0,
            "reco_occ_bins": float(np.median(sq[:, 2])) if len(sq) else 0,
            "truth_peak": float(np.median(st[:, 3])) if len(st) else 0,
            "reco_peak": float(np.median(sq[:, 3])) if len(sq) else 0}


if __name__ == "__main__":
    ensure_truth()
    res = {"tag": TAG, "variants": {}, "time_profiles": []}
    print("%-16s %7s %7s %8s %9s %8s %7s %8s" %
          ("variant", "r", "slope", "integ%", "killed", "sum_q", "nnz",
           "bins/px"))
    for name, opts in VARIANTS.items():
        m = run(name, opts)
        res["variants"][name] = m
        print("%-16s %7.4f %7.3f %8.2f %9.1f %8.1f %7d %8.2f" %
              (name, m["pearson_r"], m["slope"], m["integral_pct"],
               m["true_killed"], m["sum_q"], m["nnz"],
               m["bins_per_pixel"]), flush=True)
    print("\n%-12s %8s %8s %8s %8s %8s %8s" %
          ("sample", "t_rms", "q_rms", "t_occ", "q_occ", "t_peak", "q_peak"))
    for tag in ("mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1"):
        tp = time_profiles(tag)
        res["time_profiles"].append(tp)
        print("%-12s %8.2f %8.2f %8.1f %8.1f %8.2f %8.2f" %
              (tag, tp["truth_rms_bins"], tp["reco_rms_bins"],
               tp["truth_occ_bins"], tp["reco_occ_bins"],
               tp["truth_peak"], tp["reco_peak"]), flush=True)
    json.dump(res, open(f"{OUT}/slope_probe_{TAG}.json", "w"), indent=1)
    print(f"-> {OUT}/slope_probe_{TAG}.json", flush=True)
