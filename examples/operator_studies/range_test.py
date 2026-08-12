"""Why can the solver beat the truth? Range vs degeneracy.

Two different pathologies are often conflated:

  under-determination : the rows are too few for the unknowns the solver
        is allowed to use, so range(A P) fills the data space and almost
        ANY d can be reproduced -- including the part of d that the
        operator gets wrong. Diagnostic: the best achievable residual
        min_q ||A P q - d||^2 = ||(I - Pi) d||^2, with Pi the projector
        onto range(A P). If that is small, model error is absorbable.

  redundancy/correlation : the rows carry overlapping information, so
        many unknowns are poorly determined. Diagnostic: the effective
        rank on the ACTIVE set and the coupling profile (companion
        script channel_coupling.py). This governs which q is chosen, not
        how well d can be fit.

This script measures the first, and splits the truth's residual into the
part reachable by the support-restricted operator (removable by moving
charge, hence exploited by any least-squares fit) and the orthogonal
remainder (irreducible model error).
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")
PART = {"mu": "mu", "pos": "positron"}

from channel_coupling import gram, replay  # noqa: E402

TOL = 1e-10          # eigenvalue cut relative to the largest


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)
    supp = store.get("support").astype(bool)
    z = np.load(f"{AO}/nb1_fraccensor/{arm}/{tag}/{tag}_event_0_0.npz",
                allow_pickle=True)
    qh = np.asarray(z["deconv_q_sharp"], float)
    act = qh > 0.01

    part, ang, nb = (tag.split("_")[0], tag.split("_")[1][1:],
                     tag.split("_")[2][2:])
    f = np.load(f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz",
                allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    nx, ny, nt = op.q_shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    qg = np.zeros(op.q_shape)
    np.add.at(qg, (ix[ok], iy[ok], it[ok]), eq[ok])

    d = op.d.cpu().numpy().astype(np.float64)
    r_truth = (op.forward(op.to_tensor(qg)).cpu().numpy().astype(np.float64)
               - d)
    r_reco = (op.forward(op.to_tensor(qh)).cpu().numpy().astype(np.float64)
              - d)

    out = {"tag": tag, "rows": int(op.n_data),
           "q_voxels": int(np.prod(op.q_shape)),
           "support_voxels": int(supp.sum()),
           "active_voxels": int(act.sum()),
           "L_truth": 0.5 * float(r_truth @ r_truth),
           "L_reco": 0.5 * float(r_reco @ r_reco)}

    for name, mask in (("support", supp), ("active", act)):
        G = gram(op, op.to_tensor(mask.astype(np.float64)))
        w, V = np.linalg.eigh(G)
        keep = w > TOL * max(w.max(), 1e-30)
        U = V[:, keep]                      # orthonormal basis of range(A P)
        rank = int(keep.sum())
        proj = U @ (U.T @ d)
        res_d = d - proj
        pr = U @ (U.T @ r_truth)
        out[name] = {
            "rank": rank, "rank_frac_rows": rank / op.n_data,
            # best possible residual for ANY charge on this mask
            "L_min": 0.5 * float(res_d @ res_d),
            "L_min_over_L_truth": (0.5 * float(res_d @ res_d)
                                   / max(out["L_truth"], 1e-30)),
            # truth residual split
            "r_truth_in_range_frac": float((pr @ pr) / max(r_truth @ r_truth,
                                                           1e-30)),
        }
        del G, w, V, U
        gc.collect()
    del op, store
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
                            "pos_a50_nb1", "pos_a75_nb1"]
    res = []
    print("%-12s %5s %7s %7s | %5s %6s %9s %7s | %5s %6s %9s %7s" %
          ("sample", "rows", "L_tr", "L_reco",
           "rkS", "rkS/n", "Lmin_S", "in_rgS",
           "rkA", "rkA/n", "Lmin_A", "in_rgA"))
    for t in tags:
        a = one(t)
        res.append(a)
        s, ac = a["support"], a["active"]
        print("%-12s %5d %7.0f %7.1f | %5d %6.3f %9.2e %7.3f | %5d %6.3f "
              "%9.2e %7.3f" %
              (t, a["rows"], a["L_truth"], a["L_reco"],
               s["rank"], s["rank_frac_rows"], s["L_min"],
               s["r_truth_in_range_frac"],
               ac["rank"], ac["rank_frac_rows"], ac["L_min"],
               ac["r_truth_in_range_frac"]), flush=True)
    json.dump(res, open(f"{AO}/channel_coupling/range_test.json", "w"),
              indent=1)
    print("-> channel_coupling/range_test.json", flush=True)
