"""Does band-limiting the operator (A -> A G) remove the freedom that
lets the solver absorb model error?

Idea under test: solve for a band-limited field u, the physical estimate
being q = G u with G a known symmetric Gaussian smoother. The forward
map becomes A' = A G P_S. Because G damps the high-frequency directions,
the range of A' should no longer fill the data space, so an arbitrary d
-- including the part the operator gets wrong -- can no longer be
reproduced. The estimator's target is then G q_true, a known, symmetric,
quantifiable resolution.

Measured, for several smoothing widths (frequency-domain sigmas, as
everywhere in this project: real-space width = 1/(2 pi sigma)):

  rank(A G P_S)      how much of the data space is still reachable
  L_min              min_u ||A G P_S u - d||^2, the best fit achievable
  L_truth_sharp      0.5||A q_truth - d||^2      (no smearing; reference)
  L_truth_smeared    0.5||A G q_truth - d||^2    (what the band-limited
                     model can say about the truth)

If L_min rises to the scale of the operator error while L_truth_smeared
stays comparable, the band-limited operator can no longer hide the model
error, which is the point of the exercise.
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
TOL = 1e-10

from channel_coupling import replay  # noqa: E402
from unfoldlarpix.constrained_solver import gaussian_post_smooth  # noqa: E402

# (sigma_time, sigma_pixel) in the frequency-domain convention.
# real-space widths: 1/(2 pi sigma) ticks / pixels
WIDTHS = [(0.02, 1.0),      # 8 ticks (0.4 us), 0.16 px  -- mild
          (0.01, 0.5),      # 16 ticks (0.8 us), 0.32 px -- pixel limit
          (0.005, 0.5),     # 32 ticks (1.6 us), 0.32 px -- analysis width
          (0.0025, 0.5)]    # 64 ticks (3.2 us), 0.32 px -- strong


def smooth(q, B, st, sp):
    return gaussian_post_smooth(q, B, st, sp)


def gram_smeared(op, mask, B, st, sp):
    """G' = A G P G A^T, one column at a time (G symmetric)."""
    n = op.n_data
    G = np.zeros((n, n))
    e = torch.zeros(n, dtype=op.dtype, device=op.device)
    for r in range(n):
        e.zero_()
        e[r] = 1.0
        v = op.adjoint(e).cpu().numpy().astype(np.float64)
        v = smooth(v, B, st, sp) * mask
        v = smooth(v, B, st, sp)
        G[:, r] = op.forward(op.to_tensor(v)).cpu().numpy()
    return 0.5 * (G + G.T)


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    store, _ = replay(cfg)
    op = store.get("op")
    B = int(store.get("readout_config").adc_hold_delay)
    boff = np.asarray(store.get("block_offset"), float)
    supp = store.get("support").astype(bool).astype(np.float64)
    d = op.d.cpu().numpy().astype(np.float64)

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
    T = np.zeros(op.q_shape)
    np.add.at(T, (ix[ok], iy[ok], it[ok]), eq[ok])

    def fid(arr):
        r = op.forward(op.to_tensor(arr)).cpu().numpy() - d
        return 0.5 * float(r @ r)

    out = {"tag": tag, "rows": int(op.n_data),
           "support_voxels": int(supp.sum()),
           "L_truth_sharp": fid(T), "widths": []}
    print(f"\n{tag}: {op.n_data} rows, support {int(supp.sum())}, "
          f"L_truth(sharp) = {out['L_truth_sharp']:.1f}", flush=True)
    print("%-22s %8s %6s %11s %13s %11s" %
          ("smoothing (real space)", "rank", "rk/n", "L_min",
           "L_truth_smear", "Lmin/Ltr"))
    for st, sp in WIDTHS:
        Gm = gram_smeared(op, supp, B, st, sp)
        w, V = np.linalg.eigh(Gm)
        keep = w > TOL * max(w.max(), 1e-30)
        U = V[:, keep]
        res = d - U @ (U.T @ d)
        lmin = 0.5 * float(res @ res)
        lts = fid(smooth(T, B, st, sp))
        lab = f"{1/(2*np.pi*st):.0f} tk / {1/(2*np.pi*sp):.2f} px"
        out["widths"].append({
            "sigma_time": st, "sigma_pixel": sp, "label": lab,
            "rank": int(keep.sum()), "rank_frac": keep.sum() / op.n_data,
            "L_min": lmin, "L_truth_smeared": lts,
            "Lmin_over_Ltruth_sharp": lmin / max(out["L_truth_sharp"], 1e-30)})
        print("%-22s %8d %6.3f %11.3e %13.1f %11.3f" %
              (lab, keep.sum(), keep.sum() / op.n_data, lmin, lts,
               lmin / max(out["L_truth_sharp"], 1e-30)), flush=True)
        del Gm, w, V, U
        gc.collect()
    del op, store
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a75_nb1", "mu_a00_nb1"]
    res = [one(t) for t in tags]
    json.dump(res, open(f"{AO}/channel_coupling/smeared_operator.json", "w"),
              indent=1)
    print("\n-> channel_coupling/smeared_operator.json", flush=True)
