"""Can a tighter support make the system determined?

The under-determination measured in range_test.py is a counting
statement: the support carries far more unknowns than there are rows, so
the range of A P_S fills the data space. This script asks whether any
support choice can fix that, by measuring for several support definitions

  size            voxels in the support (compare with the row count)
  rank, L_min     row rank of A P_S and the best achievable residual
                  min_q ||A P_S q - d||^2 (zero => data fully fittable)
  truth coverage  fraction of the true charge that lies INSIDE the
                  support (what a tighter support would start to lose)

and, independently of any construction, the intrinsic requirement: how
many voxels are needed to hold 99% / 99.9% of the true charge. If that
number already exceeds the row count, no support can make the system
determined and the prior is unavoidable.

Variants: warm start thresholded at several eps with dilate 0/1, and the
data-driven hits-neighbourhood support at several dilations.
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

from channel_coupling import gram, replay  # noqa: E402
from unfoldlarpix.algs.reco_algs import BuildSupport  # noqa: E402
from unfoldlarpix.constrained_solver import gaussian_post_smooth  # noqa: E402


def warm_support(store, op, eps, dilate, smooth=True):
    rc = store.get("readout_config")
    dq = np.clip(store.get("warm.deconv_q"), 0.0, None)
    if smooth:
        dq = gaussian_post_smooth(dq, rc.adc_hold_delay, 0.005, 0.2)
    sup = dq > eps
    for _ in range(dilate):
        grown = sup.copy()
        for ax in range(3):
            for sh in (-1, 1):
                grown |= np.roll(sup, sh, axis=ax)
        sup = grown
    return sup[:, :, : op.q_shape[2]]


def hits_support(store, op, dilate, t_pad=2):
    alg = BuildSupport(source="hits", hits_dilate=dilate, t_pad=t_pad)
    return alg._from_hits(store, op, 1)


def truth_grid(store, op, tag):
    boff = np.asarray(store.get("block_offset"), float)
    B = int(store.get("readout_config").adc_hold_delay)
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
    g = np.zeros(op.q_shape)
    np.add.at(g, (ix[ok], iy[ok], it[ok]), eq[ok])
    return g


def rank_and_lmin(op, mask, d):
    G = gram(op, op.to_tensor(mask.astype(np.float64)))
    w, V = np.linalg.eigh(G)
    keep = w > TOL * max(w.max(), 1e-30)
    U = V[:, keep]
    res = d - U @ (U.T @ d)
    out = (int(keep.sum()), 0.5 * float(res @ res))
    del G, w, V, U
    gc.collect()
    return out


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    store, _ = replay(cfg)
    op = store.get("op")
    d = op.d.cpu().numpy().astype(np.float64)
    T = truth_grid(store, op, tag)
    tot = T.sum()
    flat = np.sort(T.ravel())[::-1]
    csum = np.cumsum(flat) / max(tot, 1e-30)
    need = {f"voxels_{int(100*p)}pct": int(np.searchsorted(csum, p) + 1)
            for p in (0.99, 0.999)}
    print(f"\n{tag}: {op.n_data} rows, {int(np.prod(op.q_shape))} q voxels; "
          f"truth needs {need['voxels_99pct']} voxels for 99% of its charge "
          f"({need['voxels_99pct']/op.n_data:.2f} x rows)", flush=True)
    variants = {
        "warm eps0.3 dil1 (adopted)": warm_support(store, op, 0.3, 1),
        "warm eps0.3 dil0": warm_support(store, op, 0.3, 0),
        "warm eps1.0 dil0": warm_support(store, op, 1.0, 0),
        "warm eps3.0 dil0": warm_support(store, op, 3.0, 0),
        "warm eps10 dil0": warm_support(store, op, 10.0, 0),
        "hits dil0": hits_support(store, op, 0),
        "hits dil1": hits_support(store, op, 1),
        "hits dil2 (option)": hits_support(store, op, 2),
        "hits dil3 (option)": hits_support(store, op, 3),
    }
    # how far does each support reach from the fired pixels?
    hv = store.get("hits_view")
    boff = np.asarray(store.get("block_offset"), float)
    nx, ny, _ = op.q_shape
    fired = np.zeros((nx, ny), bool)
    px = (np.asarray(hv.pixel_x) - int(boff[0])).astype(int)
    py = (np.asarray(hv.pixel_y) - int(boff[1])).astype(int)
    ok = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    fired[px[ok], py[ok]] = True
    dist = np.full((nx, ny), 99, np.int32)
    dist[fired] = 0
    cur = fired.copy()
    for k in range(1, 12):
        grown = cur.copy()
        for ax in range(2):
            for sh in (-1, 1):
                grown |= np.roll(cur, sh, axis=ax)
        new = grown & (dist == 99)
        dist[new] = k
        cur = grown
    rows = []
    print("%-26s %8s %7s %6s %10s %9s" %
          ("support", "voxels", "vox/row", "rank", "L_min", "truth in"))
    for name, m in variants.items():
        rank, lmin = rank_and_lmin(op, m, d)
        cov = float(T[m].sum() / max(tot, 1e-30))
        dm = np.repeat(dist[:, :, None], m.shape[2], axis=2)[m]
        if dm.size == 0:                      # empty support
            dm = np.array([np.nan])
        rows.append({"name": name, "voxels": int(m.sum()),
                     "pix_dist_p50": float(np.nanpercentile(dm, 50)),
                     "pix_dist_p99": float(np.nanpercentile(dm, 99)),
                     "pix_dist_max": float(np.nanmax(dm)),
                     "frac_beyond_3px": float(np.mean(dm > 3)),
                     "vox_per_row": m.sum() / op.n_data, "rank": rank,
                     "L_min": lmin, "truth_coverage": cov})
        print("%-26s %8d %7.2f %6d %10.2e %8.4f  | dist p50/p99/max "
              "%2.0f/%2.0f/%2.0f  >3px %5.1f%%" %
              (name, m.sum(), m.sum() / op.n_data, rank, lmin, cov,
               np.nanpercentile(dm, 50), np.nanpercentile(dm, 99),
               np.nanmax(dm),
               100 * (dm > 3).mean()), flush=True)
    del op, store
    gc.collect()
    torch.cuda.empty_cache()
    return {"tag": tag, "rows": int(len(d)), "truth_need": need,
            "variants": rows}


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a75_nb1"]
    res = [one(t) for t in tags]
    json.dump(res, open(f"{AO}/channel_coupling/support_dof.json", "w"),
              indent=1)
    print("\n-> channel_coupling/support_dof.json", flush=True)
