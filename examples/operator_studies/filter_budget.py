"""Can a filter no wider than 2.0 us suppress the error-absorbing modes?

Design constraint (project requirement): the analysis Gaussian is
1.6 us at baseline and must not exceed 2.0 us. Within that window, the
question is whether the smoothness prior

    min ||A G u - d||^2 + lambda ||u||^2,   q = G u

leaves the solver enough freedom to absorb the operator's model error.

Everything is done in the eigenbasis of the data-space Gram
G' = A G P G A^T = U W U^T. The fitted prediction is
U diag(f) U^T d with the filter factors f_k = w_k / (w_k + lambda), so

  effective dof   = sum_k f_k                (data directions actually used)
  signal kept     = sum_k f_k^2 (u_k . A q_true)^2 / ||A q_true||^2
  error absorbed  = sum_k f_k^2 (u_k . e)^2 / ||e||^2

with e the EXACT per-row operator error from the noiseless waveform,
e = A q_true - d_exact. A good working point keeps the signal high while
driving the absorbed error down, at a width <= 2.0 us.
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
from residual_direction import one as resid_one  # noqa: E402  (for e)
from smeared_operator import gram_smeared  # noqa: E402
from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402

# widths inside the project's window: baseline 1.6 us, cap 2.0 us
WIDTHS = {"none": None,
          "1.6 us / 0.32 px": (1.0 / (2 * np.pi * 32), 0.5),
          "2.0 us / 0.32 px": (1.0 / (2 * np.pi * 40), 0.5)}
LAMBDAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def pieces(tag, arm="B"):
    """Gram inputs plus the exact operator error per row."""
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    wf = cfg["sequence"][0]["LoadEvent"]["input"].replace(".npz", "_wf.npz")
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)
    supp = store.get("support").astype(bool)

    f = np.load(cfg["sequence"][0]["LoadEvent"]["input"], allow_pickle=True)
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
    Aqt = op.forward(op.to_tensor(T)).cpu().numpy().astype(np.float64)

    # exact per-row operator error from the waveform
    bm = [e for e in cfg["sequence"] if "BuildMeasurement" in e][0]
    bt = bm["BuildMeasurement"].get("burst_tau")
    bt = None if bt is None else resolve_burst_tau(
        rc, None if bt == "auto" else int(bt))
    windows, _ = build_latch_rows(
        ev.hits.location, ev.hits.data, B, boff,
        csa_reset_time=rc.csa_reset_time,
        split_threshold=(float(rc.threshold)
                         if bm["BuildMeasurement"].get("split_trigger", True)
                         else None),
        acq_start=getattr(ev, "acq_start", None), burst_tau=bt)
    bx, by, bnt = op.block_shape
    keep = [i for i, w in enumerate(windows)
            if 0 <= w.px < bx and 0 <= w.py < by
            and w.t_hi > max(w.t_lo, 0.0)]
    z = np.load(wf, allow_pickle=True)
    cur = np.asarray(z["current_tpc0_batch0"])
    cur = cur.reshape(-1, cur.shape[-1])
    cl = np.asarray(z["current_tpc0_batch0_location"])
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]
    cs = {k: np.concatenate([[0.0], np.cumsum(cur[i])])
          for k, i in idx.items()}
    d_ex = np.zeros(op.n_data)
    for r, i in enumerate(keep):
        w = windows[i]
        k = (int(w.px + boff[0]), int(w.py + boff[1]))
        if k not in cs:
            continue
        a = int(np.clip(max(w.t_lo, 0.0) + boff[2], 0, Nt))
        b = int(np.clip(min(w.t_hi + boff[2], Nt), 0, Nt))
        d_ex[r] = cs[k][b] - cs[k][a] if b > a else 0.0
    e = Aqt - d_ex
    return op, supp, B, op.d.cpu().numpy().astype(np.float64), Aqt, e


def run(tag):
    op, supp, B, d, Aqt, e = pieces(tag)
    print(f"\n{tag}: {op.n_data} rows, support {int(supp.sum())}, "
          f"||e|| = {np.linalg.norm(e):.1f} ke, "
          f"||A q_true|| = {np.linalg.norm(Aqt):.1f} ke", flush=True)
    print("%-18s %7s %8s %10s %11s %11s" %
          ("width", "lambda", "eff dof", "dof/rows", "signal kept",
           "error absorbed"))
    out = {"tag": tag, "rows": int(op.n_data), "rows_e_norm": float(
        np.linalg.norm(e)), "grid": []}
    for lab, wd in WIDTHS.items():
        G = (gram(op, op.to_tensor(supp.astype(np.float64))) if wd is None
             else gram_smeared(op, supp.astype(np.float64), B, wd[0], wd[1]))
        w, U = np.linalg.eigh(G)
        w = np.clip(w[::-1], 0, None)
        U = U[:, ::-1]
        cs_sig = (U.T @ Aqt) ** 2
        cs_err = (U.T @ e) ** 2
        for lam in LAMBDAS:
            f = w / (w + lam) if lam > 0 else np.ones_like(w)
            dof = float(f.sum())
            sig = float((f ** 2 * cs_sig).sum() / max(cs_sig.sum(), 1e-30))
            err = float((f ** 2 * cs_err).sum() / max(cs_err.sum(), 1e-30))
            out["grid"].append({"width": lab, "lambda": lam, "eff_dof": dof,
                                "dof_frac": dof / op.n_data,
                                "signal_kept": sig, "error_absorbed": err})
            print("%-18s %7.3g %8.1f %10.3f %11.4f %11.4f" %
                  (lab, lam, dof, dof / op.n_data, sig, err), flush=True)
        del G, w, U
        gc.collect()
    del op
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a50_nb1"]
    res = [run(t) for t in tags]
    json.dump(res, open(f"{AO}/channel_coupling/filter_budget.json", "w"),
              indent=1)
    print("\n-> channel_coupling/filter_budget.json", flush=True)
