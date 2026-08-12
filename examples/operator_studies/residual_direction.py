"""Does the solver leave the OPERATOR ERROR unfitted, or the signal?

A residual target (discrepancy principle) controls only the MAGNITUDE of
the residual, never its direction. The waveform samples make the
direction measurable: with the noiseless induced current available, each
row has an exact decomposition

    d_r  =  d_exact_r  +  n_r                (readout: noise, threshold, reset)
    (A q_truth)_r  =  d_exact_r  +  e_r      (operator model error)

so the residual the solver reports, (A q_hat - d)_r, can be correlated
against e_r (operator error) and against n_r (readout error). If the
solver's residual is aligned with e, a magnitude target leaves the model
error unfitted, which is what one wants; if it is aligned with -e (i.e.
the error has been absorbed) or with neither, stopping early leaves
signal unfitted instead, and the honest treatment is to put the model
error into the covariance.

Samples: the *_wf files (mu_a00_nb1, mu_a50_nb1), which carry both the
noiseless current and the ordinary noisy hits.
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

from channel_coupling import replay  # noqa: E402
from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402


def exact_rows(wf_path, windows, op_n):
    """Exact integral of the true current over each window."""
    z = np.load(wf_path, allow_pickle=True)
    cur = np.asarray(z["current_tpc0_batch0"])
    cur = cur.reshape(-1, cur.shape[-1])
    cl = np.asarray(z["current_tpc0_batch0_location"])
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]
    cs = {k: np.concatenate([[0.0], np.cumsum(cur[i])])
          for k, i in idx.items()}
    return z, cs, Nt


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    wf = cfg["sequence"][0]["LoadEvent"]["input"].replace(".npz", "_wf.npz")
    if not os.path.exists(wf):
        print(f"{tag}: no waveform file ({os.path.basename(wf)}) -- skip",
              flush=True)
        return None
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)
    bm = [e for e in cfg["sequence"] if "BuildMeasurement" in e][0]
    bt = bm["BuildMeasurement"].get("burst_tau")
    bt = None if bt is None else resolve_burst_tau(
        rc, None if bt == "auto" else int(bt))
    windows, metas = build_latch_rows(
        ev.hits.location, ev.hits.data, B, boff,
        csa_reset_time=rc.csa_reset_time,
        split_threshold=(float(rc.threshold)
                         if bm["BuildMeasurement"].get("split_trigger", True)
                         else None),
        acq_start=getattr(ev, "acq_start", None), burst_tau=bt)
    nx, ny, nt = op.block_shape
    keep = [i for i, w in enumerate(windows)
            if 0 <= w.px < nx and 0 <= w.py < ny
            and w.t_hi > max(w.t_lo, 0.0)]
    _, cs, Nt = exact_rows(wf, windows, op.n_data)

    d_ex = np.zeros(op.n_data)
    for r, i in enumerate(keep):
        w = windows[i]
        k = (int(w.px + boff[0]), int(w.py + boff[1]))
        if k not in cs:
            continue
        a = int(np.clip(max(w.t_lo, 0.0) + boff[2], 0, Nt))
        b = int(np.clip(min(w.t_hi + boff[2], Nt), 0, Nt))
        d_ex[r] = cs[k][b] - cs[k][a] if b > a else 0.0

    d = op.d.cpu().numpy().astype(np.float64)
    # truth on the fit grid
    f = np.load(cfg["sequence"][0]["LoadEvent"]["input"], allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    qx, qy, qt = op.q_shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < qx) & (iy >= 0) & (iy < qy)
          & (it >= 0) & (it < qt))
    qg = np.zeros(op.q_shape)
    np.add.at(qg, (ix[ok], iy[ok], it[ok]), eq[ok])
    Aqt = op.forward(op.to_tensor(qg)).cpu().numpy().astype(np.float64)

    qh = np.asarray(np.load(
        f"{AO}/nb1_fraccensor/{arm}/{tag}/{tag}_event_0_0.npz",
        allow_pickle=True)["deconv_q_sharp"], float)
    Aqh = op.forward(op.to_tensor(qh)).cpu().numpy().astype(np.float64)

    e = Aqt - d_ex            # operator model error, per row
    n = d - d_ex              # readout error, per row
    rres = Aqh - d            # what the solver leaves
    m = np.isfinite(e) & np.isfinite(n) & (d_ex != 0)

    def cc(a, b):
        a, b = a[m], b[m]
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out = {"tag": tag, "rows": int(m.sum()),
           "rms_operator_err": float(e[m].std()),
           "rms_readout_err": float(n[m].std()),
           "rms_solver_residual": float(rres[m].std()),
           "corr_e_n": cc(e, n),
           "corr_resid_e": cc(rres, e),
           "corr_resid_n": cc(rres, n),
           "mean_operator_err": float(e[m].mean()),
           "mean_readout_err": float(n[m].mean()),
           "mean_solver_residual": float(rres[m].mean())}
    # how much of the operator error survives in the solver's residual?
    ee = e[m]
    out["proj_resid_on_e"] = float((rres[m] @ ee) / max(ee @ ee, 1e-30))
    del op, store
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1"]
    res = [r for r in (one(t) for t in tags) if r]
    print("\n%-12s %6s %9s %9s %9s %9s %9s %9s" %
          ("sample", "rows", "rms_op", "rms_noise", "rms_res",
           "r(e,n)", "r(res,e)", "proj"))
    for a in res:
        print("%-12s %6d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f" %
              (a["tag"], a["rows"], a["rms_operator_err"],
               a["rms_readout_err"], a["rms_solver_residual"],
               a["corr_e_n"], a["corr_resid_e"], a["proj_resid_on_e"]))
    json.dump(res, open(f"{AO}/channel_coupling/residual_direction.json",
                        "w"), indent=1)
    print("-> channel_coupling/residual_direction.json", flush=True)
