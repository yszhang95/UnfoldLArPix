"""Does the reconstruction fit the data BETTER than the truth does?

For each sample: the data-fidelity value at the stored solution,
0.5||A q_hat - d||^2, at the truth mapped onto the fit grid,
0.5||A q_truth - d||^2, and the noise floor implied by the readout noise
model, 0.5 * sum_r var_r (the expected residual of a perfect
reconstruction with a perfect operator).

If L(q_hat) < L(q_truth) the solver is exploiting freedoms the truth does
not use; if L(q_hat) also sits below the noise floor it is fitting noise
and operator error, i.e. over-fitting in the strict sense. The truth's
own residual, by contrast, contains readout noise AND operator error and
is the honest reference.
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
AO = f"{ROOT}/examples/analysis_output"
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")
PART = {"mu": "mu", "pos": "positron"}

from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.fwk.component import ALGORITHMS  # noqa: E402
from unfoldlarpix.fwk.runner import build_job  # noqa: E402
from unfoldlarpix.fwk.store import EventStore  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402
from unfoldlarpix.model.noise import row_variances  # noqa: E402

_SVC = {}


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
    keep = [e for e in cfg["sequence"]
            if list(e)[0] in ("LoadEvent", "FFTWarmStart",
                              "BuildMeasurement", "BuildSupport")]
    skey = json.dumps(cfg["services"], sort_keys=True)
    if skey not in _SVC:
        _SVC[skey], _ = build_job({"services": cfg["services"],
                                   "sequence": keep})
    services = _SVC[skey]
    store = EventStore()
    store.put("job.config", cfg, "runner")
    for entry in keep:
        (name, props), = entry.items()
        a = ALGORITHMS[name](**(props or {}))
        a.initialize(services)
        a.execute(store)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)

    z = np.load(f"{AO}/nb1_fraccensor/{arm}/{tag}/{tag}_event_0_0.npz",
                allow_pickle=True)
    qh = op.to_tensor(np.asarray(z["deconv_q_sharp"], float))
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
    qt = op.to_tensor(qg)

    def fid(q):
        r = op.forward(q) - op.d
        return 0.5 * float((r * r).sum())

    # noise floor from the analytic row model
    bm = [e for e in cfg["sequence"] if "BuildMeasurement" in e][0]
    bt = bm["BuildMeasurement"].get("burst_tau")
    bt = None if bt is None else resolve_burst_tau(
        rc, None if bt == "auto" else int(bt))
    _, metas = build_latch_rows(
        ev.hits.location, ev.hits.data, B, boff,
        csa_reset_time=rc.csa_reset_time,
        split_threshold=(float(rc.threshold)
                         if bm["BuildMeasurement"].get("split_trigger", True)
                         else None),
        acq_start=getattr(ev, "acq_start", None), burst_tau=bt)
    var = row_variances(metas, rc)
    out = {"tag": tag, "rows": int(op.n_data),
           "L_reco": fid(qh), "L_truth": fid(qt),
           "noise_floor": 0.5 * float(np.sum(var[:op.n_data])),
           "sum_q_reco": float(qh.sum()), "sum_q_truth": float(qt.sum()),
           "truth_in_grid": float(eq[ok].sum() / eq.sum())}
    out["ratio_reco_truth"] = out["L_reco"] / max(out["L_truth"], 1e-30)
    out["reco_over_floor"] = out["L_reco"] / max(out["noise_floor"], 1e-30)
    out["truth_over_floor"] = out["L_truth"] / max(out["noise_floor"], 1e-30)
    del op, qh, qt, store
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
                            "pos_a00_nb1", "pos_a50_nb1", "pos_a75_nb1"]
    res = []
    print("%-12s %6s %10s %10s %10s %8s %8s %8s" %
          ("sample", "rows", "L_reco", "L_truth", "floor",
           "reco/tr", "reco/fl", "tr/fl"))
    for t in tags:
        a = one(t)
        res.append(a)
        print("%-12s %6d %10.1f %10.1f %10.1f %8.3f %8.3f %8.2f" %
              (t, a["rows"], a["L_reco"], a["L_truth"], a["noise_floor"],
               a["ratio_reco_truth"], a["reco_over_floor"],
               a["truth_over_floor"]), flush=True)
    json.dump(res, open(f"{AO}/channel_coupling/fit_vs_truth.json", "w"),
              indent=1)
    print("-> channel_coupling/fit_vs_truth.json", flush=True)
