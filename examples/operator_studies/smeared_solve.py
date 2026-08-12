"""Solve with the band-limited operator A' = A G and score the result.

The unknown is the coefficient field u (positivity and the l1 ladder act
on u; for u >= 0 and a mass-conserving G the l1 value is unchanged by the
smoothing, so this is the same sparsity prior with a smoothness geometry
underneath). The physical estimate is q = G u.

Convention, to smear exactly once on each side:
  truth : the standard smeared_true at (sigma_time, sigma_pixel)
  reco  : u is stored as deconv_q_sharp and the standard gaussian deposit
          supplies the same G. Nothing is smeared twice.

Usage: smeared_solve.py [--alpha-scale S] [--sigma-time ST] tag [tag...]
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
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")
PART = {"mu": "mu", "pos": "positron"}
OUT = f"{AO}/smeared_solve"

from eval_alpha_beta import SIG_T, SP  # noqa: E402
from unfoldlarpix.constrained_solver import (build_latch_rows,  # noqa: E402
                                             centroid_bin_offsets)
from unfoldlarpix.data_loader import DataLoader  # noqa: E402
from unfoldlarpix.deconv_workflow import smear_effective_charge  # noqa: E402
from unfoldlarpix.eval.universal import (metrics_from_blocks,  # noqa: E402
                                         universal_rebin)
from unfoldlarpix.fwk.component import ALGORITHMS  # noqa: E402
from unfoldlarpix.fwk.runner import build_job  # noqa: E402
from unfoldlarpix.fwk.store import EventStore  # noqa: E402
from unfoldlarpix.model.conventions import (resolve_burst_tau,  # noqa: E402
                                            solver_time_shift)
from unfoldlarpix.model.smeared_operator import SmearedOperator  # noqa: E402
from unfoldlarpix.solve.engine import Fista  # noqa: E402
from unfoldlarpix.solve.strategy import Ladder, SolveState  # noqa: E402
from unfoldlarpix.terms.censor import CensorRunningMax  # noqa: E402
from unfoldlarpix.terms.data import DataFidelity  # noqa: E402

CUT = 0.5
_SVC = {}


class RidgeProx:
    """prox of lambda/2 ||u||^2 + positivity + support, for FISTA.

    The smoothness prior of the band-limited parameterisation is an l2
    penalty on the COEFFICIENTS: q = G u, so ||u||^2 = ||G^{-1} q||^2
    weights high frequencies by G^{-2}. This is the penalty whose filter
    factors f_k = w_k/(w_k + lambda) the budget analysis measured.
    """

    def __init__(self, lam: float, support):
        self.lam = float(lam)
        self.support = support
        self.alpha = 0.0          # for tracer compatibility

    def __call__(self, v, step):
        return torch.clamp(v, min=0.0) / (1.0 + step * self.lam) \
            * self.support


def job_for(tag):
    for base in (f"{AO}/nb1_fraccensor/B", f"{AO}/angscan_tau"):
        p = f"{base}/job_{tag}.yaml"
        if os.path.exists(p):
            return yaml.safe_load(open(p)), base
    raise FileNotFoundError(tag)


def truth_npz(tag):
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


def solve(tag, ascale=1.0, sigma_time=SIG_T, sigma_pixel=SP, tagout=None,
          ridge=None):
    cfg, _ = job_for(tag)
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
    op0 = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)
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
    prepared = services["detector"].prepared(B)
    op = SmearedOperator(prepared.integrated_response, op0.block_shape,
                         windows, B, device=op0.device, dtype=op0.dtype,
                         sigma_time=sigma_time, sigma_pixel=sigma_pixel)
    sc = [e for e in cfg["sequence"] if "Solve" in e][0]["Solve"]
    terms = [DataFidelity(op)]
    for t in (sc.get("terms") or []):
        if t["type"] == "censor":
            terms.append(CensorRunningMax.from_hits(
                op, store.get("hits_view"), store.get("block_offset"),
                csa_reset_time=float(rc.csa_reset_time or 0),
                threshold=float(rc.threshold), npad_bins=50,
                beta=float(t["beta"]), margin=float(t["margin"]),
                norm=t.get("norm", "l2"), bin_ticks=B))
    support = op.to_tensor(store.get("support").astype(np.float64))
    q0 = op.to_tensor(np.clip(store.get("warm.deconv_q"), 0.0, None)
                      [:, :, : op.q_shape[2]])
    engine = Fista(n_iter=int(sc["engine"]["iters"]))
    if ridge is not None:
        prox = RidgeProx(ridge, support)
        qq = engine.minimize(op, terms, prox, q0=q0)
        st = SolveState(q=qq)
    else:
        lad = dict(sc["strategy"])
        lad.pop("type")
        lad["alphas"] = [a * ascale for a in lad["alphas"]]
        st = Ladder(n_iter=engine.n_iter, **lad).run(
            engine, op, terms, support, SolveState(q=q0))
    u = st.q.cpu().numpy().astype(np.float64)
    q_phys = op.physical(st.q)
    name = tagout or tag
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    p = f"{OUT}/{name}/{tag}_event_0_0.npz"
    # u goes in as deconv_q_sharp: the evaluation's gaussian deposit is the
    # SAME G, so the reconstruction is smeared exactly once.
    np.savez_compressed(
        p, deconv_q_sharp=u.astype(np.float32),
        deconv_q=q_phys.astype(np.float32),
        deconv_q_offsets=centroid_bin_offsets(u, window_bins=2)
        .astype(np.float32),
        boffset=np.array([boff[0], boff[1],
                          boff[2] + solver_time_shift(B)], float),
        boffset_raw=boff, adc_hold_delay=np.array(B),
        sum_q=np.array(q_phys.sum()))
    r = op.forward(st.q) - op.d
    info = {"tag": tag, "variant": name, "sum_q": float(q_phys.sum()),
            "nnz_u": int((u > 0.01).sum()), "data_fid": 0.5 * float((r*r).sum()),
            "alpha_scale": ascale, "ridge": ridge, "sigma_time": sigma_time,
            "sigma_pixel": sigma_pixel, "path": p}
    del op, op0, store, st
    gc.collect()
    torch.cuda.empty_cache()
    return info


def score(path, tag, use_offsets=True):
    z = np.load(path, allow_pickle=True)
    to = np.asarray(z["deconv_q_offsets"], float) if use_offsets else None
    T, R = universal_rebin(path, truth_npz=truth_npz(tag),
                           deposit_shape="gaussian", sigma_time=SIG_T,
                           sigma_pxl=SP, time_offsets=to)
    m = metrics_from_blocks(T, R, corr_threshold=CUT)
    out = {k: float(m[k]) for k in
           ("pearson_r", "slope", "integral_pct", "ghost_frac",
            "ghost_iso_frac", "ghost_iso_charge", "true_killed")}
    del T, R
    gc.collect()
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    ascale = 1.0
    ridge = None
    if "--alpha-scale" in args:
        i = args.index("--alpha-scale")
        ascale = float(args[i + 1])
        del args[i:i + 2]
    if "--ridge" in args:
        i = args.index("--ridge")
        ridge = float(args[i + 1])
        del args[i:i + 2]
    tags = args or ["mu_a00_nb1", "mu_a75_nb1", "pos_a00_nb1", "pos_a75_nb1",
                    "mu_a00_nb4", "mu_a75_nb4", "pos_a00_nb4", "pos_a75_nb4"]
    res = {}
    print("%-14s %8s %8s %8s %9s %9s %9s %8s" %
          ("tag", "r", "slope", "integ%", "ghost%", "isoghost", "killed",
           "nnz_u"))
    for t in tags:
        lbl = (f"ridge{ridge:g}" if ridge is not None
               else f"a{ascale:g}").replace(".", "p")
        info = solve(t, ascale=ascale, ridge=ridge, tagout=lbl)
        m = score(info["path"], t)
        res[t] = {**info, **m}
        print("%-14s %8.4f %8.3f %8.2f %9.3f %9.2f %9.1f %8d" %
              (t, m["pearson_r"], m["slope"], m["integral_pct"],
               100 * m["ghost_frac"], m["ghost_iso_charge"],
               m["true_killed"], info["nnz_u"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    lbl = (f"ridge{ridge:g}" if ridge is not None else f"a{ascale:g}")
    json.dump(res, open(f"{OUT}/smeared_solve_{lbl}.json", "w"), indent=1)
    print(f"-> {OUT}/smeared_solve_{lbl}.json", flush=True)
