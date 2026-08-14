"""Is the PRE-TRIGGER silence constraint honest, and does it bite?

``CensorRunningMax.from_hits`` covers only the interval after a pixel's
last burst.  The silent intervals BEFORE each trigger are constrained only
at their endpoint (the ``split_trigger`` pseudo row asserts the accumulator
equalled the threshold AT the trigger), which leaves the excursions on the
way free: with a bipolar response the accumulator may cross the threshold
early and be pulled back down before the recorded trigger.
``terms.censor.pre_trigger_censors`` forbids that.

Before it can be used it has to pass two gates, and this script measures
both without solving anything:

  1. TRUTH FEASIBILITY.  The statement is true of the real detector by
     construction (the discriminator fires the first time it crosses), so
     the truth mapped onto the fit grid must not violate it beyond the
     operator's own model error.  A large violation at truth means the
     boundary arithmetic or the bin quantisation is wrong, not that the
     physics is.
  2. NON-VACUITY.  How many pixel-intervals carry at least one check
     instant after bin quantisation, and how much slack the truth leaves.
     A constraint that is satisfied with huge slack everywhere adds
     nothing.

Reported alongside: the value at the STORED reconstruction, which says
whether the adopted solutions already satisfy it (as the post-latch term
is satisfied on muons) or violate it (in which case it carries new
information the solve is currently free to ignore).

Usage:  python examples/operator_studies/censor_pre_probe.py [ARM] [TAG...]
Output: analysis_output/censor_pre/censor_pre_probe.json
"""
from __future__ import annotations

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

from channel_coupling import replay  # noqa: E402
from unfoldlarpix.terms.base import IterCtx  # noqa: E402
from unfoldlarpix.terms.censor import (CensorRunningMax,  # noqa: E402
                                       pre_trigger_censors)

OUT = f"{AO}/censor_pre"
TAGS = ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
        "pos_a00_nb1", "pos_a50_nb1", "pos_a75_nb1"]
# which campaign's jobs and stored solutions to read; set CAMPAIGN=censor_pre
# to re-probe the solutions produced WITH the term (does it remove the
# violations it was added for?).
CAMPAIGN = os.environ.get("CAMPAIGN", "nb1_fraccensor")


def truth_on_fit_grid(cfg, op, boff, B):
    """Effective-charge truth summed onto the operator's q grid."""
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
    return qg, float(eq.sum()), float(eq[ok].sum())


def peaks_report(terms, q, op, thr_nominal):
    """Per-interval-kind violation summary at charge field ``q``."""
    ctx = IterCtx(op.to_tensor(q) if isinstance(q, np.ndarray) else q, op)
    rows = []
    for j, t in enumerate(terms):
        viol, _ = t._peaks(ctx)
        constrained = t.armed.any(dim=2)
        n_c = int(constrained.sum())
        # slack = threshold(+margin) - peak, over the constrained pixels
        C = torch.cumsum(t.w * ctx.block_pred, dim=2)
        Cm = torch.where(t.armed, C, t._neg_inf)
        peak = Cm.max(dim=2).values
        pk = peak[constrained]
        rows.append({
            "ordinal": j,
            "constrained_pixel_intervals": n_c,
            "check_instants": int(t.armed.sum()),
            "n_violating": int((viol > 0).sum()),
            "max_violation_ke": float(viol.max()) if n_c else 0.0,
            "value": float(t.value(ctx)),
            "peak_max_ke": float(pk.max()) if n_c else 0.0,
            "peak_median_ke": float(pk.median()) if n_c else 0.0,
            "bound_ke": t.threshold,
            "thr_nominal_ke": thr_nominal,
        })
    return rows


def one(tag, arm):
    job = f"{AO}/{CAMPAIGN}/{arm}/job_{tag}.yaml"
    solved = f"{AO}/{CAMPAIGN}/{arm}/{tag}/{tag}_event_0_0.npz"
    if not os.path.exists(job):
        print(f"{tag} [{arm}]: no job yaml -- skip", flush=True)
        return None
    cfg = yaml.safe_load(open(job))
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = float(rc.adc_hold_delay)
    thr = float(rc.threshold)

    bm = [e for e in cfg["sequence"] if "BuildMeasurement" in e][0]
    acq = bm["BuildMeasurement"].get("acq_start")
    acq = getattr(ev, "acq_start", None) if acq == "event" else acq

    def build_pre(close_back=0.0):
        return pre_trigger_censors(
            op, store.get("hits_view"), boff,
            csa_reset_time=float(rc.csa_reset_time or 0), threshold=thr,
            acq_start=acq, npad_bins=50, beta=1.0, margin=3.0, norm="l1",
            bin_ticks=int(B), one_tick=float(rc.one_tick or 1),
            close_back=close_back,
            include_post_reset=True)   # probe both kinds; the term
                                       # itself ships pre-trigger only

    pre = build_pre()
    post = CensorRunningMax.from_hits(
        op, store.get("hits_view"), boff,
        csa_reset_time=float(rc.csa_reset_time or 0), threshold=thr,
        npad_bins=50, beta=1.0, margin=3.0, norm="l1", bin_ticks=int(B))

    qt, q_all, q_in = truth_on_fit_grid(cfg, op, boff, B)
    out = {"tag": tag, "arm": arm, "rows": int(op.n_data),
           "truth_ke_total": q_all, "truth_ke_on_grid": q_in,
           "n_interval_kinds": len(pre)}
    out["truth"] = peaks_report(pre, qt, op, thr)
    out["truth_post_latch"] = peaks_report([post], qt, op, thr)
    # How far must the closing boundary be backed off before the intervals
    # are truth-feasible?  The checks nearest the crossing have ~0 slack, so
    # they report the operator's within-bin model error as a violation.
    # Reported separately for the PRE-TRIGGER interval (a pixel's first) and
    # the POST-RESET ones (between triggers), which are the ones that fail at
    # close_back = 0.
    scan = []
    for cb in (0.0, 10.0, 20.0, B, 2 * B):
        ts = build_pre(cb)
        rep = peaks_report(ts, qt, op, thr)
        pre_trigger, post_reset = rep[:1], rep[1:]
        scan.append({
            "close_back_ticks": cb,
            "pre_trigger": {
                "intervals": sum(r["constrained_pixel_intervals"] for r in pre_trigger),
                "checks": sum(r["check_instants"] for r in pre_trigger),
                "truth_violating": sum(r["n_violating"] for r in pre_trigger),
                "truth_max_viol_ke": max([r["max_violation_ke"] for r in pre_trigger],
                                         default=0.0)},
            "post_reset": {
                "intervals": sum(r["constrained_pixel_intervals"] for r in post_reset),
                "checks": sum(r["check_instants"] for r in post_reset),
                "truth_violating": sum(r["n_violating"] for r in post_reset),
                "truth_max_viol_ke": max([r["max_violation_ke"] for r in post_reset],
                                         default=0.0)},
        })
    out["close_back_scan"] = scan
    if os.path.exists(solved):
        qh = np.asarray(np.load(solved, allow_pickle=True)["deconv_q_sharp"],
                        float)
        out["reco"] = peaks_report(pre, qh, op, thr)
        out["reco_post_latch"] = peaks_report([post], qh, op, thr)

    print(f"\n{tag} [{arm}]  rows={op.n_data}  interval kinds={len(pre)}", flush=True)
    for what in ("truth", "reco"):
        if what not in out:
            continue
        for r in out[what]:
            print(f"  {what:5s} n{r['ordinal']}: "
                  f"{r['constrained_pixel_intervals']:4d} intervals, "
                  f"{r['check_instants']:6d} checks, "
                  f"violating {r['n_violating']:3d}, "
                  f"max viol {r['max_violation_ke']:8.3f} ke, "
                  f"peak med/max {r['peak_median_ke']:7.2f}/"
                  f"{r['peak_max_ke']:8.2f} vs bound {r['bound_ke']:.1f}",
                  flush=True)
        pl = out[f"{what}_post_latch"][0]
        print(f"  {what:5s} post-latch: violating {pl['n_violating']:3d}, "
              f"max viol {pl['max_violation_ke']:.3f} ke", flush=True)
    print("  close_back [ticks] | pre-trigger: int checks viol maxV "
          "| post-reset: int checks viol maxV", flush=True)
    for s in out["close_back_scan"]:
        v, p = s["pre_trigger"], s["post_reset"]
        print(f"  {s['close_back_ticks']:18.0f} | "
              f"{v['intervals']:5d} {v['checks']:6d} {v['truth_violating']:4d} "
              f"{v['truth_max_viol_ke']:5.2f} | "
              f"{p['intervals']:11d} {p['checks']:6d} {p['truth_violating']:4d} "
              f"{p['truth_max_viol_ke']:5.2f}", flush=True)
    return out


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "B"
    tags = sys.argv[2:] or TAGS
    os.makedirs(OUT, exist_ok=True)
    res = [r for r in (one(t, arm) for t in tags) if r]
    path = f"{OUT}/censor_pre_probe_{CAMPAIGN}_{arm}.json"
    json.dump(res, open(path, "w"), indent=1)
    print(f"\n-> {path}", flush=True)
