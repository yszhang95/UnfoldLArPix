"""A/B/C on the soft-seed distance metric.

The seed distance behind the weighted-L1 alpha field is measured in grid
INDICES, and it is computed with np.roll dilations, so it is also
periodic.  Two separate effects hide in that one choice:

  stock      grid-index Manhattan, periodic boundary  (the record)
  unit       grid-index Manhattan, open boundary      (isolates the wrap)
  physical   step cost = physical length, open        (isolates the
             anisotropy: one time bin is 1.5 us * 1.59645 mm/us
             = 2.395 mm against a pixel's 4.434 mm, so the time cost is
             2.395/4.434 = 0.540)

Everything else -- alphas, seed_cut, soft_len, censor, iterations -- is
copied verbatim from the record job, so the three arms differ only in the
metric.  Reported on the universal grid against the once-smeared truth,
with the roughness anisotropy that motivated the test.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import yaml

ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
OUT = f"{AO}/axis_metric_ab"

from roughness import PITCH_MM, TBIN_MM, occupancy, tv  # noqa: E402
from unfoldlarpix.eval.universal import (metrics_from_blocks,  # noqa: E402
                                         universal_rebin)
from unfoldlarpix.fwk.runner import run as run_job  # noqa: E402

TIME_COST = round(TBIN_MM / PITCH_MM, 3)
ARMS = {"stock": None, "unit": [1.0, 1.0, 1.0],
        "physical": [1.0, 1.0, TIME_COST]}


def make_cfg(tag, arm, cost):
    cfg = copy.deepcopy(yaml.safe_load(
        open(f"{AO}/nb1_fraccensor/B/job_{tag}.yaml")))
    for entry in cfg["sequence"]:
        (name, props), = entry.items()
        if name == "Solve" and cost is not None:
            props["strategy"]["soft_axis_cost"] = cost
        if name == "WriteCharges":
            props["out_dir"] = f"{OUT}/{arm}/{tag}"
            props["prefix"] = tag
    return cfg


def measure(path):
    truth, reco = universal_rebin(path)
    m = metrics_from_blocks(truth, reco)
    st, sr = truth.sum(), reco.sum()
    rough = {}
    for lab, a, s in (("truth", truth, st), ("reco", reco, sr)):
        txy = 0.5 * (tv(a, 0, s) + tv(a, 1, s))
        rough[lab] = tv(a, 2, s) / max(txy, 1e-12)
    m["TV_t_over_xy_truth"] = rough["truth"]
    m["TV_t_over_xy_reco"] = rough["reco"]
    m["excess_time_roughness"] = rough["reco"] / max(rough["truth"], 1e-12)
    m["occ90_reco"] = occupancy(reco)
    m["occ90_truth"] = occupancy(truth)
    return m


def main(tags):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for tag in tags:
        res[tag] = {}
        for arm, cost in ARMS.items():
            d = f"{OUT}/{arm}/{tag}"
            os.makedirs(d, exist_ok=True)
            cp = f"{OUT}/{arm}/job_{tag}.yaml"
            yaml.safe_dump(make_cfg(tag, arm, cost), open(cp, "w"),
                           sort_keys=False)
            run_job(cp)
            res[tag][arm] = measure(f"{d}/{tag}_event_0_0.npz")
        print(f"\n{tag}   (time step cost {TIME_COST} = 2.395/4.434 mm)",
              flush=True)
        print("  %-9s %7s %7s %8s %8s %9s %9s %8s" %
              ("arm", "r", "slope", "integ%", "ghost%", "isoghost",
               "killed", "TVt/TVxy"))
        for arm in ARMS:
            m = res[tag][arm]
            print("  %-9s %7.4f %7.3f %8.2f %8.3f %9.2f %9.1f %8.3f" %
                  (arm, m["pearson_r"], m["slope"], m["integral_pct"],
                   100.0 * m["ghost_frac"], m["ghost_iso_charge"],
                   m["true_killed"], m["excess_time_roughness"]), flush=True)
    json.dump(res, open(f"{OUT}/axis_metric_ab.json", "w"), indent=1)
    print(f"\n-> {OUT}/axis_metric_ab.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["mu_a75_nb1", "mu_a00_nb1", "mu_a50_nb1",
                          "pos_a75_nb1"])
