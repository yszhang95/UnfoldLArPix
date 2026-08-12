"""How much of the integral deficit is L1 shrinkage on the large charges?

`soft_len` only changes the CONTRAST between the penalty on the skeleton
and off it.  The direct version of "should L1 be weaker where the charge
is large?" is FinalRefit: it freezes the strong support (q > eps) and
re-solves the amplitudes with alpha = 0, i.e. no L1 bias at all on the
voxels that carry the charge.  The record nb1 configuration does NOT
include a refit stage, so whatever shrinkage the last ladder rung
(alpha = 0.3, weighted) applies is still in the answer.

Arms:
  none          the record -- ladder only
  refit         + FinalRefit(eps=0.5, alpha=0)
  refit_eps0.2  + FinalRefit(eps=0.2, alpha=0)  -- a wider frozen support,
                to see whether the gain comes from the strong voxels only

If the integral moves towards zero and r/ghost hold, the deficit was L1
shrinkage and the fix is architectural, not a matter of retuning alpha.
If the integral overshoots or the ghost fraction climbs, the prior was
doing real work and removing it lets the solver chase operator error.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import yaml

ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
OUT = f"{AO}/refit_test"

from axis_metric_ab import measure  # noqa: E402
from unfoldlarpix.fwk.runner import run as run_job  # noqa: E402

ARMS = {"none": None,
        "refit": {"eps": 0.5, "alpha": 0.0},
        "refit_eps0.2": {"eps": 0.2, "alpha": 0.0}}


def main(tags):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for tag in tags:
        res[tag] = {}
        base = yaml.safe_load(open(f"{AO}/nb1_fraccensor/B/job_{tag}.yaml"))
        for arm, rcfg in ARMS.items():
            cfg = copy.deepcopy(base)
            for entry in cfg["sequence"]:
                (name, props), = entry.items()
                if name == "Solve" and rcfg is not None:
                    props["refit"] = dict(rcfg)
                if name == "WriteCharges":
                    props["out_dir"] = f"{OUT}/{arm}/{tag}"
                    props["prefix"] = tag
            os.makedirs(f"{OUT}/{arm}/{tag}", exist_ok=True)
            cp = f"{OUT}/{arm}_job_{tag}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            run_job(cp)
            res[tag][arm] = measure(f"{OUT}/{arm}/{tag}/{tag}_event_0_0.npz")
        print(f"\n{tag}   (refit = no L1 at all on the frozen strong "
              f"support)", flush=True)
        print("  %-13s %7s %7s %8s %8s %9s %9s %8s" %
              ("arm", "r", "slope", "integ%", "ghost%", "isoghost",
               "killed", "TVt/TVxy"))
        for arm in ARMS:
            m = res[tag][arm]
            print("  %-13s %7.4f %7.3f %8.2f %8.3f %9.2f %9.1f %8.3f" %
                  (arm, m["pearson_r"], m["slope"], m["integral_pct"],
                   100.0 * m["ghost_frac"], m["ghost_iso_charge"],
                   m["true_killed"], m["excess_time_roughness"]), flush=True)
    json.dump(res, open(f"{OUT}/refit_test.json", "w"), indent=1)
    print(f"\n-> {OUT}/refit_test.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["mu_a75_nb1", "mu_a00_nb1", "mu_a50_nb1",
                          "pos_a00_nb1"])
