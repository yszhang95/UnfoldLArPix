"""Should the L1 penalty be weaker or stronger where the charge is large?

`soft_len` is exactly the near-vs-far L1 contrast knob.  The stage weight
is `alpha_v = a * exp(d_v / soft_len)` with `d_v` the distance to the
previous stage's skeleton (`q > seed_cut`), i.e. to where the charge
already is.  So

  soft_len -> 0    alpha = a on the skeleton, effectively infinite off it
                   -- maximal relaxation ON large charge (adaptive lasso
                   taken to its limit)
  soft_len -> inf  uniform alpha everywhere -- no relaxation on large
                   charge at all

Scanning it therefore answers "smaller or larger L1 near large charge?"
without touching the solver.  The record value is 2.0.

The competing considerations are real in both directions: adaptive-lasso
logic says relax on large coefficients to cut the shrinkage bias, while
the measured operator error scales with the charge in the partially
covered bins (kappa ~ 0.45, see PLAN_error_model.md §2.2), so large
charge is exactly where the model is least trustworthy and a weak prior
lets the solver chase model error.  The scan decides which dominates.
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
OUT = f"{AO}/softlen_scan"

from axis_metric_ab import measure  # noqa: E402
from unfoldlarpix.fwk.runner import run as run_job  # noqa: E402

LENS = [0.5, 1.0, 2.0, 4.0, 8.0, 1000.0]      # 1000 ~ uniform alpha


def main(tags):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for tag in tags:
        res[tag] = {}
        base = yaml.safe_load(open(f"{AO}/nb1_fraccensor/B/job_{tag}.yaml"))
        for L in LENS:
            key = f"L{L:g}"
            cfg = copy.deepcopy(base)
            for entry in cfg["sequence"]:
                (name, props), = entry.items()
                if name == "Solve":
                    props["strategy"]["soft_len"] = float(L)
                if name == "WriteCharges":
                    props["out_dir"] = f"{OUT}/{key}/{tag}"
                    props["prefix"] = tag
            cp = f"{OUT}/{key}_job_{tag}.yaml"
            os.makedirs(f"{OUT}/{key}/{tag}", exist_ok=True)
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            run_job(cp)
            res[tag][key] = measure(f"{OUT}/{key}/{tag}/{tag}_event_0_0.npz")
        print(f"\n{tag}   (L -> 0: weakest L1 on large charge; "
              f"L -> inf: uniform alpha)", flush=True)
        print("  %-9s %7s %7s %8s %8s %9s %9s %8s" %
              ("soft_len", "r", "slope", "integ%", "ghost%", "isoghost",
               "killed", "TVt/TVxy"))
        for L in LENS:
            m = res[tag][f"L{L:g}"]
            print("  %-9g %7.4f %7.3f %8.2f %8.3f %9.2f %9.1f %8.3f" %
                  (L, m["pearson_r"], m["slope"], m["integral_pct"],
                   100.0 * m["ghost_frac"], m["ghost_iso_charge"],
                   m["true_killed"], m["excess_time_roughness"]), flush=True)
    json.dump(res, open(f"{OUT}/softlen_scan.json", "w"), indent=1)
    print(f"\n-> {OUT}/softlen_scan.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["mu_a75_nb1", "mu_a00_nb1", "pos_a00_nb1"])
