#!/usr/bin/env python
"""Verify that the numbers in the burst-metrics figure are reproducible.

The figure (`plot_burst_metrics.py`) reads `eval_centers/eval_centers.json`,
which was produced by the standalone driver `eval_centers/eval_centers.py`.
This script checks those values against a SECOND, independent route: the
framework `Evaluate` algorithm re-run on the same archived solves through
`algrun.py`, which recomputes the smeared truth per event and rebins onto the
universal grid from scratch.

Agreement is therefore a statement about the pipeline, not about a cached file:
the same solves scored twice by two code paths must give the same metrics.

  --json      the file the figure reads (eval_centers.json)
  --against   one or more algrun outputs carrying `eval.metrics` per run,
              each optionally suffixed `:rtol` to give it its own tolerance.
              The two checks below are different in kind and must not share a
              tolerance:

              REPRODUCIBILITY -- the same solves scored by a second code path.
                Must be bit-exact.  killed_origin.json at the default 1e-6.

              ROBUSTNESS -- a DIFFERENT solve arm (killed_origin_noquiet.json,
                the terms:[] ablation of the quiet term).  It is not supposed
                to agree bit-exactly; it should agree within the measured size
                of the ablation.  Over its 40 nburst>=2 tags the maxima are
                2.2e-4 (r), 2.8e-4 (slope), 1.4e-2 (true_killed),
                1.9e-2 (ghost_iso_charge), so `:3e-2` is the honest gate.
  --config    which arm of --json to compare against (default B)
  --metrics   default: the seven the figure plots, plus sum_truth
  --rtol      relative tolerance, default 1e-6

Exits 1 if any metric on any tag exceeds --rtol, so it can gate a rebuild.

Usage:
  cd UnfoldLArPix
  PYTHONPATH=src .venv/bin/python examples/verify_burst_metrics.py \
      --json    examples/analysis_output/eval_centers/eval_centers.json \
      --against examples/analysis_output/killed_origin/killed_origin.json \
                examples/analysis_output/killed_origin/killed_origin_noquiet.json:3e-2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIG_METRICS = ["pearson_r", "slope", "resid_rms", "ghost_frac",
               "ghost_iso_frac", "ghost_iso_charge", "true_killed",
               "sum_truth"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--against", nargs="+", required=True)
    ap.add_argument("--config", default="B")
    ap.add_argument("--mode", default="centers")
    ap.add_argument("--metrics", nargs="+", default=FIG_METRICS)
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="default tolerance; a per-source `SRC:rtol` wins")
    a = ap.parse_args()

    E = json.loads(Path(a.json).read_text())
    fails, checked = [], 0
    for spec in a.against:
        src, _, rt = spec.partition(":")
        rtol = float(rt) if rt else a.rtol
        d = json.loads(Path(src).read_text())
        runs = {r["tag"]: r.get("products", {}) for r in d.get("runs", [])
                if "error" not in r}
        errs = [r["tag"] for r in d.get("runs", []) if "error" in r]
        missing = [t for t, p in runs.items()
                   if not isinstance(p.get("eval.metrics"), dict)]
        print(f"\n=== {Path(src).name}   (rtol {rtol:g}) ===")
        print(f"  {len(runs)} runs, {len(errs)} errored, "
              f"{len(missing)} without eval.metrics")
        if errs:
            print(f"  ERRORED: {', '.join(errs)}")
        rows = []
        for m in a.metrics:
            ds = []
            for t, p in runs.items():
                got = p.get("eval.metrics")
                if not isinstance(got, dict) or m not in got:
                    continue
                key = f"{a.config}|{t}"
                if key not in E:
                    fails.append(f"{src}: {key} absent from {a.json}")
                    continue
                want = float(E[key][a.mode][m])
                have = float(got[m])
                rel = abs(have - want) / max(abs(want), 1e-12)
                ds.append((abs(have - want), rel, t))
                checked += 1
            if not ds:
                rows.append((m, 0, None, None, None))
                continue
            mx = max(ds, key=lambda x: x[1])
            rows.append((m, len(ds), mx[0], mx[1], mx[2]))
            if mx[1] > rtol:
                fails.append(f"{Path(src).name}: {m} on {mx[2]} "
                             f"rel {mx[1]:.3g} > rtol {rtol:g}")
        print(f"  {'metric':18s}{'n':>4s}{'max |delta|':>13s}"
              f"{'max rel':>11s}{'worst tag':>15s}")
        for m, n, ad, rel, t in rows:
            if not n:
                print(f"  {m:18s}{0:4d}{'absent':>13s}")
                continue
            print(f"  {m:18s}{n:4d}{ad:13.3g}{rel:11.2g}{t:>15s}")

    print(f"\n{checked} (tag, metric) comparisons")
    if fails:
        print(f"FAIL — {len(fails)}:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("PASS — every comparison within its source's rtol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
