#!/usr/bin/env python3
"""Angle x burst-scan summary: reconstruction metrics vs nburst, one curve per
anode angle, muon and positron panels. Reads metrics_*.json + *_stats.json from
analysis_output/angscan/. Emits angscan_summary.png and a LaTeX metrics table."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output/angscan")
NB = [1, 2, 4, 8, 16, 64]
ANG = ["00", "25", "50", "75"]
PARTS = [("mu", "muon"), ("pos", "positron")]
COL = {"00": "tab:blue", "25": "tab:green", "50": "tab:orange", "75": "tab:red"}


def load(tp, ang, N):
    m = OUT / f"metrics_{tp}_a{ang}_nb{N}.json"
    s = OUT / f"{tp}_a{ang}_nb{N}_stats.json"
    if not m.exists():
        return None
    d = json.loads(m.read_text()); d = d[list(d)[0]] if isinstance(list(d.values())[0], dict) else d
    if s.exists():
        d.update(json.loads(s.read_text()))
    return d


DATA = {(tp, ang, N): load(tp, ang, N) for tp, _ in PARTS for ang in ANG for N in NB}

METRICS = [("pearson_r", "Pearson $r$", None),
           ("integral_pct", "integral bias [%]", 0),
           ("ghost_frac", "ghost fraction [%]", None),
           ("relrms_pointwise_hi", "high-$q$ rel. RMS [%]", None)]

fig, axes = plt.subplots(len(PARTS), len(METRICS), figsize=(19, 8.4), squeeze=False)
for r, (tp, pname) in enumerate(PARTS):
    for c, (key, ylab, hline) in enumerate(METRICS):
        ax = axes[r][c]
        for ang in ANG:
            xs, ys = [], []
            for N in NB:
                d = DATA[(tp, ang, N)]
                if d is None or key not in d or d[key] is None:
                    continue
                v = d[key]
                if key == "ghost_frac":
                    v = 100 * v
                if isinstance(v, float) and np.isnan(v):
                    continue
                xs.append(N); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", color=COL[ang], label=f"{int(ang)}$^\\circ$")
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.7, ls=":")
        ax.set_xscale("log", base=2); ax.set_xticks(NB); ax.set_xticklabels(NB)
        ax.grid(alpha=0.3)
        if r == len(PARTS) - 1:
            ax.set_xlabel("nburst")
        if c == 0:
            ax.set_ylabel(f"{pname}", fontsize=12, fontweight="bold")
        ax.set_title(ylab if r == 0 else "", fontsize=11)
        if r == 0 and c == len(METRICS) - 1:
            ax.legend(title="angle to anode", fontsize=8, ncol=1)
fig.suptitle("Deconvolution vs burst count, by anode angle — muon (top) and positron (bottom), "
             "v2a 25$\\times$25 FR, TPC0", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "angscan_summary.png", dpi=130)
print(f"wrote {OUT/'angscan_summary.png'}")

# ---- LaTeX metrics table (all present configs) ----
rows = []
for tp, pname in PARTS:
    for ang in ANG:
        for N in NB:
            d = DATA[(tp, ang, N)]
            if d is None:
                continue
            rows.append(f"{pname} & {int(ang)} & {N} & {d.get('integral_pct',0):+.2f} & "
                        f"{d.get('pearson_r',0):.4f} & {d.get('slope',0):.3f} & "
                        f"{100*d.get('ghost_frac',0):.2f} & {d.get('ghost_iso_charge',0):.2f} & "
                        f"{d.get('true_killed',0):.0f} & "
                        f"{d.get('relrms_pointwise_hi',float('nan')):.1f} \\\\")
(OUT / "metrics_table.tex").write_text("\n".join(rows))
n = sum(1 for v in DATA.values() if v is not None)
print(f"metrics table: {n}/48 configs -> {OUT/'metrics_table.tex'}")
