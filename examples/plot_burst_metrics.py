#!/usr/bin/env python
"""Metrics against nburst, sized for an IEEE two-column manuscript.

Reads an evaluation JSON keyed "<config>|<series>_nb<n>" (eval_centers.json or
reeval_sp0.5.json) and plots one panel per metric.  No number is transcribed:
every point comes from the JSON, and the JSON comes from
`eval_centers/eval_centers.py` (metrics_from_blocks, i.e. the `Evaluate`
algorithm's protocol).

Differs from `eval_centers/mkfigs_centers.py`, which is the JINST/technote
figure: that one splits muon and positron into two columns and overlays the
three solver configurations.  Here the configuration is FIXED (default B, the
adopted one) because the manuscript text does not discuss A/B/C, which frees
the linestyle for the particle and halves the panel count.

Encoding: angle is ordered, so it gets a single-hue ordinal ramp (light = 0
deg, dark = 75 deg) rather than four arbitrary colours; the particle gets
linestyle + marker, so identity is never colour-alone.

  --metrics  default: pearson_r slope resid_rms ghost_frac ghost_iso_frac
             (`resid_rms` is the manuscript's "RMS width")
  --width    7.16 for \\begin{figure*}, 3.5 for a single column
  --out      PDF path; a .txt table of the same numbers is written beside it

Usage:
  cd UnfoldLArPix
  PYTHONPATH=src .venv/bin/python examples/plot_burst_metrics.py \
      --json examples/analysis_output/eval_centers/eval_centers.json \
      --out  /tmp/burst_metrics.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ANG = ["00", "25", "50", "75"]
NB = [1, 2, 4, 8, 16, 64]
# ordinal single-hue ramp, light -> dark with angle (validated: monotone
# lightness, adjacent dL >= 0.06, light end 2.06:1 on a white surface)
CANG = {"00": "#86b6ef", "25": "#5598e7", "50": "#2a78d6", "75": "#184f95"}
PART = {"mu": ("muon", "-", "o"), "pos": ("positron", "--", "s")}
# Solver arms, for --x angle where the abscissa takes the angle and colour is
# free.  Okabe-Ito hues, and the linestyle repeats the identity so the arms are
# separable in greyscale too; B (production) is the solid one.
CCFG = {"A": ("#D55E00", "--", "^"), "B": ("#0072B2", "-", "o"),
        "C": ("#009E73", ":", "s")}
LABEL = {
    "pearson_r":      ("Pearson $r$", 1.0),
    "slope":          ("slope", 1.0),
    "resid_rms":      ("RMS width [ke]", 1.0),
    "ghost_frac":     ("ghost rate [%]", 100.0),
    "ghost_iso_frac": ("iso. ghost rate [%]", 100.0),
    "integral_pct":   ("integral bias [%]", 1.0),
    "true_killed":    ("killed truth [ke]", 1.0),
    "killed_pct":     ("killed truth [%]", 1.0),
    "ghost_charge_pct":     ("ghost charge [%]", 1.0),
    "ghost_iso_charge_pct": ("iso. ghost charge [%]", 1.0),
}

# Metrics the evaluation records as an absolute charge, quoted here as a
# percentage of the event's true charge -- which is how the manuscript states
# them.  Derived in the plotter, not stored, so there is one definition.
DERIVED = {
    "killed_pct":           ("true_killed", "sum_truth"),
    "ghost_charge_pct":     ("ghost_charge", "sum_truth"),
    "ghost_iso_charge_pct": ("ghost_iso_charge", "sum_truth"),
}
UNITY = {"slope"}          # draw a reference line at 1
ZERO = {"integral_pct"}    # draw a reference line at 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--extra-json", default=None,
                    help="a second evaluation JSON merged into the first, for "
                         "columns added after the fact (ghost_charge/)")
    ap.add_argument("--mode", default="centers",
                    help="'centers' / 'offsets' for a nested JSON; "
                         "'none' for a flat one")
    ap.add_argument("--config", default="B")
    ap.add_argument("--x", default="nburst", choices=["nburst", "angle"],
                    help="abscissa. 'nburst': one curve per angle at a fixed "
                         "config (the manuscript figure). 'angle': one curve "
                         "per config in --configs at a fixed --nburst, which "
                         "is the arm comparison -- colour then encodes the arm "
                         "because the angle has moved to the abscissa.")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="--x angle only: the arms to overlay (default: "
                         "whichever of A/B/C the JSON has)")
    ap.add_argument("--nburst", type=int, default=4,
                    help="--x angle only: the burst count to fix")
    ap.add_argument("--series", nargs="+", default=None,
                    help="subset of particle_angle series, e.g. mu pos, or "
                         "mu_a00 mu_a50. Default: every series in the JSON.")
    ap.add_argument("--metrics", nargs="+",
                    default=["pearson_r", "slope", "resid_rms",
                             "killed_pct", "ghost_frac", "ghost_iso_frac",
                             "ghost_iso_charge_pct"],
                    help="every default is pooled over the event's voxels, so "
                         "it has an internal sample. integral_pct is NOT: it "
                         "is one ratio of two totals per track, and with one "
                         "event per configuration it carries no spread. Pass "
                         "it explicitly if you want it.")
    ap.add_argument("--width", type=float, default=3.5,
                    help="3.5 = IEEE single column (default), "
                         "7.16 = \\begin{figure*}, ~5.8 = a 16:9 beamer frame")
    ap.add_argument("--ncols", type=int, default=2)
    ap.add_argument("--panel-height", type=float, default=1.5,
                    help="inches per panel row (default 1.5, the IEEE size)")
    ap.add_argument("--font", type=float, default=8.0,
                    help="base font size in pt (default 8, the IEEE size)")
    ap.add_argument("--share-ylim", action="store_true",
                    help="set each panel's y range from EVERY series in the "
                         "JSON, not only the plotted ones. Two figures made "
                         "with different --series are then on one scale and "
                         "can be read side by side; without it each figure "
                         "auto-scales and the two are not comparable.")
    ap.add_argument("--title", default=None,
                    help="suptitle, e.g. the particle when --series splits it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def load(path):
        """Accept either an eval_centers-style map or an algrun output.

        algrun writes {"runs":[{"tag":..., "products":{"eval.metrics":{...}}}]};
        eval_centers writes {"<cfg>|<tag>": {"centers": {...}, ...}}.  Both are
        the same numbers from the same call, so the plotter normalises rather
        than making the caller convert.
        """
        d = json.loads(Path(path).read_text())
        # eval_padfix wraps the map in {"meta": ..., "metrics": ...} so the
        # code state that produced it travels with the numbers
        if "metrics" in d and "meta" in d:
            d = d["metrics"]
        if "runs" not in d:
            return d, True                     # nested (centers/offsets)
        out = {}
        for r in d["runs"]:
            m = r.get("products", {}).get("eval.metrics")
            if isinstance(m, dict):
                out[f"{a.config}|{r['tag']}"] = m
        return out, False

    E, nested = load(a.json)
    if a.extra_json:
        # a later rescoring that adds columns (e.g. ghost_charge/); merged per
        # entry so the extra file only has to carry the new keys
        X, _ = load(a.extra_json)
        for k, v in X.items():
            if k in E:
                for m in v:
                    if m in E[k]:
                        E[k][m] = {**E[k][m], **v[m]}
            else:
                E[k] = v
    mode = (None if a.mode == "none" or not nested else a.mode)

    want = a.series
    def keep(part, ang):
        if not want:
            return True
        return any(w in (part, f"{part}_a{ang}") for w in want)

    def get(part, ang, nb, key, cfg=None):
        e = E[f"{cfg or a.config}|{part}_a{ang}_nb{nb}"]
        d = e[mode] if mode else {k: v for k, v in e.items()
                                  if not k.startswith("_")}
        if key in d:
            return float(d[key])
        if key in DERIVED:
            num, den = DERIVED[key]
            if num not in d:
                raise KeyError(
                    f"{key} needs {num}, absent from this JSON. "
                    f"ghost_charge is only in the ghost_charge/ rescoring -- "
                    f"pass it with --extra-json.")
            return 100.0 * float(d[num]) / float(d[den])
        raise KeyError(f"{key} not in the JSON and not a derived metric")

    # IEEE house style: no gridlines, ticks turned inward on all four spines
    F = a.font
    plt.rcParams.update({
        "font.size": F, "axes.labelsize": F, "xtick.labelsize": F - 0.5,
        "ytick.labelsize": F - 0.5, "axes.grid": False,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "pdf.fonttype": 42, "figure.dpi": 200,
    })

    n = len(a.metrics)
    ncols = min(a.ncols, n)
    nrows = -(-n // ncols)
    # a spare grid cell holds the legend for free; only when the metrics fill
    # the grid exactly does the figure have to grow a strip for it
    spare = nrows * ncols - n
    legend_in = 0.0 if spare else 0.62
    title_in = 0.28 if a.title else 0.0
    height = a.panel_height * nrows + 0.15 + legend_in + title_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(a.width, height),
                             squeeze=False)
    flat = axes.ravel()

    # One "curve" is (x, y-getter, colour, linestyle, marker).  Building the
    # list once, before the panel loop, is what lets the two abscissae share
    # every other decision -- panels, shared ranges, legend placement, table.
    if a.x == "nburst":
        curves = [(NB, (lambda key, p=part, g=ang: [get(p, g, nb, key)
                                                    for nb in NB]),
                   CANG[ang], ls, mk)
                  for part, (_, ls, mk) in PART.items() for ang in ANG
                  if keep(part, ang)]
        xlabel = r"$n_{\mathrm{burst}}$"
        # the arms are not on this figure; the angle is the colour
        handles = [Line2D([], [], color=CANG[g], lw=1.4,
                          label=rf"$\theta={int(g)}^\circ$") for g in ANG]
        handles += [Line2D([], [], color="0.35", lw=1.0, ls=ls, marker=mk,
                           ms=2.8, mec="white", mew=0.5, label=name)
                    for pt, (name, ls, mk) in PART.items()
                    if any(keep(pt, g) for g in ANG)]
    else:
        cfgs = a.configs or [c for c in ("A", "B", "C")
                             if any(k.startswith(f"{c}|") for k in E)]
        xs = [int(g) for g in ANG]
        curves = [(xs, (lambda key, p=part, c=cfg: [get(p, g, a.nburst, key,
                                                        cfg=c) for g in ANG]),
                   CCFG[cfg][0], CCFG[cfg][1], CCFG[cfg][2])
                  for part, _ in PART.items() for cfg in cfgs
                  if any(keep(part, g) for g in ANG)]
        xlabel = r"$\theta$ to anode plane [deg]"
        handles = [Line2D([], [], color=CCFG[c][0], lw=1.4, ls=CCFG[c][1],
                          marker=CCFG[c][2], ms=2.8, mec="white", mew=0.5,
                          label=f"arm {c}") for c in cfgs]
        handles += [Line2D([], [], color="none",
                           label=rf"$n_{{\mathrm{{burst}}}} = {a.nburst}$")]

    def spread(key, scale):
        """Every value the JSON holds for this metric, both particles and (in
        angle mode) every arm -- the union that --share-ylim uses."""
        v = []
        for part in PART:
            for g in ANG:
                if a.x == "nburst":
                    v += [get(part, g, nb, key) * scale for nb in NB
                          if f"{a.config}|{part}_a{g}_nb{nb}" in E]
                else:
                    for c in (a.configs or ("A", "B", "C")):
                        if f"{c}|{part}_a{g}_nb{a.nburst}" in E:
                            v.append(get(part, g, a.nburst, key, cfg=c) * scale)
        return v

    # the x label goes only on the lowest panel of each column: repeating it
    # under all five costs vertical space a single-column figure does not have
    bottom = {i for i in range(n) if i + ncols >= n}
    for i, (ax, key) in enumerate(zip(flat, a.metrics, strict=False)):
        ylab, scale = LABEL.get(key, (key, 1.0))
        for xs, yget, col, ls, mk in curves:
            ax.plot(xs, [y * scale for y in yget(key)], ls, color=col,
                    marker=mk, ms=2.8, lw=1.0, mew=0.5, mec="white", zorder=3)
        if a.share_ylim:
            v = spread(key, scale)
            lo, hi = min(v), max(v)
            pad = 0.05 * (hi - lo) or 0.05 * abs(hi) or 0.05
            ax.set_ylim(lo - pad, hi + pad)
        if key in UNITY:
            ax.axhline(1.0, color="0.55", lw=0.5, zorder=1)
        if key in ZERO:
            ax.axhline(0.0, color="0.55", lw=0.5, zorder=1)
        if a.x == "nburst":
            ax.set_xscale("log", base=2)
            ax.set_xticks(NB)
            ax.set_xticklabels(NB)
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.set_xticks([int(g) for g in ANG])
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if i in bottom:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)

    for ax in flat[n:]:
        ax.axis("off")
    # the title has to be set BEFORE tight_layout, and its strip reserved in
    # the same rect, or the layout pass puts the top row of panels under it
    top = 1.0 - title_in / height if a.title else 1.0
    if a.title:
        fig.suptitle(a.title, fontsize=F + 1, y=1.0, va="top")
    if spare:
        flat[n].legend(handles=handles, fontsize=F, loc="center", ncol=1,
                       handlelength=1.8, labelspacing=0.45,
                       borderaxespad=0.0)
        fig.tight_layout(pad=0.35, rect=(0, 0, 1, top))
    else:
        frac = legend_in / height
        fig.tight_layout(pad=0.35, rect=(0, frac * 0.95, 1, top))
        fig.legend(handles=handles, fontsize=F, loc="lower center", ncol=3,
                   handlelength=1.8, columnspacing=1.4, labelspacing=0.4,
                   bbox_to_anchor=(0.5, 0.0))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

    # the table view: the same numbers, so the figure is never the only source
    tab = out.with_suffix(".txt")
    with tab.open("w") as f:
        f.write(f"# {a.json}  mode {a.mode}  x {a.x}\n")
        if a.x == "nburst":
            f.write(f"# config {a.config}\n")
            f.write("metric\tparticle\tangle\t"
                    + "\t".join(f"nb{n}" for n in NB) + "\n")
            for key in a.metrics:
                for part in PART:
                    for ang in ANG:
                        if not keep(part, ang):
                            continue
                        v = [get(part, ang, nb, key) for nb in NB]
                        f.write(f"{key}\t{part}\t{ang}\t"
                                + "\t".join(f"{x:.5g}" for x in v) + "\n")
        else:
            cfgs = a.configs or [c for c in ("A", "B", "C")
                                 if any(k.startswith(f"{c}|") for k in E)]
            f.write(f"# nburst {a.nburst}\n")
            f.write("metric\tparticle\tarm\t"
                    + "\t".join(f"a{g}" for g in ANG) + "\n")
            for key in a.metrics:
                for part in PART:
                    if not any(keep(part, g) for g in ANG):
                        continue
                    for c in cfgs:
                        v = [get(part, g, a.nburst, key, cfg=c) for g in ANG]
                        f.write(f"{key}\t{part}\t{c}\t"
                                + "\t".join(f"{x:.5g}" for x in v) + "\n")
    print(f"wrote {out}\nwrote {tab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
