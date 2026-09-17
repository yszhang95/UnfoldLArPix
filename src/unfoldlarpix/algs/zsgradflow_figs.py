"""Figures for :mod:`unfoldlarpix.algs.zsgradflow_algs`.

Reads only the archived JSON and NPZ of the gradient-flow campaign, so the
job needs no event, no response file and no GPU.  Five figures:

``GF1`` the state at the created ionisation charge: the residual it leaves on
       each record row, the charge one solver step moves onto each pixel
       column, and the same step against release time on two probe pixels;
``GF2`` the flow started at the created charge: charge ratio, distance to the
       created charge, and charge on the +1 pixels, against iteration;
``GF3`` the same flow against the flow started at zero, with the closest
       approach of the zero start marked;
``GF4`` the endpoint of every scheme, as four distances from the created
       charge;
``GF5`` the probe pixel's waveform: created charge against the endpoints.

Colours are the campaign's: an arm keeps its colour everywhere (vermillion
least squares, green positivity, magenta positivity + l1), a variant its line
style (solid A, dashed D), a start its marker fill (filled from the created
charge, open from zero).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fwk.component import algorithm
from .zsbasis_algs import _ieee_axes, _JsonAlg

ARM_COLOR = {"ls": "#D55E00", "pos": "#009E73", "pos_l1": "#CC79A7"}
ARM_LABEL = {"ls": "least squares", "pos": "positivity",
             "pos_l1": r"positivity $+\ \ell_1$"}
VARIANT_STYLE = {"A": "-", "D": "--"}
KIND_COLOR = {"lumped": "#000000", "diff": "#E69F00",
              "pseudo": "#0072B2", "remainder": "#D55E00"}
TICK_US = 0.05


def _fmt(v, n=4):
    return f"{v:.{n}f}"


@algorithm("ZSGradFlowFigures")
class ZSGradFlowFigures(_JsonAlg):
    """GF1-GF5 from ``flow_*.json/.npz`` and, when given, ``fista_*.json``."""

    reads = ()
    writes = ("zs.gradflow.figs",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = self.props
        flow = json.load(open(p["flow_json"]))["result"]
        npz = np.load(p["flow_npz"], allow_pickle=False)
        # the FISTA record is optional: the flow job stands on its own, and a
        # missing file is a job that has not been run yet, not an error
        fj = p.get("fista_json")
        fn = p.get("fista_npz")
        fista = (json.load(open(fj))["result"]
                 if fj and Path(fj).exists() else None)
        fnpz = (np.load(fn, allow_pickle=False)
                if fn and Path(fn).exists() else None)
        if fj and fista is None:
            print(f"[ZSGradFlowFigures] {fj} not present; "
                  "GF4 shows the flow endpoints only")
        fig_dir = Path(p.get("fig_dir", "."))
        fig_dir.mkdir(parents=True, exist_ok=True)
        conv = str(p.get("convention", "acq_edge"))
        variants = list(p.get("variants", ["A", "D"]))
        arms = list(p.get("arms", ["ls", "pos", "pos_l1"]))
        probe = list(p.get("probe_pixels", ["141_68", "142_68"]))
        truth_total = float(flow["truth_total_ke"])
        # the pad grid: written by later runs of ZSGradientFlow; when the
        # archive predates it, take it from the FISTA NPZ of the same sample.
        src = npz if "block_offset" in npz.files else fnpz
        if src is not None and "block_offset" in src.files:
            boff = np.asarray(src["block_offset"])
            nx, ny = (int(v) for v in np.asarray(src["pad_shape"]))
        elif p.get("block_offset") and p.get("pad_shape"):
            # archives written before ZSGradientFlow stored the pad grid: the
            # props carry it, read once from the store by FFTWarmStart
            boff = np.asarray(p["block_offset"], dtype=np.int64)
            nx, ny = (int(v) for v in p["pad_shape"])
        else:
            raise ValueError("no NPZ carries 'block_offset' and no "
                             "block_offset/pad_shape prop was given")
        if fnpz is not None and "pad_shape" in fnpz.files:
            if [nx, ny] != [int(v) for v in np.asarray(fnpz["pad_shape"])]:
                raise ValueError("the flow and FISTA NPZ disagree on the pad "
                                 "grid; they are not the same block")
        b0 = int(np.asarray(npz["fine_origin_tick"])[0])
        c = int(flow["cell_ticks"])
        rec = {"figures": [], "convention": conv, "block_offset":
               [int(v) for v in boff], "pad_shape": [nx, ny]}

        def traj(V, arm, start, field):
            k = f"traj_{conv}_{V}_{arm}_{start}_flow_{field}"
            return np.asarray(npz[k], dtype=float) if k in npz.files else None

        def run_of(rec_src, V, arm, start, method):
            for r in rec_src["runs"]:
                if (r["variant"] == V and r["arm"] == arm
                        and r["start"] == start and r["method"] == method
                        and r["convention"] == conv):
                    return r
            return None

        # ---- GF1: the state at the created ionisation charge --------------
        V0 = variants[-1] if len(variants) > 1 else variants[0]
        arm0 = str(p.get("stationarity_arm", "pos_l1"))
        fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6))
        st = flow["stationarity"][f"{conv}_{V0}"]
        res = np.asarray(npz[f"resid_truth_{conv}_{V0}"], dtype=float)
        kinds = np.asarray(npz[f"rowkind_{conv}_{V0}"]).astype(str)
        bins = np.histogram_bin_edges(res, bins=40)
        for k in sorted(set(kinds)):
            m = kinds == k
            ax[0].hist(res[m], bins=bins, histtype="step", lw=1.4,
                       color=KIND_COLOR.get(k, "#666666"),
                       label=f"{k} ({int(m.sum())} rows, "
                             f"$\\Sigma$ {res[m].sum():+.1f} ke)")
        ax[0].axvline(0.0, color="#666666", lw=0.8, ls=":")
        ax[0].set_yscale("log")
        ax[0].set_xlabel(r"$A x_{\mathrm{truth}} - y$ per row [ke]")
        ax[0].set_ylabel("rows")
        ax[0].legend(fontsize=6.5, frameon=False)
        tot = float(res.sum())
        ax[0].set_title(
            f"variant {V0}: the created charge "
            f"{'over' if tot > 0 else 'under'}-predicts the records by "
            f"{abs(tot):.1f} ke\n"
            f"$\\|Ax_{{\\mathrm{{truth}}}}-y\\|/\\|y\\|$ = "
            f"{st['truth_rel_residual']:.4f}", fontsize=7.5)

        step1 = np.asarray(npz[f"step1_pad_{conv}_{V0}_{arm0}"], dtype=float)
        colx = step1.sum(axis=1)
        px = np.arange(nx) + int(boff[0])
        keep = np.abs(colx) > 1e-9
        ax[1].bar(px[keep], colx[keep], width=0.8, color="#0072B2")
        ax[1].axhline(0.0, color="#666666", lw=0.8)
        ax[1].set_xlabel("pixel_x")
        ax[1].set_ylabel("charge one step moves [ke]")
        ax[1].set_title("where the first step puts charge\n"
                        "(all created charge is on pixel_x = 141)",
                        fontsize=7.5)
        for xi, v in zip(px[keep], colx[keep]):
            ax[1].annotate(f"{v:+.2f}", (xi, v), ha="center", fontsize=6,
                           va="bottom" if v >= 0 else "top")

        ax2r = ax[2].twinx()
        handles = []
        for i, k in enumerate(probe):
            key = f"step1_cells_{conv}_{V0}_{arm0}_{k}"
            if key not in npz.files:
                continue
            s1 = np.asarray(npz[key], dtype=float)
            tt = np.asarray(npz[f"truth_cells_{conv}_{V0}_{k}"], dtype=float)
            t_us = (b0 + c * np.arange(len(s1))) * TICK_US
            m = (np.abs(s1) > 1e-6) | (tt > 0)
            if not m.any():
                continue
            lo, hi = np.argmax(m), len(m) - np.argmax(m[::-1])
            sl = slice(max(lo - 20, 0), min(hi + 20, len(m)))
            ls = "-" if i == 0 else "--"
            handles += ax[2].plot(
                t_us[sl], s1[sl], ls, color="#0072B2", lw=1.3,
                label=f"one step, pixel {k.replace('_', ', ')} (left)")
            if tt.max() > 0:
                handles += ax2r.plot(
                    t_us[sl], tt[sl], ":", color="#000000", lw=1.1,
                    label=f"created charge, pixel {k.replace('_', ', ')} "
                          f"(right)")
        ax[2].axhline(0.0, color="#666666", lw=0.8)
        ax[2].set_xlabel(r"release time at the response plane [$\mu$s]")
        ax[2].set_ylabel("charge one step moves [ke]", color="#0072B2")
        ax[2].tick_params(axis="y", colors="#0072B2")
        ax2r.set_ylabel("created ionisation charge per cell [ke]")
        ax2r.tick_params(direction="in")
        ax[2].legend(handles, [h.get_label() for h in handles], fontsize=6.5,
                     frameon=False)
        ax[2].set_title("the same step in release time", fontsize=7.5)
        for a in ax:
            _ieee_axes(a)
        fig.tight_layout()
        self._save(fig, fig_dir / "GF1_state_at_truth", rec, plt)

        # ---- GF2: the flow from the created charge ------------------------
        fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6))
        for V in variants:
            for arm in arms:
                it = traj(V, arm, "truth", "iter")
                if it is None:
                    continue
                kw = dict(color=ARM_COLOR[arm], ls=VARIANT_STYLE[V], lw=1.3,
                          label=f"{ARM_LABEL[arm]}, {V}")
                ax[0].plot(it, traj(V, arm, "truth", "sum_over_truth"), **kw)
                ax[1].plot(it, traj(V, arm, "truth",
                                    "l1_distance_over_truth"), **kw)
                ax[2].plot(it, traj(V, arm, "truth", "class_plus1_ke"), **kw)
        ax[0].axhline(1.0, color="#000000", lw=0.9, ls=":")
        ax[0].axhline(float(flow["stationarity"][f"{conv}_{variants[0]}"]
                            ["sum_y_over_truth"]), color="#666666", lw=0.9,
                      ls="-.")
        ax[0].set_ylabel(r"$\Sigma \hat{x} / \Sigma q_{\mathrm{truth}}$")
        ax[0].set_title("the flow leaves the created charge immediately\n"
                        "dotted: the created charge; dash-dot: the records",
                        fontsize=7.5)
        ax[1].set_ylabel(r"$\Sigma |\hat{x} - x_{\mathrm{truth}}| / "
                         r"\Sigma q_{\mathrm{truth}}$")
        ax[1].set_title("distance from the created charge", fontsize=7.5)
        ax[2].axhline(0.0, color="#000000", lw=0.9, ls=":")
        ax[2].set_ylabel("charge on the +1 pixels [ke]")
        ax[2].set_title("where the charge goes: the +1 pixel columns",
                        fontsize=7.5)
        for a in ax:
            a.set_xlabel("iteration")
            _ieee_axes(a)
        ax[0].legend(fontsize=6.5, frameon=False, ncol=2)
        fig.tight_layout()
        self._save(fig, fig_dir / "GF2_flow_from_truth", rec, plt)

        # ---- GF3: the two starts ------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
        for V in variants:
            for arm in arms:
                for start, ls_, mk in (("truth", "-", None),
                                       ("zero", "--", None)):
                    it = traj(V, arm, start, "iter")
                    if it is None:
                        continue
                    lab = (f"{ARM_LABEL[arm]}, {V}, from "
                           + ("the created charge" if start == "truth"
                              else "zero"))
                    ax[0].plot(np.maximum(it, 1),
                               traj(V, arm, start, "l1_distance_over_truth"),
                               ls_, color=ARM_COLOR[arm], lw=1.2, label=lab)
                    ax[1].plot(np.maximum(it, 1),
                               traj(V, arm, start, "sum_over_truth"),
                               ls_, color=ARM_COLOR[arm], lw=1.2)
                    if start == "zero":
                        r = run_of(flow, V, arm, start, "flow")
                        if r and "closest_approach" in r:
                            ca = r["closest_approach"]
                            ax[0].plot([max(ca["iteration"], 1)],
                                       [ca["l1_distance_over_truth"]], "o",
                                       ms=4.5, mfc="none",
                                       color=ARM_COLOR[arm])
        ax[0].set_xscale("log")
        ax[0].set_ylabel(r"$\Sigma |\hat{x} - x_{\mathrm{truth}}| / "
                         r"\Sigma q_{\mathrm{truth}}$")
        ax[0].set_title("open circle: the closest the zero start comes to "
                        "the created charge", fontsize=7.5)
        ax[0].legend(fontsize=5.8, frameon=False, ncol=2)
        ax[1].set_xscale("log")
        ax[1].axhline(1.0, color="#000000", lw=0.9, ls=":")
        ax[1].set_ylabel(r"$\Sigma \hat{x} / \Sigma q_{\mathrm{truth}}$")
        ax[1].set_title("solid: started at the created charge; "
                        "dashed: at zero", fontsize=7.5)
        for a in ax:
            a.set_xlabel("iteration")
            _ieee_axes(a)
        fig.tight_layout()
        self._save(fig, fig_dir / "GF3_two_starts", rec, plt)

        # ---- GF4: the endpoint of every scheme ----------------------------
        rows = []
        for src_rec, tag in ((flow, "flow"), (fista, "fista")):
            if src_rec is None:
                continue
            for r in src_rec["runs"]:
                if r["convention"] != conv:
                    continue
                rows.append(r)
        if rows:
            labels = [f"{r['arm']}, {r['variant']}, "
                      f"{'truth' if r['start'] == 'truth' else '0'}, "
                      f"{r['method']}" for r in rows]
            fields = [("sum_xhat_over_truth", r"$\Sigma\hat{x}/\Sigma q$",
                       1.0),
                      ("l1_distance_over_truth",
                       r"$\Sigma|\hat{x}-x_{\mathrm{truth}}|/\Sigma q$", None),
                      ("E_rel_1.5", r"$E_{\mathrm{rel}}(1.5\,\mu$s$)$", None),
                      (None, "charge on the +1 pixels / $\\Sigma q$", 0.0)]
            fig, ax = plt.subplots(1, 4, figsize=(14.0, 0.34 * len(rows) + 1.9),
                                   sharey=True)
            ypos = np.arange(len(rows))
            for j, (key, xlab, refline) in enumerate(fields):
                if key is None:
                    vals = [r["pixels"]["plus1"]["sum_ke"] / truth_total
                            for r in rows]
                else:
                    vals = [r[key] for r in rows]
                for yv, v, r in zip(ypos, vals, rows):
                    ax[j].barh([yv], [v], color=ARM_COLOR[r["arm"]],
                               alpha=1.0 if r["start"] == "truth" else 0.5,
                               hatch="" if r["method"] == "flow" else "///")
                    ax[j].annotate(_fmt(v, 3), (v, yv), fontsize=5.6,
                                   va="center",
                                   ha="left" if v >= 0 else "right")
                if refline is not None:
                    ax[j].axvline(refline, color="#000000", lw=0.9, ls=":")
                ax[j].set_xlabel(xlab, fontsize=8)
                ax[j].margins(x=0.18)
                _ieee_axes(ax[j])
            ax[0].set_yticks(ypos)
            ax[0].set_yticklabels(labels, fontsize=6)
            ax[0].invert_yaxis()
            fig.suptitle("Endpoint of every scheme against the created "
                         "ionisation charge.  Colour: arm.  Full opacity: "
                         "started at the created charge; half: at zero.  "
                         "Hatched: FISTA; plain: the flow.", fontsize=7.5)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            self._save(fig, fig_dir / "GF4_scheme_distance", rec, plt)

        # ---- GF5: the probe waveform --------------------------------------
        k0 = probe[0]
        if f"truth_fine_{k0}" in npz.files:
            xt = np.asarray(npz[f"truth_fine_{k0}"], dtype=float)
            t_us = (b0 + np.arange(len(xt))) * TICK_US
            m = xt > 0
            lo, hi = np.argmax(m), len(m) - np.argmax(m[::-1])
            sl = slice(max(lo - 60, 0), min(hi + 90, len(m)))
            fig, ax = plt.subplots(1, len(variants),
                                   figsize=(5.2 * len(variants), 3.6),
                                   squeeze=False)
            for j, V in enumerate(variants):
                a = ax[0][j]
                a.plot(t_us[sl], xt[sl], color="#000000", lw=1.5,
                       label="created ionisation charge")
                for arm in arms:
                    for start, ls_ in (("truth", "-"), ("zero", "--")):
                        key = (f"final_fine_{conv}_{V}_{arm}_{start}_flow_"
                               f"{k0}")
                        if key not in npz.files:
                            continue
                        a.plot(t_us[sl],
                               np.asarray(npz[key], dtype=float)[sl], ls_,
                               color=ARM_COLOR[arm], lw=1.1,
                               label=f"{ARM_LABEL[arm]}, from "
                                     + ("the created charge"
                                        if start == "truth" else "zero"))
                a.axhline(0.0, color="#666666", lw=0.8)
                a.set_xlabel(r"release time at the response plane [$\mu$s]")
                a.set_ylabel("charge per fine tick [ke]")
                a.set_title(f"pixel {k0.replace('_', ', ')}, variant {V}",
                            fontsize=8)
                a.legend(fontsize=6, frameon=False)
                _ieee_axes(a)
            fig.tight_layout()
            self._save(fig, fig_dir / "GF5_probe_waveform", rec, plt)

        self.put(store, "zs.gradflow.figs", rec)
        self._emit(store, rec)

    @staticmethod
    def _save(fig, stem: Path, rec: dict, plt) -> None:
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        rec["figures"].append(str(stem) + ".pdf")
        print(f"[ZSGradFlowFigures] wrote {stem}.pdf")
