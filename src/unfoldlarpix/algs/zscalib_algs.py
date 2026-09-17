"""Is the zero-suppressed charge normalisation calibratable?

Reads the per-event records of the calibration ladder (:mod:`zsgradflow_algs`,
one JSON per depth, lifetime and unknown basis) and asks one question: after
the solver has run, can the remaining depth dependence of the reconstructed
charge be removed by a factor that does not depend on the event?

Definitions
-----------
``r(d, tau)``
    ``sum x_hat / sum q_truth`` at drift depth ``d`` and electron lifetime
    ``tau``, for one configuration (unknown basis, start).  A configuration
    whose ``r`` is flat in ``d`` has no depth dependence left to calibrate;
    one whose ``r`` is constant but not 1 is calibratable by a single number.

``t_drift(d) = d / v``
    with ``v = 0.159645 cm/us``, and ``phi(d) = (t_drift / dt) mod B`` the
    arrival phase in fine ticks of the record window, ``B = 30``,
    ``dt = 0.05 us``.  Phase and depth are confounded in this sample -- every
    depth carries exactly one phase -- so a phase model and a depth model
    cannot be separated here; the phase model is fitted because the campaign
    found the non-linear charge ratios grouped by ``phi``, and it is reported
    as an association.

Calibration models, each fitted to the nine depths of one lifetime (or to
both lifetimes jointly where stated) by unweighted least squares:

``constant``   ``r = g``.  One number per configuration: the global factor.
``phase``      ``r = g [1 + a cos(2 pi phi / B) + b sin(2 pi phi / B)]``.
               Three numbers, the "oscillating global factor": still no
               dependence on the event's charge or position, only on the
               arrival phase, which is computable from the drift time.
``linear``     ``r = g + s * t_drift``.  Two numbers; a control, since a
               factor linear in drift time is what an unmodelled attenuation
               would look like.

For each model the figure of merit is the residual scatter of ``r / model``
about 1: ``rms`` over the nine depths and the peak-to-peak spread.  A model
that leaves a scatter smaller than the depth trend it removed is a
calibration; one that does not is a fit with more parameters.

``lambda``
    the fitted decay rate from ``ln E(d) = a - lambda t_drift(d)``,
    unweighted over the nine depths, with the residual-scatter error.  A
    depth-INDEPENDENT factor cancels in this slope and cannot change lambda;
    a phase-dependent one does change it.  Both are reported so the reader
    can see which correction touches the lifetime and which does not.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..fwk.component import algorithm
from .zsbasis_algs import _ieee_axes, _JsonAlg

VEL = 0.159645          # cm/us
DT = 0.05               # us per fine tick
B = 30                  # fine ticks per record window
NAME = re.compile(r"ladder_c(?P<c>\d+)_d(?P<d>[0-9p]+)_(?P<tau>\w+)\.json$")
TAU_LAMBDA = {"1ms": 1.0, "20ms": 0.05}     # simulated 1/tau in 1/ms
CFG_COLOR = {(5, "zero"): "#0072B2", (5, "seed_trigger"): "#56B4E9",
             (30, "zero"): "#D55E00", (30, "seed_trigger"): "#E69F00"}
CFG_LABEL = {(5, "zero"): "250 ns cells, from zero",
             (5, "seed_trigger"): "250 ns cells, trigger seed",
             (30, "zero"): r"1.5 $\mu$s flat cells, from zero",
             (30, "seed_trigger"): r"1.5 $\mu$s flat cells, trigger seed"}
MODEL_MARK = {"constant": "o", "phase": "s", "linear": "^"}


def cfg_style(c, s):
    """Colour by basis, open marker and dashed line for a seeded start."""
    seeded = s != "zero"
    return {"color": CFG_COLOR[(c, s)], "marker": "o", "ms": 5,
            "ls": "--" if seeded else "-",
            "mfc": "none" if seeded else CFG_COLOR[(c, s)],
            "lw": 1.2}


def fit_lambda(t_us: np.ndarray, E: np.ndarray) -> tuple[float, float, float]:
    """``ln E = a - lambda t`` in 1/ms, with the residual-scatter error."""
    t_ms = np.asarray(t_us, dtype=float) / 1000.0
    y = np.log(np.asarray(E, dtype=float))
    n = len(t_ms)
    A = np.vstack([np.ones(n), -t_ms]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    s2 = float((r ** 2).sum() / max(n - 2, 1))
    var_t = float(((t_ms - t_ms.mean()) ** 2).sum())
    return float(coef[1]), float(np.sqrt(s2 / max(var_t, 1e-30))), \
        float(np.sqrt(s2))


def calibration_models(phi: np.ndarray, t_us: np.ndarray,
                       r: np.ndarray) -> dict:
    """Fit the three calibration models and report the residual scatter."""
    out = {}
    n = len(r)
    designs = {
        "constant": np.ones((n, 1)),
        "phase": np.vstack([np.ones(n), np.cos(2 * np.pi * phi / B),
                            np.sin(2 * np.pi * phi / B)]).T,
        "linear": np.vstack([np.ones(n), np.asarray(t_us) / 1000.0]).T,
    }
    for name, X in designs.items():
        if n <= X.shape[1]:
            continue
        coef, *_ = np.linalg.lstsq(X, r, rcond=None)
        model = X @ coef
        ratio = r / model
        out[name] = {
            "n_parameters": int(X.shape[1]),
            "coefficients": [float(v) for v in coef],
            "global_factor": float(coef[0]),
            "residual_rms": float(np.sqrt(((ratio - 1.0) ** 2).mean())),
            "residual_peak_to_peak": float(ratio.max() - ratio.min()),
            "calibrated_ratio": [float(v) for v in ratio],
        }
    out["uncalibrated"] = {
        "n_parameters": 0,
        "residual_rms": float(np.sqrt(((r - 1.0) ** 2).mean())),
        "residual_peak_to_peak": float(r.max() - r.min()),
        "mean": float(r.mean()),
    }
    return out


@algorithm("ZSCalibFit")
class ZSCalibFit(_JsonAlg):
    """Collect the ladder, fit the calibration models, draw the figures."""

    reads = ()
    writes = ("zs.calib",)

    def execute(self, store):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = self.props
        ladder = Path(p["ladder_dir"])
        fig_dir = Path(p.get("fig_dir", ladder.parent / "figs"))
        fig_dir.mkdir(parents=True, exist_ok=True)
        arm = str(p.get("arm", "pos_l1"))
        method = str(p.get("method", "fista"))

        # ---- collect ------------------------------------------------------
        rows = []
        for f in sorted(ladder.glob("ladder_c*_d*_*.json")):
            m = NAME.search(f.name)
            if not m:
                continue
            rec = json.load(open(f))["result"]
            c = int(m["c"])
            tau = m["tau"]
            d = float(m["d"].replace("p", "."))
            t_us = d / VEL
            phi = (t_us / DT) % B
            if abs(rec.get("depth_cm", d) - d) > 1e-6:
                raise ValueError(f"{f.name}: depth_cm {rec.get('depth_cm')} "
                                 f"does not match the file name")
            for r in rec["runs"]:
                if r["arm"] != arm or r["method"] != method:
                    continue
                rows.append({
                    "cell_ticks": c, "tau": tau, "depth_cm": d,
                    "t_drift_us": t_us, "phi": phi, "start": r["start"],
                    "ratio": r["sum_xhat_over_truth"],
                    "E_rel_1.5": r.get("E_rel_1.5"),
                    "E_rel_2.0": r.get("E_rel_2.0"),
                    "l1_fine": r.get("l1_distance_fine_over_truth"),
                    "l1_cells": r.get("l1_distance_over_truth"),
                    "plus1_ke": r["class_sums_ke"]["plus1"],
                    "ionised_ke": r["class_sums_ke"]["ionised"],
                    "rel_residual": r["rel_residual"],
                    "truth_total_ke": rec["truth_total_ke"],
                })
        if not rows:
            raise ValueError(f"no runs matched arm={arm} method={method} "
                             f"under {ladder}")
        cfgs = sorted({(r["cell_ticks"], r["start"]) for r in rows})
        taus = sorted({r["tau"] for r in rows},
                      key=lambda s: TAU_LAMBDA.get(s, 0.0))
        out = {"ladder_dir": str(ladder), "arm": arm, "method": method,
               "n_rows": len(rows), "configurations":
                   [{"cell_ticks": c, "start": s} for c, s in cfgs],
               "calibration": [], "lifetime_fits": [], "figures": []}

        def sel(c, s, tau):
            v = [r for r in rows if r["cell_ticks"] == c and r["start"] == s
                 and r["tau"] == tau]
            return sorted(v, key=lambda r: r["depth_cm"])

        # ---- calibration models -------------------------------------------
        for c, s in cfgs:
            for tau in taus:
                v = sel(c, s, tau)
                if len(v) < 4:
                    continue
                r = np.array([x["ratio"] for x in v])
                phi = np.array([x["phi"] for x in v])
                t = np.array([x["t_drift_us"] for x in v])
                fits = calibration_models(phi, t, r)
                out["calibration"].append({
                    "cell_ticks": c, "start": s, "tau": tau,
                    "depth_cm": [x["depth_cm"] for x in v],
                    "ratio": [float(x) for x in r],
                    "phi": [float(x) for x in phi],
                    "models": fits})
            # one factor for BOTH lifetimes: the calibration a real detector
            # would have to use, since the lifetime is what is being measured
            v = [x for x in rows if x["cell_ticks"] == c and x["start"] == s]
            r = np.array([x["ratio"] for x in v])
            phi = np.array([x["phi"] for x in v])
            t = np.array([x["t_drift_us"] for x in v])
            out["calibration"].append({
                "cell_ticks": c, "start": s, "tau": "both",
                "depth_cm": [x["depth_cm"] for x in v],
                "ratio": [float(x) for x in r],
                "phi": [float(x) for x in phi],
                "models": calibration_models(phi, t, r)})

        # ---- lifetime fits, before and after calibration -------------------
        for c, s in cfgs:
            for tau in taus:
                v = sel(c, s, tau)
                if len(v) < 4:
                    continue
                t = np.array([x["t_drift_us"] for x in v])
                phi = np.array([x["phi"] for x in v])
                q = np.array([x["truth_total_ke"] for x in v])
                E = np.array([x["ratio"] for x in v]) * q
                lam, se, rms = fit_lambda(t, E)
                entry = {"cell_ticks": c, "start": s, "tau": tau,
                         "simulated_lambda_per_ms": TAU_LAMBDA.get(tau),
                         "lambda_per_ms": lam, "error": se,
                         "rms_resid_lnE": rms}
                # the created charge itself, as the control
                lam_t, se_t, rms_t = fit_lambda(t, q)
                entry["control_lambda_per_ms"] = lam_t
                entry["control_error"] = se_t
                # after removing each calibration model
                mods = calibration_models(phi, t,
                                          np.array([x["ratio"] for x in v]))
                for name in ("constant", "phase", "linear"):
                    if name not in mods:
                        continue
                    cal = np.array(mods[name]["calibrated_ratio"]) * q
                    lam_c, se_c, _ = fit_lambda(t, cal)
                    entry[f"lambda_after_{name}"] = lam_c
                    entry[f"lambda_after_{name}_error"] = se_c
                out["lifetime_fits"].append(entry)

        self._figures(plt, fig_dir, rows, cfgs, taus, out, sel)
        self.put(store, "zs.calib", out)
        self._emit(store, out)

    # -- figures -------------------------------------------------------------
    def _figures(self, plt, fig_dir, rows, cfgs, taus, out, sel):
        def save(fig, stem):
            for ext in ("pdf", "png"):
                fig.savefig(f"{fig_dir / stem}.{ext}", dpi=200,
                            bbox_inches="tight")
            plt.close(fig)
            out["figures"].append(str(fig_dir / stem) + ".pdf")
            print(f"[ZSCalibFit] wrote {fig_dir / stem}.pdf")

        def cal_of(c, s, tau, model):
            for e in out["calibration"]:
                if (e["cell_ticks"], e["start"], e["tau"]) == (c, s, tau):
                    return e["models"].get(model)
            return None

        # C1: the charge ratio against depth
        fig, ax = plt.subplots(1, len(taus), figsize=(5.4 * len(taus), 3.9),
                               squeeze=False)
        for j, tau in enumerate(taus):
            a = ax[0][j]
            for c, s in cfgs:
                v = sel(c, s, tau)
                a.plot([x["depth_cm"] for x in v], [x["ratio"] for x in v],
                       label=CFG_LABEL[(c, s)], **cfg_style(c, s))
            a.axhline(1.0, color="#000000", lw=0.9, ls=":")
            a.set_xlabel("drift depth [cm]")
            a.set_ylabel(r"$\Sigma \hat{x} / \Sigma q_{\mathrm{truth}}$")
            a.set_title(rf"$\tau = ${tau.replace('ms', '')} ms", fontsize=9)
            _ieee_axes(a)
        ax[0][0].legend(fontsize=6.5, frameon=False)
        save(fig, "C1_ratio_vs_depth")

        # C2: after each calibration model, per lifetime
        models = ("constant", "phase")
        fig, ax = plt.subplots(len(models), len(taus),
                               figsize=(5.4 * len(taus), 3.4 * len(models)),
                               squeeze=False, sharex=True)
        for i, model in enumerate(models):
            for j, tau in enumerate(taus):
                a = ax[i][j]
                for c, s in cfgs:
                    m = cal_of(c, s, tau, model)
                    v = sel(c, s, tau)
                    if not m:
                        continue
                    a.plot([x["depth_cm"] for x in v],
                           m["calibrated_ratio"], label=CFG_LABEL[(c, s)],
                           **cfg_style(c, s))
                a.axhline(1.0, color="#000000", lw=0.9, ls=":")
                a.set_ylabel(r"$r\,/\,$" + model + " model")
                a.set_title(rf"{model} calibration, $\tau = $"
                            rf"{tau.replace('ms', '')} ms", fontsize=8.5)
                _ieee_axes(a)
        for j in range(len(taus)):
            ax[-1][j].set_xlabel("drift depth [cm]")
        ax[0][0].legend(fontsize=6.5, frameon=False)
        save(fig, "C2_calibrated_ratio")

        # C3: the ratio against the arrival phase
        fig, ax = plt.subplots(1, len(taus), figsize=(5.4 * len(taus), 3.9),
                               squeeze=False)
        for j, tau in enumerate(taus):
            a = ax[0][j]
            for c, s in cfgs:
                v = sel(c, s, tau)
                st = cfg_style(c, s); st["ls"] = "none"
                a.plot([x["phi"] for x in v], [x["ratio"] for x in v],
                       label=CFG_LABEL[(c, s)], **st)
                m = cal_of(c, s, tau, "phase")
                if m:
                    g, aa, bb = m["coefficients"]
                    ph = np.linspace(0, B, 200)
                    a.plot(ph, g * (1 + aa * np.cos(2 * np.pi * ph / B)
                                    + bb * np.sin(2 * np.pi * ph / B)),
                           "-", lw=1.0, color=CFG_COLOR[(c, s)], alpha=0.6)
            a.axhline(1.0, color="#000000", lw=0.9, ls=":")
            a.set_xlabel(r"arrival phase $\varphi = "
                         r"(t_{\mathrm{drift}}/\Delta t)\ \mathrm{mod}\ 30$ [ticks]")
            a.set_ylabel(r"$\Sigma \hat{x} / \Sigma q_{\mathrm{truth}}$")
            a.set_title(rf"$\tau = ${tau.replace('ms', '')} ms", fontsize=9)
            _ieee_axes(a)
        ax[0][0].legend(fontsize=6.5, frameon=False)
        save(fig, "C3_ratio_vs_phase")

        # C4: residual scatter by model -- the figure of merit
        fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
        labels, xs = [], []
        for c, s in cfgs:
            for tau in list(taus) + ["both"]:
                labels.append(f"c={c}, {s}, {tau}")
        xpos = np.arange(len(labels))
        for k, model in enumerate(("uncalibrated", "constant", "phase",
                                   "linear")):
            rms, ptp = [], []
            for c, s in cfgs:
                for tau in list(taus) + ["both"]:
                    m = cal_of(c, s, tau, model)
                    rms.append(m["residual_rms"] if m else np.nan)
                    ptp.append(m["residual_peak_to_peak"] if m else np.nan)
            off = (k - 1.5) * 0.11
            ax[0].plot(xpos + off, rms, MODEL_MARK.get(model, "d"), ms=5,
                       label=model)
            ax[1].plot(xpos + off, ptp, MODEL_MARK.get(model, "d"), ms=5,
                       label=model)
        for a, lab in ((ax[0], "rms of $r/$model $-1$"),
                       (ax[1], "peak-to-peak of $r/$model")):
            a.set_yscale("log")
            a.set_xticks(xpos)
            a.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
            a.set_ylabel(lab)
            _ieee_axes(a)
        ax[0].legend(fontsize=7, frameon=False)
        save(fig, "C4_residual_scatter")

        # C5: what the coarse basis costs -- E_rel and the fine L1 distance
        fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
        for c, s in cfgs:
            for tau, ls_ in zip(taus, ("-", "--")):
                v = sel(c, s, tau)
                ax[0].plot([x["depth_cm"] for x in v],
                           [x["E_rel_1.5"] for x in v], ls_, marker="o", ms=4,
                           color=CFG_COLOR[(c, s)],
                           label=f"{CFG_LABEL[(c, s)]}, {tau}")
                ax[1].plot([x["depth_cm"] for x in v],
                           [x["l1_fine"] for x in v], ls_, marker="o", ms=4,
                           color=CFG_COLOR[(c, s)])
        ax[0].set_ylabel(r"$E_{\mathrm{rel}}(1.5\ \mu$s$)$")
        ax[1].set_ylabel(r"$\Sigma |P\hat{x} - x| / \Sigma q_{\mathrm{truth}}$"
                         " on the fine grid")
        for a in ax:
            a.set_xlabel("drift depth [cm]")
            _ieee_axes(a)
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "C5_resolution_cost")

        # C6: lambda before and after calibration
        fig, ax = plt.subplots(1, len(taus), figsize=(5.4 * len(taus), 3.9),
                               squeeze=False)
        for j, tau in enumerate(taus):
            a = ax[0][j]
            ents = [e for e in out["lifetime_fits"] if e["tau"] == tau]
            xp = np.arange(len(ents))
            for k, key in enumerate(("lambda_per_ms", "lambda_after_constant",
                                     "lambda_after_phase")):
                a.errorbar(xp + (k - 1) * 0.22,
                           [e.get(key, np.nan) for e in ents],
                           yerr=[e.get(key.replace("lambda", "error")
                                       if key == "lambda_per_ms"
                                       else key + "_error", 0.0)
                                 for e in ents],
                           fmt=MODEL_MARK.get(
                               key.replace("lambda_after_", "")
                               .replace("lambda_per_ms", "constant"), "o"),
                           ms=5, capsize=2,
                           label=key.replace("lambda_after_", "after ")
                           .replace("lambda_per_ms", "no calibration"))
            a.axhline(TAU_LAMBDA[tau], color="#000000", lw=0.9, ls="--")
            a.plot(xp, [e["control_lambda_per_ms"] for e in ents], "x",
                   color="#666666", ms=6, label="created charge")
            a.set_xticks(xp)
            a.set_xticklabels([f"c={e['cell_ticks']}, {e['start']}"
                               for e in ents], rotation=30, ha="right",
                              fontsize=6.5)
            a.set_ylabel(r"$\lambda$ [ms$^{-1}$]")
            a.set_title(rf"$\tau = ${tau.replace('ms', '')} ms; dashed is the "
                        "simulated value", fontsize=8)
            _ieee_axes(a)
        ax[0][0].legend(fontsize=6.5, frameon=False)
        save(fig, "C6_lambda")

        # C7: what the seed changes
        fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
        for c in sorted({cc for cc, _ in cfgs}):
            for tau, ls_ in zip(taus, ("-", "--")):
                z = sel(c, "zero", tau)
                s_ = sel(c, "seed_trigger", tau)
                if not z or not s_:
                    continue
                d = [x["depth_cm"] for x in z]
                ax[0].plot(d, [a["ratio"] - b["ratio"]
                               for a, b in zip(s_, z)], ls_, marker="o", ms=4,
                           color=CFG_COLOR[(c, "seed_trigger")],
                           label=f"c={c}, {tau}")
                ax[1].plot(d, [a["E_rel_1.5"] - b["E_rel_1.5"]
                               for a, b in zip(s_, z)], ls_, marker="o", ms=4,
                           color=CFG_COLOR[(c, "seed_trigger")])
        for a, lab in ((ax[0], r"$\Delta(\Sigma\hat{x}/\Sigma q)$"),
                       (ax[1], r"$\Delta E_{\mathrm{rel}}(1.5\ \mu$s$)$")):
            a.axhline(0.0, color="#000000", lw=0.9, ls=":")
            a.set_xlabel("drift depth [cm]")
            a.set_ylabel(lab + ", seed $-$ zero")
            _ieee_axes(a)
        ax[0].legend(fontsize=6.5, frameon=False)
        save(fig, "C7_seed_effect")
