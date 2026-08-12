"""Empirical row-error covariance, against the analytic noise model.

The waveform samples give the exact window integrals, so the row error
n_r = d_r - d_exact_r is known per row. Grouping rows by (pixel,
trigger) reconstructs the trigger sequences and lets every element of
the analytic covariance be checked:

    Var(diff)             = 2 s_u^2
    Corr(diff_j, diff_j+1)= -1/2                  (shared eps, 100% anti)
    Corr(remainder, diff_2) = -1/sqrt(2 * (2+r))  (shared eps_1)
    Cov(pseudo, remainder)= -(s_t^2 + s_u^2)
    Var(pseudo+remainder) = s_u^2 (+ s_r^2)       (the split is
                                                   information-free in
                                                   the sum)
    Var(lumped)           = s_u^2 (+ s_r^2)

Trigger sequences are independent, so any measured correlation ACROSS
sequences (same pixel or not) is a null test.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import yaml

warnings.filterwarnings("ignore")
ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")

from channel_coupling import replay  # noqa: E402
from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402
from unfoldlarpix.model.noise import row_variances  # noqa: E402


def rows_and_errors(tag, jobdir):
    cfg = yaml.safe_load(open(f"{jobdir}/job_{tag}.yaml"))
    wf = cfg["sequence"][0]["LoadEvent"]["input"].replace(".npz", "_wf.npz")
    if not os.path.exists(wf):
        return None
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)
    bm = [e for e in cfg["sequence"] if "BuildMeasurement" in e][0]
    bt = bm["BuildMeasurement"].get("burst_tau")
    bt = None if bt is None else resolve_burst_tau(
        rc, None if bt == "auto" else int(bt))
    windows, metas = build_latch_rows(
        ev.hits.location, ev.hits.data, B, boff,
        csa_reset_time=rc.csa_reset_time,
        split_threshold=(float(rc.threshold)
                         if bm["BuildMeasurement"].get("split_trigger", True)
                         else None),
        acq_start=getattr(ev, "acq_start", None), burst_tau=bt)
    nx, ny, nt = op.block_shape
    keep = [i for i, w in enumerate(windows)
            if 0 <= w.px < nx and 0 <= w.py < ny
            and w.t_hi > max(w.t_lo, 0.0)]
    z = np.load(wf, allow_pickle=True)
    cur = np.asarray(z["current_tpc0_batch0"])
    cur = cur.reshape(-1, cur.shape[-1])
    cl = np.asarray(z["current_tpc0_batch0_location"])
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]
    cs = {k: np.concatenate([[0.0], np.cumsum(cur[i])]) for k, i in idx.items()}
    d = op.d.cpu().numpy().astype(np.float64)
    err, info = [], []
    for r, i in enumerate(keep):
        w = windows[i]
        k = (int(w.px + boff[0]), int(w.py + boff[1]))
        if k not in cs:
            err.append(np.nan)
            info.append((k, np.nan, metas[i].kind, metas[i].post_reset))
            continue
        a = int(np.clip(max(w.t_lo, 0.0) + boff[2], 0, Nt))
        b = int(np.clip(min(w.t_hi + boff[2], Nt), 0, Nt))
        err.append(d[r] - (cs[k][b] - cs[k][a] if b > a else 0.0))
        # sequence id: pixel + the window's own trigger epoch (t_lo of the
        # first row of the burst is the reset/acq edge; use rounded t_lo of
        # the first window of the sequence via the trigger time in hits)
        info.append((k, float(w.t_lo), metas[i].kind, metas[i].post_reset))
    return (np.array(err), info, rc,
            row_variances([metas[i] for i in keep], rc))


def analyse(tag, jobdir=f"{AO}/angscan_tau"):
    got = rows_and_errors(tag, jobdir)
    if got is None:
        print(f"{tag}: no waveform file -- skip", flush=True)
        return None
    err, info, rc, var = got
    s_u = float(rc.uncorr_noise)
    s_t = float(rc.thres_noise)
    s_r = float(rc.reset_noise)
    ok = np.isfinite(err)
    # group into sequences: same pixel, rows ordered by t_lo; a new
    # sequence starts at a 'pseudo' or 'lumped' row
    seq = defaultdict(list)
    order = sorted([i for i in range(len(err)) if ok[i]],
                   key=lambda i: (info[i][0], info[i][1]))
    cur_key = None
    for i in order:
        k, t, kind, pr = info[i]
        if kind in ("pseudo", "lumped"):
            cur_key = (k, t)
        seq[cur_key].append(i)
    print(f"\n{tag}: {ok.sum()} rows, {len(seq)} sequences; "
          f"s_u {s_u}, s_t {s_t}, s_r {s_r} ke", flush=True)

    def stat(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return (len(v), float(v.mean()) if len(v) else np.nan,
                float(v.std()) if len(v) else np.nan)

    by_kind = defaultdict(list)
    for i in order:
        by_kind[info[i][2]].append(err[i])
    print("  %-11s %6s %9s %9s %11s %11s" %
          ("kind", "n", "mean", "rms", "var_meas", "var_model"))
    res = {"tag": tag, "s_u": s_u, "s_t": s_t, "s_r": s_r, "kinds": {}}
    for k, v in sorted(by_kind.items()):
        n, m, s = stat(v)
        vm = float(np.mean([var[i] for i in order if info[i][2] == k]))
        res["kinds"][k] = {"n": n, "mean": m, "rms": s, "var_meas": s * s,
                           "var_model": vm}
        print("  %-11s %6d %9.3f %9.3f %11.3f %11.3f" %
              (k, n, m, s, s * s, vm), flush=True)

    # pairwise correlations inside a sequence
    pairs = defaultdict(list)
    for key, ids in seq.items():
        ids = sorted(ids, key=lambda i: info[i][1])
        kinds = [info[i][2] for i in ids]
        for j in range(len(ids) - 1):
            pairs[(kinds[j], kinds[j + 1])].append((err[ids[j]],
                                                    err[ids[j + 1]]))
        if len(ids) >= 2 and kinds[0] == "pseudo":
            # the split pair, and its sum (which should equal a lumped row)
            pairs[("sum", "pseudo+remainder")].append(
                (err[ids[0]] + err[ids[1]], 0.0))
    print("  %-24s %6s %9s %11s" % ("pair (within sequence)", "n", "corr",
                                    "cov"))
    res["pairs"] = {}
    for (a, b), v in sorted(pairs.items()):
        arr = np.array(v)
        if len(arr) < 8:
            continue
        if a == "sum":
            n, m, s = stat(arr[:, 0])
            res["pairs"]["sum(pseudo,remainder)"] = {
                "n": n, "var": s * s,
                "model_var_postreset": s_u ** 2 + s_r ** 2,
                "model_var_virgin": s_u ** 2}
            print("  %-24s %6d %9s %11.3f  (model %.3f post-reset / "
                  "%.3f virgin)" %
                  ("var[pseudo+remainder]", n, "-", s * s,
                   s_u ** 2 + s_r ** 2, s_u ** 2), flush=True)
            continue
        c = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        cov = float(np.cov(arr[:, 0], arr[:, 1])[0, 1])
        res["pairs"][f"{a}|{b}"] = {"n": len(arr), "corr": c, "cov": cov}
        print("  %-24s %6d %9.3f %11.3f" % (f"{a} -> {b}", len(arr), c, cov),
              flush=True)

    # the threshold dispersion eta never appears in any ADC word: it
    # enters only because the pseudo row's right-hand side is the NOMINAL
    # threshold while the true crossing charge is thr_nom + eta.  It is
    # still measurable, three independent ways, because it is the only
    # term in these combinations:
    #   Var(pseudo | virgin)  = s_t^2 +   s_u^2
    #   Var(remainder)        = s_t^2 + 2 s_u^2      (beta cancels)
    #  -Cov(pseudo,remainder) = s_t^2 +   s_u^2
    est = {}
    virgin = [i for i in order if info[i][2] == "pseudo" and not info[i][3]]
    if len(virgin) >= 20:
        est["from_var_pseudo_virgin"] = float(
            np.var([err[i] for i in virgin]) - s_u ** 2)
    rem = [err[i] for i in order if info[i][2] == "remainder"]
    if len(rem) >= 20:
        est["from_var_remainder"] = float(np.var(rem) - 2 * s_u ** 2)
    pr = res["pairs"].get("pseudo|remainder")
    if pr:
        est["from_cov"] = float(-pr["cov"] - s_u ** 2)
    vals = [v for v in est.values() if v > 0]
    res["s_t_eff"] = {k: (float(np.sqrt(v)) if v > 0 else None)
                      for k, v in est.items()}
    res["s_t_eff"]["combined"] = (float(np.sqrt(np.mean(vals)))
                                  if vals else None)
    res["s_t_nominal"] = s_t
    print("  s_t_eff [ke]: " + "  ".join(
        f"{k.replace('from_', '')}={'n/a' if v is None else f'{v:.3f}'}"
        for k, v in res["s_t_eff"].items())
        + f"   (nominal {s_t}, n_virgin_pseudo={len(virgin)})", flush=True)

    # null test: rows from DIFFERENT sequences on the same pixel
    cross = []
    bypix = defaultdict(list)
    for key, ids in seq.items():
        bypix[key[0]].append(ids)
    for k, lst in bypix.items():
        for i in range(len(lst) - 1):
            cross.append((err[lst[i][0]], err[lst[i + 1][0]]))
    if len(cross) >= 8:
        arr = np.array(cross)
        c = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        res["cross_sequence_corr"] = {"n": len(arr), "corr": c}
        print("  %-24s %6d %9.3f   (null test: sequences are independent)" %
              ("across sequences", len(arr), c), flush=True)
    return res


if __name__ == "__main__":
    tags = sys.argv[1:] or ["pos_a50_nb4", "mu_a50_nb4", "mu_a25_nb4"]
    res = [r for r in (analyse(t) for t in tags) if r]
    json.dump(res, open(f"{AO}/channel_coupling/row_covariance.json", "w"),
              indent=1)
    print("\n-> channel_coupling/row_covariance.json", flush=True)
