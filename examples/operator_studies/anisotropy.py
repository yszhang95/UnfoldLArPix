"""Is the operator's structural error anisotropic in (t, pixel)?

Hypothesis under test (user's): a row whose window is LONG in time
averages the within-bin charge model over many ticks, so its structural
error should be small; a SHORT window sees the pulse shape directly and
should be worse. If true, the data weight can be made adaptive -- rows
get trusted in proportion to their time extent -- and the prior can be
made anisotropic, penalising differently along t and along the pixel
plane.

Per row we know exactly

    e_r = (A q_true)_r - d_exact_r      (operator model error)
    n_r = d_r - d_exact_r               (readout error)
    dt_r = t_hi - t_lo                  (window length, ticks)
    q_r = d_exact_r                     (charge actually in the window)
    xy_r = live neighbours of the row's pixel within +-2 px, same window
                                        (how extended the event is in the
                                         pixel plane at this row's time)

and we report |e_r| and the relative |e_r|/q_r in bins of dt and of xy,
plus the rank correlation of |e_r|/q_r with each. The two directions are
compared at fixed angle (rows of one event span a range of dt) and
across angles (theta=0 is long-t/short-xy per pixel, theta=75 is the
opposite).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from collections import defaultdict

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
from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402

JOBDIRS = [f"{AO}/nb1_fraccensor/B", f"{AO}/angscan_tau"]


def find_cfg(tag):
    for jd in JOBDIRS:
        p = f"{jd}/job_{tag}.yaml"
        if os.path.exists(p):
            cfg = yaml.safe_load(open(p))
            wf = cfg["sequence"][0]["LoadEvent"]["input"].replace(
                ".npz", "_wf.npz")
            if os.path.exists(wf):
                return cfg, wf
    return None, None


def rowdata(tag):
    cfg, wf = find_cfg(tag)
    if cfg is None:
        return None
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), float)
    B = int(rc.adc_hold_delay)

    src = cfg["sequence"][0]["LoadEvent"]["input"]
    f = np.load(src, allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    nx, ny, nt = op.q_shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    T = np.zeros(op.q_shape)
    np.add.at(T, (ix[ok], iy[ok], it[ok]), eq[ok])
    Aqt = op.forward(op.to_tensor(T)).cpu().numpy().astype(np.float64)

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
    bx, by, bnt = op.block_shape
    keep = [i for i, w in enumerate(windows)
            if 0 <= w.px < bx and 0 <= w.py < by
            and w.t_hi > max(w.t_lo, 0.0)]
    z = np.load(wf, allow_pickle=True)
    cur = np.asarray(z["current_tpc0_batch0"])
    cur = cur.reshape(-1, cur.shape[-1])
    cl = np.asarray(z["current_tpc0_batch0_location"])
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]
    cs = {k: np.concatenate([[0.0], np.cumsum(cur[i])])
          for k, i in idx.items()}
    d_ex = np.zeros(op.n_data)
    dt = np.zeros(op.n_data)
    px = np.zeros(op.n_data, int)
    py = np.zeros(op.n_data, int)
    tmid = np.zeros(op.n_data)
    qpart = np.zeros(op.n_data)      # charge in the two partially-covered
    kind = []                        # q-grid bins at the window edges
    t0 = int(boff[2])
    for r, i in enumerate(keep):
        w = windows[i]
        k = (int(w.px + boff[0]), int(w.py + boff[1]))
        a = int(np.clip(max(w.t_lo, 0.0) + boff[2], 0, Nt))
        b = int(np.clip(min(w.t_hi + boff[2], Nt), 0, Nt))
        if k in cs and b > a:
            d_ex[r] = cs[k][b] - cs[k][a]
            # q-grid bin edges are t0 + j*B; the first and last bin the
            # window touches are only partially covered, so the charge in
            # them is the part the within-bin model has to guess
            lo_e = t0 + ((a - t0) // B + 1) * B
            hi_e = t0 + ((b - t0) // B) * B
            if hi_e <= lo_e:                       # window inside one bin
                qpart[r] = d_ex[r]
            else:
                qpart[r] = ((cs[k][min(lo_e, Nt)] - cs[k][a])
                            + (cs[k][b] - cs[k][max(hi_e, 0)]))
        dt[r] = max(w.t_hi, 0.0) - max(w.t_lo, 0.0)
        px[r], py[r] = int(w.px), int(w.py)
        tmid[r] = 0.5 * (max(w.t_lo, 0.0) + w.t_hi)
        kind.append(metas[i].kind)
    d = op.d.cpu().numpy().astype(np.float64)
    e = Aqt - d_ex
    n = d - d_ex

    # pixel-plane extent at each row's time: live pixels within +-2 px
    # whose windows overlap this row's midpoint
    xy = np.zeros(op.n_data)
    bytime = defaultdict(list)
    for r in range(op.n_data):
        bytime[int(tmid[r] // 8)].append(r)
    for r in range(op.n_data):
        c = 0
        for kb in (int(tmid[r] // 8) - 1, int(tmid[r] // 8),
                   int(tmid[r] // 8) + 1):
            for s in bytime.get(kb, ()):
                if s != r and abs(px[s] - px[r]) <= 2 \
                        and abs(py[s] - py[r]) <= 2 \
                        and abs(tmid[s] - tmid[r]) <= 8:
                    c += 1
        xy[r] = c
    del op
    gc.collect()
    torch.cuda.empty_cache()
    return dict(e=e, n=n, q=d_ex, dt=dt, xy=xy, qpart=qpart,
                kind=np.array(kind), Aqt=Aqt, px=px, py=py, tlo=tmid - dt / 2)


def seq_structure(R):
    """Is the OPERATOR error correlated inside a trigger sequence?

    Sigma_op is only diagonal if it is not. Consecutive windows on one
    pixel see the same mis-modelled pulse, so the expectation is that it
    IS correlated -- which would forbid the diagonal shortcut.
    """
    px, py, tlo, kind = R["px"], R["py"], R["tlo"], R["kind"]
    e, n, q = R["e"], R["n"], R["q"]
    order = sorted(range(len(e)), key=lambda i: (px[i], py[i], tlo[i]))
    seqs, cur = [], None
    for i in order:
        if kind[i] in ("pseudo", "lumped") or cur is None:
            cur = []
            seqs.append(cur)
        cur.append(i)
    adj_e, adj_n, pix_e = [], [], []
    for s in seqs:
        for j in range(len(s) - 1):
            adj_e.append((e[s[j]], e[s[j + 1]]))
            adj_n.append((n[s[j]], n[s[j + 1]]))
    bypix = defaultdict(list)
    for s in seqs:
        bypix[(px[s[0]], py[s[0]])].append(s)
    for k, lst in bypix.items():
        for j in range(len(lst) - 1):
            pix_e.append((e[lst[j][0]], e[lst[j + 1][0]]))
    out = {"n_seq": len(seqs)}
    for lab, arr in (("adj_e", adj_e), ("adj_n", adj_n), ("xseq_e", pix_e)):
        a = np.array(arr) if len(arr) >= 8 else None
        out[lab] = (float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if a is not None
                    else np.nan)
        out[lab + "_n"] = len(arr)
    # sign coherence: does the operator error keep the same sign along a
    # sequence?  A shared bias would show up as a mean well above 0.5.
    same = [np.mean([np.sign(e[s[j]]) == np.sign(e[s[j + 1]])
                     for j in range(len(s) - 1)])
            for s in seqs if len(s) > 1]
    out["sign_coherence"] = float(np.mean(same)) if same else np.nan
    tot = [sum(e[i] for i in s) for s in seqs]
    ind = [np.sqrt(sum(e[i] ** 2 for i in s)) for s in seqs]
    out["seq_sum_vs_quad"] = float(np.sqrt(np.mean(np.square(tot)))
                                   / max(np.sqrt(np.mean(np.square(ind))),
                                         1e-9))
    print("  sequence structure: %d sequences; corr(e_j,e_j+1)=%+.3f "
          "(n=%d), corr(n_j,n_j+1)=%+.3f, corr across seq=%+.3f" %
          (out["n_seq"], out["adj_e"], out["adj_e_n"], out["adj_n"],
           out["xseq_e"]), flush=True)
    print("    sign coherence %.3f (0.5 = random);  "
          "rms(sum_seq e)/rms(quadrature) = %.3f  (1 = independent, "
          "sqrt(len) = fully coherent)" %
          (out["sign_coherence"], out["seq_sum_vs_quad"]), flush=True)
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def report(tag, R, qmin=1.0):
    e, q, dt, xy, n = R["e"], R["q"], R["dt"], R["xy"], R["n"]
    m = q > qmin
    rel = np.abs(e[m]) / q[m]
    print(f"\n{tag}: {m.sum()}/{len(q)} rows with q > {qmin} ke; "
          f"||e|| {np.linalg.norm(e):.1f}, ||n|| {np.linalg.norm(n):.1f}, "
          f"||e||/||n|| {np.linalg.norm(e)/max(np.linalg.norm(n),1e-9):.2f}",
          flush=True)
    out = {"tag": tag, "n_rows": int(m.sum()),
           "norm_e": float(np.linalg.norm(e)),
           "norm_n": float(np.linalg.norm(n)),
           "rho_rel_dt": spearman(dt[m], rel),
           "rho_rel_xy": spearman(xy[m], rel),
           "rho_abs_dt": spearman(dt[m], np.abs(e[m])),
           "bins_dt": [], "bins_xy": []}
    for lab, v, key in (("dt [ticks]", dt[m], "bins_dt"),
                        ("xy neighbours", xy[m], "bins_xy")):
        qs = np.unique(np.percentile(v, [0, 20, 40, 60, 80, 100]))
        if len(qs) < 3:
            continue
        print("  %-14s %6s %9s %10s %11s %11s" %
              (lab, "n", "median", "<q> ke", "rms e ke", "<|e|/q>"))
        for i in range(len(qs) - 1):
            sel = (v >= qs[i]) & (v <= qs[i + 1] if i == len(qs) - 2
                                  else v < qs[i + 1])
            if sel.sum() < 5:
                continue
            row = {"lo": float(qs[i]), "hi": float(qs[i + 1]),
                   "n": int(sel.sum()), "median": float(np.median(v[sel])),
                   "q_mean": float(q[m][sel].mean()),
                   "e_rms": float(np.sqrt((e[m][sel] ** 2).mean())),
                   "e_bias": float(e[m][sel].mean()),
                   "rel": float(np.median(rel[sel]))}
            out[key].append(row)
            print("  %-14s %6d %9.1f %10.2f %11.3f %11.4f" %
                  (f"{qs[i]:.0f}-{qs[i+1]:.0f}", row["n"], row["median"],
                   row["q_mean"], row["e_rms"], row["rel"]), flush=True)
    print(f"  spearman(|e|/q, dt) = {out['rho_rel_dt']:+.3f}   "
          f"spearman(|e|/q, xy) = {out['rho_rel_xy']:+.3f}", flush=True)

    # confound control: long windows are mostly pseudo/lumped rows, so
    # check the dt trend WITHIN a single row kind, and report per kind
    kd = R["kind"][m]
    out["by_kind"] = {}
    print("  %-11s %6s %10s %11s %11s %9s" %
          ("kind", "n", "<dt>", "rms e ke", "<|e|/q>", "rho(dt)"))
    for k in ("diff", "lumped", "pseudo", "remainder"):
        s = kd == k
        if s.sum() < 8:
            continue
        r = {"n": int(s.sum()), "dt_mean": float(dt[m][s].mean()),
             "e_rms": float(np.sqrt((e[m][s] ** 2).mean())),
             "rel": float(np.median(rel[s])),
             "rho_dt": spearman(dt[m][s], rel[s]),
             "rho_xy": spearman(xy[m][s], rel[s])}
        out["by_kind"][k] = r
        print("  %-11s %6d %10.1f %11.3f %11.4f %9s" %
              (k, r["n"], r["dt_mean"], r["e_rms"], r["rel"],
               "n/a" if not np.isfinite(r["rho_dt"])
               else f"{r['rho_dt']:+.3f}"), flush=True)

    # partial correlation of |e|/q with dt, controlling for xy (and v.v.)
    def partial(a, b, c):
        rab, rac, rbc = (spearman(a, b), spearman(a, c), spearman(b, c))
        den = np.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 1e-12))
        return float((rab - rac * rbc) / den)
    out["partial_dt"] = partial(dt[m], rel, xy[m])
    out["partial_xy"] = partial(xy[m], rel, dt[m])
    out["rho_dt_xy"] = spearman(dt[m], xy[m])
    print(f"  partial rho(rel,dt | xy) = {out['partial_dt']:+.3f}   "
          f"partial rho(rel,xy | dt) = {out['partial_xy']:+.3f}   "
          f"rho(dt,xy) = {out['rho_dt_xy']:+.3f}", flush=True)

    # the mechanistic predictor: charge in the partially covered q-bins.
    # dt only works because long windows have a small partial fraction;
    # if kappa = ||e|| / ||q_part|| is the same across topologies, the
    # weight built from q_part carries over to multi-prong events, and
    # the one built from dt does not.
    qp = R["qpart"][m]
    frac = qp / q[m]
    kap = float(np.sqrt((e[m] ** 2).mean()) / max(np.sqrt((qp ** 2).mean()),
                                                  1e-9))
    out["kappa"] = kap
    out["rho_rel_frac"] = spearman(frac, rel)
    out["rho_abse_qpart"] = spearman(qp, np.abs(e[m]))
    out["partial_frac"] = partial(frac, rel, dt[m])
    out["qpart_frac_mean"] = float(np.median(frac))
    print("  partial-bin charge:  kappa = rms|e|/rms(q_part) = "
          f"{kap:.3f}   median q_part/q = {out['qpart_frac_mean']:.3f}")
    print(f"  spearman(|e|/q, q_part/q) = {out['rho_rel_frac']:+.3f}  "
          f"(partial | dt: {out['partial_frac']:+.3f});  "
          f"spearman(|e|, q_part) = {out['rho_abse_qpart']:+.3f}",
          flush=True)
    qs = np.unique(np.percentile(frac, [0, 25, 50, 75, 100]))
    out["bins_frac"] = []
    if len(qs) >= 3:
        print("  %-14s %6s %10s %11s %11s %10s" %
              ("q_part/q", "n", "median", "rms e ke", "<|e|/q>", "kappa"))
        for i in range(len(qs) - 1):
            sel = ((frac >= qs[i]) & (frac <= qs[i + 1] if i == len(qs) - 2
                                      else frac < qs[i + 1]))
            if sel.sum() < 5:
                continue
            kk = float(np.sqrt((e[m][sel] ** 2).mean())
                       / max(np.sqrt((qp[sel] ** 2).mean()), 1e-9))
            out["bins_frac"].append(
                {"lo": float(qs[i]), "hi": float(qs[i + 1]),
                 "n": int(sel.sum()), "e_rms": float(
                     np.sqrt((e[m][sel] ** 2).mean())),
                 "rel": float(np.median(rel[sel])), "kappa": kk})
            print("  %-14s %6d %10.3f %11.3f %11.4f %10.3f" %
                  (f"{qs[i]:.2f}-{qs[i+1]:.2f}", sel.sum(),
                   float(np.median(frac[sel])),
                   out["bins_frac"][-1]["e_rms"],
                   out["bins_frac"][-1]["rel"], kk), flush=True)
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a25_nb4",
                            "mu_a50_nb4", "mu_a75_nb4", "pos_a50_nb4"]
    res = []
    for t in tags:
        R = rowdata(t)
        if R is None:
            print(f"{t}: no waveform sample -- skip", flush=True)
            continue
        r = report(t, R)
        r["sequence"] = seq_structure(R)
        res.append(r)
    json.dump(res, open(f"{AO}/channel_coupling/anisotropy.json", "w"),
              indent=1)
    print("\n-> channel_coupling/anisotropy.json", flush=True)
