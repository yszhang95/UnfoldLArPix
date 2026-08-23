"""SUPERSEDED as a production path -- kept as the record of what was run.

Replaced by the framework algorithms in
``src/unfoldlarpix/algs/spectrum_algs.py``:

    OperatorConditioning   matrix-free (Lanczos), cost independent of n
    OperatorSpectrum       dense Gram + eigh, full spectrum and mode geometry

They read the operator from the write-once event store, so the spectrum is
provably of the same A the solver used, and the record carries the resolved job
config.  This script rebuilt the operator from a job YAML outside the solver's
run and wrote a JSON that names neither -- which is how two archived campaigns
ended up on different events under one set of tag names (mu_a00_nb1 at 856 rows
here, 887 in charge_space_modes/), invisible until both were replayed.

Equivalence was established field by field, not asserted: see
``examples/analysis_output/spectrum_algs/replace_archived_studies.py`` and its
report.  That exercise found four defects in the PORT, not here, and three
reporting defects here:

  * ``cond_sqrt`` divides by ``max(lambda_min, 1e-30)`` and so prints ~1e15 on a
    numerically singular system.  The algorithms report null plus
    ``n_eig_at_roundoff``.
  * per-mode geometry (``weak_dirs``, ``weak_modes``) is emitted with nothing to
    say it is defined only up to a rotation inside a degenerate cluster.  The
    algorithms report ``eig_gap_to_neighbour`` and ``eigvec_well_separated`` per
    mode; quote the aggregates.
  * no provenance in the output.

DO NOT delete or edit: the JSONs in analysis_output/ were produced by this code
and the note still cites them for the archived campaigns.  New work goes through
the algorithms.

Original docstring follows.
"""
"""Which measurement channels carry the same information, over what range
in pixel space, and what do the l1/support terms do to that geometry?

Three nested linear systems are compared, all exactly (no sampling):

  free    data fidelity alone: rows of A over the WHOLE fit grid.
          G = A A^T.  This is the geometry of the measurement itself.
  support data fidelity restricted to the ROI: G = A P_S A^T.  P_S is
          the hard support mask (BuildSupport), the only place the
          support enters the solver.
  active  restricted to the voxels the solution actually uses
          (q_hat > 0.01): G = A P_act A^T.  The weighted l1 is LINEAR in
          q on the positive orthant, so it contributes no curvature at
          all -- its entire effect on channel geometry is which voxels
          survive, i.e. P_act.

The censor term is a squared hinge on the running maximum over pixels
that never fired; its Gauss-Newton curvature is zero wherever the
constraint is slack. Its activity is measured at the stored solution and
reported (for muons it is provably inactive, so it drops out here).

For each system: normalised coupling rho_ij = G_ij/sqrt(G_ii G_jj), its
profile against pixel separation (is the coupling short range, or does
it reach the 25x25 response half-width of 12 pixels?), the eigenvalue
spectrum of G (= squared singular values of A), effective rank, and the
pixel-space localisation of the near-null channel combinations.

Outputs: analysis_output/channel_coupling/channel_coupling_<tag>.{json,png,pdf}
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
AO = f"{ROOT}/examples/analysis_output"

from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.fwk.component import ALGORITHMS  # noqa: E402
from unfoldlarpix.fwk.runner import build_job  # noqa: E402
from unfoldlarpix.fwk.store import EventStore  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402
from unfoldlarpix.terms.base import IterCtx  # noqa: E402
from unfoldlarpix.terms.censor import CensorRunningMax  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "mu_a00_nb1"
ARM = sys.argv[2] if len(sys.argv) > 2 else "B"
CAMP = sys.argv[3] if len(sys.argv) > 3 else "nb1_fraccensor"
JOB = f"{AO}/{CAMP}/{ARM}/job_{TAG}.yaml" if CAMP == "nb1_fraccensor" \
    else f"{AO}/{CAMP}/job_{TAG}.yaml"
SOLVED = (f"{AO}/{CAMP}/{ARM}/{TAG}/{TAG}_event_0_0.npz"
          if CAMP == "nb1_fraccensor"
          else f"{AO}/{CAMP}/{TAG}/{TAG}_event_0_0.npz")
OUT = f"{AO}/channel_coupling"
KHALF = 12          # response half-width in pixels (25 x 25 kernel)


def replay(cfg):
    keep = [e for e in cfg["sequence"]
            if list(e)[0] in ("LoadEvent", "FFTWarmStart",
                              "BuildMeasurement", "BuildSupport")]
    services, _ = build_job({"services": cfg["services"], "sequence": keep})
    store = EventStore()
    store.put("job.config", cfg, "runner")
    for entry in keep:
        (name, props), = entry.items()
        alg = ALGORITHMS[name](**(props or {}))
        alg.initialize(services)
        alg.execute(store)
    return store, services


def gram(op, mask: torch.Tensor | None) -> np.ndarray:
    """Exact G = A P A^T, one column at a time (memory O(one q grid))."""
    n = op.n_data
    G = torch.zeros((n, n), dtype=torch.float64, device=op.device)
    e = torch.zeros(n, dtype=op.dtype, device=op.device)
    for r in range(n):
        e.zero_()
        e[r] = 1.0
        v = op.adjoint(e)
        if mask is not None:
            v = v * mask
        G[:, r] = op.forward(v).to(torch.float64)
    return ((G + G.T) * 0.5).cpu().numpy()      # symmetrise round-off


def analyse(G, rpx, rpy, kind):
    # Rows whose restriction sees (almost) no charge have G_ii ~ 0: the
    # solution's active set is blind to them. They cannot be normalised
    # (and in float32 their entries are round-off), so they are counted
    # and excluded from the correlation statistics.
    dg = np.clip(np.diag(G), 0, None)
    live = dg > 1e-8 * np.median(dg[dg > 0]) if np.any(dg > 0) else dg > 1
    n_dead = int((~live).sum())
    Gl = G[np.ix_(live, live)]
    rpx, rpy = rpx[live], rpy[live]
    kind = [k for k, m in zip(kind, live) if m]
    d = np.sqrt(np.clip(np.diag(Gl), 1e-30, None))
    RHO = Gl / d[:, None] / d[None, :]
    np.fill_diagonal(RHO, np.nan)
    dpx = np.abs(rpx[:, None] - rpx[None, :])
    dpy = np.abs(rpy[:, None] - rpy[None, :])
    dpix = np.maximum(dpx, dpy)
    prof = []
    for k in range(0, 31):
        m = (dpix == k) & np.isfinite(RHO)
        if m.sum() < 3:
            continue
        a = np.abs(RHO[m])
        prof.append({"d": k, "n": int(m.sum()), "mean": float(a.mean()),
                     "p90": float(np.percentile(a, 90)),
                     "max": float(a.max()),
                     "frac_gt_0.1": float((a > 0.1).mean()),
                     "frac_gt_0.5": float((a > 0.5).mean())})
    M = np.full((KHALF + 3, KHALF + 3), np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m = (dpx == i) & (dpy == j) & np.isfinite(RHO)
            if m.sum() >= 3:
                M[i, j] = float(np.abs(RHO[m]).mean())
    w, V = np.linalg.eigh(Gl)
    w = np.clip(w[::-1], 0, None)
    V = V[:, ::-1]
    tot = max(w.sum(), 1e-30)
    cum = np.cumsum(w) / tot
    # NB: an earlier version built these keys with int(100*p), so p=0.999
    # collided with p=0.99 (int(99.9) == 99) and the 99.9% count overwrote
    # the 99% one.  Every "rank_99pct" written before this fix is in fact
    # the 99.9% count.
    eff = {f"rank_{p*100:g}pct".replace('.0pct', 'pct'):
           int(np.searchsorted(cum, p) + 1) for p in (0.9, 0.99, 0.999)}
    # pixel-space localisation of the 20 weakest directions
    loc = []
    for i in range(1, 21):
        v2 = V[:, -i] ** 2
        cx = float((v2 * rpx).sum())
        cy = float((v2 * rpy).sum())
        rms = float(np.sqrt((v2 * ((rpx - cx) ** 2 + (rpy - cy) ** 2)).sum()))
        loc.append({"eig": float(w[-i]), "pixel_rms": rms,
                    "participation": float(1.0 / (v2 ** 2).sum())})
    # same for the 5 strongest, for contrast
    loc_top = []
    for i in range(5):
        v2 = V[:, i] ** 2
        cx = float((v2 * rpx).sum())
        cy = float((v2 * rpy).sum())
        loc_top.append({"eig": float(w[i]),
                        "pixel_rms": float(np.sqrt(
                            (v2 * ((rpx - cx) ** 2 + (rpy - cy) ** 2)).sum())),
                        "participation": float(1.0 / (v2 ** 2).sum())})
    out = {"n_rows_live": int(live.sum()), "n_rows_blind": n_dead,
           "eig_top20": [float(x) for x in w[:20]],
           "eig_tail20": [float(x) for x in w[-20:]],
           "cond_sqrt": float(np.sqrt(w[0] / max(w[-1], 1e-30))),
           **eff, "profile": prof, "map_dpx_dpy": M.tolist(),
           "weak_dirs": loc, "strong_dirs": loc_top,
           "mean_abs_rho_same_pixel": float(
               np.abs(RHO[(dpix == 0) & np.isfinite(RHO)]).mean()),
           "mean_abs_rho_beyond_kernel": float(
               np.abs(RHO[(dpix > KHALF) & np.isfinite(RHO)]).mean()),
           }
    # coupling by row-kind pair (pseudo vs remainder vs diff ...)
    ks = sorted(set(kind))
    kk = {}
    for a in ks:
        for b in ks:
            ia = np.array([i for i, x in enumerate(kind) if x == a])
            ib = np.array([i for i, x in enumerate(kind) if x == b])
            if len(ia) and len(ib):
                sub = RHO[np.ix_(ia, ib)]
                sub = sub[np.isfinite(sub)]
                if sub.size:
                    kk[f"{a}|{b}"] = float(np.abs(sub).mean())
    out["kind_pairs"] = kk
    return out, w, RHO, M


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open(JOB))
    store, _ = replay(cfg)
    op = store.get("op")
    rc = store.get("readout_config")
    ev = store.get("event")
    boff = np.asarray(store.get("block_offset"), dtype=float)
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
    assert op.n_data == len(keep), (op.n_data, len(keep))
    rpx = np.array([windows[i].px for i in keep], float)
    rpy = np.array([windows[i].py for i in keep], float)
    kind = [metas[i].kind for i in keep]

    q_hat = np.asarray(np.load(SOLVED, allow_pickle=True)["deconv_q_sharp"],
                       dtype=np.float64)
    supp = store.get("support").astype(bool)
    act = q_hat > 0.01
    print(f"{TAG} arm {ARM}: rows {op.n_data}, block {op.block_shape} "
          f"({np.prod(op.q_shape)} q voxels), support {supp.sum()}, "
          f"active {act.sum()}", flush=True)
    print(f"  row kinds: "
          f"{ {k: kind.count(k) for k in sorted(set(kind))} }", flush=True)

    # censor activity at the solution
    tcfg = [t for e in cfg["sequence"] if "Solve" in e
            for t in (e["Solve"].get("terms") or []) if t["type"] == "censor"]
    cen = {"configured": bool(tcfg)}
    if tcfg:
        t = tcfg[0]
        term = CensorRunningMax.from_hits(
            op, store.get("hits_view"), store.get("block_offset"),
            csa_reset_time=float(rc.csa_reset_time or 0),
            threshold=float(rc.threshold), npad_bins=50,
            beta=float(t["beta"]), margin=float(t["margin"]),
            norm=t.get("norm", "l2"), bin_ticks=B)
        ctx = IterCtx(op.to_tensor(q_hat), op)
        viol, _ = term._peaks(ctx)
        cen.update(value=float(term.value(ctx)),
                   n_active=int((viol > 0).sum()),
                   max_violation=float(viol.max()))
        print(f"  censor at the solution: value {cen['value']:.4g}, "
              f"active pixels {cen['n_active']}, "
              f"max violation {cen['max_violation']:.4g} ke "
              f"(zero curvature where slack)", flush=True)

    masks = {"free": None,
             "support": op.to_tensor(supp.astype(np.float64)),
             "active": op.to_tensor(act.astype(np.float64))}
    res = {"tag": TAG, "arm": ARM, "n_rows": int(op.n_data),
           "block_shape": [int(s) for s in op.block_shape],
           "q_voxels": int(np.prod(op.q_shape)),
           "support_voxels": int(supp.sum()),
           "active_voxels": int(act.sum()),
           "row_kinds": {k: kind.count(k) for k in sorted(set(kind))},
           "censor": cen, "systems": {}}
    keep_plot = {}
    for name, m in masks.items():
        G = gram(op, m)
        a, w, RHO, M = analyse(G, rpx, rpy, kind)
        res["systems"][name] = a
        keep_plot[name] = (w, M, a)
        print(f"  [{name:7s}] rank(99%) {a['rank_99pct']:4d}/{op.n_data}"
              f"  sqrt-cond {a['cond_sqrt']:.3g}"
              f"  <|rho|> same-pixel {a['mean_abs_rho_same_pixel']:.3f}"
              f"  beyond 12 px {a['mean_abs_rho_beyond_kernel']:.4f}",
              flush=True)
    json.dump(res, open(f"{OUT}/channel_coupling_{TAG}.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    COL = {"free": "#2e6fb7", "support": "#c2410c", "active": "#2e8b57"}
    fig = plt.figure(figsize=(16, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    for name, (w, M, a) in keep_plot.items():
        p = a["profile"]
        ax0.semilogy([x["d"] for x in p], [x["mean"] for x in p], "o-",
                     color=COL[name], ms=4, label=f"{name} (mean)")
    ax0.axvline(KHALF, color="k", ls="--", lw=1)
    ax0.text(KHALF + 0.4, ax0.get_ylim()[1] * 0.2,
             "25x25 response\nhalf-width", fontsize=8)
    ax0.set_xlabel("channel separation [pixels, Chebyshev]")
    ax0.set_ylabel(r"mean $|\rho_{ij}|$")
    ax0.set_title(f"{TAG}: coupling range")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3, which="both")

    ax1 = fig.add_subplot(gs[0, 1])
    for name, (w, M, a) in keep_plot.items():
        p = a["profile"]
        ax1.plot([x["d"] for x in p], [100 * x["frac_gt_0.1"] for x in p],
                 "o-", color=COL[name], ms=4, label=f"{name}")
    ax1.axvline(KHALF, color="k", ls="--", lw=1)
    ax1.set_xlabel("channel separation [pixels]")
    ax1.set_ylabel(r"% of pairs with $|\rho| > 0.1$")
    ax1.set_title("strongly coupled fraction")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    for name, (w, M, a) in keep_plot.items():
        s = np.sqrt(np.clip(w, 0, None))
        ax2.semilogy(np.arange(1, len(s) + 1), s / s[0], color=COL[name],
                     label=f"{name}: rank99 {a['rank_99pct']}")
    ax2.set_xlabel("index")
    ax2.set_ylabel(r"$\sigma_k/\sigma_1$")
    ax2.set_title("singular spectrum of A (data fidelity)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    for i, name in enumerate(("free", "support", "active")):
        ax = fig.add_subplot(gs[1, i])
        M = np.array(keep_plot[name][1], dtype=float)
        im = ax.imshow(M, origin="lower", cmap="viridis",
                       norm=LogNorm(vmin=1e-4, vmax=1.0))
        plt.colorbar(im, ax=ax, label=r"mean $|\rho|$")
        ax.set_xlabel(r"$|\Delta$ pixel$_b|$")
        ax.set_ylabel(r"$|\Delta$ pixel$_a|$")
        ax.set_title(f"pixel-space coupling: {name}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/channel_coupling_{TAG}.png", dpi=130)
    fig.savefig(f"{OUT}/channel_coupling_{TAG}.pdf")
    print(f"-> {OUT}/channel_coupling_{TAG}.{{json,png,pdf}}", flush=True)


if __name__ == "__main__":
    main()
