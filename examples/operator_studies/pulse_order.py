"""How many charge pulses does a pixel actually have?

A time-parameterised unfolding (arrival time + amplitude per pulse)
replaces the per-bin amplitudes by 2K unknowns per pixel, but needs the
order K. Two things decide whether that is practical:

  identifiability : a pixel with m_p measurement rows can determine at
        most K <= m_p/2 pulses. This script counts the rows per pixel
        from the actual row construction.
  sufficiency : what fraction of a pixel's true charge (and of the event
        charge) a K-pulse description would capture. Measured on the
        truth mapped to the fit grid: contiguous runs of occupied bins
        are the "pulses"; the top-K runs are what a K-pulse model can
        hold.

Reported per sample: the distribution of rows per pixel, of true pulses
per pixel, and the charge captured at K = 1, 2, 3, weighted by charge.
"""
from __future__ import annotations

import gc
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
NFS = ("/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/"
       "tests/pgun_farfield")
PART = {"mu": "mu", "pos": "positron"}
GAP = 1        # bins of separation that split two pulses
CUT = 0.5      # ke, a bin counts as occupied above this

from channel_coupling import replay  # noqa: E402
from unfoldlarpix.constrained_solver import build_latch_rows  # noqa: E402
from unfoldlarpix.model.conventions import resolve_burst_tau  # noqa: E402


def pulses(v, cut=CUT, gap=GAP):
    """Charges of the contiguous runs (pulses) in a time profile."""
    occ = v > cut
    out, run, hole = [], 0.0, 0
    for k, o in enumerate(occ):
        if o:
            run += v[k]
            hole = 0
        elif run > 0:
            hole += 1
            if hole > gap:
                out.append(run)
                run, hole = 0.0, 0
    if run > 0:
        out.append(run)
    return sorted(out, reverse=True)


def one(tag, arm="B"):
    cfg = yaml.safe_load(open(f"{AO}/nb1_fraccensor/{arm}/job_{tag}.yaml"))
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
    windows, _ = build_latch_rows(
        ev.hits.location, ev.hits.data, B, boff,
        csa_reset_time=rc.csa_reset_time,
        split_threshold=(float(rc.threshold)
                         if bm["BuildMeasurement"].get("split_trigger", True)
                         else None),
        acq_start=getattr(ev, "acq_start", None), burst_tau=bt)
    nx, ny, nt = op.q_shape
    rows_per_px = {}
    for w in windows:
        if 0 <= w.px < nx and 0 <= w.py < ny and w.t_hi > max(w.t_lo, 0.0):
            rows_per_px[(w.px, w.py)] = rows_per_px.get((w.px, w.py), 0) + 1

    part, ang, nb = (tag.split("_")[0], tag.split("_")[1][1:],
                     tag.split("_")[2][2:])
    f = np.load(f"{NFS}/pgun_{PART[part]}_3gev_ang{ang}_tred_nb{nb}.npz",
                allow_pickle=True)
    el = np.asarray(f["effq_tpc0_batch0_location"])
    eq = np.asarray(f["effq_tpc0_batch0"], float)[:, 3]
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    T = np.zeros(op.q_shape)
    np.add.at(T, (ix[ok], iy[ok], it[ok]), eq[ok])

    tot = T.sum()
    npul, cap = [], {1: 0.0, 2: 0.0, 3: 0.0}
    charge_px, rows_of_px = [], []
    for (px, py) in zip(*np.nonzero(T.sum(axis=2) > 1.0)):
        p = pulses(T[px, py])
        if not p:
            continue
        s = sum(p)
        npul.append(len(p))
        charge_px.append(s)
        rows_of_px.append(rows_per_px.get((px, py), 0))
        for K in cap:
            cap[K] += sum(p[:K])
    npul = np.array(npul)
    charge_px = np.array(charge_px)
    rows_of_px = np.array(rows_of_px)
    out = {"tag": tag, "pixels": int(len(npul)),
           "rows_per_pixel_median": float(np.median(rows_of_px)),
           "rows_per_pixel_max": int(rows_of_px.max()) if len(rows_of_px)
           else 0,
           "K_identifiable_median": float(np.median(rows_of_px) / 2),
           "pulses_median": float(np.median(npul)),
           "pulses_mean": float(npul.mean()),
           "frac_pixels_K1": float((npul <= 1).mean()),
           "frac_pixels_K2": float((npul <= 2).mean()),
           "charge_captured_K1": float(cap[1] / tot),
           "charge_captured_K2": float(cap[2] / tot),
           "charge_captured_K3": float(cap[3] / tot),
           "unknowns_perbin": int((T > CUT).sum()),
           "unknowns_K1": int(2 * len(npul)),
           "rows": int(op.n_data)}
    del op, store
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1", "mu_a75_nb1",
                            "pos_a00_nb1", "pos_a50_nb1", "pos_a75_nb1"]
    res = [one(t) for t in tags]
    print("\n%-12s %6s %6s %7s %7s %7s %8s %8s %8s %9s %9s" %
          ("sample", "rows", "px", "rows/px", "K_id", "pulses",
           "%px K=1", "capK1", "capK2", "unk/bin", "unk K=1"))
    for a in res:
        print("%-12s %6d %6d %7.1f %7.1f %7.1f %8.2f %8.3f %8.3f %9d %9d" %
              (a["tag"], a["rows"], a["pixels"], a["rows_per_pixel_median"],
               a["K_identifiable_median"], a["pulses_median"],
               a["frac_pixels_K1"], a["charge_captured_K1"],
               a["charge_captured_K2"], a["unknowns_perbin"],
               a["unknowns_K1"]))
    json.dump(res, open(f"{AO}/channel_coupling/pulse_order.json", "w"),
              indent=1)
    print("-> channel_coupling/pulse_order.json", flush=True)
