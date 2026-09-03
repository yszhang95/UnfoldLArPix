"""tred fixed-interval records -> solver-schema pseudo hits.

``tred.readout.fixed_interval_readout`` samples the per-pixel CSA accumulator
on a threshold-free, trigger-free grid of stride ``adc_hold_delay`` (30 fine
ticks = 1.5 us = exactly one fit bin).  That output is ALREADY the solver's
hits schema for a single sequence with ``nburst = Nsample``
(:mod:`unfoldlarpix.model.conventions`, :mod:`unfoldlarpix.io.hits`)::

    hits.location = [px, py, trigger, trigger + B, rearm]
    hits.data     = [x, y, z, q1..qN]        cumulative, differenced by
                                             build_latch_rows into windows

so NO operator change is needed.  This module only

1. puts every pixel on ONE global sample grid.  ``concatenate_waveforms``
   gives each pixel its own block start (a multiple of the 120-tick readout
   chunk, hence of B); before that start the accumulator is exactly 0 and
   after the last sample it is constant, so zero-/constant-padding is exact
   -- no interpolation;
2. selects pixels by the largest EXCURSION of the accumulator, never by its
   final value: a neighbour pixel's induced signal is bipolar and integrates
   back to ~0, yet it is exactly the row a ZS readout triggers on;
3. re-references the cumulative column to the sample just BEFORE the kept
   window and declares ``trigger = T_s``, ``latch_k = T_s + k*B``, all on
   global multiples of B -- which is what makes ``windows_to_sampling``
   return exactly one column per row at weight 1.

Pair the output with ``BuildMeasurement: {acq_start: <pseudo_trigger_tick>,
split_trigger: false}`` (there is no trigger crossing to split) and
``FFTWarmStart: {align_origin: true}`` (so the block origin is also a
multiple of B).

CLI::

    python -m unfoldlarpix.io.pseudo_hits IN.npz OUT.npz [--qcut ke]
        [--padbins n] [--fcut f] [--tpc i] [--batch i]
"""
from __future__ import annotations

import argparse

import numpy as np


def _batches(f, tpc: int):
    """Batch indices present for one TPC, in order."""
    pre = f"hits_tpc{tpc}_batch"
    return sorted(int(k[len(pre):]) for k in f.files
                  if k.startswith(pre) and not k.endswith("_location"))


def make_pseudo_hits(inp: str, out: str, tpc: int = 0, batch: int = 0,
                     qcut: float = 5.0, padbins: int = 6,
                     fcut: float = 1e-4, verbose: bool = True,
                     all_batches: bool = False) -> dict:
    """Convert one tred fixed-interval npz into a solver-schema npz.

    Parameters
    ----------
    qcut : float
        Keep pixels whose accumulator EXCURSION exceeds this [ke].
    padbins : int
        Bins of quiet record kept on each side of the signal span.  One
        sample before the window is always consumed as the trigger
        reference.  Note the block always extends one kernel length before
        the first window regardless, so padding here does not remove the
        leading zero-gain region.
    fcut : float
        Per-sample fraction of the event total that opens/closes the span.
    """
    f = np.load(inp, allow_pickle=True)
    if all_batches:
        return _make_all(f, inp, out, tpc, qcut, padbins, fcut, verbose)
    key = f"hits_tpc{tpc}_batch{batch}"
    H = np.asarray(f[key], dtype=float)
    L = np.asarray(f[f"{key}_location"])
    B = int(f["adc_hold_delay"])
    off = int(f["offset_to_align"])
    C = H[:, 3:]
    coords = H[:, :3]

    # -- one global sample grid ---------------------------------------------
    t0s = np.asarray(L[:, 2], dtype=np.int64)
    if np.any((t0s + off) % B):
        raise ValueError(f"a pixel sample grid is not a multiple of B={B}")
    Tmin = int((t0s + off).min())
    ns_src = C.shape[1]
    Tmax = int((t0s + off).max()) + (ns_src - 1) * B
    ngrid = (Tmax - Tmin) // B + 1
    shift = ((t0s + off) - Tmin) // B

    def regrid(A):
        G = np.zeros((A.shape[0], ngrid), dtype=float)
        for i in range(A.shape[0]):
            k = int(shift[i])
            G[i, k:k + ns_src] = A[i]
            if k + ns_src < ngrid:
                G[i, k + ns_src:] = A[i, -1]
        return G

    C = regrid(C)
    t0 = Tmin

    # -- pixel and time selection -------------------------------------------
    keep_px = np.abs(C).max(axis=1) > qcut
    if not keep_px.any():
        raise ValueError(f"no pixel exceeds qcut={qcut} ke")
    inc = np.diff(C[keep_px], axis=1, prepend=0.0).sum(axis=0)
    live = np.flatnonzero(np.abs(inc) > fcut * np.abs(inc).sum())
    if live.size == 0:
        raise ValueError("no live samples")
    k0 = max(int(live[0]) - padbins, 1)          # >=1: one sample as reference
    k1 = min(int(live[-1]) + padbins, C.shape[1] - 1)
    s = k0 - 1
    cum = C[np.ix_(keep_px, np.arange(k0, k1 + 1))] - C[keep_px, s][:, None]
    trigger = t0 + s * B
    if trigger % B:
        raise ValueError(f"trigger {trigger} is not on the global B={B} grid")
    nb = cum.shape[1]

    loc = np.zeros((int(keep_px.sum()), 5), dtype=np.int64)
    loc[:, :2] = L[keep_px, :2]
    loc[:, 2] = trigger
    loc[:, 3] = trigger + B            # HitsView contract: col3 == trigger + B
    loc[:, 4] = trigger + nb * B + 1   # re-arm after the last latch (unused)
    dat = np.concatenate([coords[keep_px], cum], axis=1)

    payload = {}
    for k in f.files:
        if k.startswith("hits_tpc") or k.startswith("truehits_tpc"):
            continue
        try:
            payload[k] = f[k]
        except Exception:
            pass                        # e.g. a pickled event_list
    payload[key] = dat
    payload[f"{key}_location"] = loc
    tk = f"truehits_tpc{tpc}_batch{batch}"
    if tk in f.files:
        GT = regrid(np.asarray(f[tk], dtype=float)[:, 3:])
        payload[tk] = np.concatenate(
            [coords[keep_px],
             GT[np.ix_(keep_px, np.arange(k0, k1 + 1))]
             - GT[keep_px, s][:, None]], axis=1)
    payload["nburst"] = np.array(nb)
    payload["readout_model"] = np.array("fixed_interval")
    payload["pseudo_trigger_tick"] = np.array(trigger)
    payload["pseudo_sample_range"] = np.array([k0, k1])
    payload["pseudo_qcut_ke"] = np.array(float(qcut))
    np.savez(out, **payload)

    info = {"pixels_in": int(H.shape[0]), "pixels_kept": int(keep_px.sum()),
            "samples_in": int(ns_src), "samples_kept": nb,
            "trigger_tick": int(trigger), "rows": int(keep_px.sum()) * nb,
            "charge_in_window_ke": float(cum[:, -1].sum()), "out": out}
    if verbose:
        print(f"[pseudo_hits] {inp.split('/')[-1]}  B={B} t0={t0} off={off}")
        print(f"  pixels {info['pixels_in']} -> {info['pixels_kept']} "
              f"(qcut {qcut} ke, on peak excursion)")
        print(f"  samples {ns_src} -> {nb} (k {k0}..{k1}), trigger tick "
              f"{trigger} (mod B = {trigger % B})")
        print(f"  rows {info['rows']}, charge in window "
              f"{info['charge_in_window_ke']:.2f} ke  -> {out}")
    return info


def _convert_one(f, tpc, batch, qcut, padbins, fcut):
    """One batch -> (location, data, truehits or None, trigger, nburst)."""
    key = f"hits_tpc{tpc}_batch{batch}"
    H = np.asarray(f[key], dtype=float)
    L = np.asarray(f[f"{key}_location"])
    B = int(f["adc_hold_delay"])
    off = int(f["offset_to_align"])
    C = H[:, 3:]
    coords = H[:, :3]
    t0s = np.asarray(L[:, 2], dtype=np.int64)
    if np.any((t0s + off) % B):
        raise ValueError(f"batch {batch}: sample grid not a multiple of B={B}")
    Tmin = int((t0s + off).min())
    ns_src = C.shape[1]
    ngrid = ((int((t0s + off).max()) + (ns_src - 1) * B) - Tmin) // B + 1
    shift = ((t0s + off) - Tmin) // B

    def regrid(A):
        G = np.zeros((A.shape[0], ngrid), dtype=float)
        for i in range(A.shape[0]):
            k = int(shift[i])
            G[i, k:k + ns_src] = A[i]
            if k + ns_src < ngrid:
                G[i, k + ns_src:] = A[i, -1]
        return G

    C = regrid(C)
    keep = np.abs(C).max(axis=1) > qcut
    if not keep.any():
        raise ValueError(f"batch {batch}: no pixel exceeds qcut={qcut}")
    inc = np.diff(C[keep], axis=1, prepend=0.0).sum(axis=0)
    live = np.flatnonzero(np.abs(inc) > fcut * np.abs(inc).sum())
    if live.size == 0:
        raise ValueError(f"batch {batch}: no live samples")
    k0 = max(int(live[0]) - padbins, 1)
    k1 = min(int(live[-1]) + padbins, C.shape[1] - 1)
    s_ = k0 - 1
    cum = C[np.ix_(keep, np.arange(k0, k1 + 1))] - C[keep, s_][:, None]
    trigger = Tmin + s_ * B
    nb = cum.shape[1]
    loc = np.zeros((int(keep.sum()), 5), dtype=np.int64)
    loc[:, :2] = L[keep, :2]
    loc[:, 2] = trigger
    loc[:, 3] = trigger + B
    loc[:, 4] = trigger + nb * B + 1
    dat = np.concatenate([coords[keep], cum], axis=1)
    tk = f"truehits_tpc{tpc}_batch{batch}"
    th = None
    if tk in f.files:
        GT = regrid(np.asarray(f[tk], dtype=float)[:, 3:])
        th = np.concatenate([coords[keep],
                             GT[np.ix_(keep, np.arange(k0, k1 + 1))]
                             - GT[keep, s_][:, None]], axis=1)
    return loc, dat, th, int(trigger), nb


def _make_all(f, inp, out, tpc, qcut, padbins, fcut, verbose):
    """Every batch of one TPC into one npz.

    Each event keeps its OWN trigger, written into ``global_tref`` so the
    loader's canonical ``acq_start`` resolves to it -- that is what lets one
    job iterate 100 events with ``acq_start: event`` instead of needing a
    per-event YAML.
    """
    tspace = float(f["time_spacing"])
    payload = {}
    for k in f.files:
        if k.startswith(("hits_tpc", "truehits_tpc", "global_tref_tpc")):
            continue
        try:
            payload[k] = f[k]
        except Exception:
            pass
    nbs, rows, kept = [], 0, 0
    for b in _batches(f, tpc):
        loc, dat, th, trig, nb = _convert_one(f, tpc, b, qcut, padbins, fcut)
        payload[f"hits_tpc{tpc}_batch{b}"] = dat
        payload[f"hits_tpc{tpc}_batch{b}_location"] = loc
        if th is not None:
            payload[f"truehits_tpc{tpc}_batch{b}"] = th
        # tref[-1] is in us; the loader divides by time_spacing, so this makes
        # acq_start come out exactly at this event's trigger tick
        payload[f"global_tref_tpc{tpc}_batch{b}"] = np.array(
            [0.0, trig * tspace], dtype=np.float32)
        nbs.append(nb); rows += len(loc) * nb; kept += len(loc)
    payload["nburst"] = np.array(int(np.max(nbs)))
    payload["readout_model"] = np.array("fixed_interval")
    payload["pseudo_qcut_ke"] = np.array(float(qcut))
    payload["pseudo_n_batches"] = np.array(len(nbs))
    np.savez(out, **payload)
    if verbose:
        print(f"[pseudo_hits] {inp.split('/')[-1]}: {len(nbs)} batches, "
              f"{kept} pixels total, samples {min(nbs)}-{max(nbs)}, "
              f"{rows} rows -> {out}")
        print("  per-event trigger is in global_tref; drive with "
              "`acq_start: event`")
    return {"n_batches": len(nbs), "pixels": kept, "rows": rows, "out": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--tpc", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--qcut", type=float, default=5.0)
    ap.add_argument("--padbins", type=int, default=6)
    ap.add_argument("--fcut", type=float, default=1e-4)
    ap.add_argument("--all-batches", action="store_true",
                    help="convert every batch of the TPC into one npz, each "
                         "event keeping its own trigger via global_tref")
    a = ap.parse_args(argv)
    make_pseudo_hits(a.inp, a.out, tpc=a.tpc, batch=a.batch, qcut=a.qcut,
                     padbins=a.padbins, fcut=a.fcut,
                     all_batches=a.all_batches)


if __name__ == "__main__":
    main()
