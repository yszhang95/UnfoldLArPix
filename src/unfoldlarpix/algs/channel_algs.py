"""Per-channel operator weight: how much sampling weight A puts on a pixel.

The operator is ``A = S . K``: a bin-integrated response convolution followed
by the window sampling ``S``.  ``S`` is the only part that knows about the
readout cycle, and its entries are the OVERLAP FRACTIONS

    S[r, (px, py, b)] = |[t_lo/B, t_hi/B] ∩ [b, b+1]|          (dimensionless)

so ``sum_b S[r, .]`` is the row's window LENGTH in fit bins, and summing that
over every row of one pixel gives the pixel's TOTAL RECORDED TIME COVERAGE --
the number this module calls the channel weight sum ``w_ch``.  It is a pure
readout-geometry quantity: no truth, no kernel, no solve.  What it is NOT is
the measurement gain: ``c_v = A^T 1`` folds in the kernel and the transverse
response, so a pixel's ``c`` per unit charge is reported alongside as
``cgain_ch`` and the two are kept apart.

The reason to histogram it against drift depth: the number and the length of a
pixel's windows are set by how fast its current crosses threshold, and drift
diffusion flattens that current.  If ``w_ch`` slides with depth then the
operator itself -- not just the data -- is depth dependent, and any quantity
compared across a depth ladder (a lifetime, a dQ/dx) inherits that slide.
"""
from __future__ import annotations

import numpy as np
import torch

from ..fwk.component import algorithm
from .fixedgrid_algs import _JsonRecorder

LATCH_KINDS = ("lumped", "remainder", "diff", "pseudo")


def _quant(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95))}


@algorithm("ChannelWeightProfile")
class ChannelWeightProfile(_JsonRecorder):
    """Per-channel and per-row sampling weight of the operator the solver uses.

    Reads only ``op`` and ``row_meta``, so it runs on any job that reached
    ``BuildMeasurement`` -- no support, no solve, no truth.  Accumulates over
    events and pools at ``finalize``; the JSON at ``out`` carries the pooled
    quantiles and (with ``hist_edges``) the histogram, the NPZ at ``npz``
    carries the pooled per-channel and per-row arrays so several runs can be
    overlaid on COMMON edges afterwards.

    Per channel (a pixel with at least one row in the operator):

    ``w_sum``     ``sum_r sum_b S[r, (chan, b)]`` -- recorded time coverage in
                  fit bins.  THE quantity of the module.
    ``n_rows``    windows the operator has on that pixel.
    ``n_bins``    distinct fit bins the pixel's windows touch (``w_sum`` counts
                  a bin twice if two windows share it; this does not).
    ``span``      last touched bin - first touched bin + 1, so
                  ``n_bins / span`` is the duty cycle of the pixel.
    ``d_sum``     recorded charge on the pixel [ke] -- data, reported for
                  context only, never as part of the weight.
    ``cgain``     ``sum_t c[px, py, t]`` with ``c = A^T 1``: the kernel-folded
                  measurement gain of the pixel's own column stack.  Needs the
                  response; skipped when ``gain: false``.

    Props
    -----
    out : str, optional          JSON path (pooled record).
    npz : str, optional          NPZ path (pooled arrays).
    gain : bool                  compute ``cgain`` (default True; one adjoint).
    hist_edges : list, optional  ``[lo, hi, nbins]`` or explicit edges, applied
                                 to ``w_sum``.  Give the SAME value in every
                                 job of a depth ladder or the histograms are
                                 not comparable.
    """

    reads = ("op", "row_meta")
    writes = ("channel.weight",)

    def initialize(self, services):
        super().initialize(services)
        self.npz_path = self.props.get("npz")
        self._pool: dict[str, list] = {k: [] for k in (
            "w_sum", "n_rows", "n_bins", "span", "d_sum", "cgain", "event",
            "px", "py", "w_row", "row_kind", "row_event")}

    # -- per event ---------------------------------------------------------
    def execute(self, store):
        op = store.get("op")
        rm = store.get("row_meta")
        nx, ny, nt = (int(s) for s in op.block_shape)

        rows = op._rows.cpu().numpy()
        cols = op._cols.cpu().numpy()
        w = op._weights.cpu().numpy().astype(np.float64)
        d = op.d.detach().cpu().numpy().astype(np.float64)

        # channel of an entry, and of a row: col = (px * ny + py) * nt + b,
        # and every entry of a row shares the row's pixel by construction.
        chan_e = cols // nt
        bin_e = cols - chan_e * nt
        nchan = nx * ny
        chan_r = np.zeros(op.n_data, dtype=np.int64)
        chan_r[rows] = chan_e                     # last write wins; all equal
        # guard the "all equal" claim rather than assume it
        if not np.array_equal(chan_r[rows], chan_e):
            raise AssertionError("a row spans more than one channel; the "
                                 "col = (px*ny+py)*nt + b decode is wrong")
        # cross-check against row_meta, which carries px/py independently
        px_rm = np.asarray(rm["px"], dtype=np.int64)
        py_rm = np.asarray(rm["py"], dtype=np.int64)
        if not np.array_equal(chan_r, px_rm * ny + py_rm):
            raise AssertionError("channel decode disagrees with row_meta")

        w_sum = np.bincount(chan_e, weights=w, minlength=nchan)
        n_rows = np.bincount(chan_r, minlength=nchan)
        d_sum = np.bincount(chan_r, weights=d, minlength=nchan)
        w_row = np.bincount(rows, weights=w, minlength=op.n_data)
        # distinct bins per channel, and the touched span
        uniq = np.unique(cols)
        n_bins = np.bincount(uniq // nt, minlength=nchan)
        b_lo = np.full(nchan, nt, dtype=np.int64)
        b_hi = np.full(nchan, -1, dtype=np.int64)
        np.minimum.at(b_lo, chan_e, bin_e)
        np.maximum.at(b_hi, chan_e, bin_e)
        span = np.where(b_hi >= 0, b_hi - b_lo + 1, 0)

        live = n_rows > 0
        idx = np.flatnonzero(live)
        cgain = np.full(nchan, np.nan)
        if bool(self.props.get("gain", True)):
            ones = torch.ones(op.n_data, dtype=op.dtype, device=op.device)
            c = op.adjoint(ones).detach().cpu().numpy().astype(np.float64)
            cg = c.sum(axis=2).reshape(-1)        # q_shape (nx, ny, qt)
            if cg.size != nchan:
                raise AssertionError(
                    f"gain map has {cg.size} pixels against "
                    f"{nchan} operator channels")
            cgain = cg

        ev = len(self._pool["event"])          # events seen so far
        for key, arr in (("w_sum", w_sum[idx]), ("n_rows", n_rows[idx]),
                         ("n_bins", n_bins[idx]), ("span", span[idx]),
                         ("d_sum", d_sum[idx]), ("cgain", cgain[idx]),
                         ("px", idx // ny), ("py", idx % ny)):
            self._pool[key].append(np.asarray(arr))
        self._pool["event"].append(np.full(idx.size, ev, dtype=np.int64))
        self._pool["w_row"].append(w_row)
        self._pool["row_kind"].append(np.asarray(rm["kind"], dtype=object))
        self._pool["row_event"].append(np.full(op.n_data, ev, dtype=np.int64))

        rec = {"event_index": ev, "rows": int(op.n_data),
               "block_shape": [nx, ny, nt],
               "n_channels_live": int(idx.size),
               "row_weights_applied": op.row_weights is not None,
               "w_sum_total": float(w_sum.sum()),
               "w_sum_per_channel": _quant(w_sum[idx]),
               "n_rows_per_channel": _quant(n_rows[idx]),
               "w_row": _quant(w_row)}
        print(f"[{self.name}] event {ev}: {op.n_data} rows on "
              f"{idx.size} channels | w_sum/chan median "
              f"{rec['w_sum_per_channel']['median']:.3f} "
              f"mean {rec['w_sum_per_channel']['mean']:.3f} fit bins | "
              f"rows/chan mean {rec['n_rows_per_channel']['mean']:.2f} | "
              f"w_row median {rec['w_row']['median']:.3f}")
        self._emit(store, rec)

    # -- pooled ------------------------------------------------------------
    def _edges(self, a: np.ndarray) -> np.ndarray | None:
        e = self.props.get("hist_edges")
        if e is None:
            return None
        e = list(e)
        if len(e) == 3 and float(e[2]) == int(e[2]) and int(e[2]) > 1:
            return np.linspace(float(e[0]), float(e[1]), int(e[2]) + 1)
        return np.asarray(e, dtype=float)

    def finalize(self):
        if not self._pool["w_sum"]:
            return {}
        cat = {k: (np.concatenate(v) if v else np.array([]))
               for k, v in self._pool.items()}
        n_ev = int(cat["event"].max()) + 1 if cat["event"].size else 0

        pooled = {"n_events": n_ev,
                  "n_channels": int(cat["w_sum"].size),
                  "n_rows": int(cat["w_row"].size),
                  "w_sum_per_channel": _quant(cat["w_sum"]),
                  "n_rows_per_channel": _quant(cat["n_rows"]),
                  "n_bins_per_channel": _quant(cat["n_bins"]),
                  "span_per_channel": _quant(cat["span"]),
                  "duty_cycle": _quant(cat["n_bins"] /
                                       np.maximum(cat["span"], 1)),
                  "d_sum_per_channel_ke": _quant(cat["d_sum"]),
                  "cgain_per_channel": _quant(
                      cat["cgain"][np.isfinite(cat["cgain"])]),
                  "w_row": _quant(cat["w_row"]),
                  "w_row_by_kind": {
                      k: _quant(cat["w_row"][cat["row_kind"] == k])
                      for k in LATCH_KINDS
                      if bool((cat["row_kind"] == k).any())},
                  }
        edges = self._edges(cat["w_sum"])
        if edges is not None:
            cnt, _ = np.histogram(cat["w_sum"], bins=edges)
            pooled["hist_w_sum"] = {
                "edges": [float(v) for v in edges],
                "counts": [int(v) for v in cnt],
                "n_below": int((cat["w_sum"] < edges[0]).sum()),
                "n_above": int((cat["w_sum"] > edges[-1]).sum())}

        if self.npz_path:
            np.savez_compressed(
                self.npz_path,
                w_sum=cat["w_sum"].astype(np.float64),
                n_rows=cat["n_rows"].astype(np.int32),
                n_bins=cat["n_bins"].astype(np.int32),
                span=cat["span"].astype(np.int32),
                d_sum=cat["d_sum"].astype(np.float64),
                cgain=cat["cgain"].astype(np.float64),
                px=cat["px"].astype(np.int32), py=cat["py"].astype(np.int32),
                event=cat["event"].astype(np.int32),
                w_row=cat["w_row"].astype(np.float64),
                row_kind=cat["row_kind"].astype(str),
                row_event=cat["row_event"].astype(np.int32))
            print(f"[{self.name}] wrote {self.npz_path}")
            pooled["npz"] = self.npz_path

        q = pooled["w_sum_per_channel"]
        print(f"[{self.name}] POOLED {n_ev} events, {q['n']} channels: "
              f"w_sum/chan mean {q['mean']:.4f} median {q['median']:.4f} "
              f"[p05 {q['p05']:.3f}, p95 {q['p95']:.3f}] fit bins")

        # _JsonRecorder.finalize writes the per-event body; carry the pooled
        # block into the same file rather than a second one.
        self._records = [{"per_event": self._records, "pooled": pooled}]
        return super().finalize()
