"""Readout-level diagnostics: properties of the recorded hits alone.

Nothing here reads an operator, a truth or a solution, so these algorithms
run on a sequence truncated at ``LoadEvent`` -- no response file, no GPU, no
solve.  That is the point: a claim about the READOUT should not be derivable
only from a reconstruction of it.
"""
from __future__ import annotations

import numpy as np

from ..fwk.component import Algorithm, algorithm
from ..model.conventions import resolve_burst_tau


@algorithm("ImmediateFraction")
class ImmediateFraction(Algorithm):
    """How often a trigger is an immediate (suppression-limited) re-trigger.

    The gap of a trigger sequence is the distance from the previous sequence's
    LAST latch on the same pixel to this trigger::

        last_latch = trigger + nburst * adc_hold_delay     (HitsView)
        gap_j      = trigger_j - last_latch_{j-1}

    A gap below ``burst_tau`` means the pixel was still above threshold when
    the discriminator re-armed: the crossing instant is set by the re-arm
    clock, not by the signal, and the pre-trigger window holds the whole
    suppression-time pile-up instead of the threshold.  This is exactly the
    condition ``build_latch_rows`` uses to refuse the split-trigger
    pseudo-measurement (``threshold_limited``), and ``docs/BURST_TAU.md``
    carries the measurement of what happens when it is ignored.

    The word "immediate" has been used for three different populations and
    they differ by a factor of three, so all of them are reported and none is
    the default reading by accident:

    ``frac_immediate``
        ``gap < tau``, over ALL sequences -- each pixel's first sequence has
        no previous latch and counts in the denominator as not-immediate.
        **This is the published fraction** (the note's ``tab:immediate``).
    ``frac_immediate_inclusive``
        the same with ``gap <= tau``.  The gate is strict, so this is only a
        reading of the prose, never of the code.
    ``frac_immediate_of_retriggers``
        ``gap < tau`` over sequences that HAVE a previous latch.  Roughly
        2-3x the published number; it answers a different question ("given
        that the pixel re-fired, was it suppression-limited?").
    ``frac_at_rearm``
        fired at the earliest instant the readout allows (``trigger`` equal to
        the previous sequence's recorded ``rearm`` column).  This is
        "immediate" read literally, and it is about a third of
        ``frac_immediate`` -- the gate's window is one ``adc_hold_delay``
        wide, not one tick.
    ``frac_lumped_in_B``
        the fraction of first windows the split-trigger configuration leaves
        lumped, i.e. ``(gap < tau) OR (first burst charge < threshold)``.
        This is NOT the immediate fraction: the second branch is a
        sub-threshold first window, it survives to ``nburst = 64`` where the
        immediate population is exactly zero, and conflating the two is why
        ``ab_anatomy/ab_rows.py``'s ``imm_frac`` disagrees with the published
        table on every cell.  Both branches are reported separately.

    Props: ``tau`` (default ``auto`` -- the physical floor from the readout
    config; an integer is resolved through :func:`resolve_burst_tau` and so is
    clamped and warned about like ``BuildMeasurement``'s).

    Accumulates over events; ``finalize`` pools the counts.
    """

    reads = ("hits_view", "readout_config")
    writes = ("immediate.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self._acc: list[dict] = []

    def execute(self, store):
        hv = store.get("hits_view")
        rc = store.get("readout_config")
        tau_prop = self.props.get("tau", "auto")
        tau = float(resolve_burst_tau(
            rc, None if tau_prop in (None, "auto") else int(tau_prop)))
        thr = float(rc.threshold)

        px, py = hv.pixel_x, hv.pixel_y
        trig = np.asarray(hv.trigger, dtype=np.float64)
        last = np.asarray(hv.last_latch, dtype=np.float64)
        rearm = np.asarray(hv.rearm, dtype=np.float64)
        c0 = np.asarray(hv.burst_charges[:, 0], dtype=np.float64)

        # same ordering as build_latch_rows: pixel-major, then time
        order = np.lexsort((trig, py, px))
        n = order.size
        gap = np.full(n, np.nan)
        at_rearm = np.zeros(n, dtype=bool)
        prev_pixel = None
        prev_last = prev_rearm = None
        for k, i in enumerate(order):
            pixel = (int(px[i]), int(py[i]))
            if pixel != prev_pixel:
                prev_last = prev_rearm = None
            if prev_last is not None:
                gap[k] = trig[i] - prev_last
                at_rearm[k] = trig[i] == prev_rearm
            prev_pixel = pixel
            prev_last, prev_rearm = last[i], rearm[i]

        retrig = ~np.isnan(gap)
        imm = retrig & (gap < tau)
        imm_le = retrig & (gap <= tau)
        sub = (c0 < thr)[order]
        g = gap[retrig]
        npix = len({(int(px[i]), int(py[i])) for i in order})

        rec = {
            "tau_ticks": tau, "threshold_ke": thr,
            "adc_hold_delay": int(rc.adc_hold_delay),
            "adc_down_time": int(rc.adc_down_time),
            "csa_reset_time": (None if rc.csa_reset_time is None
                               else int(rc.csa_reset_time)),
            "nburst": int(hv.nburst),
            "n_sequences": int(n), "n_retriggers": int(retrig.sum()),
            "n_pixels": npix,
            "sequences_per_pixel": (n / npix) if npix else None,
            "n_immediate": int(imm.sum()),
            "frac_immediate": float(imm.sum() / n) if n else None,
            "frac_immediate_inclusive": float(imm_le.sum() / n) if n else None,
            "frac_immediate_of_retriggers": (
                float(imm.sum() / retrig.sum()) if retrig.sum() else 0.0),
            "frac_at_rearm": float(at_rearm.sum() / n) if n else None,
            "frac_lumped_in_B": float((imm | sub).sum() / n) if n else None,
            "frac_gate_only": float((imm & ~sub).sum() / n) if n else None,
            "frac_subthreshold_only": float((sub & ~imm).sum() / n) if n else None,
            "frac_both": float((imm & sub).sum() / n) if n else None,
            "frac_subthreshold": float(sub.sum() / n) if n else None,
            "gap_ticks": (None if g.size == 0 else {
                "min": float(g.min()), "max": float(g.max()),
                "median": float(np.median(g)),
                "earliest_allowed": float(int(rc.adc_down_time)
                                          + (0 if rc.csa_reset_time is None
                                             else int(rc.csa_reset_time)))}),
        }
        self._acc.append(rec)
        print("[ImmediateFraction] nburst %d: %d/%d sequences immediate "
              "(gap < %d) = %.3f  [of re-triggers %.3f, at re-arm %.3f, "
              "lumped in B %.3f]"
              % (rec["nburst"], rec["n_immediate"], n, tau,
                 rec["frac_immediate"], rec["frac_immediate_of_retriggers"],
                 rec["frac_at_rearm"], rec["frac_lumped_in_B"]))
        self.put(store, "immediate.summary", rec)

    def finalize(self) -> dict:
        if len(self._acc) <= 1:
            return {}
        n = sum(r["n_sequences"] for r in self._acc)
        return {"n_events": len(self._acc), "n_sequences": n,
                "n_immediate": sum(r["n_immediate"] for r in self._acc),
                "frac_immediate": (sum(r["n_immediate"] for r in self._acc) / n
                                   if n else None)}
