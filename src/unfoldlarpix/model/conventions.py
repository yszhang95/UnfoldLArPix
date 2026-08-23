"""Single home for every tick/bin/phase convention in the reconstruction.

The dominant bug class in this project has been CONVENTION bugs
(half-bin declaration, deposit phase, hits column semantics, window
rounding).  Every convention lives here, with its measured
justification; nothing else in the package may hard-code them.
"""
from __future__ import annotations

import warnings

# ---------------------------------------------------------------------
# Deposit phase (block building)
# ---------------------------------------------------------------------
# Burst charges are deposited on the coarse grid with a -0.5 bin phase:
# the window CENTER of mass, not the window end.  Measured: phase repair
# alone moved the pipeline r 0.9709 -> 0.9853 (FINDINGS, tier-1a).
DEPOSIT_PHASE: float = -0.5

DEPOSIT_MODE: str = "linear"    # charge-conserving linear split


# Time convention of the written q grid.  "release_point" declares bin k at
# the operator's own release instant, boffset_raw + k*B, and the evaluator
# deposits it there -- one statement instead of two half bins that cancelled.
# Files written before 2026-08-16 carry no marker and use the LEGACY pair:
# boffset = raw - B + B//2, deposited at boffset + (k+1/2)*B.  For even B the
# two are identical; the readers keep the legacy branch until it is dropped.
TIME_CONVENTION: str = "release_point"


def solver_time_shift(adc_hold_delay: int) -> int:
    """LEGACY declared time offset of the solver's q grid vs the raw block.

    Superseded by ``TIME_CONVENTION = "release_point"``: the writer now
    declares the raw corner and the evaluator deposits at ``b_off + k*B``,
    which is the same instant without the two cancelling half bins.  Kept
    so that files written before the change can still be read.

    The window->bin overlap convention of the fit sits half a bin later
    than the linear pipeline's deposit-phase(-0.5) convention, so solver
    outputs are declared at ``-B + B//2``.  Measured (nb4, Phase-0
    centroid diagnostic): the naive ``-B`` declaration leaves a
    systematic -0.55-bin reco-early offset; ``+B//2`` removes it
    (r 0.944 -> 0.980, ghost 10.4 -> 5.4%; FINDINGS item on the
    half-bin fix).
    """
    return -adc_hold_delay + adc_hold_delay // 2


def burst_tau_min(readout_config) -> int:
    """Physical floor [ticks] for the burst-merge gap ``tau``.

    ``tau`` is the largest trigger-to-trigger gap at which consecutive
    triggers are still treated as ONE continuous charge deposit and merged
    by dead-time compensation (charge-conserving); larger gaps go to
    template (shape) compensation.  Its floor is set by the readout: after
    a latch the CSA is dead for ``adc_down_time`` and re-arms at
    ``hold + adc_down_time + one_tick``; if charge is still above threshold
    it re-fires immediately, so the smallest possible gap of a genuine
    continuous re-trigger is::

        adc_hold_delay + adc_down_time + one_tick

    Below this floor such immediate re-triggers are misrouted to template
    compensation, which cannot place a sub-``adc_hold_delay`` pre-trigger
    ramp and DELETES ~threshold of charge per re-trigger (FINDINGS: burst
    template charge non-conservation).  ``tau`` may be raised ABOVE the
    floor (up to ``2*adc_hold_delay``, see ``resolve_burst_tau``) to match a
    broader field response — a wide waveform keeps a pixel's cloud alive
    longer, so re-triggers separated by more than the floor can still
    belong to one deposit and should merge.
    """
    return int(readout_config.adc_hold_delay
               + readout_config.adc_down_time
               + readout_config.one_tick)


# ---------------------------------------------------------------------
# Hits column semantics  (see io/hits.py for the enforced accessors)
# ---------------------------------------------------------------------
# hits.location columns: [pixel_x, pixel_y, trigger, trigger+B, re-arm]
#   - col3 is the FIRST latch (always trigger + B).  It equals the last
#     latch ONLY for nburst = 1.  Reading col3 as "last latch" caused
#     the censor reset-reference bug (FINDINGS item 19).
#   - later latches are DERIVED: latch_k = trigger + k*B,
#     last latch = trigger + nburst*B.
#   - col4 = discriminator re-arm AFTER the last burst
#     (= last latch + adc_down_time + 1 tick).  Between CSA restart
#     (last latch + csa_reset) and col4 the discriminator CANNOT fire.
# hits.data columns: [x, y, z, q1..q_nburst]; nburst = shape[1] - 3;
#   charges are cumulative per burst — difference across columns for
#   per-burst charges.
HITS_LOCATION_COLUMNS = ("pixel_x", "pixel_y", "trigger", "first_latch",
                         "rearm")

# ---------------------------------------------------------------------
# Window sampling
# ---------------------------------------------------------------------
# Latch windows are mapped onto coarse bins with FRACTIONAL overlap
# weights (uniform-within-bin, first order).  Kernel note: the
# "integrated response" is the response current integrated PER COARSE
# BIN (bin-integrated current) — it is NOT the CSA cumulative; the
# cumulative nature of the measurement lives in the window sampling.


# ---------------------------------------------------------------------
# Burst-merge gap (tau)
# ---------------------------------------------------------------------
def resolve_burst_tau(readout_config, tau: int | None = None) -> int:
    """Resolve the tunable burst-merge gap ``tau`` against its physical bounds.

    Bounds (both set by the readout, ticks):
    - FLOOR = adc_hold_delay + adc_down_time + one_tick (:func:`burst_tau_min`).
      Below it, immediate re-triggers are misrouted to template compensation
      and lose ~threshold of charge each.
    - CAP = 2 * adc_hold_delay.  Above it, dead-time compensation is applied
      across a gap longer than one burst window plus one dead window — a
      genuine multi-bin silence.  Dead-time merge does not create the
      intermediate gap bins that template compensation does, so the merged
      sequence comes out with a different length and its start displaced by
      ~one bin (measured: len 130->128, start +1*adc_hold_delay for a
      gap-82 pixel).  Gaps beyond the cap are real silences and belong to
      template compensation.

    ``tau=None`` -> FLOOR.  A value in [FLOOR, CAP] is used as given (tune
    upward for broader responses).  Out-of-range values warn and clamp.  If
    FLOOR > CAP (adc_down_time too large relative to adc_hold_delay) the two
    constraints cannot both hold; warn and keep the FLOOR (charge
    conservation over timing).
    """
    tau_min = burst_tau_min(readout_config)
    tau_max = 2 * int(readout_config.adc_hold_delay)
    if tau_min > tau_max:
        warnings.warn(
            f"burst tau floor {tau_min} exceeds cap {tau_max} "
            f"(=2*adc_hold_delay): adc_down_time={readout_config.adc_down_time} "
            f"is too large relative to adc_hold_delay="
            f"{readout_config.adc_hold_delay} for a well-posed burst-merge "
            f"window (no tau conserves charge AND avoids the >2B shift). "
            f"Keeping the floor {tau_min}.", RuntimeWarning, stacklevel=2)
        return tau_min
    if tau is None:
        return tau_min
    tau = int(tau)
    if tau < tau_min:
        warnings.warn(
            f"burst tau={tau} is below the physical floor {tau_min} "
            f"(adc_hold_delay+adc_down_time+one_tick): immediate re-triggers "
            f"would be routed to template compensation and lose ~threshold of "
            f"charge each. Clamping to {tau_min}.",
            RuntimeWarning, stacklevel=2)
        return tau_min
    if tau > tau_max:
        warnings.warn(
            f"burst tau={tau} exceeds the cap {tau_max} (=2*adc_hold_delay): "
            f"gaps >2*adc_hold_delay are genuine silences; dead-time merge "
            f"across them changes the merged-sequence length and shifts the "
            f"following sequence by ~one bin. Clamping to {tau_max}.",
            RuntimeWarning, stacklevel=2)
        return tau_max
    return tau
