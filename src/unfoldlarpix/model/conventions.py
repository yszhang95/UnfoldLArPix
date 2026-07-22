"""Single home for every tick/bin/phase convention in the reconstruction.

The dominant bug class in this project has been CONVENTION bugs
(half-bin declaration, deposit phase, hits column semantics, window
rounding).  Every convention lives here, with its measured
justification; nothing else in the package may hard-code them.
"""
from __future__ import annotations

# ---------------------------------------------------------------------
# Deposit phase (block building)
# ---------------------------------------------------------------------
# Burst charges are deposited on the coarse grid with a -0.5 bin phase:
# the window CENTER of mass, not the window end.  Measured: phase repair
# alone moved the pipeline r 0.9709 -> 0.9853 (FINDINGS, tier-1a).
DEPOSIT_PHASE: float = -0.5

DEPOSIT_MODE: str = "linear"    # charge-conserving linear split


def solver_time_shift(adc_hold_delay: int) -> int:
    """Declared time offset of the solver's q grid vs the raw block.

    The window->bin overlap convention of the fit sits half a bin later
    than the linear pipeline's deposit-phase(-0.5) convention, so solver
    outputs are declared at ``-B + B//2``.  Measured (nb4, Phase-0
    centroid diagnostic): the naive ``-B`` declaration leaves a
    systematic -0.55-bin reco-early offset; ``+B//2`` removes it
    (r 0.944 -> 0.980, ghost 10.4 -> 5.4%; FINDINGS item on the
    half-bin fix).
    """
    return -adc_hold_delay + adc_hold_delay // 2


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
