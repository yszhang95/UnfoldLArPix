"""Measurement-row noise model of the zero-suppressed readout.

Derived from ``tred.readout.nd_readout`` and verified against simulation
output that stores both the noisy hits and the noiseless current
(``pos_a50 nb4 *_wf.npz``):

* **uncorrelated noise** (``uncorr_noise``): one draw per fine tick added
  to the accumulator VALUE, so every latched read carries an independent
  error ``eps`` — reads 30 ticks apart do not share noise.
* **kTC baseline** (``reset_noise``): one draw per reset epoch, constant
  within it — enters every cumulative read of a post-reset sequence
  equally and cancels in all within-sequence differences.
* **threshold dispersion** (``thres_noise``): one draw per trigger.

Row errors (``d - true window integral``) in the diff space of
:func:`~unfoldlarpix.constrained_solver.build_latch_rows`:

========== =========================== ===========================
kind        error                       variance
========== =========================== ===========================
diff        ``eps_j - eps_{j-1}``       ``2 s_u^2``
lumped      ``eps_1 (+ beta)``          ``s_u^2 (+ s_r^2)``
pseudo      ``-eta + eps_* (+ beta)``   ``s_t^2 + s_u^2 (+ s_r^2)``
remainder   ``eps_1 + eta - eps_*``     ``2 s_u^2 + s_t^2``
========== =========================== ===========================

(parenthesised terms only for post-reset sequences).  Adjacent rows are
anti-correlated — ``Cov(diff_j, diff_{j+1}) = -s_u^2`` (measured
correlation exactly -0.500) and ``Cov(pseudo, remainder) =
-(s_t^2 + s_u^2)`` — which a DIAGONAL weighting deliberately ignores:
full whitening was measured to trade ghost fraction and muon r/slope for
integral in a scale-dependent way and is NOT offered here.

The crossing-selection effect (the trigger fires on the first checked
tick above threshold) makes the measured pseudo/remainder variances
~30% smaller than these analytic values; the analytic values are used
as conservative weights.
"""
from __future__ import annotations

import numpy as np

__all__ = ["row_variances", "row_weights"]

_KINDS = ("pseudo", "remainder", "lumped", "diff")


def row_variances(metas, readout_config) -> np.ndarray:
    """Analytic error variance [ke^2] per measurement row.

    ``metas`` is the :class:`~unfoldlarpix.constrained_solver.RowMeta`
    list emitted by ``build_latch_rows``; the noise scales are read from
    the readout config (they travel with the data file).
    """
    s_u = float(readout_config.uncorr_noise or 0.0)
    s_t = float(readout_config.thres_noise or 0.0)
    s_r = float(readout_config.reset_noise or 0.0)
    if s_u <= 0.0:
        raise ValueError(
            "row_variances needs uncorr_noise > 0 from the readout config; "
            "a noiseless file has no meaningful row weighting")
    su2, st2, sr2 = s_u * s_u, s_t * s_t, s_r * s_r
    out = np.empty(len(metas))
    for i, m in enumerate(metas):
        if m.kind == "diff":
            out[i] = 2 * su2
        elif m.kind == "lumped":
            out[i] = su2 + (sr2 if m.post_reset else 0.0)
        elif m.kind == "pseudo":
            out[i] = st2 + su2 + (sr2 if m.post_reset else 0.0)
        elif m.kind == "remainder":
            out[i] = 2 * su2 + st2
        else:
            raise ValueError(f"unknown row kind: {m.kind!r}")
    return out


def row_weights(metas, readout_config, mode: str) -> np.ndarray:
    """Diagonal data-fidelity weights, referenced to the burst-diff row.

    The reference variance is ``2 * uncorr_noise^2`` (a burst difference),
    so ordinary latched measurements keep weight 1 and the l1/censor
    balance of the unweighted configuration is preserved by construction.

    mode ``'split'``
        Only the two trigger-split rows are re-weighted by their
        equivalent noise; every real measurement keeps weight 1.
    mode ``'diag'``
        Every row is weighted ``ref_var / var`` (post-reset first windows
        down-weighted by the kTC baseline, virgin first windows
        up-weighted).
    """
    if mode not in ("split", "diag"):
        raise ValueError(f"row_weights mode must be 'split' or 'diag', "
                         f"got {mode!r}")
    var = row_variances(metas, readout_config)
    ref = 2 * float(readout_config.uncorr_noise) ** 2
    # The burst-diff reference presumes burst-diff rows exist.  A self-
    # trigger event (nburst = 1) has NONE -- every row is a split or lumped
    # first window -- and anchoring to the absent row type down-weights the
    # whole system (~2x), which acts as a hidden l1 rescale (measured:
    # integral -2.5..-3.5 pp on the nb1 scan).  Fall back to the mean
    # variance so the average data-term scale is preserved instead.
    if not any(m.kind == "diff" for m in metas):
        ref = float(var.mean())
    w = ref / var
    if mode == "split":
        keep = np.array([m.kind not in ("pseudo", "remainder")
                         for m in metas])
        w[keep] = 1.0
    return w
