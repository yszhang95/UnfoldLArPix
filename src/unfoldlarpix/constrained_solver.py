"""Measurement building and shared numpy utilities.

SLIMMED in the torch-only refactor: the numpy solver (ZSOperator,
solve_fista, ladder/DR/subbin strategies) and its torch mirror were
removed — the single implementation now lives in ``model.operator`` +
``terms`` + ``solve`` (engine/strategies).  What remains here is
measurement construction (latch windows, sampling), the weighted-L1
soft-seed field, the sum-preserving split deposit, the truth-free
centroid position estimator, analysis-filter smoothing, and
diagnostics — numpy is appropriate for all of these (IO-adjacent or
post-processing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import fft


@dataclass
class LatchWindow:
    """One recorded burst charge as an integration window on a pixel."""

    px: int          # pixel index within the block
    py: int
    t_lo: float      # window start, block-local fine ticks
    t_hi: float      # window end (the latch time), block-local fine ticks
    value: float     # recorded charge in the window


def build_latch_windows(
    hits_location: np.ndarray,
    hits_data: np.ndarray,
    adc_hold_delay: int,
    block_offset: np.ndarray,
    csa_reset_time: int | None = None,
    split_threshold: float | None = None,
) -> list[LatchWindow]:
    """Convert raw hits into per-burst integration windows.

    ``hits_data`` columns are ``[x, y, z, q1, q2, ...]`` with cumulative
    charges; charges are differenced per burst.  The first burst of a
    sequence integrates from the pixel's previous integration restart
    (or from the beginning of time for the first sequence) up to its first
    latch; later bursts integrate exact ``adc_hold_delay`` windows.

    When ``csa_reset_time`` is given, the integration restart is computed
    as ``previous last latch + csa_reset_time`` — the point where the CSA
    actually resumes accumulating in ``nd_readout``.  The
    ``next_integration_start`` column (used as fallback) is the
    DISCRIMINATOR re-arm time (last latch + adc_down_time + one tick),
    which is ~1 us later; charge arriving in between is in the recorded
    value, so using it as the window edge misattributes that charge.

    When ``split_threshold`` is given, each sequence's first window is
    split at the trigger time (following ``fit_deconv3d.py`` in tred):
    the pre-trigger window carries the trigger threshold as an equality
    pseudo-measurement, and (trigger, first latch] carries the remainder.
    This injects the threshold-crossing information the lumped window
    discards, at the cost of ignoring crossing overshoot.
    """
    B = float(adc_hold_delay)
    loc = np.asarray(hits_location)
    dat = np.asarray(hits_data, dtype=float)
    order = np.lexsort((loc[:, 2], loc[:, 1], loc[:, 0]))
    windows: list[LatchWindow] = []
    prev_pixel = None
    prev_restart = None
    for i in order:
        px = int(loc[i, 0] - block_offset[0])
        py = int(loc[i, 1] - block_offset[1])
        trigger = float(loc[i, 2] - block_offset[2])
        cumulative = dat[i, 3:]
        charges = np.diff(cumulative, prepend=0.0)
        pixel = (px, py)
        if pixel != prev_pixel:
            first_lo = -np.inf
        else:
            first_lo = prev_restart if prev_restart is not None else -np.inf
        t_first = trigger + B
        if split_threshold is not None and float(charges[0]) >= split_threshold:
            windows.append(
                LatchWindow(px, py, first_lo, trigger, float(split_threshold))
            )
            windows.append(
                LatchWindow(px, py, trigger, t_first,
                            float(charges[0]) - float(split_threshold))
            )
        else:
            windows.append(
                LatchWindow(px, py, first_lo, t_first, float(charges[0]))
            )
        for j in range(1, len(charges)):
            lo = t_first + (j - 1) * B
            windows.append(LatchWindow(px, py, lo, lo + B, float(charges[j])))
        prev_pixel = pixel
        last_latch = t_first + (len(charges) - 1) * B
        if csa_reset_time is not None:
            prev_restart = last_latch + float(csa_reset_time)
        elif loc.shape[1] > 4:
            prev_restart = float(loc[i, 4] - block_offset[2])
        else:
            prev_restart = None
    return windows


def build_cumulative_windows(
    hits_location: np.ndarray,
    hits_data: np.ndarray,
    adc_hold_delay: int,
    block_offset: np.ndarray,
    csa_reset_time: int | None = None,
    split_threshold: float | None = None,
) -> tuple[list[LatchWindow], np.ndarray]:
    """Cumulative-space data rows: (restart, latch_k] with cumulative values.

    Exact-likelihood alternative to :func:`build_latch_windows`: each latch
    READ carries one independent per-tick noise sample, so the cumulative
    values have i.i.d. errors and plain least squares in this space is the
    exact ML.  Differencing them (the diff-space model) manufactures
    MA(1) anti-correlated noise (Var 2*sigma^2, Cov -sigma^2 between
    neighbours) that the unweighted diff fit silently ignores.

    With ``split_threshold``, the trigger-crossing pseudo-measurement
    (restart, trigger] = threshold is added as its own row; its error is
    the crossing overshoot, NOT the latch noise, so it should carry its
    own weight (see the returned ``is_pseudo`` mask and the operator's
    ``row_weights``).

    Returns ``(windows, is_pseudo)``.
    """
    B = float(adc_hold_delay)
    loc = np.asarray(hits_location)
    dat = np.asarray(hits_data, dtype=float)
    order = np.lexsort((loc[:, 2], loc[:, 1], loc[:, 0]))
    windows: list[LatchWindow] = []
    is_pseudo: list[bool] = []
    prev_pixel = None
    prev_restart = None
    for i in order:
        px = int(loc[i, 0] - block_offset[0])
        py = int(loc[i, 1] - block_offset[1])
        trigger = float(loc[i, 2] - block_offset[2])
        cumulative = dat[i, 3:]
        pixel = (px, py)
        first_lo = -np.inf if pixel != prev_pixel else (
            prev_restart if prev_restart is not None else -np.inf
        )
        t_first = trigger + B
        if split_threshold is not None and float(cumulative[0]) >= split_threshold:
            windows.append(
                LatchWindow(px, py, first_lo, trigger, float(split_threshold))
            )
            is_pseudo.append(True)
        for k, c in enumerate(cumulative):
            windows.append(
                LatchWindow(px, py, first_lo, t_first + k * B, float(c))
            )
            is_pseudo.append(False)
        prev_pixel = pixel
        last_latch = t_first + (len(cumulative) - 1) * B
        if csa_reset_time is not None:
            prev_restart = last_latch + float(csa_reset_time)
        elif loc.shape[1] > 4:
            prev_restart = float(loc[i, 4] - block_offset[2])
        else:
            prev_restart = None
    return windows, np.asarray(is_pseudo, dtype=bool)


def windows_to_sampling(
    windows: list[LatchWindow],
    block_shape: tuple[int, int, int],
    adc_hold_delay: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse sampling matrix from windows onto block bins.

    Each block bin holds the current integral over that bin; a window
    integral is the overlap-fraction-weighted sum of bin values.  Returns
    ``(rows, flat_cols, weights)`` in COO-like parallel arrays plus the
    data vector is taken from the windows by the caller.
    """
    B = float(adc_hold_delay)
    nx, ny, nt = block_shape
    rows, cols, weights = [], [], []
    for r, w in enumerate(windows):
        if not (0 <= w.px < nx and 0 <= w.py < ny):
            continue
        lo = max(w.t_lo / B, 0.0)
        hi = w.t_hi / B
        if hi <= lo:
            continue
        b0 = int(np.floor(lo))
        b1 = int(np.ceil(hi))
        for b in range(max(b0, 0), min(b1, nt)):
            frac = min(hi, b + 1) - max(lo, b)
            if frac <= 0:
                continue
            rows.append(r)
            cols.append((w.px * ny + w.py) * nt + b)
            weights.append(frac)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def manhattan_distance_from(mask: np.ndarray, d_max: int) -> np.ndarray:
    """Manhattan (L1) distance in voxels from a seed mask, capped at d_max.

    Computed by successive one-voxel dilations; voxels farther than
    ``d_max`` (including the case of an empty seed) get ``d_max``.
    """
    dist = np.full(mask.shape, int(d_max), dtype=np.int32)
    reached = mask.copy()
    dist[reached] = 0
    for d in range(1, int(d_max)):
        grown = _dilate_mask(reached, 1)
        new = grown & ~reached
        if not new.any():
            break
        dist[new] = d
        reached = grown
    return dist


def exponential_alpha_field(
    seed_mask: np.ndarray,
    alpha: float,
    decay_len: float,
    d_max: int | None = None,
    exponent: float = 1.0,
) -> np.ndarray:
    """Weighted-L1 field for a soft seed prior:
    ``alpha * exp((d / decay_len)**exponent)``.

    Encodes 'the probability of true charge decays with distance from the
    deghosted positions'.  ``exponent=1`` is the Laplace-tail
    (exponential) prior; ``exponent=2`` the GAUSSIAN-tail prior —
    physically motivated when charge missed by the deghost is displaced by
    diffusion (a Gaussian displacement kernel).  The Gaussian profile is
    flatter near the skeleton (gentler on one-offset voxels) and cuts off
    much harder beyond ~2*decay_len.  On the skeleton the penalty is the
    plain ``alpha``; a voxel activates only when |A^T residual| exceeds
    its alpha_i.
    """
    if decay_len <= 0:
        raise ValueError("decay_len must be positive.")
    if exponent <= 0:
        raise ValueError("exponent must be positive.")
    if d_max is None:
        d_max = int(np.ceil(8 * decay_len))
    d = manhattan_distance_from(seed_mask, d_max)
    return alpha * np.exp((d.astype(np.float64) / decay_len) ** exponent)


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Grow a boolean mask by one voxel per iteration along all axes."""
    out = mask.copy()
    for _ in range(int(iterations)):
        grown = out.copy()
        for ax in range(out.ndim):
            for shift in (-1, 1):
                grown |= np.roll(out, shift, axis=ax)
        out = grown
    return out


def split_deposit(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sum-preserving linear split of each charge toward a neighbour bin.

    A charge ``q[x, y, k]`` with fractional offset ``u[x, y, k]`` (in
    bins, bounded to [-1/2, 1/2]) is deposited as ``(1 - |u|) q`` in bin
    ``k`` and ``|u| q`` in bin ``k + sign(u)`` — the first-order (linear
    interpolation) representation of a charge at continuous position
    ``k + u`` on the coarse grid.  ``u == 0`` is the identity.
    """
    pos = np.clip(u, 0.0, 0.5)
    neg = np.clip(-u, 0.0, 0.5)
    out = q * (1.0 - pos - neg)
    out[:, :, 1:] += (q * pos)[:, :, :-1]
    out[:, :, :-1] += (q * neg)[:, :, 1:]
    return out


def split_adjoint(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Adjoint of ``split_deposit`` in ``q`` for fixed ``u``."""
    pos = np.clip(u, 0.0, 0.5)
    neg = np.clip(-u, 0.0, 0.5)
    out = g * (1.0 - pos - neg)
    out[:, :, :-1] += pos[:, :, :-1] * g[:, :, 1:]
    out[:, :, 1:] += neg[:, :, 1:] * g[:, :, :-1]
    return out


def centroid_bin_offsets(
    q: np.ndarray,
    window_bins: int = 1,
    min_charge: float = 0.05,
) -> np.ndarray:
    """TRUTH-FREE sub-bin position estimator: local reco centroids.

    Measured fact (nb4, 2026-07-17): fitting per-charge offsets against
    the windows is UNIDENTIFIABLE — the likelihood constrains the
    deposit FIELD only, and a charge between bins is already fitted as
    an amplitude split across the two bins, so a split-parameterized
    offset is a redundant re-parameterization (fitted offsets came out
    ~0.2 ticks vs ~10 needed).  The honest position estimator is the
    charge-weighted centroid of the fitted sharp field itself: for each
    active voxel, the centroid over the same pixel within
    ``+-window_bins``, clipped to half a bin.

    Returns offsets in BIN units (q-shaped array, bounded to [-1/2, 1/2])
    — multiply by ``adc_hold_delay`` for ticks; deposit with
    :func:`split_deposit` or Gaussian shapes at the shifted centers.
    ``window_bins=1`` is conservative (slope stays ~1); ``2`` merges
    more aggressively (fewer ghost voxels, more killed truth).
    """
    offsets = np.zeros_like(q)
    nt = q.shape[2]
    w = int(window_bins)
    xs, ys, ks = np.nonzero(q > 1e-6)
    for x, y, k in zip(xs, ys, ks):
        lo, hi = max(k - w, 0), min(k + w + 1, nt)
        col = q[x, y, lo:hi]
        tot = float(col.sum())
        if tot < min_charge:
            continue
        idx = np.arange(lo, hi, dtype=np.float64)
        offsets[x, y, k] = np.clip(
            float((col * idx).sum() / tot) - k, -0.5, 0.5)
    return offsets


def gaussian_post_smooth(
    q: np.ndarray, adc_hold_delay: int, sigma_time: float, sigma_pixel: float
) -> np.ndarray:
    """Apply the pipeline's Gaussian filter to a solver output.

    The linear pipeline's ``deconv_q`` carries the Gaussian regularization
    filter; applying the same filter to the solver's sharp output makes the
    two directly comparable against the same smeared truth.
    """
    from .deconv import gaussian_filter_3d

    F = gaussian_filter_3d(
        q.shape, dt=(1, 1, adc_hold_delay),
        sigma=(sigma_pixel, sigma_pixel, sigma_time),
    )
    axes = (0, 1, 2)
    return fft.irfftn(fft.rfftn(q, axes=axes) * F, s=q.shape, axes=axes)


def wiener_spectral_weight(
    freqs: np.ndarray,
    P_truth: np.ndarray,
    P_deconv: np.ndarray,
    n_time: int,
    cap: float = 100.0,
    smear_sigma_f: float | None = None,
) -> np.ndarray:
    """Spectra-aware Wiener weight w(f) for the temporal quadratic prior.

    ``w(f) = max(P_deconv/P_truth - 1, 0)`` is the measured noise-to-signal
    power ratio in deconvolved space (≈0 where the reconstruction already
    matches truth, large in the noise-dominated band), interpolated onto
    the ``rfftfreq(n_time)`` grid of the fit's time axis and capped —
    uncapped weights where P_truth → 0 would forbid all high-frequency
    content outright.

    IMPORTANT: ``P_truth`` is the SMEARED truth spectrum, so the raw ratio
    counts everything the analysis Gaussian removes as noise — an
    anti-sharpness penalty that would crush the sharp fit coefficients.
    Pass ``smear_sigma_f`` (the analysis Gaussian's frequency-domain sigma
    on THIS time grid, e.g. 0.005 cycles/tick * 30 ticks/bin = 0.15
    cycles/bin) to refer the weight to the sharp-space signal spectrum
    ``S_sharp = P_truth/|G|^2``: the weight becomes ``w * |G(f)|^2``.

    By Parseval the resulting penalty ``lam * sum_f w(f)|Q(f)|^2`` is a
    TIME-DOMAIN quadratic form: a circular convolution with the symmetric
    kernel ``irfft(w)``.  The fit never leaves temporal space; its gradient
    is ``2*lam*irfft(w*rfft(q))`` per iteration.  ``w == 1`` reduces
    exactly to the flat ``lam_l2`` ridge.
    """
    target = np.fft.rfftfreq(int(n_time))
    P_t = np.interp(target, freqs, P_truth)
    P_d = np.interp(target, freqs, P_deconv)
    eps = 1e-12 * float(np.max(P_t)) if np.max(P_t) > 0 else 1e-12
    w = np.clip(P_d / (P_t + eps) - 1.0, 0.0, float(cap))
    if smear_sigma_f is not None:
        w = w * np.exp(-(target ** 2) / float(smear_sigma_f) ** 2)
    return w


def difference_spectral_weight(n_time: int, order: int = 2) -> np.ndarray:
    """Spectral weight of the temporal difference seminorm ``|D^order q|^2``.

    ``w(f) = |2 sin(pi f)|^(2*order)`` — the exact spectrum of the
    order-th discrete difference operator.  Zero at DC (order 1: constants
    free; order 2: constants and linear ramps free), so window sums carry
    NO shrinkage by construction; only voxel-to-voxel wiggle — the
    near-null 'adjacent-bin exchange' directions of the window-sampling
    operator — is penalized.  Geometry-blind: acts along time only.
    """
    f = np.fft.rfftfreq(int(n_time))
    return np.abs(2.0 * np.sin(np.pi * f)) ** (2 * int(order))


def probe_support_conditioning(
    op,
    support: np.ndarray,
    n_iter: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate (lam_max, lam_min) of P A^T A P on the given support.

    Power iteration for the largest eigenvalue, shifted power iteration
    for the smallest (restricted to the support subspace).  The ratio
    bounds the conditioning of the UNREGULARIZED amplitude fit: a modest
    ratio means positivity alone suffices; a large one quantifies how much
    variance an unregularized fit admits along the worst direction.
    """
    rng = np.random.default_rng(seed)

    def apply_m(x: np.ndarray) -> np.ndarray:
        return np.where(support, op.adjoint(op.forward(np.where(support, x, 0.0))), 0.0)

    x = np.where(support, rng.standard_normal(support.shape), 0.0)
    x /= np.linalg.norm(x)
    lam_max = 1.0
    for _ in range(int(n_iter)):
        y = apply_m(x)
        lam_max = float(np.linalg.norm(y))
        if lam_max <= 0:
            return 0.0, 0.0
        x = y / lam_max

    z = np.where(support, rng.standard_normal(support.shape), 0.0)
    z /= np.linalg.norm(z)
    mu_max = 0.0
    for _ in range(int(n_iter)):
        y = lam_max * z - apply_m(z)
        mu_max = float(np.linalg.norm(y))
        if mu_max <= 0:
            break
        z = y / mu_max
    lam_min = max(lam_max - mu_max, 0.0)
    return lam_max, lam_min

