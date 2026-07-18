"""Constrained linear inversion of zero-suppressed LArPix readout.

Zero-suppression makes the *dense* convolution model wrong, but the
measurement is still LINEAR once the latch times are known: every burst
charge is the integral of the induced current over a known window.  This
module solves

    min_q  || A q - d ||^2  +  beta * || relu(S_quiet(K q) - thr) ||^2
                            +  alpha * || q ||_1     s.t. q >= 0

where

- ``A`` = (field-response convolution) followed by (window sampling at the
  recorded latch windows) — no template compensation anywhere,
- ``d`` = the recorded per-burst charges,
- the quiet term encodes what silence means: any single window integral
  reaching the trigger threshold would have fired, so unfired bins are
  inequality data, not zeros,
- positivity + L1 regularize the (heavily underdetermined) unobserved
  regions.

Solved with monotone FISTA (projected proximal gradient).  All operators
are FFT-based; the whole event fits comfortably on CPU.
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


class ZSOperator:
    """Forward/adjoint operator: charge q -> recorded burst integrals."""

    def __init__(
        self,
        kernel: np.ndarray,
        block_shape: tuple[int, int, int],
        windows: list[LatchWindow],
        adc_hold_delay: int,
        row_weights: np.ndarray | None = None,
    ):
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.block_shape = tuple(int(s) for s in block_shape)
        nx, ny, nt = self.block_shape
        kx, ky, kt = self.kernel.shape
        self.q_shape = (nx, ny, nt - kt + 1)
        self.fft_shape = (nx + kx - 1, ny + ky - 1, nt)
        self.cx = (kx - 1) // 2
        self.cy = (ky - 1) // 2
        axes = (0, 1, 2)
        self._K = fft.rfftn(self.kernel, s=self.fft_shape, axes=axes)
        self._axes = axes
        self.n_data = len(windows)
        self.d = np.array([w.value for w in windows], dtype=np.float64)
        self._rows, self._cols, self._weights = windows_to_sampling(
            windows, self.block_shape, adc_hold_delay
        )
        if row_weights is not None:
            # fold per-row weights into the rows as sqrt(w): the plain LS
            # objective on the scaled system equals the weighted LS on the
            # original one — no solver changes needed.
            sw = np.sqrt(np.asarray(row_weights, dtype=np.float64))
            self.d = self.d * sw
            self._weights = self._weights * sw[self._rows]

    # -- block-space convolution ------------------------------------------
    def conv(self, q: np.ndarray) -> np.ndarray:
        pred = fft.irfftn(
            fft.rfftn(q, s=self.fft_shape, axes=self._axes) * self._K,
            s=self.fft_shape,
            axes=self._axes,
        )
        pred = np.roll(pred, -self.cx, axis=0)
        pred = np.roll(pred, -self.cy, axis=1)
        nx, ny, nt = self.block_shape
        return pred[:nx, :ny, :nt]

    def conv_adjoint(self, r_block: np.ndarray) -> np.ndarray:
        nx, ny, nt = self.block_shape
        padded = np.zeros(self.fft_shape, dtype=np.float64)
        padded[:nx, :ny, :nt] = r_block
        padded = np.roll(padded, self.cy, axis=1)
        padded = np.roll(padded, self.cx, axis=0)
        out = fft.irfftn(
            fft.rfftn(padded, axes=self._axes) * np.conj(self._K),
            s=self.fft_shape,
            axes=self._axes,
        )
        qx, qy, qt = self.q_shape
        return out[:qx, :qy, :qt]

    # -- sampling -----------------------------------------------------------
    def sample(self, block: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_data, dtype=np.float64)
        np.add.at(out, self._rows, self._weights * block.reshape(-1)[self._cols])
        return out

    def sample_adjoint(self, r: np.ndarray) -> np.ndarray:
        flat = np.zeros(int(np.prod(self.block_shape)), dtype=np.float64)
        np.add.at(flat, self._cols, self._weights * r[self._rows])
        return flat.reshape(self.block_shape)

    # -- full operator ------------------------------------------------------
    def forward(self, q: np.ndarray) -> np.ndarray:
        return self.sample(self.conv(q))

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        return self.conv_adjoint(self.sample_adjoint(r))

    def lipschitz(self, n_iter: int = 12, seed: int = 0) -> float:
        """Power iteration estimate of ||A^T A||."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.q_shape)
        x /= np.linalg.norm(x)
        lam = 1.0
        for _ in range(n_iter):
            y = self.adjoint(self.forward(x))
            lam = float(np.linalg.norm(y))
            if lam <= 0:
                return 1.0
            x = y / lam
        return lam


def _tv_gradient(x: np.ndarray, eps: float = 1e-6) -> tuple[float, np.ndarray]:
    """Value and gradient of the isotropic TV-like term sqrt(sum |grad x|^2).

    Matches the ``lam_dx * sqrt(sum dx^2 + dy^2 + dz^2)`` regularizer of
    tred's ``fit_deconv3d.py`` (a single global sqrt, so the gradient is
    the negative divergence of the gradient field divided by the norm).
    """
    diffs = [np.diff(x, axis=ax) for ax in range(3)]
    norm = np.sqrt(sum(float((d ** 2).sum()) for d in diffs) + eps)
    grad = np.zeros_like(x)
    for ax, d in enumerate(diffs):
        pad_lo = [(0, 0)] * 3
        pad_hi = [(0, 0)] * 3
        pad_lo[ax] = (1, 0)
        pad_hi[ax] = (0, 1)
        grad += np.pad(d, pad_hi) - np.pad(d, pad_lo)
    return norm, -grad / norm


def solve_fista(
    op: ZSOperator,
    *,
    alpha: float = 0.0,
    beta_quiet: float = 0.0,
    quiet_mask: np.ndarray | None = None,
    quiet_threshold: float = np.inf,
    n_iter: int = 200,
    q0: np.ndarray | None = None,
    L: float | None = None,
    support_mask: np.ndarray | None = None,
    lam_l2: float = 0.0,
    lam_tv: float = 0.0,
    lam_spectral: float = 0.0,
    spectral_weight: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Non-negative L1-regularized least squares via FISTA.

    Args:
        alpha: L1 weight (per charge unit) — scalar, or an array of
            q-shape for a spatially varying (weighted) L1 prior such as
            :func:`exponential_alpha_field`.  With noisy data this must be
            of order the per-window noise scale: positivity RECTIFIES
            zero-mean noise into positive charge, and the L1 soft
            threshold is what removes that bias.
        beta_quiet: Weight of the quiet-bin inequality penalty
            ``beta * relu(conv(q)[quiet] - thr)^2``.
        quiet_mask: Bool block-shape mask of bins whose window integral
            must stay below ``quiet_threshold`` (unfired discriminators).
        n_iter: FISTA iterations.
        q0: Warm start (e.g. clipped first-pass deconvolution).
        support_mask: Optional bool array of q-shape; the solution is
            projected onto this support every step (ROI-style).  Kills the
            noise-rectification bias outside the signal region.
        lam_l2: Ridge weight (adds ``2*lam_l2*q`` to the gradient).
        lam_tv: Weight of the isotropic gradient-norm smoothness term
            (see :func:`_tv_gradient`; from tred's ``fit_deconv3d.py``).
        lam_spectral / spectral_weight: Spectra-aware Wiener prior
            ``lam * sum_f w(f)|Q(f)|^2`` along the time axis (see
            :func:`wiener_spectral_weight`); ``w == 1`` reduces to lam_l2.
    """
    if L is None:
        L = op.lipschitz()
    if beta_quiet > 0 and quiet_mask is not None:
        # crude bound accounting for the quadratic quiet penalty curvature
        L_total = L + 2.0 * beta_quiet
    else:
        L_total = L
    if lam_spectral > 0 and spectral_weight is not None:
        L_total = L_total + 2.0 * lam_spectral * float(np.max(spectral_weight))
    step = 1.0 / (L_total * 1.05)

    x = np.zeros(op.q_shape) if q0 is None else np.clip(q0, 0.0, None).copy()
    if support_mask is not None:
        x = x * support_mask
    y = x.copy()
    t = 1.0
    for k in range(int(n_iter)):
        block_pred = op.conv(y)
        resid = op.sample(block_pred) - op.d
        grad = op.conv_adjoint(op.sample_adjoint(resid))
        if beta_quiet > 0 and quiet_mask is not None:
            viol = np.where(
                quiet_mask, np.clip(block_pred - quiet_threshold, 0.0, None), 0.0
            )
            # the adjoint costs two full-volume FFTs — skip it once the
            # inequality constraints are satisfied
            if viol.any():
                grad += beta_quiet * op.conv_adjoint(viol)
        if lam_l2 > 0:
            grad += 2.0 * lam_l2 * y
        if lam_tv > 0:
            _, g_tv = _tv_gradient(y)
            grad += lam_tv * g_tv
        if lam_spectral > 0 and spectral_weight is not None:
            grad += 2.0 * lam_spectral * np.fft.irfft(
                spectral_weight * np.fft.rfft(y, axis=2),
                n=y.shape[2], axis=2,
            )
        x_new = np.clip(y - step * grad - step * alpha, 0.0, None)
        if support_mask is not None:
            x_new = x_new * support_mask
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if verbose and (k % 25 == 0 or k == n_iter - 1):
            loss = 0.5 * float(np.sum(resid**2))
            print(f"  fista iter {k:4d}  data-loss {loss:.4e}  "
                  f"nnz {(x > 0).mean():.3f}")
    return x


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


class _NumpyBoundary:
    """Uniform numpy-in/numpy-out view of a numpy OR torch ZS operator.

    The torch operator's ``conv``/``sample`` methods are tensor-native;
    this shim converts at every call so backend-agnostic stages (the
    sub-bin split) can be written once in numpy.  Costs one host/device
    round trip per call — acceptable for short refinement stages.
    """

    def __init__(self, op):
        self.op = op
        self.q_shape = op.q_shape
        self.n_data = op.n_data
        self._torch = hasattr(op, "to_tensor")
        self.d = op.d.cpu().numpy() if hasattr(op.d, "cpu") else op.d

    def _out(self, t):
        return t.cpu().numpy().astype(np.float64) if self._torch else t

    def _in(self, a):
        return self.op.to_tensor(a) if self._torch else a

    def conv(self, q):
        return self._out(self.op.conv(self._in(q)))

    def conv_adjoint(self, g):
        return self._out(self.op.conv_adjoint(self._in(g)))

    def sample(self, block):
        return self._out(self.op.sample(self._in(block)))

    def sample_adjoint(self, vec):
        return self._out(self.op.sample_adjoint(self._in(vec)))


class _SplitOperator:
    """Adapter: the ZS operator composed with a fixed sub-bin split.

    Presents the :class:`ZSOperator` interface expected by
    :func:`solve_fista` (conv / conv_adjoint / sample / sample_adjoint /
    d / q_shape / lipschitz), with the deposit ``S(u)`` folded into the
    convolution.  ``op`` must already be numpy-boundary (wrap torch
    operators in :class:`_NumpyBoundary` first).
    """

    def __init__(self, op, u: np.ndarray):
        self.op = op
        self.u = u
        self.q_shape = op.q_shape
        self.d = op.d
        self.n_data = op.n_data

    def conv(self, q):
        return self.op.conv(split_deposit(np.asarray(q), self.u))

    def conv_adjoint(self, g):
        return split_adjoint(np.asarray(self.op.conv_adjoint(g)), self.u)

    def sample(self, block):
        return self.op.sample(block)

    def sample_adjoint(self, vec):
        return self.op.sample_adjoint(vec)

    def forward(self, q):
        return self.sample(self.conv(q))

    def adjoint(self, vec):
        return self.conv_adjoint(self.sample_adjoint(vec))

    def lipschitz(self, n_iter: int = 12, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.q_shape)
        x /= np.linalg.norm(x)
        lam = 1.0
        for _ in range(n_iter):
            y = self.adjoint(self.forward(x))
            lam = float(np.linalg.norm(y))
            if lam <= 0:
                return 1.0
            x = y / lam
        return lam


def solve_subbin_positions(
    op,
    q0: np.ndarray,
    *,
    skeleton: np.ndarray,
    n_rounds: int = 3,
    q_iters: int = 60,
    u_iters: int = 8,
    alpha: float | np.ndarray = 0.0,
    beta_quiet: float = 0.0,
    quiet_mask: np.ndarray | None = None,
    quiet_threshold: float = np.inf,
    u_step: float = 1.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine per-charge amplitudes AND bounded sub-bin time offsets.

    Post-ladder stage on a FROZEN skeleton: each active charge gains a
    continuous offset ``u in [-1/2, 1/2]`` bins, represented on the
    coarse grid by the sum-preserving linear split
    (:func:`split_deposit`).  Alternates FISTA amplitude passes (via
    :func:`solve_fista` on the composed operator) with projected-
    gradient offset passes.  The data leverage for ``u`` is the
    fine-tick misalignment of the latch-window edges with the q grid —
    windows sample the two split bins with different fractional
    overlaps, so the likelihood depends on the offsets.

    Motivation (truth-informed ceiling, subbin_ceiling_diag.py):
    per-charge truth-centroid offsets are bimodal at the +-1/2 bound
    (40-47% clip) and halve the ghost fraction at fixed integral.

    Returns ``(q, u)``; the physical time offset is ``u * B`` ticks.
    """
    op = _NumpyBoundary(op)
    q = np.where(skeleton, np.clip(q0, 0.0, None), 0.0)
    u = np.zeros(op.q_shape)
    L = _SplitOperator(op, u).lipschitz()

    def _grid_grad(qq, uu):
        block_pred = op.conv(split_deposit(qq, uu))
        resid = op.sample(block_pred) - op.d
        G = op.conv_adjoint(op.sample_adjoint(resid))
        if beta_quiet > 0 and quiet_mask is not None:
            viol = np.where(
                quiet_mask,
                np.clip(block_pred - quiet_threshold, 0.0, None),
                0.0,
            )
            if viol.any():
                G = G + beta_quiet * op.conv_adjoint(viol)
        loss = 0.5 * float(np.sum(resid**2))
        return G, loss

    for rnd in range(int(n_rounds)):
        # -- amplitude pass on the frozen skeleton, split fixed.
        # ||S(u)|| <= sqrt(2), so 2*L bounds the composed curvature —
        # avoids re-running the power iteration every round.
        sop = _SplitOperator(op, u)
        q = solve_fista(
            sop,
            alpha=alpha,
            beta_quiet=beta_quiet,
            quiet_mask=quiet_mask,
            quiet_threshold=quiet_threshold,
            n_iter=q_iters,
            q0=q,
            L=2.0 * L,
            support_mask=skeleton,
        )
        # -- offset pass, amplitudes fixed: projected gradient with an
        # exact line search along the gradient direction (the loss is
        # piecewise quadratic in u; kinks and bounds are handled by the
        # clip + backtracking guard).
        G, loss_prev = _grid_grad(q, u)
        for _ in range(int(u_iters)):
            Gp = np.zeros_like(G)
            Gm = np.zeros_like(G)
            Gp[:, :, :-1] = G[:, :, 1:]      # G at k+1
            Gm[:, :, 1:] = G[:, :, :-1]      # G at k-1
            right = q * (Gp - G)             # d(loss)/du on the u>0 branch
            left = q * (G - Gm)              # d(loss)/du on the u<0 branch
            g_u = np.where(
                u > 0, right,
                np.where(
                    u < 0, left,
                    np.where(right < 0, right,
                             np.where(left > 0, left, 0.0)),
                ),
            )
            g_u = g_u * skeleton
            gnorm2 = float(np.sum(g_u * g_u))
            if gnorm2 < 1e-20:
                break
            # deposit change per unit step along -g_u: on the active
            # branch d(out)/du_k = q_k (e_{k+s} - e_k), s = branch sign
            s_pos = (u > 0) | ((u == 0) & (right < 0))
            s_neg = (u < 0) | ((u == 0) & ~s_pos & (left > 0))
            w = q * g_u
            D = np.where(s_pos | s_neg, w, 0.0)
            Dsh = np.zeros_like(D)
            Dsh[:, :, 1:] += np.where(s_pos, w, 0.0)[:, :, :-1]
            Dsh[:, :, :-1] += np.where(s_neg, w, 0.0)[:, :, 1:]
            D_grid = D - Dsh                 # = +Sum g_k q_k (e_k - e_{k+s})
            v = op.sample(op.conv(D_grid))
            vnorm2 = float(np.sum(np.asarray(v) ** 2))
            if vnorm2 <= 0:
                break
            t_star = gnorm2 / vnorm2 * float(u_step)
            accepted = False
            for _bt in range(8):
                u_new = np.clip(u - t_star * g_u, -0.5, 0.5)
                u_new[:, :, 0] = np.clip(u_new[:, :, 0], 0.0, 0.5)
                u_new[:, :, -1] = np.clip(u_new[:, :, -1], -0.5, 0.0)
                u_new *= skeleton
                G_new, loss_new = _grid_grad(q, u_new)
                if loss_new <= loss_prev:
                    u, G, loss_prev = u_new, G_new, loss_new
                    accepted = True
                    break
                t_star *= 0.5
            if not accepted:
                break
        if verbose:
            moved = float((np.abs(u[skeleton]) > 1e-3).mean()) * 100
            at_bound = float((np.abs(u[skeleton]) >= 0.5 - 1e-9).mean()) * 100
            print(f"  subbin round {rnd}: data-loss {loss_prev:.4e}, "
                  f"moved {moved:.1f}% of charges "
                  f"({at_bound:.1f}% at bound), "
                  f"total q {q.sum():.1f} ke-")
    return q, u


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


def smear_kernel_gaussian(
    kernel: np.ndarray,
    adc_hold_delay: int,
    sigma_time: float,
    sigma_pixel: float,
    pad_pixel: int = 4,
    pad_time: int = 12,
) -> tuple[np.ndarray, int]:
    """Fold the analysis Gaussian into the response kernel: ``K_eff = K ⊛ G``.

    With ``K_eff`` in the measurement operator, the fitted coefficients ``c``
    parameterize the charge as Gaussian blobs, ``q = G ⊛ c`` — the unknown
    signal is *smeared by construction* to the analysis resolution, so the
    L1 penalty acts on blob amplitudes (deghosting) instead of fighting the
    kernel ambiguity spike-by-spike, and the output is directly comparable
    to the Gaussian-smeared truth.

    The Gaussian's time response is acausal, so the padded kernel is shifted
    by ``time_shift = pad_time // 2`` before smearing and NOT shifted back
    (shifting back would wrap the early tail).  Returns
    ``(smeared_kernel, time_shift)``.  A fitted coefficient at fit-grid
    index ``t`` corresponds to physical block index ``t + time_shift``:
    roll the fitted array by ``+time_shift`` along time to restore block
    alignment (and roll physical-grid warm starts by ``-time_shift`` onto
    the fit grid).

    Spatial padding is symmetric, so the (odd) spatial center convention of
    :class:`ZSOperator` is preserved.
    """
    from .deconv import gaussian_filter_3d

    kx, ky, kt = kernel.shape
    time_shift = pad_time // 2
    out_shape = (kx + 2 * pad_pixel, ky + 2 * pad_pixel, kt + pad_time)
    padded = np.zeros(out_shape, dtype=np.float64)
    padded[
        pad_pixel: pad_pixel + kx,
        pad_pixel: pad_pixel + ky,
        time_shift: time_shift + kt,
    ] = kernel
    F = gaussian_filter_3d(
        out_shape, dt=(1, 1, adc_hold_delay),
        sigma=(sigma_pixel, sigma_pixel, sigma_time),
    )
    axes = (0, 1, 2)
    smeared = fft.irfftn(
        fft.rfftn(padded, axes=axes) * F, s=out_shape, axes=axes
    )
    # preserve total integral exactly (G is normalised, but guard numerics)
    total_in = float(kernel.sum())
    total_out = float(smeared.sum())
    if total_out != 0.0:
        smeared *= total_in / total_out
    return smeared, time_shift


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


def solve_fista_ladder(
    op: ZSOperator,
    alphas: list[float],
    *,
    base_support: np.ndarray | None = None,
    seed_cut: float | None = None,
    seed_dilate: int = 2,
    soft_decay_len: float | None = None,
    soft_exponent: float = 1.0,
    n_iter_per_stage: int = 150,
    q0: np.ndarray | None = None,
    L: float | None = None,
    verbose: bool = False,
    **fista_kwargs,
) -> np.ndarray:
    """Strong-charge-first homotopy: descend an alpha ladder, seeding each
    refinement stage from the previous stage's strong charges.

    Stage 0 solves with the largest ``alphas[0]`` on ``base_support`` — only
    strong, unambiguous charges survive.  Each later stage lowers alpha
    (admitting smaller charges) and, when ``seed_cut`` is given, restricts
    the support to the dilated neighbourhood of the previous solution above
    ``seed_cut`` (intersected with ``base_support``): small charges may only
    appear where the strong-charge skeleton and the residual data demand
    them.  ``seed_cut=None`` anneals alpha on a fixed support instead.

    ``soft_decay_len`` replaces the HARD seeded support with a SOFT
    exponential prior: the stage's L1 weight becomes
    ``alpha * exp(d / soft_decay_len)`` with ``d`` the Manhattan distance
    from the previous stage's strong charges — 'the probability of true
    charge decays exponentially away from the deghosted positions'.
    Charge far from the skeleton is not forbidden, it just needs
    exponentially stronger data evidence to enter.  The base support (if
    given) still applies.

    The warm start of each stage is the previous stage's solution, so
    established strong charges stay stable while the fit refines.
    """
    if not alphas:
        raise ValueError("alphas ladder cannot be empty.")
    if L is None:
        L = op.lipschitz()
    x = q0
    support = base_support
    for k, alpha in enumerate(alphas):
        alpha_eff = alpha
        if seed_cut is not None and x is not None:
            seed = np.asarray(x) > seed_cut
            if soft_decay_len is not None:
                alpha_eff = exponential_alpha_field(
                    seed, alpha, soft_decay_len, exponent=soft_exponent
                )
                support = base_support
            else:
                seeded = _dilate_mask(seed, seed_dilate)
                support = seeded if base_support is None else (seeded & base_support)
        if verbose:
            frac = float(support.mean()) if support is not None else 1.0
            kind = "soft" if (soft_decay_len is not None and k > 0) else "hard"
            print(f"  ladder stage {k}: alpha={alpha} ({kind})  "
                  f"support={100 * frac:.2f}%")
        x = solve_fista(
            op,
            alpha=alpha_eff,
            n_iter=n_iter_per_stage,
            q0=x,
            L=L,
            support_mask=support,
            verbose=verbose,
            **fista_kwargs,
        )
    return x


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


def solve_deghost_regress(
    op: ZSOperator,
    *,
    n_rounds: int = 3,
    alpha_deghost: float = 0.5,
    alpha_regress: float = 0.02,
    seed_cut: float = 0.5,
    decay_len: float = 2.0,
    base_support: np.ndarray | None = None,
    n_iter_deghost: int = 120,
    n_iter_regress: int = 120,
    q0: np.ndarray | None = None,
    L: float | None = None,
    verbose: bool = False,
    **fista_kwargs,
) -> np.ndarray:
    """Alternate the two sub-tasks: DEGHOST (position selection) and
    REGRESS (charge amounts), iteratively.

    Each round:

    1. DEGHOST — strong uniform L1 (``alpha_deghost``): only positions
       with real data evidence survive (a voxel stays active only while
       |A^T residual| > alpha).
    2. REGRESS — refit amplitudes with a small L1 (``alpha_regress``)
       shaped by the exponential soft prior around the surviving
       positions: on the skeleton the amplitudes are nearly unbiased;
       away from it the required evidence grows as exp(d / decay_len).

    The regressed solution feeds the next deghost round, whose residual
    is now computed with realistic amplitudes — ghosts whose apparent
    charge collapses once real charges absorb the data weight get pruned,
    and genuinely demanded new positions can enter.  This closes the
    failure mode of a single-shot debias (noise rectification on the
    fixed support): every regression is followed by another pruning.
    """
    if L is None:
        L = op.lipschitz()
    x = q0
    support = base_support
    for r in range(int(n_rounds)):
        if verbose:
            print(f"  D/R round {r}: deghost alpha={alpha_deghost}")
        x = solve_fista(
            op, alpha=alpha_deghost, n_iter=n_iter_deghost, q0=x, L=L,
            support_mask=support, verbose=verbose, **fista_kwargs,
        )
        seed = x > seed_cut
        alpha_field = exponential_alpha_field(seed, alpha_regress, decay_len)
        if verbose:
            print(f"  D/R round {r}: regress alpha_min={alpha_regress}, "
                  f"skeleton {100 * float(seed.mean()):.3f}%")
        x = solve_fista(
            op, alpha=alpha_field, n_iter=n_iter_regress, q0=x, L=L,
            support_mask=base_support, verbose=verbose, **fista_kwargs,
        )
    return x


def debias_on_support(
    op: ZSOperator,
    q_hat: np.ndarray,
    *,
    support_eps: float = 1e-3,
    n_iter: int = 150,
    beta_quiet: float = 0.0,
    quiet_mask: np.ndarray | None = None,
    quiet_threshold: float = np.inf,
    L: float | None = None,
) -> np.ndarray:
    """Refit with alpha=0 restricted to the L1 solution's active set.

    The L1 soft threshold biases every surviving amplitude down by
    ~alpha/sensitivity; refitting on the (sparse) final support removes
    that bias while keeping the support selection — the standard
    two-stage LASSO debias.
    """
    support = q_hat > support_eps
    return solve_fista(
        op,
        alpha=0.0,
        beta_quiet=beta_quiet,
        quiet_mask=quiet_mask,
        quiet_threshold=quiet_threshold,
        n_iter=n_iter,
        q0=q_hat,
        L=L,
        support_mask=support,
    )


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
