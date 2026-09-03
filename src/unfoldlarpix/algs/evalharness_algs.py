"""Evaluation harness (``algo_plan.md`` Sec. 3) and the empirical resolution matrix.

Three algorithms for the fixed-interval readout, which is the reference case:
the accumulator is sampled on a fixed grid of stride ``B`` fine ticks, there is
no trigger, no reset and no zero suppression, so every record is an exact
linear functional of the fine charge and the estimator is a linear (or, with
positivity, a piecewise-linear) map of the data.

:class:`LinearArms`
    Runs the estimators and publishes their coarse charge arrays.

:class:`ResolutionScore`
    Applies the harness ``xhat = H P xbar`` to every arm, every prolongation
    and every smoothing width, and reports the decomposition
    ``xhat - H x = H P (xbar - R x) + H (P R - I) x`` term by term.

:class:`ResolutionProbe`
    Measures the empirical resolution matrix ``M`` by unit impulses: one fine
    cell of charge in, the estimator's answer out, scored by the same harness.

Nothing here scores a reconstruction against an unsmeared truth and nothing
here uses the legacy ``universal_rebin`` metrics except to print them beside
the new ones, once per arm, for continuity with ``jobs/METHODS.md`` Sec. 6.

Definitions
-----------
Every derived quantity is defined here before it is used.  Names marked NEW
are introduced by this module and are not established terminology.  Charges
are in ke throughout, time in fine ticks of ``Delta_t = 0.05 us``, and
``B = adc_hold_delay = 30`` fine ticks = 1.5 us.

``pads p``
    the operator's transverse block, ``op.q_shape[:2] = (nx, ny)``.  Block
    index ``(ix, iy)`` is hardware pixel ``(ix + block_offset[0],
    iy + block_offset[1])``.

``fine ticks j``
    absolute fine ticks, the same frame as ``event.effq.location[:, 2]`` and
    as the operator's ``block_offset[2]``.

``x_p(j)``
    the fine truth: ``effq`` charge [ke] on pad ``p`` at fine tick ``j``.  The
    tick is the instant the charge crosses the response plane, i.e. ``tau = 0``
    of the field-response kernel (measured exact to better than one fine tick
    in ``exactrows_isoline``).

``c_k``, the coarse cell centres
    ``c_k = block_offset[2] + k * B + release_offset_ticks(store)`` for
    ``k = 0 .. op.q_shape[2] - 1``: the instant at which coarse cell ``k``
    releases its charge under ``TIME_CONVENTION = "release_point"``
    (``model/conventions.py``).  ``release_offset_ticks`` is ``0`` for the
    shipped delta kernel and ``B (S-1) / (2 S)`` for
    ``detector: within_bin: uniform, subbin: S``.

``C_k``, the coarse cells
    ``C_k = [c_k - B/2, c_k + B/2)``, lower edge closed, upper edge open.

``R``, the target coarsening
    ``(R x)_{p,k} = sum_{j in C_k} x_p(j)``.  Transverse: identity on pads.
    NOTE ``R`` as defined here and ``grid_truth(store, op, mode="round")``
    agree on every fine tick EXCEPT one tick in ``B``: ``grid_truth`` uses
    ``np.rint``, which breaks the exact tie at ``j = c_k - B/2`` to the
    nearest EVEN cell index, while ``C_k`` is lower-closed and always sends it
    to ``k``.  The two are reported side by side and their difference is
    measured on the event, never assumed to be zero.

``P``, prolongations (fine x coarse; time axis only, identity on pads)
    ``P_delta[j, k] = 1`` if ``j == floor(c_k + 1/2)``, else 0.  This is what
    ``eval/universal.py:universal_rebin`` deposits before it smears, i.e. the
    convention the production metrics carry.
    ``P_0[j, k] = 1/B`` for ``j in C_k``, else 0.
    ``P_hat[j, k] = (1 - |j - c_k| / B) / B`` for ``|j - c_k| < B``, else 0 --
    a triangle of half-width ``B`` centred on ``c_k``, unit mass on the
    unbounded fine line.
    ``P_1 = P_hat (R P_hat)^{-1}``, the corrected hat of ``algo_plan`` Sec. 4.0.
    ``R P_hat`` is computed numerically, not assumed; on this grid it is
    tridiagonal with entries ``(1/8, 3/4, 1/8)``.
    All three satisfy ``R P = I`` and ``1^T P = 1^T``; both are checked
    numerically and reported.

``prolongation locality`` (NEW)
    ``a_m(k) = sum_{j in C_{k+m}} |P[j, k]|``: the absolute mass that cell
    ``k``'s prolongation places in the cell ``m`` cells away.  ``a_0`` near 1
    and ``a_m`` falling fast is what "local prolongation" means as a number.

``H``, the evaluation smoothing
    transverse box over one pad (identity on pads) times, in time, a Gaussian
    of standard deviation ``sigma_H`` (given in us; ``sigma_H = 0`` means no
    time smoothing, ``H = I``).  Discretised as
    ``g[m] propto exp(-m^2 / (2 sigma^2))`` on ``|m| <= ceil(5 sigma)`` fine
    ticks, normalised to unit sum, and applied by LINEAR convolution with zero
    padding -- never circularly.  Optionally a transverse Gaussian of a stated
    pitch fraction may be enabled for comparison with the legacy metric; it is
    off by default because a box transverse ``H`` is what keeps pads separate,
    which is the whole content of requirement (Q1).

``xhat``, ``e``
    ``xhat = H P xbar`` for a candidate ``(xbar, P)``, and
    ``e = xhat - H x``.  Decomposition, exact and linear:

        e = H P (xbar - R x)  +  H (P R - I) x
            \\_ estimation _/     \\_ representation _/

    Both terms are reported separately at every ``sigma_H``.

``time window [T0, T1)``
    the fine ticks kept in the evaluation: the union of the coarse cells
    ``C_k``, ``k0 <= k <= k1``, with ``k0``/``k1`` the smallest/largest cell
    holding truth, widened by ``margin_windows`` cells each side.  The
    fraction of coarse mass outside it,

        mass_outside_window_frac = sum_{p, k not in [k0,k1]} |xbar_{p,k}|
                                 / sum_{p,k} |xbar_{p,k}| ,

    is reported for EVERY candidate; if it exceeds ``max_outside_frac``
    (default 0.01) the window is widened automatically and the fact is
    recorded in the output.

``E_rel(sigma_H)`` (NEW)
    ``sum_{p,j} |e_p(j)| / sum_{p,j} (H x)_p(j)``: the relative absolute error
    at resolution ``sigma_H``.  The same ratio is formed for each of the two
    terms of the decomposition, with the same denominator.

``line-averaged time profile`` (NEW)
    the mean over the interior line pads (truth pads with ``pixel_y`` inside
    ``line_pixel_y_range``, default 5..131) of ``xhat``, ``H x`` and ``e`` as
    functions of the fine tick.  From it,

        E_max_line = max_j | mean_p e_p(j) | ,
        E_rms_line = sqrt( mean_j ( mean_p e_p(j) )^2 ) ,

    both in ke per fine tick.

``zero-preservation, (Q1)``
    on pads carrying no truth charge, grouped by Chebyshev distance to the
    nearest truth pad (ring 1, ring 2, ring >= 3): ``sum xhat^+`` and
    ``sum xhat^-`` -- the positive and negative parts summed separately -- in
    ke and per pad.

``global conservation``
    ``sum xhat - sum H x`` over the kept window, in ke and relative.

``segment sums``
    a segment is ``segment_pixels`` consecutive ``pixel_y`` (default 7,
    = 3.10 cm at the 0.4434 cm pitch) by ALL pads in ``pixel_x`` by ALL fine
    ticks in the window.  Segments tile the truth's ``pixel_y`` span after
    dropping ``segment_edge_exclude`` pads (default 3) at each end.  Per
    segment, ``(sum_S xhat - sum_S H x) / sum_S H x``; reported with its mean
    and RMS over segments.

``M``, the empirical resolution matrix
    column ``(p*, t*)`` of ``M`` is the estimator's answer to a single fine
    cell of charge ``Q`` at pad ``p*`` and fine tick ``t*``, pushed through
    the same measurement functional the data went through and scored by the
    same harness.  ``ResolutionProbe`` measures the 30 sub-window phases
    ``phi``, ``t* = c_{k*} - B/2 + phi``, on one pad; the operator is
    shift-invariant by ``B`` in time and by one pad transversely, so those 30
    columns characterise ``M``.  Reported per column:

    ``normalisation`` ``sum_{p,j} xhat / Q``, target 1;
    ``time shift``    ``(sum xhat * j / sum xhat) - t*``, in fine ticks;
    ``width``         ``sqrt( sum xhat (j - mean)^2 / sum xhat )``, reported as
                      ``null`` when the weights make the variance negative
                      (they can: ``xhat`` is signed);
    ``out-of-cell``   charge on the probed pad at fine ticks outside
                      ``[c_{k*} - 3B/2, c_{k*} + 3B/2)``, i.e. outside the
                      probed cell and its two neighbours;
    ``out-of-pad``    charge on every other pad, positive and negative parts
                      separately, by Chebyshev ring;
    ``A_alt`` (NEW)   at ``sigma_H = 0``, ``max_k |xbar_{p*,k} - Q delta_{k,k*}| / Q``
                      on the probed pad -- the largest single-cell departure of
                      the raw coarse column from the ideal column.

``current_zero_before_tick``
    tred deletes every current sample before the event time reference
    (``graph_effq.py:148-159``), tick 0 in every campaign.  It is part of the
    realised measurement functional, so the probe's synthetic records carry it
    explicitly: ``A_p(t) = 0`` for ``t < t_z``, and
    ``A_p(t) - sum_j x(j) Kcum(t_z - 1 - j)`` for ``t >= t_z``.

``Kcum``, ``phi``, ``impact index``
    as defined in :mod:`unfoldlarpix.algs.exactrows_algs`; the probe reuses
    that module's kernel loader and ``kcum_at`` rather than re-deriving them.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ..fwk.component import algorithm
from ..model.warm_start import deconv_fft_torch, gaussian_filter_3d_torch
from .exactrows_algs import COLLECTION_PIXEL, _Recorder, kcum_at, load_impact_response
from .fixedgrid_algs import (block_from_rows, fit_bin_ticks, grid_truth, loss,
                             release_offset_ticks, resolve_support,
                             score_universal, solve_arm)

PITCH_CM = 0.4434          # pixel pitch, for the segment length only
TICK_US = 0.05             # fine tick, for sigma_H us -> ticks


# ---------------------------------------------------------------------------
# grid geometry
# ---------------------------------------------------------------------------
def coarse_centers(store, op) -> tuple[np.ndarray, float, float]:
    """``(c_k, B, release_offset)`` in absolute fine ticks.

    ``c_k = block_offset[2] + k B + release_offset_ticks(store)`` -- the ONE
    formula the whole module uses, taken from
    ``model/conventions.py:reco_bin_centers`` under
    ``TIME_CONVENTION = "release_point"``.
    """
    boff = np.asarray(store.get("block_offset"), dtype=float)
    B = float(fit_bin_ticks(store))
    off = float(release_offset_ticks(store))
    n = int(op.q_shape[2])
    return boff[2] + np.arange(n) * B + off, B, off


def coarse_index(j, c0: float, B: float) -> np.ndarray:
    """Index ``k`` of the cell ``C_k = [c_k - B/2, c_k + B/2)`` holding tick ``j``."""
    return np.floor((np.asarray(j, dtype=float) - c0) / B + 0.5).astype(np.int64)


def cell_fine_ticks(c_k: float, B: float) -> np.ndarray:
    """The integer fine ticks in ``[c_k - B/2, c_k + B/2)``."""
    lo = int(np.ceil(c_k - B / 2.0))
    hi = int(np.ceil(c_k + B / 2.0))
    return np.arange(lo, hi)


# ---------------------------------------------------------------------------
# prolongations
# ---------------------------------------------------------------------------
def prolongation(name: str, fine: np.ndarray, c: np.ndarray, B: float,
                 tri_inverse: np.ndarray | None = None) -> np.ndarray:
    """``(n_fine, n_coarse)`` prolongation matrix restricted to ``fine`` rows.

    ``name`` is ``"delta"``, ``"uniform"``, ``"hat"`` (the raw ``P_hat``) or
    ``"corrected_hat"`` (``P_1``, which needs ``tri_inverse`` = ``(R P_hat)^{-1}``).
    """
    nf, nc = len(fine), len(c)
    if name == "delta":
        P = np.zeros((nf, nc))
        tgt = np.floor(c + 0.5).astype(np.int64)
        row = np.searchsorted(fine, tgt)
        ok = (row >= 0) & (row < nf)
        ok &= fine[np.clip(row, 0, nf - 1)] == tgt
        P[row[ok], np.arange(nc)[ok]] = 1.0
        return P
    if name == "uniform":
        P = np.zeros((nf, nc))
        u = (np.asarray(fine, float)[:, None] - c[None, :]) / B
        P[(u >= -0.5) & (u < 0.5)] = 1.0 / B
        return P
    if name in ("hat", "corrected_hat"):
        u = np.abs(np.asarray(fine, float)[:, None] - c[None, :]) / B
        P = np.where(u < 1.0, (1.0 - u) / B, 0.0)
        if name == "hat":
            return P
        if tri_inverse is None:
            raise ValueError("corrected_hat needs tri_inverse = (R P_hat)^-1")
        return P @ tri_inverse
    raise ValueError(f"unknown prolongation {name!r}")


def restriction_of_hat(c: np.ndarray, B: float) -> np.ndarray:
    """``T = R P_hat``, computed numerically on the full fine line of the grid.

    The fine line is the union of the cells ``C_0 .. C_{n-1}``; a hat centred
    on a boundary cell loses the part of its mass that falls outside, so the
    boundary COLUMNS of ``T`` do not sum to one -- which is exactly what makes
    ``1^T P_1 = 1^T`` still hold there (``1^T_coarse T = 1^T_fine P_hat``).
    """
    n = len(c)
    T = np.zeros((n, n))
    for k in range(n):
        j = np.arange(int(np.ceil(c[k] - B)), int(np.floor(c[k] + B)) + 1)
        v = np.maximum(0.0, 1.0 - np.abs(j - c[k]) / B) / B
        kk = coarse_index(j, c[0], B)
        ok = (kk >= 0) & (kk < n) & (v > 0)
        np.add.at(T, (kk[ok], np.full(ok.sum(), k)), v[ok])
    return T


def hat_column_sums(c: np.ndarray, B: float) -> np.ndarray:
    """``1^T P_hat`` on the same full fine line -- the companion of ``T``."""
    n = len(c)
    s = np.zeros(n)
    for k in range(n):
        j = np.arange(int(np.ceil(c[k] - B)), int(np.floor(c[k] + B)) + 1)
        v = np.maximum(0.0, 1.0 - np.abs(j - c[k]) / B) / B
        kk = coarse_index(j, c[0], B)
        s[k] = v[(kk >= 0) & (kk < n)].sum()
    return s


# ---------------------------------------------------------------------------
# the smoothing H
# ---------------------------------------------------------------------------
def time_kernel(sigma_ticks: float, n_sigma: float = 5.0) -> np.ndarray:
    """Unit-sum discrete Gaussian on fine ticks; ``[1.0]`` for ``sigma = 0``."""
    if sigma_ticks <= 0:
        return np.array([1.0])
    half = int(np.ceil(n_sigma * sigma_ticks))
    m = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (m / sigma_ticks) ** 2)
    return g / g.sum()


def smooth_columns(mat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Linear convolution of every column of ``mat`` with ``g``, SAME length.

    Zero padded, never wrapped: ``scipy.signal.fftconvolve`` in ``full`` mode
    followed by the centre slice, which is linear convolution by definition.
    """
    if len(g) == 1:
        return mat
    from scipy.signal import fftconvolve
    full = fftconvolve(mat, g[:, None], mode="full", axes=0)
    off = (len(g) - 1) // 2
    return full[off:off + mat.shape[0]]


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------
class EvalHarness:
    """Geometry, prolongations, ``H`` and the metric, for one event.

    Construction is pure geometry plus the fine truth; no estimator and no
    data enter it, so the same instance scores every arm and every probe.
    """

    def __init__(self, store, op, *, margin_windows: int = 40,
                 line_pixel_y_range=(5, 131), segment_pixels: int = 7,
                 segment_edge_exclude: int = 3,
                 transverse: str | None = None,
                 truth: tuple[np.ndarray, np.ndarray] | None = None):
        self.op = op
        self.c, self.B, self.release_off = coarse_centers(store, op)
        self.n_coarse = len(self.c)
        self.boff = np.asarray(store.get("block_offset"), dtype=float)
        self.nx, self.ny = int(op.q_shape[0]), int(op.q_shape[1])
        self.n_pads = self.nx * self.ny
        self.transverse = transverse

        # ---- fine truth on the pad x fine-tick grid ------------------------
        if truth is None:
            ev = store.get("event")
            el = np.asarray(ev.effq.location)
            eq = np.asarray(ev.effq.data, dtype=float)[:, -1]
            ix = el[:, 0].astype(int) - int(self.boff[0])
            iy = el[:, 1].astype(int) - int(self.boff[1])
            tj = el[:, 2].astype(np.int64)
        else:
            (ix, iy, tj), eq = truth
            ix = np.asarray(ix, dtype=int)
            iy = np.asarray(iy, dtype=int)
            tj = np.asarray(tj, dtype=np.int64)
            eq = np.asarray(eq, dtype=float)
        keep = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.truth_ix, self.truth_iy = ix[keep], iy[keep]
        self.truth_tick, self.truth_q = tj[keep], eq[keep]
        self.truth_total = float(self.truth_q.sum())
        self.truth_dropped = float(eq.sum() - self.truth_total)

        # ---- R x, and grid_truth for comparison ----------------------------
        kk = coarse_index(self.truth_tick, self.c[0], self.B)
        ok = (kk >= 0) & (kk < self.n_coarse)
        self.Rx = np.zeros(op.q_shape)
        np.add.at(self.Rx, (self.truth_ix[ok], self.truth_iy[ok], kk[ok]),
                  self.truth_q[ok])

        # ---- time window ----------------------------------------------------
        occ = np.nonzero(self.Rx.sum(axis=(0, 1)) != 0)[0]
        self.k_truth_lo, self.k_truth_hi = int(occ.min()), int(occ.max())
        self.margin_windows = int(margin_windows)
        self._set_window(self.margin_windows)

        # ---- pad classification ---------------------------------------------
        tp = np.zeros((self.nx, self.ny), dtype=bool)
        tp[self.truth_ix, self.truth_iy] = True
        self.truth_pad = tp
        self.truth_pad_rows = np.nonzero(tp.reshape(-1))[0]
        tpads = np.argwhere(tp)
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny),
                             indexing="ij")
        allp = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        d = np.abs(allp[:, None, :] - tpads[None, :, :]).max(axis=2)
        self.chebyshev = d.min(axis=1)              # per flat pad index
        self.pixel_y_of_pad = allp[:, 1] + int(self.boff[1])
        self.pixel_x_of_pad = allp[:, 0] + int(self.boff[0])

        ylo, yhi = int(line_pixel_y_range[0]), int(line_pixel_y_range[1])
        self.line_rows = np.array(
            [r for r in self.truth_pad_rows
             if ylo <= self.pixel_y_of_pad[r] <= yhi], dtype=np.int64)

        # ---- segments --------------------------------------------------------
        ty = np.unique(self.truth_iy) + int(self.boff[1])
        lo, hi = int(ty.min()) + segment_edge_exclude, \
            int(ty.max()) - segment_edge_exclude
        self.segments = []
        y = lo
        while y + segment_pixels - 1 <= hi:
            rows = np.nonzero((self.pixel_y_of_pad >= y)
                              & (self.pixel_y_of_pad < y + segment_pixels))[0]
            self.segments.append((y, y + segment_pixels - 1, rows))
            y += segment_pixels
        self.segment_pixels = int(segment_pixels)
        self.segment_edge_exclude = int(segment_edge_exclude)

    # -- window -------------------------------------------------------------
    def _set_window(self, margin: int) -> None:
        self.margin_windows = int(margin)
        self.k0 = max(0, self.k_truth_lo - int(margin))
        self.k1 = min(self.n_coarse - 1, self.k_truth_hi + int(margin))
        lo = cell_fine_ticks(self.c[self.k0], self.B)[0]
        hi = cell_fine_ticks(self.c[self.k1], self.B)[-1] + 1
        self.fine = np.arange(lo, hi)
        self.n_fine = len(self.fine)
        self._Pcache: dict = {}
        self._Hxcache: dict = {}

    def mass_outside_window(self, xbar: np.ndarray) -> float:
        """Fraction of ``sum |xbar|`` in coarse cells outside ``[k0, k1]``."""
        a = np.abs(xbar).sum(axis=(0, 1))
        tot = float(a.sum())
        if tot <= 0:
            return 0.0
        return float((tot - a[self.k0:self.k1 + 1].sum()) / tot)

    # -- matrices ------------------------------------------------------------
    def M(self, pname: str, sigma_us: float) -> np.ndarray:
        """``(n_fine, n_coarse)`` matrix of ``H P`` restricted to the window.

        ``P`` is built on an EXTENDED fine grid (the window plus ``5 sigma``
        each side) so the smoothing at the window edge sees the real
        neighbourhood and not a zero-padded artefact; the result is then cropped
        back to the window.
        """
        key = (pname, float(sigma_us))
        if key in self._Pcache:
            return self._Pcache[key]
        sig = float(sigma_us) / TICK_US
        g = time_kernel(sig)
        pad = (len(g) - 1) // 2
        ext = np.arange(self.fine[0] - pad, self.fine[-1] + pad + 1)
        if pname == "corrected_hat":
            P = prolongation(pname, ext, self.c, self.B, self.tri_inverse)
        else:
            P = prolongation(pname, ext, self.c, self.B)
        Msm = smooth_columns(P, g)
        out = np.ascontiguousarray(Msm[pad:pad + self.n_fine])
        self._Pcache[key] = out
        return out

    @property
    def tri_inverse(self) -> np.ndarray:
        if not hasattr(self, "_tri_inv"):
            T = restriction_of_hat(self.c, self.B)
            self._tri = T
            self._tri_inv = np.linalg.inv(T)
        return self._tri_inv

    def prolongation_report(self, k_ref: int | None = None) -> dict:
        """``R P = I``, ``1^T P = 1^T`` and the locality table, all measured."""
        T = restriction_of_hat(self.c, self.B)
        Tinv = self.tri_inverse
        n = self.n_coarse
        k = int(n // 2) if k_ref is None else int(k_ref)
        rep = {
            "R_Phat_diagonal": float(T[k, k]),
            "R_Phat_offdiagonal": [float(T[k - 1, k]), float(T[k + 1, k])],
            "R_Phat_is_tridiagonal_max_outside": float(
                np.abs(T - np.diag(np.diag(T))
                       - np.diag(np.diag(T, 1), 1)
                       - np.diag(np.diag(T, -1), -1)).max()),
            "R_P1_minus_I_max": float(np.abs(T @ Tinv - np.eye(n)).max()),
            "colsum_P1_minus_1_max": float(
                np.abs(hat_column_sums(self.c, self.B) @ Tinv - 1.0).max()),
        }
        # R P = I and 1^T P = 1^T for the two simple prolongations, measured on
        # the full fine line of the grid (not the analysis window).
        full = np.arange(int(cell_fine_ticks(self.c[0], self.B)[0]),
                         int(cell_fine_ticks(self.c[-1], self.B)[-1]) + 1)
        kk = coarse_index(full, self.c[0], self.B)
        for pname in ("delta", "uniform", "corrected_hat"):
            P = (prolongation(pname, full, self.c, self.B, Tinv)
                 if pname == "corrected_hat"
                 else prolongation(pname, full, self.c, self.B))
            RP = np.zeros((n, n))
            np.add.at(RP, kk, P)
            rep[f"{pname}_RP_minus_I_max"] = float(
                np.abs(RP - np.eye(n)).max())
            rep[f"{pname}_colsum_minus_1_max"] = float(
                np.abs(P.sum(axis=0) - 1.0).max())
            if pname == "corrected_hat":
                col = P[:, k]
                loc = {}
                for m in range(-6, 7):
                    sel = kk == (k + m)
                    loc[str(m)] = float(np.abs(col[sel]).sum())
                rep["P1_locality_abs_mass_per_cell"] = loc
                rep["P1_locality_cell"] = k
                rep["P1_column_max"] = float(col.max())
                rep["P1_column_min"] = float(col.min())
                self._P1_column = col
                self._P1_column_ticks = full
        return rep

    # -- H x ------------------------------------------------------------------
    def Hx(self, sigma_us: float) -> np.ndarray:
        """``(n_truth_pads, n_fine)`` smoothed fine truth on the window."""
        key = float(sigma_us)
        if key in self._Hxcache:
            return self._Hxcache[key]
        sig = key / TICK_US
        g = time_kernel(sig)
        pad = (len(g) - 1) // 2
        ext_lo = self.fine[0] - pad
        n_ext = self.n_fine + 2 * pad
        rowmap = {int(r): i for i, r in enumerate(self.truth_pad_rows)}
        X = np.zeros((len(self.truth_pad_rows), n_ext))
        flat = self.truth_ix * self.ny + self.truth_iy
        col = self.truth_tick - ext_lo
        ok = (col >= 0) & (col < n_ext)
        np.add.at(X, ([rowmap[int(f)] for f in flat[ok]], col[ok]),
                  self.truth_q[ok])
        Xs = smooth_columns(X.T, g).T if pad else X
        out = np.ascontiguousarray(Xs[:, pad:pad + self.n_fine])
        self._Hxcache[key] = out
        return out

    def Hx_direct(self, sigma_us: float) -> np.ndarray:
        """``H x`` again, by an INDEPENDENT code path: ``np.convolve`` per pad.

        :meth:`Hx` builds the smoothed truth with a batched FFT convolution;
        this one deposits the same truth and convolves each pad separately with
        ``np.convolve(..., mode="same")``, which for an odd-length kernel is
        centred linear convolution by definition.  The two must agree to
        machine precision -- that is control (i) of the brief, and it tests the
        grid registration and the kernel centring, not just the algebra.
        """
        sig = float(sigma_us) / TICK_US
        g = time_kernel(sig)
        pad = (len(g) - 1) // 2
        ext_lo = self.fine[0] - pad
        n_ext = self.n_fine + 2 * pad
        rowmap = {int(r): i for i, r in enumerate(self.truth_pad_rows)}
        X = np.zeros((len(self.truth_pad_rows), n_ext))
        flat = self.truth_ix * self.ny + self.truth_iy
        col = self.truth_tick - ext_lo
        ok = (col >= 0) & (col < n_ext)
        np.add.at(X, ([rowmap[int(f)] for f in flat[ok]], col[ok]),
                  self.truth_q[ok])
        if pad == 0:
            out = X
        else:
            out = np.stack([np.convolve(row, g, mode="same") for row in X])
        return np.ascontiguousarray(out[:, pad:pad + self.n_fine])

    def control_passthrough(self, sigma_us: float) -> dict:
        """Control (i): the fine truth passed straight through gives ``e = 0``.

        ``xhat = H x`` is taken from :meth:`Hx_direct` and differenced against
        the harness's own reference :meth:`Hx` by exactly the accumulation
        ``measure`` uses.  ``sum |e| / sum H x`` must be zero to machine
        precision.
        """
        Hx = self.Hx(sigma_us)
        Hd = self.Hx_direct(sigma_us)
        e = Hd - Hx
        sh = float(Hx.sum())
        return {"sigma_H_us": float(sigma_us),
                "sum_Hx_ke": sh,
                "sum_Hx_over_sum_x": sh / self.truth_total,
                "sum_abs_e_ke": float(np.abs(e).sum()),
                "max_abs_e_ke": float(np.abs(e).max()),
                "E_rel": float(np.abs(e).sum() / sh) if sh else float("nan")}

    # -- the metric -----------------------------------------------------------
    def measure(self, xbar: np.ndarray, pname: str, sigma_us: float, *,
                with_truth: bool = True, chunk: int = 2000,
                keep_profiles: bool = False) -> dict:
        """Score one candidate ``(xbar, P)`` at one ``sigma_H``.

        ``with_truth=False`` scores ``H P xbar`` against zero, which is what
        the estimation term ``H P (xbar - R x)`` needs.
        """
        Mt = self.M(pname, sigma_us).T                      # (n_coarse, n_fine)
        Xb = np.asarray(xbar, dtype=float).reshape(self.n_pads, self.n_coarse)
        Hx = self.Hx(sigma_us) if with_truth else None
        truth_row_of = {int(r): i for i, r in enumerate(self.truth_pad_rows)}
        line_set = set(int(r) for r in self.line_rows)

        sum_abs_e = 0.0
        sum_e = 0.0
        sum_xhat = 0.0
        sum_hx = float(Hx.sum()) if with_truth else 0.0
        ring_pos = {1: 0.0, 2: 0.0, 3: 0.0}
        ring_neg = {1: 0.0, 2: 0.0, 3: 0.0}
        ring_n = {1: 0, 2: 0, 3: 0}
        seg_hat = np.zeros(len(self.segments))
        seg_hx = np.zeros(len(self.segments))
        pad_total = np.zeros(self.n_pads)
        line_hat = np.zeros(self.n_fine)
        line_hx = np.zeros(self.n_fine)
        line_e = np.zeros(self.n_fine)
        allpad_hat = np.zeros(self.n_fine)
        max_abs_e = 0.0

        for lo in range(0, self.n_pads, chunk):
            hi = min(lo + chunk, self.n_pads)
            xhat = Xb[lo:hi] @ Mt
            sum_xhat += float(xhat.sum())
            allpad_hat += xhat.sum(axis=0)
            pad_total[lo:hi] = xhat.sum(axis=1)
            e = xhat
            if with_truth:
                e = xhat.copy()
                for r in range(lo, hi):
                    i = truth_row_of.get(r)
                    if i is not None:
                        e[r - lo] -= Hx[i]
            sum_abs_e += float(np.abs(e).sum())
            sum_e += float(e.sum())
            max_abs_e = max(max_abs_e, float(np.abs(e).max()))
            # zero-preservation: pads with no truth, by Chebyshev ring
            ch = self.chebyshev[lo:hi]
            notruth = ch > 0
            for ring in (1, 2, 3):
                sel = (ch == ring) if ring < 3 else (ch >= 3)
                sel &= notruth
                if sel.any():
                    blk = xhat[sel]
                    ring_pos[ring] += float(blk[blk > 0].sum())
                    ring_neg[ring] += float(blk[blk < 0].sum())
                    ring_n[ring] += int(sel.sum())
            for r in range(lo, hi):
                if r in line_set:
                    line_hat += xhat[r - lo]
                    line_e += e[r - lo]
                    line_hx += Hx[truth_row_of[r]] if with_truth else 0.0
            del xhat, e

        nline = max(len(self.line_rows), 1)
        line_hat /= nline
        line_hx /= nline
        line_e /= nline
        for i, (_, _, rows) in enumerate(self.segments):
            seg_hat[i] = pad_total[rows].sum()
            if with_truth:
                seg_hx[i] = sum(float(Hx[truth_row_of[int(r)]].sum())
                                for r in rows if int(r) in truth_row_of)

        out = {
            "sigma_H_us": float(sigma_us), "prolongation": pname,
            "sum_xbar_all_cells_ke": float(Xb.sum()),
            "sum_xbar_in_window_ke": float(Xb[:, self.k0:self.k1 + 1].sum()),
            "sum_xhat_ke": sum_xhat, "sum_Hx_ke": sum_hx,
            "sum_abs_e_ke": sum_abs_e, "sum_e_ke": sum_e,
            "max_abs_e_ke_per_tick": max_abs_e,
            "E_rel": (sum_abs_e / sum_hx) if sum_hx else float("nan"),
            "conservation_ke": sum_xhat - sum_hx,
            "conservation_rel": ((sum_xhat - sum_hx) / sum_hx)
            if sum_hx else float("nan"),
            "E_max_line_ke_per_tick": float(np.abs(line_e).max()),
            "E_rms_line_ke_per_tick": float(np.sqrt((line_e ** 2).mean())),
            "n_line_pads": int(len(self.line_rows)),
        }
        z = {}
        for ring in (1, 2, 3):
            lab = f"ring{ring}" if ring < 3 else "ring_ge3"
            n = max(ring_n[ring], 1)
            z[lab] = {"n_pads": ring_n[ring],
                      "sum_pos_ke": ring_pos[ring],
                      "sum_neg_ke": ring_neg[ring],
                      "pos_per_pad_ke": ring_pos[ring] / n,
                      "neg_per_pad_ke": ring_neg[ring] / n}
        out["zero_preservation"] = z
        if with_truth:
            rel = np.where(seg_hx != 0, (seg_hat - seg_hx)
                           / np.where(seg_hx != 0, seg_hx, 1.0), np.nan)
            out["segments"] = {
                "n": len(self.segments),
                "pixels_per_segment": self.segment_pixels,
                "length_cm": self.segment_pixels * PITCH_CM,
                "edge_excluded_pads": self.segment_edge_exclude,
                "rel_error_mean": float(np.nanmean(rel)),
                "rel_error_rms": float(np.sqrt(np.nanmean(rel ** 2))),
                "rel_error_max_abs": float(np.nanmax(np.abs(rel))),
                "rel_error": [float(v) for v in rel]}
        if keep_profiles:
            out["_profiles"] = {"line_xhat": line_hat, "line_Hx": line_hx,
                                "line_e": line_e, "allpad_xhat": allpad_hat}
        return out


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
@algorithm("LinearArms")
class LinearArms(_Recorder):
    """The estimators whose resolution is to be measured, and the two controls.

    Three arms, and the reason each one is here.

    ``LS_nopos`` -- least squares on the support, no positivity, no l1.  It is
    the only arm that is a LINEAR map of the data, so it is the only one whose
    resolution matrix is a matrix: one column measured at charge ``Q`` predicts
    every other charge by scaling, and the impact-averaging error, which is a
    zero-sum dipole, passes through it without rectification
    (``algo_plan`` Sec. 4.4).  Its cost is that it is free to oscillate in
    time, which is exactly the thing ``H`` is supposed to price.

    ``pos_a0`` -- the same system with ``q >= 0`` and no l1.  It is the
    cheapest estimator that violates the linearity assumption, and
    ``jobs/METHODS.md`` Sec. 9 measures its integral bias at +73 % on this
    event.  Including it says whether the smoothing ``H`` prices that bias or
    hides it, which no linear-only study can answer.  Its resolution matrix is
    charge dependent by construction, and the probe reports it as such.

    ``FFT`` -- the unfiltered FFT inverse of the same system.  It is linear,
    it uses no support and no prior at all, and its total charge is pinned to
    ``sum d / sum K`` for any data, so it separates "what the operator can
    invert" from "what the support and the prior contributed".  ``FFT_1p59us``
    adds the legacy analysis filter (``sigma_time`` 0.005, ``sigma_pixel`` 0.5)
    so the published production arm appears in the same table.

    Controls: ``truth_R`` is ``grid_truth(store, op, mode="round")``, the
    production truth-on-grid; ``truth_Rbox`` is ``R x`` with the lower-closed
    cell ``[c_k - B/2, c_k + B/2)``.  They differ only where a fine tick lands
    exactly on a cell boundary; the difference is measured, not assumed away.

    Props
    -----
    arms : list of dict, optional
        ``{label, kind, alpha, positivity, support, iters, sigma_time,
        sigma_pixel}``; ``kind`` is ``solve`` or ``fft``.  Defaults are the
        four arms above.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset",
             "charge_model")
    writes = ("arms.q",)

    DEFAULT_ARMS = [
        {"label": "LS_nopos", "kind": "solve", "alpha": 0.0,
         "positivity": False, "support": "gain:0.5", "iters": 2000},
        {"label": "pos_a0", "kind": "solve", "alpha": 0.0,
         "positivity": True, "support": "gain:0.5", "iters": 2000},
        {"label": "FFT", "kind": "fft", "sigma_time": None,
         "sigma_pixel": 0.0},
        {"label": "FFT_1p59us", "kind": "fft", "sigma_time": 0.005,
         "sigma_pixel": 0.5},
    ]

    def execute(self, store):
        op = store.get("op")
        specs = self.props.get("arms") or self.DEFAULT_ARMS
        qg = grid_truth(store, op, mode="round")
        c, B, off = coarse_centers(store, op)

        arms: dict = {}
        rec_arms = []
        for spec in specs:
            lab = str(spec["label"])
            t0 = time.time()
            if str(spec.get("kind", "solve")) == "fft":
                q = self._fft_arm(store, op, spec)
                declared = spec.get("declared_P", "delta")
                note = ("linear FFT inverse of the same system; "
                        "sum q = sum d / sum K by construction")
            else:
                supp = resolve_support(store, op, spec.get("support"))
                q = solve_arm(op, supp, spec.get("alpha", 0.0),
                              bool(spec.get("positivity", False)),
                              int(spec.get("iters", 2000)))
                declared = spec.get("declared_P", "delta")
                note = (f"FISTA from q0=0, {int(spec.get('iters', 2000))} "
                        f"iterations, support {spec.get('support')}, "
                        f"positivity {bool(spec.get('positivity', False))}")
            wall = time.time() - t0
            arms[lab] = {"q": q, "declared_P": declared, "note": note}
            r = self._row(lab, q, op, wall, spec)
            rec_arms.append(r)
            self._print(r)
            torch.cuda.empty_cache()

        # controls
        Rx = np.zeros(op.q_shape)
        ev = store.get("event")
        el = np.asarray(ev.effq.location)
        eq = np.asarray(ev.effq.data, dtype=float)[:, -1]
        ix = el[:, 0].astype(int) - int(np.asarray(store.get("block_offset"))[0])
        iy = el[:, 1].astype(int) - int(np.asarray(store.get("block_offset"))[1])
        kk = coarse_index(el[:, 2], c[0], B)
        ok = ((ix >= 0) & (ix < op.q_shape[0]) & (iy >= 0)
              & (iy < op.q_shape[1]) & (kk >= 0) & (kk < op.q_shape[2]))
        np.add.at(Rx, (ix[ok], iy[ok], kk[ok]), eq[ok])
        arms["truth_R"] = {"q": qg, "declared_P": "delta",
                           "note": "grid_truth(round): np.rint, ties to even"}
        arms["truth_Rbox"] = {"q": Rx, "declared_P": "delta",
                              "note": "R x with C_k = [c_k - B/2, c_k + B/2)"}
        for lab in ("truth_R", "truth_Rbox"):
            r = self._row(lab, arms[lab]["q"], op, 0.0, {})
            rec_arms.append(r)
            self._print(r)

        diff = qg - Rx
        rec = {
            "release_offset_ticks": off, "fit_bin_ticks": B,
            "n_coarse": int(op.q_shape[2]),
            "coarse_center_first": float(c[0]),
            "truth_convention": {
                "grid_truth_round_total_ke": float(qg.sum()),
                "R_box_total_ke": float(Rx.sum()),
                "max_abs_cell_difference_ke": float(np.abs(diff).max()),
                "sum_abs_cell_difference_ke": float(np.abs(diff).sum()),
                "n_cells_differing": int((np.abs(diff) > 1e-9).sum()),
                "note": ("grid_truth uses np.rint, which sends a fine tick "
                         "landing exactly on c_k - B/2 to the nearest EVEN "
                         "cell index; the box C_k = [c_k - B/2, c_k + B/2) "
                         "always sends it to k.  On an integral release "
                         "offset this is one fine tick in B.")},
            "arms": rec_arms,
        }
        self.put(store, "arms.q", arms)
        self._records.append(rec)
        if not self._recipe:
            self._recipe = {"job_config": store.get("job.config"),
                            "provenance": store.provenance()}

    def _fft_arm(self, store, op, spec):
        B = fit_bin_ticks(store)
        blk = op.to_tensor(block_from_rows(op))
        bs = tuple(blk.shape)
        prep = self.services["detector"].prepared(int(round(B)))
        kern = torch.as_tensor(prep.integrated_response, dtype=op.dtype,
                               device=op.device)
        st_ = spec.get("sigma_time")
        sp = float(spec.get("sigma_pixel", 0.0) or 0.0)
        filt = None
        if st_ is not None:
            filt = gaussian_filter_3d_torch(
                (bs[0] + kern.shape[0] - 1, bs[1] + kern.shape[1] - 1, bs[2]),
                dt=(1, 1, B), sigma=(sp, sp, float(st_)),
                device=op.device, dtype=op.dtype)
        q = deconv_fft_torch(blk, kern, filt).detach().cpu().numpy()
        return q.astype(np.float64)[:, :, :op.q_shape[2]]

    def _row(self, lab, q, op, wall, spec):
        return {"label": lab, "sum_q_ke": float(q.sum()),
                "sum_q_pos_ke": float(q[q > 0].sum()),
                "sum_q_neg_ke": float(q[q < 0].sum()),
                "max_q_ke": float(q.max()), "min_q_ke": float(q.min()),
                "nnz_abs_gt_0p01": int((np.abs(q) > 0.01).sum()),
                "L": loss(op, q), "wall_s": float(wall),
                "spec": {k: v for k, v in spec.items() if k != "label"}}

    def _print(self, r):
        print(f"[{self.name}] {r['label']:12s} sum {r['sum_q_ke']:10.2f} ke  "
              f"q+ {r['sum_q_pos_ke']:10.2f}  q- {r['sum_q_neg_ke']:11.2f}  "
              f"L {r['L']:.5g}  max {r['max_q_ke']:8.2f}  "
              f"min {r['min_q_ke']:8.2f}  wall {r['wall_s']:6.1f} s")


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------
@algorithm("ResolutionScore")
class ResolutionScore(_Recorder):
    """The harness of ``algo_plan.md`` Sec. 3 applied to every arm.

    For each arm, each prolongation ``P`` in ``{delta, uniform,
    corrected_hat}`` and each ``sigma_H``, computes ``xhat = H P xbar``,
    ``e = xhat - H x`` and the two terms of the decomposition, and reports
    ``E_rel``, the line-averaged profiles, the zero-preservation ledger, the
    global conservation residual and the segment sums.  The representation
    term depends only on ``(P, sigma_H)`` and is computed once per pair.

    Two controls are asserted, not merely reported:
    (i) the fine truth passed straight through must give ``e = 0`` exactly;
    (ii) ``xbar = R x`` isolates the representation term, which is a RESULT --
    the resolution cost of a 1.5 us basis -- and not a failure.

    Props
    -----
    sigma_H_us : list of float
        Default ``[0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0]``.
    prolongations : list of str
        Default ``["delta", "uniform", "corrected_hat"]``.
    margin_windows : int
        Half-width of the kept time window, in coarse cells (default 40).
    max_outside_frac : float
        Widen the window automatically if any arm leaves more than this
        fraction of ``sum |xbar|`` outside it (default 0.01).
    line_pixel_y_range, segment_pixels, segment_edge_exclude
    legacy_metrics : bool
        Print and store ``score_universal`` once per arm (default true).
    out_json, out_npz : str
    """

    reads = ("arms.q", "event", "readout_config", "op", "block_offset",
             "charge_model")
    writes = ("eval.score",)

    def execute(self, store):
        op = store.get("op")
        arms = store.get("arms.q")
        sigmas = [float(v) for v in self.props.get(
            "sigma_H_us", [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0])]
        pnames = [str(v) for v in self.props.get(
            "prolongations", ["delta", "uniform", "corrected_hat"])]
        margin = int(self.props.get("margin_windows", 40))
        max_out = float(self.props.get("max_outside_frac", 0.01))

        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))

        # -- time window, widened automatically if any arm needs it ----------
        widened = []
        while True:
            out = {lab: H.mass_outside_window(a["q"]) for lab, a in arms.items()}
            worst = max(out.values())
            if worst <= max_out or H.k0 == 0 or H.k1 == H.n_coarse - 1:
                break
            widened.append({"margin_windows": H.margin_windows,
                            "worst_outside_frac": worst})
            H._set_window(H.margin_windows * 2)
        rec: dict = {
            "window": {
                "margin_windows": H.margin_windows,
                "coarse_cells": [H.k0, H.k1],
                "fine_ticks": [int(H.fine[0]), int(H.fine[-1]) + 1],
                "n_fine": int(H.n_fine),
                "truth_cells": [H.k_truth_lo, H.k_truth_hi],
                "widened": widened,
                "max_outside_frac_allowed": max_out,
                "mass_outside_window_frac": {k: float(v)
                                             for k, v in out.items()}},
            "truth": {"total_ke": H.truth_total,
                      "dropped_outside_block_ke": H.truth_dropped,
                      "n_truth_pads": int(len(H.truth_pad_rows)),
                      "n_line_pads": int(len(H.line_rows))},
            "prolongation_checks": H.prolongation_report(),
        }
        print(f"[{self.name}] window cells {H.k0}..{H.k1}, fine ticks "
              f"{H.fine[0]}..{H.fine[-1]}, n_fine {H.n_fine}; "
              f"worst outside fraction {max(out.values()):.5f}")
        pc = rec["prolongation_checks"]
        print(f"[{self.name}] R P_hat diag {pc['R_Phat_diagonal']:.6f} "
              f"offdiag {pc['R_Phat_offdiagonal']}  "
              f"|R P1 - I| {pc['R_P1_minus_I_max']:.3e}  "
              f"|1^T P1 - 1| {pc['colsum_P1_minus_1_max']:.3e}")

        # -- control (i): the fine truth passed straight through --------------
        ctrl = []
        for s in sigmas:
            cp = H.control_passthrough(s)
            ctrl.append(cp)
            print(f"[{self.name}] control passthrough sigma_H {s:4.2f} us: "
                  f"sum|e| {cp['sum_abs_e_ke']:.3e} ke, E_rel {cp['E_rel']:.3e},"
                  f" sum Hx / sum x {cp['sum_Hx_over_sum_x']:.9f}")
            if cp["E_rel"] > 1e-12:
                raise AssertionError(
                    f"fine passthrough control failed at sigma_H={s}: "
                    f"E_rel {cp['E_rel']:.3e} > 1e-12")
        rec["control_fine_passthrough"] = ctrl

        arrays: dict = {}
        results = []
        for s in sigmas:
            for pname in pnames:
                t0 = time.time()
                repr_term = H.measure(H.Rx, pname, s, with_truth=True,
                                      keep_profiles=True)
                rp = repr_term.pop("_profiles")
                arrays[f"prof_repr_{pname}_s{s:g}_line_xhat"] = \
                    rp["line_xhat"].astype(np.float32)
                arrays[f"prof_repr_{pname}_s{s:g}_line_Hx"] = \
                    rp["line_Hx"].astype(np.float32)
                arrays[f"prof_repr_{pname}_s{s:g}_line_e"] = \
                    rp["line_e"].astype(np.float32)
                results.append({"arm": "representation_term", **repr_term})
                self._print_row(results[-1])
                for lab, a in arms.items():
                    full = H.measure(a["q"], pname, s, with_truth=True,
                                     keep_profiles=True)
                    pr = full.pop("_profiles")
                    est = H.measure(a["q"] - H.Rx, pname, s, with_truth=False)
                    full["arm"] = lab
                    full["declared_P"] = a["declared_P"]
                    full["estimation_term_sum_abs_ke"] = est["sum_abs_e_ke"]
                    full["estimation_term_E_rel"] = (
                        est["sum_abs_e_ke"] / full["sum_Hx_ke"]
                        if full["sum_Hx_ke"] else float("nan"))
                    full["representation_term_sum_abs_ke"] = \
                        repr_term["sum_abs_e_ke"]
                    full["representation_term_E_rel"] = repr_term["E_rel"]
                    full["mass_outside_window_frac"] = float(out[lab])
                    # the decomposition is exact and linear, so the SIGNED
                    # sums must add: sum e = sum e_est + sum e_repr.
                    full["decomposition_signed_sum_residual_ke"] = float(
                        full["sum_e_ke"] - est["sum_e_ke"]
                        - repr_term["sum_e_ke"])
                    results.append(full)
                    self._print_row(full)
                    tag = f"{lab}_{pname}_s{s:g}"
                    arrays[f"prof_{tag}_line_xhat"] = \
                        pr["line_xhat"].astype(np.float32)
                    arrays[f"prof_{tag}_line_Hx"] = \
                        pr["line_Hx"].astype(np.float32)
                    arrays[f"prof_{tag}_line_e"] = \
                        pr["line_e"].astype(np.float32)
                    if s == 0.0 and pname == pnames[0]:
                        arrays[f"prof_{lab}_allpad_s0"] = \
                            pr["allpad_xhat"].astype(np.float32)
                print(f"[{self.name}]   sigma_H {s:4.2f} us  P {pname:14s} "
                      f"{time.time() - t0:6.1f} s")
        rec["rows"] = results

        # -- the coarse oscillation, in ke per 1.5 us -------------------------
        osc = {}
        ka = int(np.argmax(np.abs(H.Rx).sum(axis=(0, 1))))
        ks = list(range(max(ka - 5, 0), min(ka + 6, H.n_coarse)))
        osc["cells"] = ks
        osc["cell_center_ticks"] = [float(H.c[k]) for k in ks]
        osc["truth_Rbox_ke"] = [float(H.Rx[:, :, k].sum()) for k in ks]
        for lab, a in arms.items():
            osc[lab + "_ke"] = [float(a["q"][:, :, k].sum()) for k in ks]
            osc[lab + "_minus_truth_ke"] = [
                float(a["q"][:, :, k].sum() - H.Rx[:, :, k].sum()) for k in ks]
        rec["coarse_pad_summed_profile"] = osc
        arrays["coarse_cells"] = np.asarray(ks)
        for lab, a in arms.items():
            arrays["coarse_padsum_" + lab] = \
                a["q"].sum(axis=(0, 1)).astype(np.float64)
        arrays["coarse_padsum_truth_Rbox"] = \
            H.Rx.sum(axis=(0, 1)).astype(np.float64)
        arrays["fine_ticks"] = H.fine.astype(np.int64)
        if hasattr(H, "_P1_column"):
            arrays["P1_column"] = H._P1_column.astype(np.float64)
            arrays["P1_column_ticks"] = H._P1_column_ticks.astype(np.int64)

        # -- legacy metrics, once per arm -------------------------------------
        if bool(self.props.get("legacy_metrics", True)):
            leg = {}
            for lab, a in arms.items():
                sc = score_universal(store, op, a["q"])
                sc.pop("transport", None)
                leg[lab] = {k: sc[k] for k in
                            ("integral_pct", "pearson_r", "slope",
                             "ghost_charge", "ghost_iso_charge", "true_killed")
                            if k in sc}
                print(f"[{self.name}] legacy {lab:12s} int% "
                      f"{leg[lab]['integral_pct']:+8.2f} r "
                      f"{leg[lab]['pearson_r']:+.4f} slope "
                      f"{leg[lab]['slope']:+.4f} ghostQ "
                      f"{leg[lab]['ghost_charge']:9.1f} killed "
                      f"{leg[lab]['true_killed']:8.1f}")
            rec["legacy_universal"] = leg

        self._emit(store, rec, arrays)

    def _print_row(self, r):
        print(f"[{self.name}] s{r['sigma_H_us']:4.2f} {r['prolongation']:14s} "
              f"{r['arm']:20s} E_rel {r['E_rel']:9.5f}  "
              f"cons {r['conservation_rel']:+9.5f}  "
              f"Emax_line {r['E_max_line_ke_per_tick']:9.4f}  "
              f"seg mean {r.get('segments', {}).get('rel_error_mean', float('nan')):+8.5f}"
              f" rms {r.get('segments', {}).get('rel_error_rms', float('nan')):8.5f}")


# ---------------------------------------------------------------------------
# Algorithm 3
# ---------------------------------------------------------------------------
def row_lookup(store, op) -> dict:
    """``(px_block, py_block, t_hi_block) -> operator row index``.

    ``row_meta`` is published by ``BuildMeasurement`` with the SAME filter
    ``windows_to_sampling`` applies, so index ``r`` here is row ``r`` of
    ``op``.  ``t_hi`` is the window's upper edge in block-local fine ticks,
    which for a fixed-interval readout is the latch time.
    """
    rm = store.get("row_meta")
    return {(int(px), int(py), int(round(th))): r
            for r, (px, py, th) in enumerate(
                zip(rm["px"], rm["py"], rm["t_hi"]))}


def records_from_impulse(kcum: np.ndarray, offsets: np.ndarray,
                         latch_abs: np.ndarray, t_star: int, Q: float,
                         t_zero: int | None) -> np.ndarray:
    """Window charges ``y[d, k]`` on pads at pixel offsets ``offsets`` from ``p*``.

    ``kcum`` is ``(25, 25, Nt)``, the cumulative kernel indexed
    ``[12 + dx, 12 + dy]``.  ``A_p(l) = Q Kcum_{p-p*}(l - t*)``, with the tred
    truncation applied when ``t_zero`` is not ``None``:
    ``A_p(l) = 0`` for ``l < t_zero`` and ``A_p(l) - Q Kcum(t_zero - 1 - t*)``
    otherwise.  ``y[d, k] = A(l_k) - A(l_{k-1})`` for ``k = 1 .. len(latch)-1``.
    """
    c = COLLECTION_PIXEL
    kc = kcum[c + offsets[:, 0], c + offsets[:, 1]]        # (nd, Nt)
    arg = np.asarray(latch_abs) - t_star                  # (n_latch,)
    A = Q * kcum_at(kc, arg)                              # (nd, n_latch)
    if t_zero is not None:
        ped = Q * kcum_at(kc, np.array(t_zero - 1 - t_star))[:, None]
        A = A - ped
        A[:, latch_abs < t_zero] = 0.0
    return np.diff(A, axis=1)


@algorithm("ResolutionProbe")
class ResolutionProbe(_Recorder):
    """Empirical resolution matrix by unit impulses, scored by the harness.

    For each probe the fine charge is ``x = Q e_{p*, t*}`` with ``Q`` stated
    (default 30 ke).  Its records are built with the EXACT-FUNCTIONAL forward
    of :mod:`unfoldlarpix.algs.exactrows_algs` -- the cumulative kernel
    evaluated at the event's own latch times, including tred's
    ``current_zero_before_tick`` deletion -- so the probe sees the same
    data-generating process as the data.  Those records are mapped onto the
    operator's rows through ``row_meta``, and the mapping is verified against
    the REAL event's ``op.d`` before any probe is solved; the check is in the
    output.

    The linear arm's answer scales with ``Q``; the positivity arm's does not,
    so its column is only valid at the ``Q`` it was measured at.

    Props
    -----
    probe_pad : [pixel_x, pixel_y]
        Hardware pixel (default ``[141, 68]``).
    probe_phases : list of int
        Sub-window phases ``phi``; ``t* = c_{k*} - B/2 + phi``.
        Default ``[0, 5, 10, 15, 20, 25]``; ``range(30)`` is the full set.
    probe_impacts_at : dict
        ``{phi: [kernel, ...]}``; kernel is ``"bar"`` (impact-averaged) or
        ``"ix,iy"`` (one impact cell).  Default ``{15: [bar, 4,4, 0,0]}``.
    probe_cell : int, optional
        ``k*``; defaults to the cell carrying most truth charge.
    charge_ke : float
        ``Q`` (default 30.0).
    arms : list of dict
        Same schema as :class:`LinearArms`; default ``LS_nopos`` and ``FFT``.
    sigma_H_us, prolongations, margin_windows : as ResolutionScore.
    current_zero_before_tick : int or null (default 0).
    out_json, out_npz : str
    """

    reads = ("op", "support", "readout_config", "block_offset",
             "charge_model", "row_meta", "hits_view", "event")
    writes = ("eval.resolution",)

    DEFAULT_ARMS = [
        {"label": "LS_nopos", "kind": "solve", "alpha": 0.0,
         "positivity": False, "support": "gain:0.5", "iters": 2000},
        {"label": "FFT", "kind": "fft", "sigma_time": None,
         "sigma_pixel": 0.0},
    ]

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        c, B, off = coarse_centers(store, op)
        Bi = int(round(B))
        Q = float(self.props.get("charge_ke", 30.0))
        tz = self.props.get("current_zero_before_tick", 0)
        tz = None if tz is None else int(tz)
        phases = [int(v) for v in self.props.get(
            "probe_phases", [0, 5, 10, 15, 20, 25])]
        imp_at = {int(k): [str(x) for x in v] for k, v in
                  (self.props.get("probe_impacts_at")
                   or {15: ["bar", "4,4", "0,0"]}).items()}
        ppad = [int(v) for v in self.props.get("probe_pad", [141, 68])]
        specs = self.props.get("arms") or self.DEFAULT_ARMS
        sigmas = [float(v) for v in self.props.get(
            "sigma_H_us", [0.0, 0.5, 1.0, 1.5, 3.0, 6.0])]
        pnames = [str(v) for v in self.props.get(
            "prolongations", ["corrected_hat"])]
        margin = int(self.props.get("margin_windows", 40))

        # -- row mapping, verified on the real event --------------------------
        look = row_lookup(store, op)
        hv = store.get("hits_view")
        loc = np.asarray(hv.location)
        Cq = np.asarray(hv.cumulative_charges, dtype=np.float64)
        yreal = np.diff(np.concatenate([np.zeros((len(Cq), 1)), Cq], axis=1),
                        axis=1)
        N = yreal.shape[1]
        trig = int(np.unique(hv.trigger)[0])
        d_check = np.zeros(op.n_data)
        miss = 0
        for i in range(len(loc)):
            px = int(loc[i, 0] - boff[0]); py = int(loc[i, 1] - boff[1])
            for k in range(1, N + 1):
                key = (px, py, int(trig + k * Bi - boff[2]))
                r = look.get(key)
                if r is None:
                    miss += 1
                    continue
                d_check[r] = yreal[i, k - 1]
        d_op = op.d.detach().cpu().numpy().astype(np.float64)
        map_err = float(np.abs(d_check - d_op).max())
        rec: dict = {
            "row_mapping_check": {
                "n_rows": int(op.n_data), "n_unmapped_windows": int(miss),
                "max_abs_difference_ke": map_err,
            "max_abs_difference_relative_to_max_d": float(
                map_err / max(float(np.abs(d_op).max()), 1e-30)),
            "note": ("op.d is stored float32, so the residual floor of this "
                     "check is the float32 rounding of the largest record"),
                "sum_d_op_ke": float(d_op.sum()),
                "sum_d_mapped_ke": float(d_check.sum()),
                "tolerance": 1e-6,
                "passed": bool(map_err < 1e-6 and miss == 0)},
            "probe_charge_ke": Q,
            "current_zero_before_tick": tz,
            "probe_pad_pixel": ppad,
        }
        print(f"[{self.name}] row mapping: max|d_mapped - op.d| = "
              f"{map_err:.3e} ke over {op.n_data} rows, {miss} unmapped")
        if not rec["row_mapping_check"]["passed"]:
            raise AssertionError(
                f"row_meta -> window mapping does not reproduce op.d "
                f"(max {map_err:.3e} ke, {miss} unmapped)")

        # -- kernels -----------------------------------------------------------
        path = self.props.get("response")
        if path is None:
            path = self.services["detector"].response_path
        need = {"bar"}
        for v in imp_at.values():
            need |= set(v)
        R, meta = load_impact_response(str(path))
        dt = meta["time_tick_us"]
        kcums = {}
        if "bar" in need:
            kcums["bar"] = np.cumsum(R.mean(axis=(1, 3), dtype=np.float64),
                                     axis=-1) * dt
        for nm in sorted(need - {"bar"}):
            ix, iy = (int(v) for v in nm.split(","))
            kcums[nm] = np.cumsum(R[:, ix, :, iy, :].astype(np.float64),
                                  axis=-1) * dt
        del R

        # -- probe geometry -----------------------------------------------------
        kstar = self.props.get("probe_cell")
        if kstar is None:
            qg = grid_truth(store, op, mode="round")
            kstar = int(np.argmax(qg.sum(axis=(0, 1))))
        kstar = int(kstar)
        px_b = ppad[0] - int(boff[0]); py_b = ppad[1] - int(boff[1])
        ring = 12
        dxy = np.array([(dx, dy) for dx in range(-ring, ring + 1)
                        for dy in range(-ring, ring + 1)], dtype=int)
        pads_abs = np.stack([dxy[:, 0] + px_b, dxy[:, 1] + py_b], axis=1)
        inblk = ((pads_abs[:, 0] >= 0) & (pads_abs[:, 0] < op.q_shape[0])
                 & (pads_abs[:, 1] >= 0) & (pads_abs[:, 1] < op.q_shape[1]))
        dxy, pads_abs = dxy[inblk], pads_abs[inblk]
        latch_abs = trig + np.arange(N + 1) * Bi
        rows_for = []
        for (bx, by) in pads_abs:
            rr = np.full(N, -1, dtype=np.int64)
            for k in range(1, N + 1):
                r = look.get((int(bx), int(by), int(trig + k * Bi - boff[2])))
                if r is not None:
                    rr[k - 1] = r
            rows_for.append(rr)
        rows_for = np.asarray(rows_for)
        n_unmapped = int((rows_for < 0).sum())
        rec["probe_geometry"] = {
            "probe_cell_k": kstar, "cell_center_tick": float(c[kstar]),
            "n_pads_receiving": int(len(pads_abs)),
            "n_probe_windows_unmapped": n_unmapped,
            "latch_first": int(latch_abs[0]), "latch_last": int(latch_abs[-1]),
            "n_latches": int(N + 1)}

        supports = {}
        results = []
        n_total = sum(len(specs) * (1 if ph not in imp_at else len(imp_at[ph]))
                      for ph in phases)
        done, t_start = 0, time.time()
        arrays: dict = {}
        for phi in phases:
            kern_names = imp_at.get(phi, ["bar"])
            t_star = int(np.floor(c[kstar] - B / 2.0)) + int(phi)
            for kn in kern_names:
                y = records_from_impulse(kcums[kn], dxy, latch_abs, t_star,
                                         Q, tz)
                dvec = np.zeros(op.n_data)
                ok = rows_for >= 0
                dvec[rows_for[ok]] = y[ok]
                op_p = copy.copy(op)
                op_p.d = torch.as_tensor(dvec, dtype=op.dtype,
                                         device=op.device)
                truth = ((np.array([px_b]), np.array([py_b]),
                          np.array([t_star])), np.array([Q]))
                Hh = EvalHarness(store, op_p, margin_windows=margin,
                                 line_pixel_y_range=(-10 ** 9, 10 ** 9),
                                 truth=truth)
                for spec in specs:
                    lab = str(spec["label"])
                    t0 = time.time()
                    if str(spec.get("kind", "solve")) == "fft":
                        q = LinearArms._fft_arm(self, store, op_p, spec)
                    else:
                        ss = spec.get("support")
                        if ss not in supports:
                            supports[ss] = resolve_support(store, op, ss)
                        q = solve_arm(op_p, supports[ss],
                                      spec.get("alpha", 0.0),
                                      bool(spec.get("positivity", False)),
                                      int(spec.get("iters", 2000)))
                    wall = time.time() - t0
                    r = self._score_probe(Hh, q, sigmas, pnames, Q, t_star,
                                          kstar, px_b, py_b, B)
                    r.update({"phi": int(phi), "kernel": kn, "arm": lab,
                              "t_star": int(t_star), "wall_s": float(wall),
                              "sum_d_probe_ke": float(dvec.sum()),
                              "sum_abs_d_probe_ke": float(np.abs(dvec).sum()),
                              "L": loss(op_p, q)})
                    results.append(r)
                    arrays[f"coarse_col_phi{phi}_{kn}_{lab}"] = \
                        q[px_b, py_b, :].astype(np.float64)
                    done += 1
                    el = time.time() - t_start
                    eta = el / done * (n_total - done)
                    print(f"[{self.name}] phi {phi:2d} {kn:5s} {lab:10s} "
                          f"norm {r['sigmas'][0]['normalisation']:8.4f} "
                          f"shift {r['sigmas'][0]['shift_ticks']:+8.2f} "
                          f"A_alt {r['A_alt']:8.4f} | {done}/{n_total} "
                          f"elapsed {el / 60:.1f} min ETA {eta / 60:.1f} min")
                    torch.cuda.empty_cache()
                del op_p, Hh
        rec["probes"] = results
        rec["sigma_H_us"] = sigmas
        rec["prolongations"] = pnames
        self._emit(store, rec, arrays)

    def _score_probe(self, H, q, sigmas, pnames, Q, t_star, kstar,
                     px_b, py_b, B) -> dict:
        out = {"sigmas": [], "mass_outside_window_frac":
               float(H.mass_outside_window(q))}
        col = q[px_b, py_b, :]
        ideal = np.zeros_like(col); ideal[kstar] = Q
        out["A_alt"] = float(np.abs(col - ideal).max() / Q)
        lo, hi = max(kstar - 4, 0), min(kstar + 4, len(col))
        out["coarse_column_cells"] = list(range(lo, hi))
        out["coarse_column_ke"] = [float(v) for v in col[lo:hi]]
        out["coarse_column_over_Q"] = [float(v / Q) for v in col[lo:hi]]
        pad_flat = px_b * H.ny + py_b
        for pn in pnames:
            for s in sigmas:
                Mt = H.M(pn, s).T
                Xb = np.asarray(q, dtype=float).reshape(H.n_pads, H.n_coarse)
                xhat_pad = Xb[pad_flat] @ Mt
                tot_all = 0.0
                ringp = {1: 0.0, 2: 0.0, 3: 0.0}
                ringn = {1: 0.0, 2: 0.0, 3: 0.0}
                mom0 = mom1 = mom2 = 0.0
                # moments are taken in the coordinate u = j - t*, which keeps
                # the second moment free of the cancellation that j ~ -1740
                # would otherwise force.
                u = H.fine.astype(float) - float(t_star)
                for lo_ in range(0, H.n_pads, 2000):
                    hi_ = min(lo_ + 2000, H.n_pads)
                    xh = Xb[lo_:hi_] @ Mt
                    tot_all += float(xh.sum())
                    w = xh.sum(axis=0)
                    mom0 += float(w.sum())
                    mom1 += float((w * u).sum())
                    mom2 += float((w * u ** 2).sum())
                    ch = H.chebyshev[lo_:hi_]
                    for rr in (1, 2, 3):
                        sel = (ch == rr) if rr < 3 else (ch >= 3)
                        if sel.any():
                            blk = xh[sel]
                            ringp[rr] += float(blk[blk > 0].sum())
                            ringn[rr] += float(blk[blk < 0].sum())
                    del xh
                mean = mom1 / mom0 if mom0 else float("nan")   # ticks from t*
                var = mom2 / mom0 - mean ** 2 if mom0 else float("nan")
                cw = cell_fine_ticks(H.c[kstar], H.B)
                in3 = (H.fine >= cw[0] - H.B) & (H.fine < cw[-1] + 1 + H.B)
                out["sigmas"].append({
                    "prolongation": pn, "sigma_H_us": s,
                    "normalisation": tot_all / Q,
                    "own_pad_sum_over_Q": float(xhat_pad.sum() / Q),
                    "shift_ticks": mean,
                    "shift_us": mean * TICK_US,
                    "width_ticks": (float(np.sqrt(var)) if var >= 0
                                    else None),
                    "own_pad_outside_3cells_over_Q":
                        float(xhat_pad[~in3].sum() / Q),
                    "other_pads_pos_over_Q":
                        {f"ring{r}" if r < 3 else "ring_ge3":
                         ringp[r] / Q for r in (1, 2, 3)},
                    "other_pads_neg_over_Q":
                        {f"ring{r}" if r < 3 else "ring_ge3":
                         ringn[r] / Q for r in (1, 2, 3)},
                })
        return out
