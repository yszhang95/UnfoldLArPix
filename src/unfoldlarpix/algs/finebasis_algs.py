"""Fine-binned (50 ns) linear operator and its closed-form Tikhonov inverse.

The production operator carries one unknown per pad per 1.5 us bin.  This
module carries one unknown per pad per 50 ns fine tick and inverts the
resulting rank-deficient system in closed form, so that the two estimators can
be compared at a stated resolution ``sigma_H`` on the same event, with the same
records, the same evaluation map and the same metrics.

:class:`FineBasisInverse`
    Builds the fine operator, validates it against the exact-functional direct
    sum and against its own adjoint, scans the Tikhonov floor, and scores every
    arm (fine and coarse) with the evaluation harness of
    :mod:`unfoldlarpix.algs.evalharness_algs`.

:class:`FineBasisPlots`
    Reads the products of the same job and writes the figures at ``finalize``.

:class:`FineBasisProbe`
    Resolution matrix of the two estimators by unit fine impulses, using the
    same exact-functional record generator as
    :class:`~unfoldlarpix.algs.evalharness_algs.ResolutionProbe`.

:class:`FineBasisProbePlots`
    Figures for the probe.

Definitions
-----------
Charges are in ke, time in fine ticks of ``Delta_t = 0.05 us``,
``B = adc_hold_delay = 30`` fine ticks = 1.5 us.  Names marked NEW are
introduced here and are not established terminology.  Everything defined in
the module docstrings of :mod:`unfoldlarpix.algs.exactrows_algs` and
:mod:`unfoldlarpix.algs.evalharness_algs` (``Kcum``, ``phi``, ``c_k``, ``C_k``,
``R``, ``P_delta``/``P_0``/``P_1``, ``H``, ``E_rel``, the decomposition, the
zero-preservation ledger, the segment sums) is used with the same meaning and
is not repeated.

``b``, the block time origin
    ``b = block_offset[2]``, a multiple of ``B`` in absolute fine ticks
    (``-7050`` on this event).  The block has ``M = block_shape[2]`` record
    windows and ``N = 30 M`` fine ticks.

``window w``
    record window ``w`` of the block covers the absolute fine ticks
    ``(b + 30 w, b + 30 (w+1)]`` -- lower edge OPEN, upper edge CLOSED.  tred's
    accumulator is an INCLUSIVE cumulative sum sampled at the latch times
    (``tred/readout.py:fixed_interval_readout``), so the charge of a window is
    the current summed over ``b + 30w + 1 ... b + 30(w+1)``.  This is the
    convention that makes ``FORWARD_MODEL_revised.md`` Sec. 3.5's tables come
    out at arrival phase ``phi = 1`` (``exactrows_isoline`` A.2) and that makes
    the production operator invert a fine impulse exactly at ``phi = 16``
    (``evalharness_isoline`` G.1).

``padding pads``
    the operator's transverse FFT grid is ``(nx + 24, ny + 24) = (73, 188)``
    for a ``(49, 164)`` block: 12 padding pixels each side, which carry no
    records.  Both operators here treat the padding pads' records as EXACTLY
    ZERO, which is what the production FFT arm does (``block_from_rows``
    leaves them at zero).  It is a model statement, not a measurement: a pad
    just outside the block does induce charge, and this operator asserts it
    records none.

``Kbar_d(tau)``, ``Kcumbar_d(tau)``
    the impact-averaged response and its inclusive cumulative, as in
    ``exactrows_algs``.  ``Kbar`` is the charge induced on the pad at offset
    ``d`` during response tick ``tau`` by one unit of charge crossing the
    response plane at ``tau = 0``; ``sum_{d,tau} Kbar_d(tau) = 1.000266``.

``h_d(tau)`` (NEW), the fine window function
    ``h_d(tau) = Kcumbar_d(tau) - Kcumbar_d(tau - 30)
              = sum_{m = tau-29}^{tau} Kbar_d(m)`` ,
    the 30-tick moving sum of the impact-averaged response.  Support
    ``tau = 0 .. 3929``.  It is what one window of the readout sees from one
    unit of charge released ``tau`` fine ticks before that window's latch.

``A_fine`` (the fine-binned operator)
    unknown ``x_p(j)`` on every fine tick ``j`` of the block,
    ``N = 30 M`` per pad.  Forward

        y_p[w] = sum_{p'} sum_j x_{p'}(j) h_{p-p'}( b + 30(w+1) - j ) ,

    i.e. a 3-D convolution with ``h`` followed by 30-fold decimation in time,
    ``A_fine = D_30 . conv(h)``.  Implemented as a CIRCULAR convolution on the
    block's own periodic grid ``(73, 188, N)``; see ``circularity`` below.
    Adjoint: zero-insertion upsampling (record ``w`` placed at fine tick
    ``30(w+1) mod N``) followed by circular correlation with ``h``.

``A_coarse`` (the production bin-integrated operator)
    unknown ``q_p[k]``, one per pad per coarse bin; ``A_coarse = Sel .
    conv(Kbar_bin)`` with ``Kbar_bin[d, j] = sum_{tau in [30j, 30(j+1))}
    Kbar_d(tau)`` (``integrate_kernel_over_time(full_response, 30)``, shape
    ``25 x 25 x 130``), the FFT convolution of
    :class:`~unfoldlarpix.model.operator.ZSOperator`, and a selection sampling
    whose every weight is exactly 1 under a fixed-interval readout.
    Because the record window is ``(b + 30w, b + 30(w+1)]`` while
    ``Kbar_bin[m]`` integrates ``[30m, 30m + 30)``, cell ``k`` of ``A_coarse``
    is EXACTLY the fine delta at ``j = c_k + 1``:

        A_coarse[., k] = A_fine[., c_k + 1] ,   c_k = b + 30 k .

    This is measured, not assumed (``coarse_column_identity`` in the output).

``circularity``
    ``A_fine`` is circular in time on the length-``N`` grid and circular
    transversely on the padded ``(73, 188)`` grid.  Circularity is what makes
    ``A A^T`` exactly diagonal in the DFT of the record grid, which is what
    makes the closed-form inverse below exact rather than approximate.  It is
    a deviation from a linear (zero-padded) convolution ONLY where the support
    of the signal plus the 3930-tick support of ``h`` reaches around the grid;
    on this event it does not, and the statement is verified against the
    exact-functional direct sum, which uses no periodicity at all.

``G(kx, ky, nu)`` (the record-grid symbol of ``A A^T``)
    with ``hhat(kx, ky, f)`` the 3-D DFT of ``h`` on the fine grid (time length
    ``N``) and ``nu`` the record-grid frequency index (period ``M = N/30``),

        G(kx, ky, nu) = (1/30) sum_{m=0}^{29} | hhat(kx, ky, nu + m M) |^2 .

    Derivation: for ``y[w] = z(30(w+1))`` with ``z = x * h``,
    ``yhat(nu) = e^{2 pi i nu / M} (1/30) sum_m xhat(nu + mM) hhat(nu + mM)``
    and ``(A^T y)hat(nu + mM) = e^{-2 pi i nu / M} conj(hhat(nu + mM)) yhat(nu)``,
    so the phases cancel and ``(A A^T y)hat(nu) = G(nu) yhat(nu)`` exactly.
    Equivalently ``A A^T`` is circular convolution on the record grid with the
    30-fold decimated autocorrelation of ``h``.  The formula is TESTED against
    the validated forward and adjoint on random vectors.

``lambda``, the Tikhonov floor
    ``lambda = lambda_rel * max G``.  The estimator is the closed-form
    minimum-norm Tikhonov inverse

        xhat = A^T (A A^T + lambda I)^{-1} y ,
        what(nu) = yhat(nu) / (G(nu) + lambda) ,   xhat = A^T w .

    Per frequency the whole map is ``xhat(f) = e^{-2 pi i . 30 f / N}
    conj(hhat(f)) yhat(f mod M) / (G(f mod M) + lambda)``, i.e. the
    Wiener/Tikhonov filter ``conj(hhat) / (G + lambda)``: the matched filter
    ``conj(hhat)`` divided by the aliased power ``G`` regularised by
    ``lambda``.  At ``lambda = 0`` it is the Moore-Penrose pseudo-inverse
    ``A^+ y`` -- the minimum-2-norm solution of ``A x = y`` -- so the fine
    basis is inverted with NO positivity, NO support and NO sparsity prior.

``conservation identity``
    ``h`` is the 30-tick moving sum of ``Kbar``, so
    ``hhat(f) = Kbarhat(f) . S_30(f)`` with ``S_30`` the DFT of the 30-tick
    box, and ``S_30(mM) = 0`` for ``m = 1..29``.  Hence
    ``G(0) = hhat(0)^2 / 30`` exactly (no aliases at DC) and at
    ``lambda = 0``

        sum_j xhat(j) = 30 sum_w y[w] / sum_tau h(tau)
                      = sum_w y[w] / sum_{d,tau} Kbar_d(tau) .

    The reconstructed total is the recorded total divided by the kernel's own
    charge gain, for ANY data.  At ``lambda > 0`` it is multiplied by
    ``G(0) / (G(0) + lambda)``, which is reported.

``base-band share`` (NEW)
    ``|hhat(nu)|^2 / sum_m |hhat(nu + m M)|^2`` for ``nu`` in the base band
    ``|nu| <= M/2``: the fraction of the record-grid power at frequency ``nu``
    that comes from the base band rather than from its 29 aliases.  It is the
    determinacy of the fine basis, frequency by frequency; it is 1 at DC by
    the identity above.

``P = I`` for the fine candidates
    a fine candidate already lives on the fine grid, so its declared
    prolongation is the identity and the representation term
    ``H (P R - I) x`` of ``algo_plan.md`` Sec. 3.2 is identically zero.  The
    whole score is estimation error.  Coarse candidates keep ``P_delta`` and
    ``P_1`` as before.

``R xhat_fine``
    the fine estimate summed into the coarse cells ``C_k``, used only to put
    the two estimators on one axis in the bin-space figure.

The INTERMEDIATE (cell) time basis
----------------------------------
Everything above is the two extremes ``cell_ticks = 1`` (one unknown per
50 ns tick) and the production ``1.5 us`` bin.  The intermediate basis puts one
unknown on every block of ``c`` fine ticks, ``c`` a divisor of ``B = 30``.  It
is reached through the props ``cell_ticks`` (default ``1``, which reproduces
everything above bit for bit) and ``cell_model``.

``cell m``, ``x_p[m]``
    the unknown ``x_p[m]`` is the charge on pad ``p`` released in the ``c``
    fine ticks ``[b + c m, b + c (m+1))`` of the block.  There are
    ``N_c = (B/c) M`` cells per pad; ``D = B/c`` is the DECIMATION STRIDE of
    the operator on the cell grid (``D = 30`` for ``c = 1``, ``D = 1`` for
    ``c = 30``).

``cc_m``, the cell centre
    ``cc_m = b + c m + (c-1)/2``, the arithmetic mean of the ``c`` integer
    fine ticks of the cell.  It is INTEGER for odd ``c`` (so ``c = 1`` has
    ``cc_m = b + m``, the tick itself) and half-integer for even ``c``.  With
    this centre the box ``[cc_m - c/2, cc_m + c/2)`` is exactly the cell, so
    ``R_c`` and the prolongations of :mod:`~unfoldlarpix.algs.evalharness_algs`
    apply unchanged under ``(c_k, B) -> (cc_m, c)``.

``h_c(tau)`` (NEW), the cell window function
    the two WITHIN-CELL CHARGE MODELS, prop ``cell_model``:

    ``uniform`` (pairs with ``P_0``): the cell's charge is spread evenly over
    its ``c`` fine ticks, so
    ``h_c(tau) = (1/c) sum_{u=0}^{c-1} h(tau - u)`` -- the ``c``-tick moving
    AVERAGE of the fine window function.

    ``delta`` (pairs with ``P_delta``): the cell's charge is released as a
    point at the cell's LOWER EDGE ``b + c m``, so ``h_c(tau) = h(tau)``.

``g[s]`` (NEW), the operator kernel on the cell grid
    ``g[s] = h_c(c s - release_shift)``, i.e. ``h_c`` sampled every ``c`` fine
    ticks.  ``release_shift`` (constructor argument, default ``0``) moves the
    release point inside the cell by whole fine ticks and exists for ONE
    purpose: with ``c = 30``, ``cell_model = "delta"`` and
    ``release_shift = 1`` the operator IS the production ``A_coarse``
    (``A_coarse[., k] = A_fine[., c_k + 1]``, the ``phi = 1`` convention),
    which is checked in the job output and in ``tests/test_finebasis.py``.

``A_c`` (the cell operator)
    ``y_p[w] = sum_{p'} sum_m x_{p'}[m] g_{p-p'}[(B/c)(w+1) - m]``, i.e.
    ``A_c = D_{B/c} . conv(g)`` on the cell grid: the SAME structure as
    ``A_fine`` with ``(h, B) -> (g, D)``, honouring the same
    inclusive-upper-edge record convention (record ``w`` is latched at
    absolute fine tick ``b + B(w+1)``).  Derivation: the release point of cell
    ``m`` is ``b + c m`` and ``B(w+1) - c m = c ((B/c)(w+1) - m)`` because
    ``c`` divides ``B``, so the fine-tick argument of ``h_c`` is always a
    multiple of ``c`` and ``g`` carries every value the operator needs.

``G_c(nu)``
    ``G_c(nu) = (1/D) sum_{m=0}^{D-1} |ghat(nu + m M)|^2`` with ``ghat`` the
    DFT of ``g`` on the length-``N_c`` cell grid; ``D`` aliases instead of 30.
    The derivation of ``G`` above used only "convolution followed by
    decimation by ``D``", so it carries over verbatim.

``conservation on the cell basis``
    ``box_B = box_c * comb`` with ``comb`` the ``D`` unit samples at
    ``0, c, ..., (D-1)c``, so for BOTH charge models ``g`` is the ``D``-tap
    moving SUM on the cell grid of a shorter kernel; hence
    ``ghat(m M) = 0`` for ``m = 1..D-1``, ``G_c(0) = ghat(0)^2 / D`` exactly,
    ``sum_s g[s] = D sum_{d,tau} Kbar_d(tau)``, and at ``lambda -> 0``
    ``sum_m xhat[m] = D sum_w y[w] / sum_s g[s] = sum_w y[w] / sum Kbar`` --
    the same total as on the fine basis, at every ``c`` and for both models.

``R_c``
    box coarsening of the fine truth onto the cells:
    ``(R_c x)_p[m] = sum_{j in [b+cm, b+c(m+1))} x_p(j)``.  ``R_c`` is NOT the
    production ``R``: the production cells are ``[c_k - 15, c_k + 15)`` about
    the integer centre ``c_k = b + 30k``, while the ``c = 30`` cell basis here
    is ``[b + 30m, b + 30m + 30)`` -- the same width, offset by 15 fine ticks.

``P_0``, ``P_1`` on the cell basis
    ``P_0``: ``1/c`` on each of the cell's ``c`` fine ticks.
    ``P_1 = P_hat (R_c P_hat)^{-1}`` with ``P_hat`` the triangle of half-width
    ``c`` centred on ``cc_m``, normalised by ``1/c``.  ``T = R_c P_hat`` is
    tridiagonal by construction (the triangle reaches only the neighbouring
    cells) and, because ``cc_m`` is the cell's own centre, SYMMETRIC:
    ``(0.12, 0.76, 0.12)`` for ``c = 5``, ``(0.125, 0.75, 0.125)`` for
    ``c = 30``, and exactly ``I`` for ``c = 1`` (the triangle collapses onto
    the single tick, so ``P_1 = P_0 = I`` and both representation terms
    vanish identically).  ``R_c P = I`` and ``1^T P = 1^T`` are measured, not
    assumed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..deconv_workflow import integrate_kernel_over_time, prepare_field_response
from ..fwk.component import algorithm
from .evalharness_algs import (EvalHarness, TICK_US, cell_fine_ticks,
                               coarse_centers, coarse_index,
                               records_from_impulse, row_lookup,
                               smooth_columns, time_kernel)
from .exactrows_algs import COLLECTION_PIXEL, _Recorder, load_impact_response
from .fixedgrid_algs import (block_from_rows, fit_bin_ticks, resolve_support,
                             solve_arm)

KRAD = COLLECTION_PIXEL            # 12: kernel half-width in pads
NTICK_RESPONSE = 3900              # response length in fine ticks


# ---------------------------------------------------------------------------
# the fine operator
# ---------------------------------------------------------------------------
def fine_window_kernel(full_response: np.ndarray, B: int) -> np.ndarray:
    """``h_d(tau) = Kcumbar_d(tau) - Kcumbar_d(tau - 30)``, shape ``(25,25,3900+B)``.

    ``full_response`` is ``FieldResponseProcessor.process_response()``: the
    impact-averaged CHARGE induced per fine tick (it already carries the
    ``Delta_t`` factor -- it sums to 1.000266 over all pads and ticks).  So
    ``Kcumbar = cumsum(full_response)`` and ``h`` is the ``B``-tick moving sum.
    """
    K = np.asarray(full_response, dtype=np.float64)
    kc = np.cumsum(K, axis=-1)
    nt = kc.shape[-1]
    out = np.zeros(K.shape[:-1] + (nt + B,), dtype=np.float64)
    out[..., :nt] = kc
    out[..., nt:] = kc[..., -1:]
    shifted = np.zeros_like(out)
    shifted[..., B:] = out[..., :-B]
    return out - shifted


def truncated_response(full_response: np.ndarray,
                       kernel_cut_tick: int | None) -> np.ndarray:
    """``Kbar_d(tau) . [tau >= kernel_cut_tick]`` -- the kernel with its
    leading part deleted.

    tred zeroes every CURRENT sample before the event time reference
    ``t_0`` (``graph_effq.py:148-159``).  The fine truth tick ``j`` of a
    charge is the time it crosses the RESPONSE PLANE, so the absolute time of
    response tick ``tau`` is ``j + tau`` and the deletion removes exactly the
    kernel ticks ``tau < t_0 - j``.  For an isochronous single-depth event
    every charge shares one ``j`` up to longitudinal diffusion, so the
    deletion is ONE fixed kernel modification, ``kernel_cut_tick =
    tau_cut = t_0 - j``, and the operator stays shift invariant.

    ``kernel_cut_tick = None`` (or ``<= 0``) returns ``full_response``
    itself, unmodified and not copied.
    """
    if kernel_cut_tick is None or int(kernel_cut_tick) <= 0:
        return np.asarray(full_response)
    cut = int(kernel_cut_tick)
    fr = np.array(full_response, dtype=np.float64, copy=True)
    if cut >= fr.shape[-1]:
        raise ValueError(f"kernel_cut_tick {cut} deletes the whole response "
                         f"({fr.shape[-1]} ticks)")
    fr[..., :cut] = 0.0
    return fr


def cell_window_kernel(h: np.ndarray, cell_ticks: int,
                       cell_model: str = "uniform",
                       release_shift: int = 0) -> np.ndarray:
    """``g[s] = h_c(c s - release_shift)``, the operator kernel on the cell grid.

    ``h`` is the fine window function of :func:`fine_window_kernel`;
    ``cell_model`` is ``"uniform"`` (``h_c`` = the ``c``-tick moving AVERAGE of
    ``h``, the cell's charge spread evenly over its ``c`` fine ticks) or
    ``"delta"`` (``h_c = h``, the cell's charge released at its lower edge).
    With ``cell_ticks = 1`` and ``release_shift = 0`` both models return ``h``
    itself, element for element.
    """
    c = int(cell_ticks)
    if c < 1:
        raise ValueError("cell_ticks must be >= 1")
    hh = np.asarray(h, dtype=np.float64)
    if cell_model == "uniform":
        L = hh.shape[-1]
        hc = np.zeros(hh.shape[:-1] + (L + c - 1,), dtype=np.float64)
        for u in range(c):
            hc[..., u:u + L] += hh
        if c > 1:
            hc /= float(c)
    elif cell_model == "delta":
        hc = hh
    else:
        raise ValueError(f"unknown cell_model {cell_model!r}")
    Lc = hc.shape[-1]
    shift = int(release_shift)
    S = (Lc - 1 + shift) // c + 1
    idx = c * np.arange(S) - shift
    out = np.zeros(hc.shape[:-1] + (S,), dtype=np.float64)
    ok = (idx >= 0) & (idx < Lc)
    out[..., ok] = hc[..., idx[ok]]
    return out


class FineOperator:
    """``A_c = D_{B/c} . conv(g)`` on the block's periodic grid, with ``A A^T``.

    Circular in all three axes (see the module docstring).  ``forward`` maps an
    unknown array ``(nxp, nyp, N)`` to records ``(nxp, nyp, M)``; ``adjoint``
    maps back.  ``solve`` is the closed-form Tikhonov inverse.

    ``cell_ticks = 1`` (the default) is the fine 50 ns basis: ``g = h``,
    ``D = B``, ``N = B M`` fine ticks, and every array, symbol and result is
    identical element for element to the fine operator these studies started
    from.  ``cell_ticks = c > 1`` puts one unknown on every ``c`` fine ticks;
    ``N`` is then the number of CELLS, ``N = (B/c) M``.
    """

    def __init__(self, full_response: np.ndarray, block_shape, B: int,
                 device="cuda", dtype=torch.float64, cell_ticks: int = 1,
                 cell_model: str = "uniform", release_shift: int = 0,
                 kernel_cut_tick: int | None = None):
        nx, ny, M = (int(v) for v in block_shape)
        kx, ky = int(full_response.shape[0]), int(full_response.shape[1])
        self.krad = (kx - 1) // 2
        self.nx, self.ny, self.M = nx, ny, M
        self.nxp, self.nyp = nx + kx - 1, ny + ky - 1
        self.B = int(B)
        self.cell_ticks = int(cell_ticks)
        self.cell_model = str(cell_model)
        self.release_shift = int(release_shift)
        if self.B % self.cell_ticks:
            raise ValueError(f"cell_ticks {self.cell_ticks} does not divide "
                             f"B = {self.B}")
        self.D = self.B // self.cell_ticks       # decimation stride, cell grid
        self.N = self.D * M                      # unknowns per pad
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        self.kernel_cut_tick = (None if kernel_cut_tick is None
                                else int(kernel_cut_tick))
        fr_used = truncated_response(full_response, self.kernel_cut_tick)
        self.K_np = np.asarray(fr_used, dtype=np.float64)
        h = fine_window_kernel(fr_used, self.B)            # (25,25,3930)
        self.h_np = h
        g = cell_window_kernel(h, self.cell_ticks, self.cell_model,
                               self.release_shift)
        self.g_np = g
        if g.shape[-1] > self.N:
            raise ValueError("kernel longer than the block's unknown grid")
        hg = torch.zeros((self.nxp, self.nyp, self.N), dtype=dtype,
                         device=self.device)
        hg[:kx, :ky, :g.shape[-1]] = torch.as_tensor(g, dtype=dtype,
                                                     device=self.device)
        self.Hr = torch.fft.rfftn(hg, dim=(0, 1, 2))
        del hg
        torch.cuda.empty_cache()
        self._build_G()

    # -- A A^T symbol --------------------------------------------------------
    def _build_G(self) -> None:
        """``G(kx, ky, nu) = (1/D) sum_{m<D} |ghat(kx, ky, nu + m M)|^2``.

        ``Hr`` holds only ``f = 0 .. N//2``; the missing half is recovered from
        the Hermitian symmetry of a real kernel,
        ``|ghat(kx, ky, N - f)|^2 = |ghat(-kx, -ky, f)|^2``.  ``D = B/c`` is
        the decimation stride, so this is 30 aliases on the fine basis and
        ``30/c`` on the ``c``-tick cell basis.
        """
        N, M, D = self.N, self.M, self.D
        P = (self.Hr.real ** 2 + self.Hr.imag ** 2)
        Pf = torch.roll(torch.flip(P, dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
        nu = torch.arange(M // 2 + 1, device=self.device)
        G = torch.zeros((self.nxp, self.nyp, M // 2 + 1), dtype=self.dtype,
                        device=self.device)
        for m in range(D):
            f = nu + m * M
            lo = f <= N // 2
            if bool(lo.all()):
                G += P[:, :, f]
            elif bool((~lo).all()):
                G += Pf[:, :, N - f]
            else:
                G[:, :, lo] += P[:, :, f[lo]]
                G[:, :, ~lo] += Pf[:, :, N - f[~lo]]
        self.G = G / D
        del P, Pf
        torch.cuda.empty_cache()
        self.G_max = float(self.G.max())
        self.G_dc = float(self.G[0, 0, 0])

    # -- forward / adjoint ---------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(nxp, nyp, N) -> (nxp, nyp, M)``."""
        Xf = torch.fft.rfftn(x, dim=(0, 1, 2))
        Xf *= self.Hr
        z = torch.fft.irfftn(Xf, s=(self.nxp, self.nyp, self.N), dim=(0, 1, 2))
        del Xf
        # decimate FIRST, then roll: the transverse roll on the full fine grid
        # would need a second array of 240.9 M cells for no reason.
        zd = z[:, :, ::self.D].contiguous()
        del z
        y = torch.roll(zd, (-self.krad, -self.krad, -1), dims=(0, 1, 2))
        del zd
        return y

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """``(nxp, nyp, M) -> (nxp, nyp, N)``: zero insertion then correlation."""
        u = torch.zeros((self.nxp, self.nyp, self.N), dtype=self.dtype,
                        device=self.device)
        # roll the RECORD grid, then insert: same reason as in ``forward``.
        u[:, :, ::self.D] = torch.roll(r, (self.krad, self.krad, 1),
                                       dims=(0, 1, 2))
        Uf = torch.fft.rfftn(u, dim=(0, 1, 2))
        del u
        Uf *= torch.conj(self.Hr)
        x = torch.fft.irfftn(Uf, s=(self.nxp, self.nyp, self.N), dim=(0, 1, 2))
        del Uf
        return x

    def AAt_fft(self, r: torch.Tensor) -> torch.Tensor:
        """``A A^T r`` by the diagonal formula alone (the object under test)."""
        Rf = torch.fft.rfftn(r, dim=(0, 1, 2))
        Rf *= self.G.to(Rf.dtype)
        return torch.fft.irfftn(Rf, s=(self.nxp, self.nyp, self.M), dim=(0, 1, 2))

    def solve(self, y: torch.Tensor, lam: float) -> torch.Tensor:
        """``xhat = A^T (A A^T + lambda I)^{-1} y``."""
        Yf = torch.fft.rfftn(y, dim=(0, 1, 2))
        Yf = Yf / (self.G + lam).to(Yf.dtype)
        w = torch.fft.irfftn(Yf, s=(self.nxp, self.nyp, self.M), dim=(0, 1, 2))
        del Yf
        return self.adjoint(w)

    # -- diagnostics ---------------------------------------------------------
    def hhat_own_1d(self) -> np.ndarray:
        """``|ghat_(0,0)(f)|`` on the unknown grid, time axis, ``f = 0..N//2``."""
        v = np.zeros(self.N)
        v[:self.g_np.shape[-1]] = self.g_np[self.krad, self.krad]
        return np.abs(np.fft.rfft(v))


def embed_pads(a: np.ndarray | torch.Tensor, nxp: int, nyp: int, nt: int,
               device, dtype) -> torch.Tensor:
    """Put a ``(nx, ny, nt)`` block into the padded ``(nxp, nyp, nt)`` grid."""
    out = torch.zeros((nxp, nyp, nt), dtype=dtype, device=device)
    t = a if torch.is_tensor(a) else torch.as_tensor(np.ascontiguousarray(a),
                                                     dtype=dtype, device=device)
    out[:t.shape[0], :t.shape[1], :] = t.to(dtype)
    return out


def direct_sum_records(h: np.ndarray, truth_ix, truth_iy, truth_tick, truth_q,
                       b: int, B: int, nx: int, ny: int, M: int,
                       device, dtype) -> np.ndarray:
    """``y_p[w] = sum_{p',j} x_{p'}(j) h_{p-p'}(b + 30(w+1) - j)`` by direct sum.

    No FFT, no periodicity, no operator: the defining formula evaluated term by
    term.  Grouped by distinct truth tick, because ``h``'s argument depends on
    the tick and not on the pad.
    """
    ticks = np.unique(np.asarray(truth_tick))
    w = np.arange(M)
    ht = torch.as_tensor(h, dtype=dtype, device=device)          # (25,25,L)
    L = ht.shape[-1]
    out = torch.zeros((nx, ny, M), dtype=dtype, device=device)
    K = int(h.shape[0])
    krad = (K - 1) // 2
    for t in ticks:
        arg = b + B * (w + 1) - int(t)
        idx = np.clip(arg, 0, L - 1)
        val = torch.zeros((K, K, M), dtype=dtype, device=device)
        good = (arg >= 0) & (arg < L)
        v = ht[:, :, torch.as_tensor(idx, device=device)]
        v[:, :, torch.as_tensor(~good, device=device)] = 0.0
        val = v
        sel = np.asarray(truth_tick) == t
        qx = np.asarray(truth_ix)[sel]
        qy = np.asarray(truth_iy)[sel]
        qq = np.asarray(truth_q)[sel]
        Q = torch.zeros((nx, ny), dtype=dtype, device=device)
        Q.index_put_((torch.as_tensor(qx, device=device),
                      torch.as_tensor(qy, device=device)),
                     torch.as_tensor(qq, dtype=dtype, device=device),
                     accumulate=True)
        for dx in range(-krad, krad + 1):
            xs = slice(max(0, dx), min(nx, nx + dx))
            xq = slice(max(0, -dx), min(nx, nx - dx))
            for dy in range(-krad, krad + 1):
                ys = slice(max(0, dy), min(ny, ny + dy))
                yq = slice(max(0, -dy), min(ny, ny - dy))
                c = Q[xq, yq]
                if not bool(c.any()):
                    continue
                out[xs, ys, :] += (c[:, :, None]
                                   * val[krad + dx, krad + dy][None, None, :])
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# scoring: one code path for both bases
# ---------------------------------------------------------------------------
def score_rows(H: EvalHarness, xhat: np.ndarray, sigma_us: float, *,
               with_truth: bool = True) -> dict:
    """Score an ALREADY-SMOOTHED ``xhat`` of shape ``(n_pads, n_fine)``.

    Identical metric definitions to
    :meth:`~unfoldlarpix.algs.evalharness_algs.EvalHarness.measure`; the only
    difference is that the caller supplies ``xhat = H P xbar`` (coarse) or
    ``xhat = H x`` (fine) rather than the coarse array plus a prolongation.
    Cross-checked against ``measure`` in the tests and in the job output.
    """
    Hx = H.Hx(sigma_us) if with_truth else None
    truth_row_of = {int(r): i for i, r in enumerate(H.truth_pad_rows)}
    e = xhat.copy()
    if with_truth:
        for r, i in truth_row_of.items():
            e[r] -= Hx[i]
    sum_hx = float(Hx.sum()) if with_truth else 0.0
    sum_abs_e = float(np.abs(e).sum())
    sum_xhat = float(xhat.sum())

    z, ch = {}, H.chebyshev
    for ring in (1, 2, 3):
        sel = (ch == ring) if ring < 3 else (ch >= 3)
        blk = xhat[sel]
        n = max(int(sel.sum()), 1)
        z["ring%d" % ring if ring < 3 else "ring_ge3"] = {
            "n_pads": int(sel.sum()),
            "sum_pos_ke": float(blk[blk > 0].sum()),
            "sum_neg_ke": float(blk[blk < 0].sum()),
            "pos_per_pad_ke": float(blk[blk > 0].sum()) / n,
            "neg_per_pad_ke": float(blk[blk < 0].sum()) / n}

    line_e = e[H.line_rows].mean(axis=0)
    line_hat = xhat[H.line_rows].mean(axis=0)
    line_hx = (Hx[[truth_row_of[int(r)] for r in H.line_rows]].mean(axis=0)
               if with_truth else np.zeros(H.n_fine))
    pad_total = xhat.sum(axis=1)
    out = {
        "sigma_H_us": float(sigma_us),
        "sum_xhat_ke": sum_xhat, "sum_Hx_ke": sum_hx,
        "sum_abs_e_ke": sum_abs_e, "sum_e_ke": float(e.sum()),
        "max_abs_e_ke_per_tick": float(np.abs(e).max()),
        "E_rel": (sum_abs_e / sum_hx) if sum_hx else float("nan"),
        "conservation_ke": sum_xhat - sum_hx,
        "conservation_rel": ((sum_xhat - sum_hx) / sum_hx) if sum_hx else float("nan"),
        "E_max_line_ke_per_tick": float(np.abs(line_e).max()),
        "E_rms_line_ke_per_tick": float(np.sqrt((line_e ** 2).mean())),
        "n_line_pads": int(len(H.line_rows)),
        "zero_preservation": z,
    }
    if with_truth:
        sh = np.array([pad_total[rows].sum() for _, _, rows in H.segments])
        sx = np.array([sum(float(Hx[truth_row_of[int(r)]].sum())
                           for r in rows if int(r) in truth_row_of)
                       for _, _, rows in H.segments])
        rel = np.where(sx != 0, (sh - sx) / np.where(sx != 0, sx, 1.0), np.nan)
        out["segments"] = {
            "n": len(H.segments),
            "rel_error_mean": float(np.nanmean(rel)),
            "rel_error_rms": float(np.sqrt(np.nanmean(rel ** 2))),
            "rel_error_max_abs": float(np.nanmax(np.abs(rel))),
            "rel_error": [float(v) for v in rel]}
    out["_profiles"] = {"line_xhat": line_hat, "line_Hx": line_hx,
                        "line_e": line_e, "allpad_xhat": xhat.sum(axis=0)}
    return out


def coarse_xhat(H: EvalHarness, q: np.ndarray, pname: str,
                sigma_us: float) -> np.ndarray:
    """``H P q`` on ``(n_pads, n_fine)``."""
    Mt = H.M(pname, sigma_us).T
    return np.asarray(q, dtype=float).reshape(H.n_pads, H.n_coarse) @ Mt


def fine_xhat(H: EvalHarness, xwin: np.ndarray, win_lo: int,
              sigma_us: float) -> np.ndarray:
    """``H x`` on ``(n_pads, n_fine)`` for a fine array given on ``[win_lo, ...)``.

    ``P = I``: the candidate is already on the fine grid, so ``H`` is applied
    directly and the representation term is identically zero.
    """
    g = time_kernel(float(sigma_us) / TICK_US)
    pad = (len(g) - 1) // 2
    lo = int(H.fine[0]) - win_lo - pad
    hi = lo + H.n_fine + 2 * pad
    if lo < 0 or hi > xwin.shape[-1]:
        raise ValueError(f"stored fine window too narrow for sigma_H={sigma_us}"
                         f" (need [{lo}, {hi}) of {xwin.shape[-1]})")
    blk = np.asarray(xwin, dtype=float).reshape(-1, xwin.shape[-1])[:, lo:hi]
    if pad == 0:
        return np.ascontiguousarray(blk)
    sm = smooth_columns(blk.T, g).T
    return np.ascontiguousarray(sm[:, pad:pad + H.n_fine])


# ---------------------------------------------------------------------------
# the intermediate (cell) time basis: geometry, R_c, P_0 and P_1
# ---------------------------------------------------------------------------
def cell_centers(b: int, cell_ticks: int, n_cells: int) -> np.ndarray:
    """``cc_m = b + c m + (c-1)/2``, the mean of the cell's ``c`` fine ticks."""
    c = int(cell_ticks)
    return float(b) + c * np.arange(int(n_cells)) + (c - 1) / 2.0


class CellGrid:
    """The ``c``-tick cell basis on one block: ``R_c``, ``P_0`` and ``P_1``.

    Pure geometry -- no operator, no data, no estimator -- so one instance
    serves every arm of a job.  ``c = 1`` is the fine grid and every map here
    is the identity, which is asserted rather than assumed
    (:meth:`report`).

    Definitions are those of the module docstring, section "The INTERMEDIATE
    (cell) time basis".  Both prolongations are stored as a list of TAPS: a
    prolongation is ``P[b + c m + r, m] = w_r``, one weight per integer offset
    ``r`` inside the cell (``P_0``: ``r = 0..c-1``, ``w = 1/c``) or across it
    (``P_hat``: ``|r - (c-1)/2| < c``, ``w = (1 - |r - (c-1)/2|/c)/c``).  Both
    tap sets sum to 1 exactly, so ``1^T P = 1^T`` on the interior.

    ``T = R_c P_hat`` follows from the taps alone: the tap at offset ``r``
    lands in cell ``m + floor(r/c)``, and ``floor(r/c)`` is only ``-1``, ``0``
    or ``+1``, so ``T`` is tridiagonal and Toeplitz with the three band values
    ``T_sup``, ``T_diag``, ``T_sub`` reported below.  ``P_1 = P_hat T^{-1}`` is
    never formed as a matrix: ``T u = xbar`` is solved as a banded system and
    the taps are then applied to ``u``.
    """

    def __init__(self, b: int, cell_ticks: int, n_cells: int):
        self.b = int(b)
        self.c = int(cell_ticks)
        self.n = int(n_cells)
        self.cc = cell_centers(self.b, self.c, self.n)
        c = self.c
        r = np.arange(-c, 2 * c + 1)
        w = np.maximum(0.0, 1.0 - np.abs(r - (c - 1) / 2.0) / c) / c
        keep = w > 0
        self.hat_taps, self.hat_w = r[keep], w[keep]
        off = np.floor(self.hat_taps / float(c)).astype(int)
        if not set(np.unique(off)).issubset({-1, 0, 1}):
            raise AssertionError("P_hat is not tridiagonal on this cell grid")
        self.T_sub = float(self.hat_w[off == 1].sum())     # T[m+1, m]
        self.T_diag = float(self.hat_w[off == 0].sum())    # T[m,   m]
        self.T_sup = float(self.hat_w[off == -1].sum())    # T[m-1, m]
        self.box_taps = np.arange(c)
        self.box_w = np.full(c, 1.0 / c)

    # -- geometry -----------------------------------------------------------
    def index(self, tick) -> np.ndarray:
        """Cell index of the fine tick(s): ``m = floor((tick - b)/c)``."""
        return np.floor((np.asarray(tick, dtype=float) - self.b)
                        / self.c).astype(np.int64)

    def window(self, lo_tick: int, hi_tick: int) -> tuple[int, int]:
        """The cell range ``[m_lo, m_hi)`` covering the fine ticks
        ``[lo_tick, hi_tick)``, clipped to the grid."""
        m_lo = int(np.floor((lo_tick - self.b) / self.c))
        m_hi = int(np.ceil((hi_tick - self.b) / self.c))
        return max(m_lo, 0), min(m_hi, self.n)

    def fine_origin(self, m_lo: int) -> int:
        return self.b + self.c * int(m_lo)

    def taps(self, pname: str) -> tuple[np.ndarray, np.ndarray]:
        if pname in ("uniform", "P0", "box"):
            return self.box_taps, self.box_w
        if pname in ("corrected_hat", "hat", "P1"):
            return self.hat_taps, self.hat_w
        raise ValueError(f"unknown cell prolongation {pname!r}")

    # -- R_c ----------------------------------------------------------------
    def restrict(self, ix, iy, tick, q, nx: int, ny: int) -> np.ndarray:
        """``R_c x`` from the fine truth list, shape ``(nx, ny, n)``."""
        m = self.index(tick)
        ok = (m >= 0) & (m < self.n)
        out = np.zeros((nx, ny, self.n))
        np.add.at(out, (np.asarray(ix)[ok], np.asarray(iy)[ok], m[ok]),
                  np.asarray(q, dtype=float)[ok])
        return out

    # -- T^{-1} -------------------------------------------------------------
    def solve_T(self, x: np.ndarray) -> np.ndarray:
        """``u = T^{-1} xbar`` for ``xbar`` of shape ``(n_rows, n_cols)``.

        ``n_cols`` is the number of cells actually carried by ``x``: the whole
        grid, or the stored window when a candidate has been cropped to it.
        ``T`` is symmetric tridiagonal Toeplitz with band ratio
        ``T_sub / T_diag <= 0.17``, so ``T^{-1}`` decays by at least that
        factor per cell and a crop hundreds of cells away from the signal
        changes ``P_1 xbar`` by nothing a float carries; the cropping distance
        is the job's ``margin_windows``, which is reported.
        """
        if self.c == 1:
            return np.asarray(x, dtype=float)
        from scipy.linalg import solve_banded
        xf = np.asarray(x, dtype=float)
        n = xf.shape[-1]
        ab = np.zeros((3, n))
        ab[0, 1:] = self.T_sup
        ab[1, :] = self.T_diag
        ab[2, :-1] = self.T_sub
        return np.ascontiguousarray(solve_banded((1, 1), ab, xf.T).T)

    # -- P ------------------------------------------------------------------
    def to_fine(self, x: np.ndarray, pname: str, m_lo: int, m_hi: int,
                x_lo: int = 0) -> np.ndarray:
        """``P xbar`` on the fine ticks ``[b + c m_lo, b + c m_hi)``.

        ``x`` is ``(..., n_cols)`` covering cells ``[x_lo, x_lo + n_cols)``
        (the whole grid when ``x_lo = 0`` and ``n_cols = n``); the result is
        ``(n_rows, c (m_hi - m_lo))`` with ``n_rows`` the flattened leading
        axes.  The fine origin is ``fine_origin(m_lo)``, which is what
        :func:`fine_xhat` wants as ``win_lo``.
        """
        taps, w = self.taps(pname)
        xf = np.asarray(x, dtype=float)
        xf = xf.reshape(-1, xf.shape[-1])
        u = xf if pname in ("uniform", "P0", "box") else self.solve_T(xf)
        nf = self.c * (int(m_hi) - int(m_lo))
        out = np.zeros((u.shape[0], nf))
        base = self.c * (np.arange(xf.shape[-1]) + int(x_lo) - int(m_lo))
        for r, wr in zip(taps, w):
            idx = base + int(r)
            ok = (idx >= 0) & (idx < nf)
            if ok.any():
                out[:, idx[ok]] += wr * u[:, ok]
        return out

    # -- what is measured rather than assumed -------------------------------
    def report(self, n_probe: int = 3) -> dict:
        """``R_c P = I``, ``1^T P = 1^T`` and ``T T^{-1} = I``, all measured.

        ``R_c P = I`` is measured end to end: unit charge is put on one cell,
        prolonged to the fine ticks by the very code the arms use, and box
        coarsened back with :meth:`index`.
        """
        out = {"cell_ticks": self.c, "n_cells": self.n,
               "cell_center_first": float(self.cc[0]),
               "cell_center_offset_in_cell": (self.c - 1) / 2.0,
               "T_bands_sup_diag_sub": [self.T_sup, self.T_diag, self.T_sub],
               "T_band_sum": self.T_sup + self.T_diag + self.T_sub,
               "hat_tap_offsets": [int(v) for v in self.hat_taps],
               "hat_tap_weights": [float(v) for v in self.hat_w],
               "hat_tap_weight_sum": float(self.hat_w.sum()),
               "box_tap_weight_sum": float(self.box_w.sum())}
        mid = self.n // 2
        ms = [mid + 3 * i for i in range(int(n_probe))]
        for pname in ("uniform", "corrected_hat"):
            worst_rp, worst_col = 0.0, 0.0
            for m in ms:
                m_lo, m_hi = max(m - 40, 0), min(m + 41, self.n)
                e = np.zeros((1, self.n))
                e[0, m] = 1.0
                pf = self.to_fine(e, pname, m_lo, m_hi)[0]
                j = self.fine_origin(m_lo) + np.arange(len(pf))
                kk = self.index(j)
                rp = np.zeros(self.n)
                np.add.at(rp, kk, pf)
                tgt = np.zeros(self.n)
                tgt[m] = 1.0
                worst_rp = max(worst_rp, float(np.abs(rp - tgt).max()))
                worst_col = max(worst_col, abs(float(pf.sum()) - 1.0))
            out[f"{pname}_Rc_P_minus_I_max"] = worst_rp
            out[f"{pname}_colsum_minus_1_max"] = worst_col
            out[f"{pname}_probed_cells"] = ms
        if self.c == 1:
            out["c1_identity"] = {
                "hat_taps_is_single_zero": (list(self.hat_taps) == [0]
                                            and float(self.hat_w[0]) == 1.0),
                "T_is_identity": (self.T_diag == 1.0 and self.T_sub == 0.0
                                  and self.T_sup == 0.0),
                "note": ("at c = 1 the triangle collapses onto the tick "
                         "itself, so P_hat = T = I and P_1 = P_0 = I")}
        return out


def cell_xhat(H: EvalHarness, grid: CellGrid, x: np.ndarray, pname: str,
              m_lo: int, m_hi: int, sigma_us: float,
              x_lo: int = 0) -> np.ndarray:
    """``H P xbar`` on ``(n_pads, n_fine)`` for a cell-basis candidate."""
    pf = grid.to_fine(x, pname, m_lo, m_hi, x_lo)
    return fine_xhat(H, pf, grid.fine_origin(m_lo), sigma_us)


def cell_charge_model_taps(grid: CellGrid, cell_model: str
                           ) -> tuple[np.ndarray, np.ndarray]:
    """The fine-tick representative of one unit of cell charge, as taps.

    ``uniform`` -> ``1/c`` on each of the cell's ``c`` ticks (``P_0``);
    ``delta`` -> unit mass at the cell's LOWER EDGE, which is where the delta
    operator releases it.  This is the fine signal whose exact functional the
    cell operator must reproduce, so it is what the forward validation uses.
    """
    if cell_model == "uniform":
        return grid.box_taps, grid.box_w
    if cell_model == "delta":
        return np.array([0]), np.array([1.0])
    raise ValueError(f"unknown cell_model {cell_model!r}")


def prolong_truth_to_fine(grid: CellGrid, Rx: np.ndarray, taps: np.ndarray,
                          w: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray]:
    """A cell-basis charge array as a sparse fine-tick list ``(ix, iy, t, q)``.

    Used to feed :func:`direct_sum_records`, which evaluates the exact
    functional term by term on a fine-tick charge list.
    """
    nx, ny, n = Rx.shape
    ix, iy, mm = np.nonzero(Rx)
    q = Rx[ix, iy, mm]
    IX = np.repeat(ix, len(taps))
    IY = np.repeat(iy, len(taps))
    TT = (grid.b + grid.c * np.repeat(mm, len(taps))
          + np.tile(taps, len(mm)))
    QQ = np.repeat(q, len(taps)) * np.tile(w, len(mm))
    return IX, IY, TT.astype(np.int64), QQ


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
@algorithm("FineBasisInverse")
class FineBasisInverse(_Recorder):
    """Build ``A_fine``, validate it, scan ``lambda``, score every arm.

    Validations, all asserted and all in the output:

    1. ``A_fine`` applied to the effq truth against the exact-functional
       DIRECT SUM of the same formula (no FFT, no periodicity), relative to
       ``sum |y|``.
    2. the dot-product test ``<A x, y> = <x, A^T y>`` on random vectors.
    3. ``(A A^T) y`` by the diagonal formula against ``A(A^T y)`` by the
       validated forward and adjoint, on random ``y``.
    4. the coarse-column identity ``A_coarse[., k] = A_fine[., c_k + 1]``.

    Props
    -----
    lambda_rel : list of float
        Tikhonov floors as fractions of ``max G``.  Default
        ``[1e-2, 1e-3, 1e-4, 1e-6, 1e-8]``.
    sigma_H_us : list of float, default ``[0.0, 1.5, 2.0]``.
    prolongations : list of str, for the COARSE arms only; default
        ``["delta", "corrected_hat"]``.
    margin_windows, line_pixel_y_range, segment_pixels, segment_edge_exclude
        as :class:`~unfoldlarpix.algs.evalharness_algs.ResolutionScore`.
    dtype : ``"float64"`` (default) or ``"float32"``.
    out_json, out_npz : str
    """

    reads = ("op", "event", "readout_config", "block_offset", "charge_model",
             "arms.q")
    writes = ("fine.result", "fine.solutions")

    def execute(self, store):
        op = store.get("op")
        arms = store.get("arms.q")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        B = int(round(fit_bin_ticks(store)))
        dtype = (torch.float64 if str(self.props.get("dtype", "float64"))
                 == "float64" else torch.float32)
        # validation tolerances follow the dtype: float32 FFTs over 17550
        # ticks accumulate ~1e-5 relative rounding, float64 ~1e-14
        tol_fwd = 1e-6 if dtype == torch.float64 else 1e-4
        tol_lin = 1e-5 if dtype == torch.float64 else 1e-3
        lam_rel = [float(v) for v in self.props.get(
            "lambda_rel", [1e-2, 1e-3, 1e-4, 1e-6, 1e-8])]
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [0.0, 1.5, 2.0])]
        pnames = [str(v) for v in self.props.get(
            "prolongations", ["delta", "corrected_hat"])]
        margin = int(self.props.get("margin_windows", 40))
        dev = op.device

        prep = self.services["detector"].prepared(B)
        fr = np.asarray(prep.full_response, dtype=np.float64)
        kbin = np.asarray(prep.integrated_response, dtype=np.float64)

        t0 = time.time()
        F = FineOperator(fr, op.block_shape, B, device=dev, dtype=dtype)
        t_build = time.time() - t0
        print(f"[{self.name}] fine operator: pads {F.nxp}x{F.nyp}, "
              f"N {F.N} fine ticks, M {F.M} windows, built in {t_build:.1f} s; "
              f"max G {F.G_max:.6g}, G(0) {F.G_dc:.6g}")

        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))
        # stored fine window: the evaluation window plus room for 5 sigma_H
        pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
        win_lo = int(H.fine[0]) - pad_ext
        win_hi = int(H.fine[-1]) + 1 + pad_ext
        n_win = win_hi - win_lo

        rec: dict = {
            "geometry": {
                "block_shape": [int(v) for v in op.block_shape],
                "q_shape": [int(v) for v in op.q_shape],
                "padded_pads": [F.nxp, F.nyp],
                "n_padding_each_side": KRAD,
                "block_offset": [float(v) for v in boff],
                "B_fine_ticks": B, "M_windows": F.M, "N_fine_ticks": F.N,
                "n_fine_unknowns_total": int(F.nxp * F.nyp * F.N),
                "kernel_h_support_ticks": int(F.h_np.shape[-1]),
                "window_convention":
                    "window w covers absolute fine ticks (b+30w, b+30(w+1)]",
                "stored_fine_window": [win_lo, win_hi],
                "eval_window_fine_ticks": [int(H.fine[0]), int(H.fine[-1]) + 1],
                "eval_window_cells": [H.k0, H.k1]},
            "kernel": {
                "sum_Kbar_all_pads": float(fr.sum()),
                "sum_Kbar_own_pad": float(fr[KRAD, KRAD].sum()),
                "sum_h_all_pads": float(F.h_np.sum()),
                "sum_h_over_B_all_pads": float(F.h_np.sum() / B),
                "sum_Kbin_all_pads": float(kbin.sum()),
                "G_max": F.G_max, "G_dc": F.G_dc,
                "G_dc_over_hhat0_sq_over_B": float(
                    F.G_dc / (F.h_np.sum() ** 2 / B)),
                "note": ("G(0) = hhat(0)^2 / 30 exactly, because the 30-tick "
                         "moving sum has an exact zero at every non-zero "
                         "multiple of the record-grid Nyquist period")},
            "build_wall_s": t_build,
        }

        # -------- validation 1: forward vs the exact-functional direct sum ---
        t0 = time.time()
        xt = torch.zeros((F.nxp, F.nyp, F.N), dtype=dtype, device=dev)
        jj = H.truth_tick - b
        keep = (jj >= 0) & (jj < F.N)
        xt.index_put_((torch.as_tensor(H.truth_ix[keep], device=dev),
                       torch.as_tensor(H.truth_iy[keep], device=dev),
                       torch.as_tensor(jj[keep], device=dev)),
                      torch.as_tensor(H.truth_q[keep], dtype=dtype, device=dev),
                      accumulate=True)
        y_fft = F.forward(xt).cpu().numpy()[:H.nx, :H.ny]
        del xt
        torch.cuda.empty_cache()
        y_dir = direct_sum_records(F.h_np, H.truth_ix[keep], H.truth_iy[keep],
                                   H.truth_tick[keep], H.truth_q[keep],
                                   b, B, H.nx, H.ny, F.M, dev, dtype)
        den = float(np.abs(y_dir).sum())
        rec["validation_forward_vs_direct_sum"] = {
            "sum_abs_difference_ke": float(np.abs(y_fft - y_dir).sum()),
            "relative_to_sum_abs": float(np.abs(y_fft - y_dir).sum() / den),
            "max_abs_difference_ke": float(np.abs(y_fft - y_dir).max()),
            "max_relative_to_max": float(np.abs(y_fft - y_dir).max()
                                         / np.abs(y_dir).max()),
            "sum_y_fft_ke": float(y_fft.sum()),
            "sum_y_direct_ke": float(y_dir.sum()),
            "sum_abs_y_direct_ke": den,
            "tolerance": tol_fwd, "wall_s": time.time() - t0,
            "note": ("the direct sum uses no FFT and no periodicity, so this "
                     "also measures that the circular grid does not wrap on "
                     "this event")}
        v1 = rec["validation_forward_vs_direct_sum"]["relative_to_sum_abs"]
        print(f"[{self.name}] forward vs direct sum: {v1:.3e} of sum|y|")
        if not v1 < tol_fwd:
            raise AssertionError(f"forward vs direct sum {v1:.3e} > {tol_fwd:g}")

        # -------- validation 2: adjoint dot product -------------------------
        g = torch.Generator(device="cpu").manual_seed(11)
        dots = []
        for _ in range(3):
            xr = torch.randn((F.nxp, F.nyp, F.N), generator=g,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            yr = torch.randn((F.nxp, F.nyp, F.M), generator=g,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            a = float((F.forward(xr) * yr).sum())
            bb = float((xr * F.adjoint(yr)).sum())
            dots.append({"Ax_y": a, "x_Aty": bb,
                         "relative_difference": abs(a - bb) / max(abs(a), 1e-30)})
            del xr, yr
            torch.cuda.empty_cache()
        rec["validation_adjoint"] = {"trials": dots, "tolerance": tol_lin,
                                     "worst": max(d["relative_difference"]
                                                  for d in dots)}
        print(f"[{self.name}] adjoint dot product worst "
              f"{rec['validation_adjoint']['worst']:.3e}")
        if not rec["validation_adjoint"]["worst"] < tol_lin:
            raise AssertionError("adjoint dot-product test failed")

        # -------- validation 3: A A^T diagonal formula ----------------------
        aat = []
        for _ in range(3):
            yr = torch.randn((F.nxp, F.nyp, F.M), generator=g,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            u = F.AAt_fft(yr)
            v = F.forward(F.adjoint(yr))
            aat.append({"relative_difference": float(
                torch.abs(u - v).sum() / torch.abs(v).sum())})
            del yr, u, v
            torch.cuda.empty_cache()
        rec["validation_AAt"] = {"trials": aat, "tolerance": tol_lin,
                                 "worst": max(d["relative_difference"] for d in aat)}
        print(f"[{self.name}] A A^T formula worst "
              f"{rec['validation_AAt']['worst']:.3e}")
        if not rec["validation_AAt"]["worst"] < tol_lin:
            raise AssertionError("A A^T diagonal-formula test failed")

        # -------- validation 4: the coarse column is a fine delta -----------
        k_id = int(H.k_truth_hi)
        e_c = np.zeros(op.q_shape)
        e_c[H.nx // 2, H.ny // 2, k_id] = 1.0
        y_c = op.conv(op.to_tensor(e_c)).cpu().numpy()
        best = None
        for shift in (0, 1, 2, -1):
            xd = torch.zeros((F.nxp, F.nyp, F.N), dtype=dtype, device=dev)
            xd[H.nx // 2, H.ny // 2, int(H.c[k_id]) - b + shift] = 1.0
            y_f = F.forward(xd).cpu().numpy()[:H.nx, :H.ny]
            d = float(np.abs(y_f - y_c).sum())
            if best is None or d < best[1]:
                best = (shift, d)
            rec.setdefault("coarse_column_identity", {})[f"shift_{shift:+d}"] = {
                "sum_abs_difference": d,
                "relative_to_sum_abs": d / float(np.abs(y_c).sum())}
            del xd
        rec["coarse_column_identity"]["best_shift_ticks"] = best[0]
        rec["coarse_column_identity"]["note"] = (
            "A_coarse cell k equals the fine operator applied to a delta at "
            "c_k + best_shift; the measured value is +1 fine tick, which is "
            "the phi = 1 convention")
        print(f"[{self.name}] coarse column = fine delta at c_k "
              f"{best[0]:+d} ticks (sum|diff| {best[1]:.3e})")
        torch.cuda.empty_cache()

        # -------- the data --------------------------------------------------
        blk = block_from_rows(op)                       # (nx, ny, M), records
        y_t = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
        y_norm = float(torch.linalg.vector_norm(y_t))
        rec["data"] = {"sum_records_ke": float(blk.sum()),
                       "sum_abs_records_ke": float(np.abs(blk).sum()),
                       "n_nonzero_windows": int((blk != 0).sum()),
                       "padding_pads_records": 0.0}

        # -------- lambda scan -------------------------------------------------
        sol: dict = {}
        scan = []
        for lr in lam_rel:
            lam = lr * F.G_max
            t0 = time.time()
            xh = F.solve(y_t, lam)
            wall = time.time() - t0
            r = F.forward(xh) - y_t
            row = {
                "lambda_rel": lr, "lambda": lam,
                "sum_xhat_ke": float(xh.sum()),
                "sum_xhat_pos_ke": float(xh[xh > 0].sum()),
                "sum_xhat_neg_ke": float(xh[xh < 0].sum()),
                "sum_xhat_real_pads_ke": float(xh[:H.nx, :H.ny].sum()),
                "sum_xhat_padding_pads_ke": float(
                    xh.sum() - xh[:H.nx, :H.ny].sum()),
                "residual_rel": float(torch.linalg.vector_norm(r) / y_norm),
                "conservation_predicted": float(
                    F.G_dc / (F.G_dc + lam)),
                "sum_y_over_sum_Kbar": float(blk.sum() / fr.sum()),
                "wall_s": wall}
            scan.append(row)
            print(f"[{self.name}] lambda_rel {lr:8.1e}  sum {row['sum_xhat_ke']:10.2f} "
                  f"ke  x+ {row['sum_xhat_pos_ke']:10.2f}  x- "
                  f"{row['sum_xhat_neg_ke']:11.2f}  |Ax-y|/|y| "
                  f"{row['residual_rel']:.4e}  {wall:5.1f} s")
            sol[lr] = xh[:H.nx, :H.ny, win_lo - b:win_hi - b].cpu().numpy()
            del xh, r
            torch.cuda.empty_cache()
        rec["lambda_scan"] = scan
        del y_t
        torch.cuda.empty_cache()

        # -------- scoring -----------------------------------------------------
        # the ring-1 pad row: pixel_x = 140 (one pad off the line at 141)
        rows_r1 = np.array([r for r in range(H.n_pads)
                            if H.pixel_x_of_pad[r] == 140
                            and 5 <= H.pixel_y_of_pad[r] <= 131])
        rows, arrays = [], {}

        def _do(tag, xh, meta):
            m = score_rows(H, xh, meta["sigma_H_us"])
            pr = m.pop("_profiles")
            rows.append({**meta, **m})
            self._store_prof(arrays, tag, pr)
            arrays[f"ring1prof_{tag}"] = xh[rows_r1].mean(axis=0).astype(np.float32)
            self._print_row(rows[-1])

        for s in sigmas:
            for pname in pnames:
                _do(f"repr_{pname}_s{s:g}", coarse_xhat(H, H.Rx, pname, s),
                    {"arm": "representation_term", "basis": "coarse",
                     "prolongation": pname, "sigma_H_us": s})
                for lab, a in arms.items():
                    _do(f"{lab}_{pname}_s{s:g}",
                        coarse_xhat(H, a["q"], pname, s),
                        {"arm": lab, "basis": "coarse", "prolongation": pname,
                         "sigma_H_us": s})
            for lr in lam_rel:
                _do(f"fine_lrel{lr:g}_s{s:g}", fine_xhat(H, sol[lr], win_lo, s),
                    {"arm": f"fine_lrel{lr:g}", "basis": "fine",
                     "prolongation": "identity", "lambda_rel": lr,
                     "sigma_H_us": s})
        rec["rows"] = rows

        # -------- bin-space comparison (F4) ----------------------------------
        ka = int(np.argmax(np.abs(H.Rx).sum(axis=(0, 1))))
        ks = list(range(max(ka - 5, 0), min(ka + 6, H.n_coarse)))
        osc = {"cells": ks, "cell_center_ticks": [float(H.c[k]) for k in ks],
               "truth_Rbox_ke": [float(H.Rx[:, :, k].sum()) for k in ks]}
        for lab, a in arms.items():
            osc[lab + "_ke"] = [float(a["q"][:, :, k].sum()) for k in ks]
        jw = np.arange(win_lo, win_hi)
        kw = coarse_index(jw, H.c[0], H.B)
        for lr in lam_rel:
            tot = sol[lr].sum(axis=(0, 1))
            osc[f"fine_lrel{lr:g}_Rxhat_ke"] = [
                float(tot[kw == k].sum()) for k in ks]
        rec["coarse_pad_summed_profile"] = osc

        # -------- arrays for the figures --------------------------------------
        arrays["kbar_own"] = fr[KRAD, KRAD].astype(np.float64)
        arrays["kbar_ring1"] = fr[KRAD + 1, KRAD].astype(np.float64)
        arrays["kcum_own"] = np.cumsum(fr[KRAD, KRAD])
        arrays["kcum_ring1"] = np.cumsum(fr[KRAD + 1, KRAD])
        arrays["h_own"] = F.h_np[KRAD, KRAD].astype(np.float64)
        arrays["h_ring1"] = F.h_np[KRAD + 1, KRAD].astype(np.float64)
        arrays["kbin_own"] = kbin[KRAD, KRAD].astype(np.float64)
        arrays["kbin_ring1"] = kbin[KRAD + 1, KRAD].astype(np.float64)
        hh = F.hhat_own_1d()
        arrays["hhat_own_abs"] = hh
        full = np.concatenate([hh, hh[-2:0:-1]])
        p = full ** 2
        arrays["alias_power_sum"] = p.reshape(B, F.M).sum(axis=0)
        arrays["base_band_power"] = p[:F.M]
        arrays["G_own_transverse_dc"] = F.G[0, 0].cpu().numpy()
        arrays["fine_ticks"] = H.fine.astype(np.int64)
        arrays["stored_window_ticks"] = np.arange(win_lo, win_hi)
        arrays["coarse_centers"] = H.c.astype(np.float64)
        arrays["truth_padsum_fine"] = self._truth_padsum(H, win_lo, n_win)
        for lab, a in arms.items():
            arrays["coarse_padsum_" + lab] = a["q"].sum(axis=(0, 1))
        arrays["coarse_padsum_truth_Rbox"] = H.Rx.sum(axis=(0, 1))
        # transverse profiles, sum over the stored window
        arrays["transverse_truth"] = self._transverse_truth(H)
        for lr in lam_rel:
            arrays[f"transverse_fine_lrel{lr:g}"] = sol[lr].sum(axis=(1, 2))
            arrays[f"fine_line_lrel{lr:g}"] = \
                sol[lr].reshape(H.n_pads, -1)[H.line_rows].mean(axis=0)
        for lab, a in arms.items():
            arrays["transverse_" + lab] = a["q"].sum(axis=(1, 2))
        # line-averaged and ring-1-row-averaged COARSE cell values, for the
        # step plots (the coarse basis has no finer object than a cell)
        for lab, a in arms.items():
            qq = a["q"].reshape(H.n_pads, H.n_coarse)
            arrays["line_coarse_" + lab] = qq[H.line_rows].mean(axis=0)
            arrays["ring1_coarse_" + lab] = qq[rows_r1].mean(axis=0)
        arrays["line_coarse_truth_Rbox"] = \
            H.Rx.reshape(H.n_pads, H.n_coarse)[H.line_rows].mean(axis=0)
        arrays["ring1_row_pads"] = rows_r1
        rec["ring1_row"] = {"pixel_x": 140, "n_pads": int(len(rows_r1)),
                            "pixel_y_range": [5, 131]}
        rec["lambda_rel"] = lam_rel
        rec["sigma_H_us"] = sigmas
        rec["prolongations_coarse"] = pnames
        self._emit(store, rec, arrays)
        # in-memory products for the plotting algorithm of the same job
        self.put(store, "fine.solutions",
                 {"x": sol, "win_lo": win_lo, "lambda_rel": lam_rel,
                  "harness": H, "arms": arms, "arrays": arrays})

    def _truth_padsum(self, H, win_lo, n_win):
        v = np.zeros(n_win)
        j = H.truth_tick - win_lo
        ok = (j >= 0) & (j < n_win)
        np.add.at(v, j[ok], H.truth_q[ok])
        return v

    def _transverse_truth(self, H):
        v = np.zeros(H.nx)
        np.add.at(v, H.truth_ix, H.truth_q)
        return v

    @staticmethod
    def _store_prof(arrays, tag, pr):
        for k in ("line_xhat", "line_Hx", "line_e", "allpad_xhat"):
            arrays[f"prof_{tag}_{k}"] = pr[k].astype(np.float32)

    def _print_row(self, r):
        print(f"[{self.name}] s{r['sigma_H_us']:4.2f} {r['basis']:6s} "
              f"{str(r.get('prolongation')):14s} {r['arm']:20s} "
              f"E_rel {r['E_rel']:9.5f}  cons {r['conservation_rel']:+9.5f}  "
              f"Emax_line {r['E_max_line_ke_per_tick']:9.4f}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
# Okabe-Ito, colour-blind safe.  Fixed roles across every figure of this
# campaign: truth black, bin-integrated (production) blue, fine-binned
# vermillion, H-smoothed truth grey.
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
C_TRUTH, C_COARSE, C_FINE, C_HTRUTH = (OI["black"], OI["blue"],
                                       OI["vermillion"], OI["grey"])


def ieee_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": False, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.linewidth": 1.0, "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def save(fig, outdir: Path, name: str, made: list):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    made.append(str(outdir / f"{name}.png"))
    print(f"[figure] {outdir / (name + '.png')}")
    import matplotlib.pyplot as plt
    plt.close(fig)


@algorithm("FineBasisPlots")
class FineBasisPlots(_Recorder):
    """Figures F1-F5, F7, F8 from the products of :class:`FineBasisInverse`.

    Reads the store, not a file, so the figures cannot drift from the numbers
    of the same job.  Written at ``finalize``.

    Props
    -----
    figdir : str            where the PNG/PDF pairs go.
    lambda_best : float     the ``lambda_rel`` used wherever one curve per
                            estimator is plotted; default the smallest
                            ``E_rel`` at ``sigma_H = 1.5`` us.
    coarse_arm : str        the bin-integrated arm shown; default ``LS_nopos``.
    out_json : str
    """

    reads = ("fine.result", "fine.solutions")
    writes = ("fine.figures",)

    def execute(self, store):
        self._res = store.get("fine.result")
        self._sol = store.get("fine.solutions")
        self._arr = self._sol["arrays"]
        self._store = store
        self.put(store, "fine.figures", {"pending": True})

    def finalize(self):
        plt = ieee_style()
        rec, sol, A = self._res, self._sol, self._arr
        outdir = Path(self.props.get("figdir", "figs"))
        made: list = []
        lam_rel = rec["lambda_rel"]
        sigmas = rec["sigma_H_us"]
        arm = str(self.props.get("coarse_arm", "LS_nopos"))
        rows = rec["rows"]

        def get(armname, s, pn=None):
            for r in rows:
                if (r["arm"] == armname and abs(r["sigma_H_us"] - s) < 1e-12
                        and (pn is None or r.get("prolongation") == pn)):
                    return r
            return None

        lb = self.props.get("lambda_best")
        if lb is None:
            cand = [(get(f"fine_lrel{lr:g}", 1.5)["E_rel"], lr) for lr in lam_rel]
            lb = min(cand)[1]
        self._lambda_best = lb
        B = rec["geometry"]["B_fine_ticks"]
        dt = TICK_US

        # ---------------- F1: operator construction ----------------------
        tau = np.arange(len(A["kbar_own"])) * dt
        fig, ax = plt.subplots(1, 4, figsize=(11.0, 2.4))
        m = (tau >= 3812 * dt - 12.0) & (tau <= 3812 * dt + 2.0)
        t0 = 3812 * dt
        ax[0].plot(tau[m] - t0, A["kbar_own"][m] / dt, color=C_TRUTH, lw=0.9,
                   label=r"$\bar{K}_{(0,0)}/\Delta t$ [e/$\mu$s]")
        ax0b = ax[0].twinx()
        ax0b.plot(tau[m] - t0, A["kcum_own"][m], color=OI["green"], lw=0.9,
                  label=r"$\overline{Kcum}_{(0,0)}$")
        ax0b.set_ylabel(r"$\overline{Kcum}_{(0,0)}$", color=OI["green"])
        ax0b.tick_params(axis="y", colors=OI["green"], direction="in")
        ax[0].set_xlabel(r"$\tau-\tau_{\rm arrival}$ [$\mu$s]")
        ax[0].set_ylabel(r"current [e/$\mu$s]")
        ax[0].set_title("(a) own pad: current and cumulative")

        for k, (lab, hk, bk, ttl) in enumerate([
                ("own", "h_own", "kbin_own", r"(b) $d=(0,0)$"),
                ("ring1", "h_ring1", "kbin_ring1", r"(c) $d=(1,0)$")]):
            a = ax[1 + k]
            th = np.arange(len(A[hk])) * dt - t0
            mm = (th >= -12.0) & (th <= 3.0)
            a.plot(th[mm], A[hk][mm], color=C_FINE, lw=1.0,
                   label=r"$h_d(\tau)$, fine")
            kb = A[bk]
            edges = (np.arange(len(kb) + 1) * B) * dt - t0
            a.stairs(kb, edges, color=C_COARSE, lw=1.0,
                     label=r"$\bar{K}^{\rm bin}_d$, coarse")
            a.set_xlim(-12.0, 3.0)
            a.set_xlabel(r"$\tau-\tau_{\rm arrival}$ [$\mu$s]")
            a.set_ylabel("charge per unit released")
            a.set_title(ttl)
            a.legend(frameon=False, loc="upper left")

        a = ax[3]
        # one record row as weights on the unknown's release time, zoomed on
        # three 1.5 us cells around the peak.  The fine row gives every one of
        # the 30 ticks in a cell its own weight; the coarse row has ONE number
        # per cell, and that number is the fine row sampled at tau = 30m + 29.
        th = (np.arange(len(A["h_own"])) - 3812) * dt
        mm = (th >= -3.2) & (th <= 1.6)
        a.plot(th[mm], A["h_own"][mm], color=C_FINE, lw=1.0,
               label=r"$A_{\rm fine}$ row: $h_{(0,0)}(b{+}30(w{+}1)-j)$")
        kb = A["kbin_own"]
        tc = (np.arange(len(kb)) * B + B - 1 - 3812) * dt
        sel = (tc >= -3.2) & (tc <= 1.6)
        a.vlines(tc[sel], 0.0, kb[sel], color=C_COARSE, lw=0.9)
        a.plot(tc[sel], kb[sel], ls="none", marker="o", ms=3.0,
               color=C_COARSE,
               label=r"$A_{\rm coarse}$ row: $\bar{K}^{\rm bin}_{(0,0)}[w-k]$")
        for e in np.arange(-3, 2) * B * dt - (3812 % B) * dt:
            a.axvline(e, color=OI["grey"], lw=0.4, ls=":")
        a.axhline(0, color="k", lw=0.5)
        a.set_xlim(-3.2, 1.6)
        a.set_xlabel(r"unknown release time $\tau-\tau_{\rm arrival}$ [$\mu$s]")
        a.set_ylabel("operator row weight")
        a.set_title("(d) one record row, zoomed on three cells")
        a.legend(frameon=False, loc="upper left", fontsize=6)
        ax[0].legend(frameon=False, loc="upper left")
        fig.tight_layout()
        save(fig, outdir, "F1_operator_construction", made)

        # ---------------- F2: determinacy --------------------------------
        N = rec["geometry"]["N_fine_ticks"]
        M = rec["geometry"]["M_windows"]
        f_MHz = np.arange(len(A["hhat_own_abs"])) / (N * dt)   # 1/us = MHz
        fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.6))
        a = ax[0]
        a.semilogy(f_MHz, np.maximum(A["hhat_own_abs"], 1e-12), color=C_FINE,
                   lw=0.8, label=r"$|\hat{h}_{(0,0)}(f)|$")
        for mm in range(1, 16):
            a.axvline(mm / (B * dt), color=OI["grey"], lw=0.4, ls=":")
        a.axvline(1.0 / (2 * B * dt), color=C_COARSE, lw=0.9,
                  label=r"record Nyquist $1/(3\,\mu$s)")
        for s, cc in ((1.5, OI["green"]), (2.0, OI["purple"])):
            a.semilogy(f_MHz, np.exp(-0.5 * (2 * np.pi * f_MHz * s) ** 2),
                       color=cc, lw=0.9, ls="--",
                       label=rf"$H$ gain, $\sigma_H={s}\,\mu$s")
        a.set_xlim(0, 10.0)
        a.set_ylim(1e-8, 60)
        a.set_xlabel("frequency [MHz]")
        a.set_ylabel(r"$|\hat{h}|$,  $H$ gain")
        a.set_title("(a) fine-grid transfer function")
        a.legend(frameon=False, fontsize=6, loc="upper right")

        a = ax[1]
        # the BASE BAND is the signed frequency nu in [-M/2, M/2), i.e. the
        # fine-grid index f = nu mod N; |hhat| is stored for f = 0..N/2 only,
        # so f > N/2 is read back through the Hermitian symmetry |hhat(N-f)|.
        hh = A["hhat_own_abs"]
        nu = np.arange(M)
        nu_s = np.where(nu <= M // 2, nu, nu - M)
        base = np.where(nu <= M // 2, hh[np.minimum(nu, len(hh) - 1)] ** 2,
                        hh[np.abs(nu - M)] ** 2)
        share = base / np.maximum(A["alias_power_sum"], 1e-300)
        o = np.argsort(nu_s)
        fnu = nu_s[o] / (M * B * dt)
        a.semilogy(fnu, np.maximum(share[o], 1e-12), color=C_FINE, lw=0.8)
        a.set_xlabel(r"record-grid frequency $\nu$ [MHz]")
        a.set_ylabel("base-band share")
        a.set_xlim(-1.0 / (2 * B * dt), 1.0 / (2 * B * dt))
        a.set_title("(b) share of the base band vs its 29 aliases")

        a = ax[2]
        G = A["G_own_transverse_dc"]
        nug = np.arange(len(G)) / (M * B * dt)
        a.semilogy(nug, np.maximum(G, 1e-30), color=C_FINE, lw=0.8,
                   label=r"$\hat{G}(0,0,\nu)$")
        gm = rec["kernel"]["G_max"]
        for lr in lam_rel:
            a.axhline(lr * gm, color=OI["grey"], lw=0.5, ls="--")
            a.text(nug[-1], lr * gm, rf"$\lambda_{{\rm rel}}=${lr:g}",
                   fontsize=5.5, va="bottom", ha="right", color=OI["grey"])
        a.set_xlabel(r"record-grid frequency $\nu$ [MHz]")
        a.set_ylabel(r"$\hat{G}$")
        a.set_title(r"(c) $\hat{G}$ and the $\lambda$ levels")
        a.legend(frameon=False, loc="lower left")
        fig.tight_layout()
        save(fig, outdir, "F2_determinacy", made)

        # ---------------- F3: time profiles on the line -------------------
        ft = A["fine_ticks"].astype(float)
        cen = float((H_centroid := np.sum(A["truth_padsum_fine"]
                                          * A["stored_window_ticks"])
                     / max(A["truth_padsum_fine"].sum(), 1e-30)))
        tus = (ft - cen) * dt
        fig, ax = plt.subplots(1, 3, figsize=(10.0, 2.6), sharex=True)
        for i, s in enumerate([0.0, 1.5, 2.0]):
            a = ax[i]
            tag_c = f"{arm}_corrected_hat_s{s:g}"
            tag_f = f"fine_lrel{lb:g}_s{s:g}"
            a.plot(tus, A[f"prof_{tag_c}_line_Hx"], color=C_TRUTH if s == 0
                   else C_HTRUTH, lw=1.1,
                   label=(r"$x$ (fine truth)" if s == 0 else r"$Hx$"))
            if s == 0:
                cc = A["coarse_centers"]
                edges = np.concatenate([cc - B / 2.0, [cc[-1] + B / 2.0]])
                sel = (edges >= ft[0]) & (edges <= ft[-1] + 1)
                kk = np.nonzero(sel[:-1] & sel[1:])[0]
                a.stairs(A["line_coarse_" + arm][kk] / B,
                         (edges[kk[0]:kk[-1] + 2] - cen) * dt,
                         color=C_COARSE, lw=1.0,
                         label=rf"{arm} cells /1.5 $\mu$s")
            else:
                a.plot(tus, A[f"prof_{tag_c}_line_xhat"], color=C_COARSE,
                       lw=1.0, label=rf"$H P_1$ {arm}")
            a.plot(tus, A[f"prof_{tag_f}_line_xhat"], color=C_FINE, lw=1.0,
                   label=rf"$H\,\hat{{x}}_{{\rm fine}}$ ($\lambda_{{\rm rel}}$={lb:g})")
            a.set_xlabel(r"time from the truth centroid [$\mu$s]")
            a.set_title(rf"$\sigma_H = {s:g}\,\mu$s" if s else
                        r"$\sigma_H = 0$ (raw)")
            a.legend(frameon=False, fontsize=6)
        ax[0].set_ylabel("mean over interior line pads [ke / fine tick]")
        ax[0].set_xlim(-9, 9)
        fig.tight_layout()
        save(fig, outdir, "F3_line_time_profiles", made)

        # ---------------- F4: oscillation in bin space --------------------
        osc = rec["coarse_pad_summed_profile"]
        ks = osc["cells"]
        x = np.arange(len(ks))
        series = [("$R\\,x$ (truth)", osc["truth_Rbox_ke"], C_TRUTH),
                  (arm, osc[arm + "_ke"], C_COARSE),
                  (rf"$R\,\hat{{x}}_{{\rm fine}}$ ($\lambda_{{\rm rel}}$={lb:g})",
                   osc[f"fine_lrel{lb:g}_Rxhat_ke"], C_FINE)]
        fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.6))
        wdt = 0.27
        for i, (lab, v, cc) in enumerate(series):
            ax[0].bar(x + (i - 1) * wdt, v, wdt, color=cc, label=lab)
            if i:
                ax[1].bar(x + (i - 1.5) * wdt, np.asarray(v)
                          - np.asarray(osc["truth_Rbox_ke"]), wdt, color=cc,
                          label=lab)
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels([f"{k}" for k in ks], fontsize=6)
            a.set_xlabel(r"coarse cell $k$ (1.5 $\mu$s)")
            a.axhline(0, color="k", lw=0.5)
            a.legend(frameon=False, fontsize=6)
        ax[0].set_ylabel("pad-summed charge [ke]")
        ax[1].set_ylabel(r"difference to $R\,x$ [ke]")
        ax[0].set_title("(a) charge per 1.5 $\\mu$s cell")
        ax[1].set_title("(b) difference to the truth cells")
        fig.tight_layout()
        save(fig, outdir, "F4_bin_space", made)

        # ---------------- F5: E_rel at the goal ---------------------------
        entries = [("representation ($R\\,x$)", "representation_term",
                    "corrected_hat", OI["grey"]),
                   (f"{arm}, $P_\\delta$", arm, "delta", OI["sky"]),
                   (f"{arm}, $P_1$", arm, "corrected_hat", C_COARSE),
                   ("FFT, $P_1$", "FFT", "corrected_hat", OI["green"])]
        entries += [(rf"fine, $\lambda_{{\rm rel}}$={lr:g}",
                     f"fine_lrel{lr:g}", None, C_FINE) for lr in lam_rel]
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        xs = np.arange(len(entries))
        for i, s in enumerate((1.5, 2.0)):
            v = []
            for _, an, pn, _c in entries:
                r = get(an, s, pn)
                v.append(r["E_rel"] if r else np.nan)
            ax.bar(xs + (i - 0.5) * 0.4, v, 0.4,
                   color=[e[3] for e in entries],
                   alpha=1.0 if i == 0 else 0.55,
                   edgecolor="k", linewidth=0.4,
                   label=rf"$\sigma_H={s:g}\,\mu$s")
            for xi, vv in zip(xs, v):
                ax.text(xi + (i - 0.5) * 0.4, vv, f"{vv:.3f}", fontsize=5.5,
                        ha="center", va="bottom", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([e[0] for e in entries], rotation=25, ha="right",
                           fontsize=6)
        ax.set_ylabel(r"$E_{\rm rel}$")
        ax.set_ylim(0, max(1e-3, np.nanmax([get(e[1], 1.5, e[2])["E_rel"]
                                            for e in entries])) * 1.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, outdir, "F5_Erel_at_goal", made)

        # ---------------- F7: zero preservation ---------------------------
        fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.7))
        a = ax[0]
        for s, ls in ((0.0, "-"), (1.5, "--")):
            a.plot(tus, A[f"ring1prof_{arm}_corrected_hat_s{s:g}"],
                   color=C_COARSE, ls=ls, lw=0.9,
                   label=rf"{arm}, $\sigma_H={s:g}$")
            a.plot(tus, A[f"ring1prof_fine_lrel{lb:g}_s{s:g}"], color=C_FINE,
                   ls=ls, lw=0.9, label=rf"fine, $\sigma_H={s:g}$")
        a.axhline(0, color="k", lw=0.5)
        a.set_xlim(-9, 9)
        a.set_xlabel(r"time from the truth centroid [$\mu$s]")
        a.set_ylabel("mean over the pixel_x = 140 row [ke / fine tick]")
        a.set_title("(a) ring-1 pad row (no truth charge)")
        a.legend(frameon=False, fontsize=6)

        a = ax[1]
        px = np.arange(rec["geometry"]["q_shape"][0]) \
            + int(rec["geometry"]["block_offset"][0])
        a.plot(px, A["transverse_truth"], color=C_TRUTH, lw=1.0, marker="o",
               ms=2.2, label="truth")
        a.plot(px, A["transverse_" + arm], color=C_COARSE, lw=1.0, marker="s",
               ms=2.2, label=arm)
        a.plot(px, A[f"transverse_fine_lrel{lb:g}"], color=C_FINE, lw=1.0,
               marker="^", ms=2.2, label=rf"fine ($\lambda_{{\rm rel}}$={lb:g})")
        a.set_yscale("symlog", linthresh=1e-2)
        a.set_xlim(129, 153)
        a.set_xlabel("pixel_x")
        a.set_ylabel(r"$\sum_{y,t} \hat{x}$ per pixel_x row [ke]")
        a.set_title("(b) transverse profile")
        a.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, outdir, "F7_zero_preservation", made)

        # ---------------- F8: lambda scan ---------------------------------
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        lr = np.array(lam_rel)
        for s, cc, mk in ((1.5, C_FINE, "o"), (2.0, OI["purple"], "s")):
            v = [get(f"fine_lrel{x:g}", s)["E_rel"] for x in lam_rel]
            ax.semilogx(lr, v, color=cc, marker=mk, ms=3,
                        label=rf"$E_{{\rm rel}}(\sigma_H={s:g}\,\mu$s)")
            r = get(arm, s, "corrected_hat")
            ax.axhline(r["E_rel"], color=cc, ls=":", lw=0.8)
            ax.text(lr.min(), r["E_rel"], f" {arm} $P_1$", fontsize=5.5,
                    color=cc, va="bottom")
        ax.set_xlabel(r"$\lambda_{\rm rel} = \lambda / \max\hat{G}$")
        ax.set_ylabel(r"$E_{\rm rel}$")
        axb = ax.twinx()
        res = [r["residual_rel"] for r in rec["lambda_scan"]]
        axb.loglog([r["lambda_rel"] for r in rec["lambda_scan"]], res,
                   color=OI["green"], marker="v", ms=3, ls="--",
                   label=r"$\|A\hat{x}-y\|/\|y\|$")
        axb.set_ylabel(r"$\|A\hat{x}-y\|/\|y\|$", color=OI["green"])
        axb.tick_params(axis="y", colors=OI["green"], direction="in")
        ax.legend(frameon=False, fontsize=6, loc="center left")
        fig.tight_layout()
        save(fig, outdir, "F8_lambda_scan", made)

        out = {"figures": made, "lambda_best_rel": lb, "coarse_arm": arm}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": out}, fh, indent=1,
                          default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return out


# ---------------------------------------------------------------------------
# the resolution probe, for both bases
# ---------------------------------------------------------------------------
def probe_metrics(H: EvalHarness, xhat: np.ndarray, Q: float, t_star: int,
                  pad_flat: int, sigma_us: float) -> dict:
    """Column metrics of the resolution matrix, definitions of ``ResolutionProbe``.

    ``normalisation`` ``sum xhat / Q``; ``shift`` the first moment in fine
    ticks measured from ``t*``; ``width`` the square root of the second central
    moment (``null`` when the signed weights make it negative); ``ring k`` the
    positive and negative parts on pads at Chebyshev distance ``k`` from the
    probed pad, in units of ``Q``.
    """
    u = H.fine.astype(float) - float(t_star)
    w = xhat.sum(axis=0)
    m0 = float(w.sum())
    m1 = float((w * u).sum())
    m2 = float((w * u ** 2).sum())
    mean = m1 / m0 if m0 else float("nan")
    var = m2 / m0 - mean ** 2 if m0 else float("nan")
    ch = H.chebyshev
    out = {"sigma_H_us": float(sigma_us), "normalisation": m0 / Q,
           "own_pad_sum_over_Q": float(xhat[pad_flat].sum() / Q),
           "shift_ticks": mean, "shift_us": mean * TICK_US,
           "width_ticks": float(np.sqrt(var)) if var >= 0 else None}
    for r in (1, 2, 3):
        sel = (ch == r) if r < 3 else (ch >= 3)
        blk = xhat[sel]
        lab = f"ring{r}" if r < 3 else "ring_ge3"
        out[lab + "_pos_over_Q"] = float(blk[blk > 0].sum() / Q)
        out[lab + "_neg_over_Q"] = float(blk[blk < 0].sum() / Q)
    return out


@algorithm("FineBasisProbe")
class FineBasisProbe(_Recorder):
    """Resolution matrix of both estimators by unit fine impulses.

    One fine cell of charge ``Q`` at pad ``p*`` and fine tick
    ``t* = c_{k*} - B/2 + phi``; its records are built by the SAME
    exact-functional generator as
    :class:`~unfoldlarpix.algs.evalharness_algs.ResolutionProbe`
    (``records_from_impulse``: the cumulative kernel at the event's own latch
    times, with tred's tick-0 current deletion), mapped onto the operator rows
    through ``row_meta``, and the mapping is verified against the real event's
    ``op.d`` before any probe is solved.  The bin-integrated arm is solved by
    FISTA on the support exactly as before; the fine arm is the closed-form
    Tikhonov inverse of :class:`FineOperator` at one stated ``lambda_rel``.

    Props
    -----
    probe_pad, probe_phases, probe_impacts_at, probe_cell, charge_ke,
    current_zero_before_tick, sigma_H_us, margin_windows : as ``ResolutionProbe``.
    lambda_rel : float      the fine arm's Tikhonov floor (default 1e-6).
    coarse_arm : dict       spec for the bin-integrated arm.
    dtype, out_json, out_npz
    """

    reads = ("op", "support", "readout_config", "block_offset", "charge_model",
             "row_meta", "hits_view", "event")
    writes = ("fine.resolution",)

    def execute(self, store):
        import copy
        op = store.get("op")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        c, Bf, off = coarse_centers(store, op)
        B = int(round(Bf))
        Q = float(self.props.get("charge_ke", 30.0))
        tz = self.props.get("current_zero_before_tick", 0)
        tz = None if tz is None else int(tz)
        phases = [int(v) for v in self.props.get(
            "probe_phases", [0, 5, 10, 15, 20, 25, 29])]
        imp_at = {int(k): [str(x) for x in v] for k, v in
                  (self.props.get("probe_impacts_at")
                   or {15: ["bar", "4,4", "0,0"]}).items()}
        ppad = [int(v) for v in self.props.get("probe_pad", [141, 68])]
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [0.0, 1.5, 2.0])]
        margin = int(self.props.get("margin_windows", 40))
        lam_rel = float(self.props.get("lambda_rel", 1e-6))
        cspec = self.props.get("coarse_arm") or {
            "label": "LS_nopos", "alpha": 0.0, "positivity": False,
            "support": "gain:0.5", "iters": 2000}
        dtype = (torch.float64 if str(self.props.get("dtype", "float64"))
                 == "float64" else torch.float32)
        dev = op.device

        # -- row mapping, verified on the real event -------------------------
        look = row_lookup(store, op)
        hv = store.get("hits_view")
        loc = np.asarray(hv.location)
        Cq = np.asarray(hv.cumulative_charges, dtype=np.float64)
        yreal = np.diff(np.concatenate([np.zeros((len(Cq), 1)), Cq], axis=1),
                        axis=1)
        Nl = yreal.shape[1]
        trig = int(np.unique(hv.trigger)[0])
        d_check = np.zeros(op.n_data)
        miss = 0
        for i in range(len(loc)):
            px = int(loc[i, 0] - boff[0]); py = int(loc[i, 1] - boff[1])
            for k in range(1, Nl + 1):
                r = look.get((px, py, int(trig + k * B - boff[2])))
                if r is None:
                    miss += 1
                    continue
                d_check[r] = yreal[i, k - 1]
        d_op = op.d.detach().cpu().numpy().astype(np.float64)
        map_err = float(np.abs(d_check - d_op).max())
        rec: dict = {"row_mapping_check": {
            "n_rows": int(op.n_data), "n_unmapped_windows": int(miss),
            "max_abs_difference_ke": map_err, "tolerance": 1e-6,
            "passed": bool(map_err < 1e-6 and miss == 0)}}
        print(f"[{self.name}] row mapping: max|d_mapped - op.d| = {map_err:.3e} ke")
        if not rec["row_mapping_check"]["passed"]:
            raise AssertionError("row_meta -> window mapping does not reproduce op.d")

        # -- kernels ------------------------------------------------------------
        path = self.props.get("response") or self.services["detector"].response_path
        need = {"bar"} | {x for v in imp_at.values() for x in v}
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

        prep = self.services["detector"].prepared(B)
        F = FineOperator(np.asarray(prep.full_response, dtype=np.float64),
                         op.block_shape, B, device=dev, dtype=dtype)
        lam = lam_rel * F.G_max
        rec["fine_arm"] = {"lambda_rel": lam_rel, "lambda": lam,
                           "G_max": F.G_max}

        # -- geometry -----------------------------------------------------------
        kstar = int(self.props.get("probe_cell") or
                    np.argmax(np.abs(np.zeros(1)) if False else
                              self._truth_cell(store, op, c, Bf)))
        px_b = ppad[0] - int(boff[0]); py_b = ppad[1] - int(boff[1])
        ring = 12
        dxy = np.array([(dx, dy) for dx in range(-ring, ring + 1)
                        for dy in range(-ring, ring + 1)], dtype=int)
        pads_abs = np.stack([dxy[:, 0] + px_b, dxy[:, 1] + py_b], axis=1)
        ok = ((pads_abs[:, 0] >= 0) & (pads_abs[:, 0] < op.q_shape[0])
              & (pads_abs[:, 1] >= 0) & (pads_abs[:, 1] < op.q_shape[1]))
        dxy, pads_abs = dxy[ok], pads_abs[ok]
        latch_abs = trig + np.arange(Nl + 1) * B
        rows_for = np.full((len(pads_abs), Nl), -1, dtype=np.int64)
        bins_for = np.full((len(pads_abs), Nl), -1, dtype=np.int64)
        for i, (bx, by) in enumerate(pads_abs):
            for k in range(1, Nl + 1):
                r = look.get((int(bx), int(by), int(trig + k * B - boff[2])))
                if r is not None:
                    rows_for[i, k - 1] = r
                    bins_for[i, k - 1] = (trig + k * B - int(boff[2])) // B - 1
        rec["probe_geometry"] = {"probe_cell_k": kstar,
                                 "cell_center_tick": float(c[kstar]),
                                 "probe_pad_pixel": ppad,
                                 "n_pads_receiving": int(len(pads_abs)),
                                 "n_unmapped": int((rows_for < 0).sum())}
        supports = {}
        results, arrays = [], {}
        t_start = time.time()
        for phi in phases:
            for kn in imp_at.get(phi, ["bar"]):
                t_star = int(np.floor(c[kstar] - Bf / 2.0)) + int(phi)
                y = records_from_impulse(kcums[kn], dxy, latch_abs, t_star, Q, tz)
                dvec = np.zeros(op.n_data)
                good = rows_for >= 0
                dvec[rows_for[good]] = y[good]
                blk = np.zeros(op.block_shape)
                blk[pads_abs[:, 0][:, None].repeat(Nl, 1)[good],
                    pads_abs[:, 1][:, None].repeat(Nl, 1)[good],
                    bins_for[good]] = y[good]
                truth = ((np.array([px_b]), np.array([py_b]),
                          np.array([t_star])), np.array([Q]))
                Hh = EvalHarness(store, op, margin_windows=margin,
                                 line_pixel_y_range=(-10 ** 9, 10 ** 9),
                                 truth=truth)
                pad_flat = px_b * Hh.ny + py_b
                pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
                wlo = int(Hh.fine[0]) - pad_ext
                whi = int(Hh.fine[-1]) + 1 + pad_ext

                # coarse arm
                op_p = copy.copy(op)
                op_p.d = torch.as_tensor(dvec, dtype=op.dtype, device=op.device)
                ss = cspec.get("support")
                if ss not in supports:
                    supports[ss] = resolve_support(store, op, ss)
                t0 = time.time()
                qc = solve_arm(op_p, supports[ss], cspec.get("alpha", 0.0),
                               bool(cspec.get("positivity", False)),
                               int(cspec.get("iters", 2000)))
                wall_c = time.time() - t0
                del op_p
                torch.cuda.empty_cache()

                # fine arm
                torch.cuda.empty_cache()
                t0 = time.time()
                yt = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
                xf = F.solve(yt, lam)
                xwin = xf[:Hh.nx, :Hh.ny, wlo - b:whi - b].cpu().numpy()
                del xf, yt
                torch.cuda.empty_cache()
                wall_f = time.time() - t0

                col = qc[px_b, py_b, :]
                ideal = np.zeros_like(col); ideal[kstar] = Q
                lo, hi = max(kstar - 4, 0), min(kstar + 4, len(col))
                base = {"phi": int(phi), "kernel": kn, "t_star": int(t_star),
                        "sum_d_probe_ke": float(dvec.sum()),
                        "sum_abs_d_probe_ke": float(np.abs(dvec).sum())}
                for lab, mk in (
                        (str(cspec["label"]), lambda s: coarse_xhat(
                            Hh, qc, "corrected_hat", s)),
                        (f"fine_lrel{lam_rel:g}",
                         lambda s: fine_xhat(Hh, xwin, wlo, s))):
                    r = {**base, "arm": lab,
                         "basis": "fine" if lab.startswith("fine") else "coarse",
                         "wall_s": wall_f if lab.startswith("fine") else wall_c,
                         "sigmas": []}
                    if not lab.startswith("fine"):
                        r["A_alt"] = float(np.abs(col - ideal).max() / Q)
                        r["coarse_column_over_Q"] = [float(v / Q)
                                                     for v in col[lo:hi]]
                        r["coarse_column_cells"] = list(range(lo, hi))
                    for s in sigmas:
                        xh = mk(s)
                        r["sigmas"].append(probe_metrics(Hh, xh, Q, t_star,
                                                         pad_flat, s))
                        if s in (0.0, 1.5):
                            arrays[f"imp_phi{phi}_{kn}_{lab}_s{s:g}"] = \
                                xh[pad_flat].astype(np.float32)
                    results.append(r)
                    m15 = [d for d in r["sigmas"] if abs(d["sigma_H_us"] - 1.5) < 1e-9]
                    m15 = m15[0] if m15 else r["sigmas"][0]
                    print(f"[{self.name}] phi {phi:2d} {kn:5s} {lab:16s} "
                          f"norm {m15['normalisation']:8.5f} shift "
                          f"{m15['shift_ticks']:+7.2f} width "
                          f"{m15['width_ticks'] if m15['width_ticks'] else float('nan'):7.2f}"
                          f" | {time.time() - t_start:.0f} s")
                arrays[f"fine_ticks_phi{phi}_{kn}"] = Hh.fine.astype(np.int64)
                # H delta reference
                for s in (1.5,):
                    gk = time_kernel(s / TICK_US)
                    ref = np.zeros(Hh.n_fine)
                    j0 = t_star - int(Hh.fine[0])
                    half = (len(gk) - 1) // 2
                    a0, a1 = max(0, j0 - half), min(Hh.n_fine, j0 + half + 1)
                    ref[a0:a1] = gk[a0 - (j0 - half):a1 - (j0 - half)] * Q
                    arrays[f"Hdelta_phi{phi}_{kn}_s{s:g}"] = ref.astype(np.float32)
        rec["probes"] = results
        rec["sigma_H_us"] = sigmas
        self._emit(store, rec, arrays)

    @staticmethod
    def _truth_cell(store, op, c, B):
        from .fixedgrid_algs import grid_truth
        return grid_truth(store, op, mode="round").sum(axis=(0, 1))


@algorithm("FineBasisProbePlots")
class FineBasisProbePlots(_Recorder):
    """Figure F6 from the products of :class:`FineBasisProbe` in the same job."""

    reads = ("fine.resolution",)
    writes = ("fine.probe_figures",)

    def execute(self, store):
        self._res = store.get("fine.resolution")
        self._arr = {}
        self.put(store, "fine.probe_figures", {"pending": True})

    def finalize(self):
        plt = ieee_style()
        rec = self._res
        npz = self.props.get("in_npz")
        A = dict(np.load(npz, allow_pickle=True)) if npz else {}
        outdir = Path(self.props.get("figdir", "figs"))
        made: list = []
        pr = rec["probes"]
        arms = sorted({r["arm"] for r in pr})
        c_arm = [a for a in arms if not a.startswith("fine")][0]
        f_arm = [a for a in arms if a.startswith("fine")][0]
        phases = sorted({r["phi"] for r in pr if r["kernel"] == "bar"})

        def g(arm, phi, kn, s):
            for r in pr:
                if r["arm"] == arm and r["phi"] == phi and r["kernel"] == kn:
                    for d in r["sigmas"]:
                        if abs(d["sigma_H_us"] - s) < 1e-9:
                            return d
            return None

        fig = plt.figure(figsize=(10.5, 5.2))
        gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)
        show = [p for p in (0, 10, 20, 29) if p in phases]
        for i, phi in enumerate(show):
            a = fig.add_subplot(gs[0, i])
            t = A.get(f"fine_ticks_phi{phi}_bar")
            if t is None:
                continue
            tstar = [r["t_star"] for r in pr if r["phi"] == phi][0]
            tu = (t - tstar) * TICK_US
            a.plot(tu, A[f"Hdelta_phi{phi}_bar_s1.5"], color=C_TRUTH, lw=1.0,
                   label=r"$H\delta$")
            a.plot(tu, A[f"imp_phi{phi}_bar_{c_arm}_s1.5"], color=C_COARSE,
                   lw=1.0, label=c_arm)
            a.plot(tu, A[f"imp_phi{phi}_bar_{f_arm}_s1.5"], color=C_FINE,
                   lw=1.0, label="fine")
            a.set_xlim(-6, 6)
            a.set_xlabel(r"$t-t^*$ [$\mu$s]")
            if i == 0:
                a.set_ylabel(r"$\hat{x}$ on the probed pad [ke/tick]")
                a.legend(frameon=False, fontsize=6)
            a.set_title(rf"(a) $\varphi={phi}$, $\sigma_H=1.5\,\mu$s")

        a = fig.add_subplot(gs[1, 0])
        for arm, cc in ((c_arm, C_COARSE), (f_arm, C_FINE)):
            a.plot(phases, [g(arm, p, "bar", 0.0)["shift_ticks"] for p in phases],
                   color=cc, marker="o", ms=2.5, label=arm)
        a.plot(phases, [14.5 - p for p in phases], color=C_HTRUTH, ls="--",
               lw=0.9, label=r"coarse-basis ideal $14.5-\varphi$")
        a.axhline(0, color=C_TRUTH, lw=0.7, ls=":")
        a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
        a.set_ylabel("time shift [fine ticks]")
        a.set_title(r"(b) first moment, $\sigma_H=0$")
        a.legend(frameon=False, fontsize=6)

        a = fig.add_subplot(gs[1, 1])
        for arm, cc in ((c_arm, C_COARSE), (f_arm, C_FINE)):
            v = [g(arm, p, "bar", 1.5)["width_ticks"] for p in phases]
            a.plot(phases, [np.nan if x is None else x for x in v], color=cc,
                   marker="o", ms=2.5, label=arm)
        a.axhline(1.5 / TICK_US, color=C_HTRUTH, ls="--", lw=0.9,
                  label=r"ideal $\sigma_H/\Delta t = 30$")
        a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
        a.set_ylabel("width [fine ticks]")
        a.set_title(r"(c) second moment, $\sigma_H=1.5\,\mu$s")
        a.legend(frameon=False, fontsize=6)

        kns = [k for k in ("bar", "4,4", "0,0")
               if any(r["kernel"] == k and r["phi"] == 15 for r in pr)]
        styles = {"bar": ("-", "impact-averaged"), "4,4": ("--", "impact (4,4)"),
                  "0,0": (":", "impact (0,0)")}
        for i, (arm, cc, ttl) in enumerate(
                ((c_arm, C_COARSE, c_arm), (f_arm, C_FINE, "fine"))):
            a = fig.add_subplot(gs[1, 2 + i])
            tstar = [r["t_star"] for r in pr if r["phi"] == 15][0]
            for kn in kns:
                t = A.get(f"fine_ticks_phi15_{kn}")
                if t is None:
                    continue
                tu = (t - tstar) * TICK_US
                ls, lab = styles[kn]
                a.plot(tu, A[f"imp_phi15_{kn}_{arm}_s1.5"], color=cc, ls=ls,
                       lw=1.0, label=lab)
            t = A.get("fine_ticks_phi15_bar")
            if t is not None:
                a.plot((t - tstar) * TICK_US, A["Hdelta_phi15_bar_s1.5"],
                       color=C_TRUTH, lw=1.0, label=r"$H\delta$")
            a.set_xlim(-6, 6)
            a.set_xlabel(r"$t-t^*$ [$\mu$s]")
            a.set_title(rf"(d) $\varphi=15$, {ttl}")
            a.legend(frameon=False, fontsize=6)
        save(fig, outdir, "F6_resolution_matrix", made)
        out = {"figures": made}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": out}, fh, indent=1,
                          default=str)
        return out


# ---------------------------------------------------------------------------
# the intermediate basis: closed-form filtered inverse and the ladder in c
# ---------------------------------------------------------------------------
@algorithm("CellBasisInverse")
class CellBasisInverse(_Recorder):
    """``A_c`` on the ``c``-tick cell basis: validate, invert, score, ladder.

    The same closed-form filtered inverse as
    :class:`FineBasisInverse` -- ``xhat = A^T (A A^T + lambda I)^{-1} y`` with
    the symbol ``G_c`` of the module docstring -- on the intermediate basis.
    It does NOT read the bin-integrated arms, so the job needs no solver and
    the archived coarse and fine references are quoted, not recomputed.

    Validations, all asserted and all in the output:

    1. ``A_c`` applied to ``R_c x`` against the exact-functional DIRECT SUM of
       the FINE formula applied to the cell truth's fine representative
       (``P_0 R_c x`` for ``cell_model = uniform``, unit mass at the cell's
       lower edge for ``delta``).  This is ``A_c (R_c x)`` against
       ``A_fine (P_0 R_c x)`` evaluated without any operator, FFT or
       periodicity.
    2. the dot-product test ``<A x, y> = <x, A^T y>``.
    3. ``(A A^T) y`` by the diagonal formula ``G_c`` against ``A(A^T y)``.
    4. ``G_c(0) = ghat(0)^2 / D`` and ``sum_s g[s] = D sum Kbar``, the two
       identities the conservation statement rests on.
    5. ``c = 30`` with ``cell_model = delta`` against the PRODUCTION operator
       ``op.conv``, scanning the release shift: the production column is the
       cell column at ``release_shift = +1`` (the ``phi = 1`` convention).
    6. ``R_c P = I`` and ``1^T P = 1^T`` for ``P_0`` and ``P_1``
       (:meth:`CellGrid.report`).

    Props
    -----
    cell_ticks : int, default 5.       cell_model : ``"uniform"`` or ``"delta"``.
    lambda_rel : list of float, default ``[1e-4, 1e-6, 1e-8]``.
    sigma_H_us : list, default ``[0.0, 1.5, 2.0]``.
    cell_prolongations : list, default ``["uniform", "corrected_hat"]``.
    ladder : list of int, default ``[1, 5, 10, 30]`` -- the cell widths whose
        REPRESENTATION term is measured.
    ladder_solve : list of int, default ``[5, 10, 30]`` -- the subset whose
        linear inverse is also solved and scored.  ``c = 1`` is left out by
        default because its operator needs ~11 GB and its result is archived.
    ladder_lambda_rel : float, default 1e-6.
    check_coarse_identity : bool, default True.
    margin_windows, line_pixel_y_range, segment_pixels, segment_edge_exclude
    dtype : ``"float64"`` (default) or ``"float32"``.
    out_json, out_npz : str
    """

    reads = ("op", "event", "readout_config", "block_offset", "charge_model")
    writes = ("cell.result", "cell.solutions")

    def execute(self, store):
        op = store.get("op")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        B = int(round(fit_bin_ticks(store)))
        dtype = (torch.float64 if str(self.props.get("dtype", "float64"))
                 == "float64" else torch.float32)
        tol_fwd = 1e-6 if dtype == torch.float64 else 1e-4
        tol_lin = 1e-5 if dtype == torch.float64 else 1e-3
        cell_ticks = int(self.props.get("cell_ticks", 5))
        cell_model = str(self.props.get("cell_model", "uniform"))
        lam_rel = [float(v) for v in self.props.get(
            "lambda_rel", [1e-4, 1e-6, 1e-8])]
        sigmas = [float(v) for v in self.props.get("sigma_H_us",
                                                   [0.0, 1.5, 2.0])]
        pnames = [str(v) for v in self.props.get(
            "cell_prolongations", ["uniform", "corrected_hat"])]
        ladder = [int(v) for v in self.props.get("ladder", [1, 5, 10, 30])]
        ladder_solve = [int(v) for v in self.props.get("ladder_solve",
                                                       [5, 10, 30])]
        lam_ladder = float(self.props.get("ladder_lambda_rel", 1e-6))
        margin = int(self.props.get("margin_windows", 40))
        dev = op.device

        prep = self.services["detector"].prepared(B)
        fr = np.asarray(prep.full_response, dtype=np.float64)

        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))
        pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
        win_lo = int(H.fine[0]) - pad_ext
        win_hi = int(H.fine[-1]) + 1 + pad_ext
        rows_r1 = np.array([r for r in range(H.n_pads)
                            if H.pixel_x_of_pad[r] == 140
                            and 5 <= H.pixel_y_of_pad[r] <= 131])

        rec: dict = {
            "basis": {"cell_ticks": cell_ticks, "cell_model": cell_model,
                      "B_fine_ticks": B, "cells_per_record_window":
                          B // cell_ticks,
                      "dtype": str(dtype)},
            "geometry": {
                "block_shape": [int(v) for v in op.block_shape],
                "q_shape_production": [int(v) for v in op.q_shape],
                "block_offset": [float(v) for v in boff],
                "eval_window_fine_ticks": [int(H.fine[0]), int(H.fine[-1]) + 1],
                "eval_window_cells_production": [H.k0, H.k1],
                "stored_fine_window": [win_lo, win_hi]},
            "kernel": {"sum_Kbar_all_pads": float(fr.sum())},
        }
        arrays: dict = {}
        rows: list = []

        # ------------------------------------------------------------------
        # the main operator
        # ------------------------------------------------------------------
        t0 = time.time()
        F = FineOperator(fr, op.block_shape, B, device=dev, dtype=dtype,
                         cell_ticks=cell_ticks, cell_model=cell_model)
        t_build = time.time() - t0
        grid = CellGrid(b, cell_ticks, F.N)
        m_lo, m_hi = grid.window(win_lo, win_hi)
        Rx_c = grid.restrict(H.truth_ix, H.truth_iy, H.truth_tick, H.truth_q,
                             H.nx, H.ny)
        rec["basis"].update({
            "n_cells_per_pad": F.N, "decimation_stride_D": F.D,
            "M_windows": F.M, "padded_pads": [F.nxp, F.nyp],
            "n_unknowns_total": int(F.nxp * F.nyp * F.N),
            "n_unknowns_real_pads": int(H.nx * H.ny * F.N),
            "kernel_g_support_cells": int(F.g_np.shape[-1]),
            "build_wall_s": t_build,
            "stored_cell_window": [m_lo, m_hi]})
        rec["kernel"].update({
            "sum_g_all_pads": float(F.g_np.sum()),
            "sum_g_over_D_all_pads": float(F.g_np.sum() / F.D),
            "sum_g_over_D_minus_sum_Kbar": float(F.g_np.sum() / F.D - fr.sum()),
            "G_max": F.G_max, "G_dc": F.G_dc,
            "G_dc_over_ghat0_sq_over_D": float(
                F.G_dc / (F.g_np.sum() ** 2 / F.D)),
            "note": ("g is the D-tap moving sum on the cell grid of a shorter "
                     "kernel, so ghat vanishes at every non-zero multiple of "
                     "M and G_c(0) = ghat(0)^2 / D exactly")})
        rec["cell_grid"] = grid.report()
        rec["cell_grid"].update({
            "truth_total_ke": float(H.truth_total),
            "Rc_truth_total_ke": float(Rx_c.sum()),
            "Rc_truth_minus_truth_ke": float(Rx_c.sum() - H.truth_total),
            "n_nonzero_cells": int((Rx_c != 0).sum())})
        print(f"[{self.name}] cell basis c = {cell_ticks} ({cell_model}): "
              f"{F.N} cells/pad, D = {F.D}, {F.nxp}x{F.nyp} pads, built in "
              f"{t_build:.1f} s; max G {F.G_max:.6g}, G(0) {F.G_dc:.6g}")

        # -------- validation 1: forward vs the exact-functional direct sum --
        taps, wts = cell_charge_model_taps(grid, cell_model)
        IX, IY, TT, QQ = prolong_truth_to_fine(grid, Rx_c, taps, wts)
        t0 = time.time()
        xc = embed_pads(Rx_c, F.nxp, F.nyp, F.N, dev, dtype)
        y_fft = F.forward(xc).cpu().numpy()[:H.nx, :H.ny]
        del xc
        torch.cuda.empty_cache()
        y_dir = direct_sum_records(F.h_np, IX, IY, TT, QQ, b, B, H.nx, H.ny,
                                   F.M, dev, dtype)
        den = float(np.abs(y_dir).sum())
        v1 = float(np.abs(y_fft - y_dir).sum() / den)
        rec["validation_forward_vs_direct_sum"] = {
            "sum_abs_difference_ke": float(np.abs(y_fft - y_dir).sum()),
            "relative_to_sum_abs": v1,
            "max_abs_difference_ke": float(np.abs(y_fft - y_dir).max()),
            "sum_y_cell_operator_ke": float(y_fft.sum()),
            "sum_y_direct_fine_sum_ke": float(y_dir.sum()),
            "sum_abs_y_direct_ke": den,
            "n_fine_ticks_in_representative": int(len(TT)),
            "tolerance": tol_fwd, "wall_s": time.time() - t0,
            "note": ("A_c (R_c x) against the fine exact functional applied to "
                     "the cell truth's fine representative; the direct sum "
                     "uses no FFT and no periodicity")}
        print(f"[{self.name}] forward vs direct sum: {v1:.3e} of sum|y|")
        if not v1 < tol_fwd:
            raise AssertionError(f"forward vs direct sum {v1:.3e} > {tol_fwd:g}")

        # -------- validations 2 and 3 ---------------------------------------
        g = torch.Generator(device="cpu").manual_seed(11)
        dots, aat = [], []
        for _ in range(3):
            xr = torch.randn((F.nxp, F.nyp, F.N), generator=g,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            yr = torch.randn((F.nxp, F.nyp, F.M), generator=g,
                             dtype=torch.float64).to(device=dev, dtype=dtype)
            a = float((F.forward(xr) * yr).sum())
            bb = float((xr * F.adjoint(yr)).sum())
            dots.append({"Ax_y": a, "x_Aty": bb,
                         "relative_difference": abs(a - bb) / max(abs(a), 1e-30)})
            u = F.AAt_fft(yr)
            v = F.forward(F.adjoint(yr))
            aat.append({"relative_difference": float(
                torch.abs(u - v).sum() / torch.abs(v).sum())})
            del xr, yr, u, v
            torch.cuda.empty_cache()
        rec["validation_adjoint"] = {
            "trials": dots, "tolerance": tol_lin,
            "worst": max(d["relative_difference"] for d in dots)}
        rec["validation_AAt"] = {
            "trials": aat, "tolerance": tol_lin,
            "worst": max(d["relative_difference"] for d in aat)}
        print(f"[{self.name}] adjoint worst "
              f"{rec['validation_adjoint']['worst']:.3e}; A A^T formula worst "
              f"{rec['validation_AAt']['worst']:.3e}")
        if not rec["validation_adjoint"]["worst"] < tol_lin:
            raise AssertionError("adjoint dot-product test failed")
        if not rec["validation_AAt"]["worst"] < tol_lin:
            raise AssertionError("G_c diagonal-formula test failed")

        # -------- the data ---------------------------------------------------
        blk = block_from_rows(op)
        y_t = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
        y_norm = float(torch.linalg.vector_norm(y_t))
        rec["data"] = {"sum_records_ke": float(blk.sum()),
                       "sum_abs_records_ke": float(np.abs(blk).sum()),
                       "sum_y_over_sum_Kbar_ke": float(blk.sum() / fr.sum())}

        # -------- lambda scan -------------------------------------------------
        sol: dict = {}
        scan = []
        for lr in lam_rel:
            lam = lr * F.G_max
            t0 = time.time()
            xh = F.solve(y_t, lam)
            wall = time.time() - t0
            r = F.forward(xh) - y_t
            row = {"lambda_rel": lr, "lambda": lam,
                   "sum_xhat_ke": float(xh.sum()),
                   "sum_xhat_pos_ke": float(xh[xh > 0].sum()),
                   "sum_xhat_neg_ke": float(xh[xh < 0].sum()),
                   "sum_xhat_real_pads_ke": float(xh[:H.nx, :H.ny].sum()),
                   "sum_xhat_padding_pads_ke": float(
                       xh.sum() - xh[:H.nx, :H.ny].sum()),
                   "residual_rel": float(torch.linalg.vector_norm(r) / y_norm),
                   "conservation_predicted": float(F.G_dc / (F.G_dc + lam)),
                   "sum_y_over_sum_Kbar": float(blk.sum() / fr.sum()),
                   "wall_s": wall}
            scan.append(row)
            print(f"[{self.name}] lambda_rel {lr:8.1e}  sum "
                  f"{row['sum_xhat_ke']:10.2f} ke  x+ "
                  f"{row['sum_xhat_pos_ke']:10.2f}  x- "
                  f"{row['sum_xhat_neg_ke']:11.2f}  |Ax-y|/|y| "
                  f"{row['residual_rel']:.4e}  {wall:5.2f} s")
            sol[lr] = xh[:H.nx, :H.ny].cpu().numpy()
            del xh, r
            torch.cuda.empty_cache()
        rec["lambda_scan"] = scan
        del y_t
        torch.cuda.empty_cache()

        # -------- scoring -----------------------------------------------------
        def _score(tag, xcells, meta, keep_arrays=True):
            for pname in pnames:
                pf = grid.to_fine(xcells, pname, m_lo, m_hi)
                for s in sigmas:
                    xh = fine_xhat(H, pf, grid.fine_origin(m_lo), s)
                    m = score_rows(H, xh, s)
                    pr = m.pop("_profiles")
                    rows.append({**meta, "prolongation": pname,
                                 "sigma_H_us": s, **m})
                    if keep_arrays:
                        t2 = f"{tag}_{pname}_s{s:g}"
                        for k in ("line_xhat", "line_Hx", "line_e"):
                            arrays[f"prof_{t2}_{k}"] = pr[k].astype(np.float32)
                        arrays[f"ring1prof_{t2}"] = \
                            xh[rows_r1].mean(axis=0).astype(np.float32)
                    r = rows[-1]
                    print(f"[{self.name}] s{s:4.2f} c{meta['cell_ticks']:3d} "
                          f"{pname:14s} {r['arm']:22s} E_rel {r['E_rel']:9.5f} "
                          f" cons {r['conservation_rel']:+9.5f}  ring1+ "
                          f"{r['zero_preservation']['ring1']['pos_per_pad_ke']:8.4f}")
                del pf

        meta0 = {"basis": "cell", "cell_ticks": cell_ticks,
                 "cell_model": cell_model}
        _score("repr", Rx_c, {**meta0, "arm": "representation_term"})
        for lr in lam_rel:
            _score(f"minnorm_lrel{lr:g}", sol[lr],
                   {**meta0, "arm": f"cell{cell_ticks}_minnorm_lrel{lr:g}",
                    "lambda_rel": lr})

        # raw cell-space line profiles, for the raw panel of the profile figure
        arrays["cell_centers"] = grid.cc[m_lo:m_hi].astype(np.float64)
        arrays["cell_window"] = np.array([m_lo, m_hi])
        arrays["fine_ticks"] = H.fine.astype(np.int64)
        v = np.zeros(m_hi - m_lo)
        kt = grid.index(H.truth_tick) - m_lo
        ok = (kt >= 0) & (kt < m_hi - m_lo)
        np.add.at(v, kt[ok], H.truth_q[ok])
        arrays["truth_padsum_cells"] = v
        arrays["truth_cell_line"] = \
            Rx_c.reshape(H.n_pads, -1)[H.line_rows].mean(axis=0)[m_lo:m_hi]
        for lr in lam_rel:
            arrays[f"cell_line_minnorm_lrel{lr:g}"] = \
                sol[lr].reshape(H.n_pads, -1)[H.line_rows].mean(
                    axis=0)[m_lo:m_hi]
            arrays[f"transverse_minnorm_lrel{lr:g}"] = \
                sol[lr][:, :, m_lo:m_hi].sum(axis=(1, 2))
        tv = np.zeros(H.nx)
        np.add.at(tv, H.truth_ix, H.truth_q)
        arrays["transverse_truth"] = tv
        # the truth's fine-tick profile on the line, for the raw panel
        jw = np.arange(win_lo, win_hi)
        tf = np.zeros(len(jw))
        jt = H.truth_tick - win_lo
        ok = (jt >= 0) & (jt < len(jw))
        np.add.at(tf, jt[ok], H.truth_q[ok])
        arrays["stored_window_ticks"] = jw
        arrays["truth_padsum_fine"] = tf

        del sol
        del F
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # validation 5: c = 30 + delta is the production operator
        # ------------------------------------------------------------------
        if bool(self.props.get("check_coarse_identity", True)):
            k_id = int(H.k_truth_hi)
            e_c = np.zeros(op.q_shape)
            e_c[H.nx // 2, H.ny // 2, k_id] = 1.0
            y_c = op.conv(op.to_tensor(e_c)).cpu().numpy()
            ident, best = {}, None
            for shift in (0, 1, 2, -1):
                Fc = FineOperator(fr, op.block_shape, B, device=dev,
                                  dtype=dtype, cell_ticks=B,
                                  cell_model="delta", release_shift=shift)
                xd = torch.zeros((Fc.nxp, Fc.nyp, Fc.N), dtype=dtype,
                                 device=dev)
                xd[H.nx // 2, H.ny // 2, k_id] = 1.0
                y_f = Fc.forward(xd).cpu().numpy()[:H.nx, :H.ny]
                d = float(np.abs(y_f - y_c).sum())
                ident[f"shift_{shift:+d}"] = {
                    "sum_abs_difference": d,
                    "relative_to_sum_abs": d / float(np.abs(y_c).sum())}
                if best is None or d < best[1]:
                    best = (shift, d, d / float(np.abs(y_c).sum()))
                del xd, Fc
                torch.cuda.empty_cache()
            ident["best_shift_ticks"] = best[0]
            ident["best_relative_to_sum_abs"] = best[2]
            ident["probed_production_cell"] = k_id
            ident["note"] = (
                "the cell operator at c = 30, cell_model = delta and "
                "release_shift = s releases cell m at fine tick b + 30m + s; "
                "the production A_coarse releases cell k at c_k + 1 = "
                "b + 30k + 1, so the best shift is +1 and the residual is the "
                "float floor of this dtype")
            rec["coarse_operator_identity"] = ident
            print(f"[{self.name}] c=30 delta vs production A_coarse: best "
                  f"shift {best[0]:+d}, {best[2]:.3e} of sum|y|")

        # ------------------------------------------------------------------
        # the ladder in c
        # ------------------------------------------------------------------
        lad: list = []
        for cc in ladder:
            gc = CellGrid(b, cc, (B // cc) * int(op.block_shape[2]))
            ml, mh = gc.window(win_lo, win_hi)
            Rxc = gc.restrict(H.truth_ix, H.truth_iy, H.truth_tick, H.truth_q,
                              H.nx, H.ny)
            ent = {"cell_ticks": cc, "n_cells_per_pad": gc.n,
                   "decimation_stride_D": B // cc,
                   "n_unknowns_real_pads": int(H.nx * H.ny * gc.n),
                   "grid": gc.report(n_probe=1), "representation": {},
                   "linear_inverse": {}}
            for pname in pnames:
                pf = gc.to_fine(Rxc, pname, ml, mh)
                for s in sigmas:
                    m = score_rows(H, fine_xhat(H, pf, gc.fine_origin(ml), s), s)
                    m.pop("_profiles")
                    ent["representation"][f"{pname}_s{s:g}"] = {
                        "E_rel": m["E_rel"],
                        "conservation_rel": m["conservation_rel"],
                        "segments_rel_error_rms":
                            m.get("segments", {}).get("rel_error_rms")}
                    rows.append({"basis": "cell", "cell_ticks": cc,
                                 "cell_model": cell_model,
                                 "arm": "representation_term_ladder",
                                 "prolongation": pname, "sigma_H_us": s, **m})
                del pf
            print(f"[{self.name}] ladder c = {cc:2d}: representation "
                  + "  ".join(
                      f"{k} {v['E_rel']:.5f}"
                      for k, v in ent["representation"].items()
                      if k.endswith("s1.5") or k.endswith("s2")))
            if cc in ladder_solve:
                t0 = time.time()
                Fl = FineOperator(fr, op.block_shape, B, device=dev,
                                  dtype=dtype, cell_ticks=cc,
                                  cell_model=cell_model)
                tb = time.time() - t0
                yl = embed_pads(blk, Fl.nxp, Fl.nyp, Fl.M, dev, dtype)
                t0 = time.time()
                xl = Fl.solve(yl, lam_ladder * Fl.G_max)
                tw = time.time() - t0
                xln = xl[:H.nx, :H.ny].cpu().numpy()
                ent["linear_inverse"] = {
                    "lambda_rel": lam_ladder, "build_wall_s": tb,
                    "solve_wall_s": tw, "G_max": Fl.G_max, "G_dc": Fl.G_dc,
                    "sum_xhat_ke": float(xl.sum()),
                    "sum_xhat_pos_ke": float(xl[xl > 0].sum()),
                    "sum_xhat_neg_ke": float(xl[xl < 0].sum()),
                    "scores": {}}
                del xl, yl, Fl
                torch.cuda.empty_cache()
                for pname in pnames:
                    pf = gc.to_fine(xln, pname, ml, mh)
                    for s in sigmas:
                        m = score_rows(H, fine_xhat(H, pf,
                                                    gc.fine_origin(ml), s), s)
                        m.pop("_profiles")
                        ent["linear_inverse"]["scores"][f"{pname}_s{s:g}"] = {
                            "E_rel": m["E_rel"],
                            "conservation_rel": m["conservation_rel"],
                            "ring1_pos_per_pad_ke":
                                m["zero_preservation"]["ring1"]["pos_per_pad_ke"],
                            "ring1_neg_per_pad_ke":
                                m["zero_preservation"]["ring1"]["neg_per_pad_ke"],
                            "segments_rel_error_rms":
                                m.get("segments", {}).get("rel_error_rms")}
                        rows.append({"basis": "cell", "cell_ticks": cc,
                                     "cell_model": cell_model,
                                     "arm": "minnorm_ladder",
                                     "lambda_rel": lam_ladder,
                                     "prolongation": pname,
                                     "sigma_H_us": s, **m})
                    del pf
                print(f"[{self.name}] ladder c = {cc:2d}: linear inverse "
                      + "  ".join(
                          f"{k} {v['E_rel']:.5f}"
                          for k, v in ent["linear_inverse"]["scores"].items()
                          if k.endswith("s1.5")))
            lad.append(ent)
        rec["ladder"] = lad
        rec["rows"] = rows
        rec["lambda_rel"] = lam_rel
        rec["sigma_H_us"] = sigmas
        rec["cell_prolongations"] = pnames
        self._emit(store, rec, arrays)
        self.put(store, "cell.solutions", {"harness": H, "grid": grid,
                                           "arrays": arrays})


# ---------------------------------------------------------------------------
# figures C1-C5 of the intermediate-basis campaign
# ---------------------------------------------------------------------------
@algorithm("CellFigures")
class CellFigures(_Recorder):
    """Figures C1-C5: the ``c``-tick basis beside its fine and coarse references.

    Unlike :class:`FineBasisPlots`, this algorithm reads FILES, not the store:
    the figures put three jobs of this campaign (the linear scan and ladder,
    the nonlinear arms, the probe) beside the ARCHIVED fine and bin-integrated
    results, which were produced by earlier jobs and are quoted rather than
    re-run.  Every input is named in the YAML, so a reader can see exactly
    which file each number came from, and the algorithm has no store
    dependencies at all.

    Props
    -----
    cell_linear_json/_npz, cell_nonlinear_json/_npz, cell_probe_json/_npz
    ref_fine_scan_json/_npz     the archived fine linear campaign.
    ref_fine_nl_json/_npz       the archived fine nonlinear campaign.
    ref_fine_probe_npz/_json    the archived fine nonlinear probe.
    cell_lambda_rel : float, default 1e-6   -- the cell min-norm arm shown.
    fine_lambda_rel : float, default 1e-6   -- the fine min-norm arm shown.
    cell_prolongation : str, default "uniform"  -- the cell P drawn in the
        figures where one curve per arm is drawn (the tables carry both).
    l1_middle : str, default "cell5_pos_l1_0.01".
    fine_iteration_time_s : float, default 0.178 -- the archived wall time per
        FISTA iteration on the fine basis (fine_nl_probe.json: 177.9 s for
        1000 iterations), which this campaign does not re-measure.
    figdir, out_json
    """

    reads = ()
    writes = ("cell.figures",)

    def execute(self, store):
        self._store = store
        self.put(store, "cell.figures", {"pending": True})

    # -- small readers -------------------------------------------------------
    @staticmethod
    def _json(path):
        with open(path) as fh:
            return json.load(fh)["result"]

    @staticmethod
    def _npz(path):
        return dict(np.load(path, allow_pickle=True))

    @staticmethod
    def _row(rows, **sel):
        for r in rows:
            if all(abs(r.get(k, None) - v) < 1e-12
                   if isinstance(v, float) else r.get(k, None) == v
                   for k, v in sel.items()):
                return r
        return None

    def finalize(self):
        from .finebasis_nonlinear_algs import C_POS, C_POSL1, L1_TINTS, tint
        plt = ieee_style()
        P = self.props
        CL = self._json(P["cell_linear_json"])
        CLA = self._npz(P["cell_linear_npz"])
        CN = self._json(P["cell_nonlinear_json"])
        CNA = self._npz(P["cell_nonlinear_npz"])
        FS = self._json(P["ref_fine_scan_json"])
        FSA = self._npz(P["ref_fine_scan_npz"])
        FN = self._json(P["ref_fine_nl_json"])
        FNA = self._npz(P["ref_fine_nl_npz"])
        cprobe = self._json(P["cell_probe_json"]) if P.get("cell_probe_json") \
            else None
        cprobeA = self._npz(P["cell_probe_npz"]) if P.get("cell_probe_npz") \
            else {}
        fprobe = self._json(P["ref_fine_probe_json"]) \
            if P.get("ref_fine_probe_json") else None
        fprobeA = self._npz(P["ref_fine_probe_npz"]) \
            if P.get("ref_fine_probe_npz") else {}
        outdir = Path(P.get("figdir", "figs_cell5"))
        made: list = []
        lb = float(P.get("cell_lambda_rel", 1e-6))
        fb = float(P.get("fine_lambda_rel", 1e-6))
        pn = str(P.get("cell_prolongation", "uniform"))
        mid = str(P.get("l1_middle", "cell5_pos_l1_0.01"))
        t_fine_iter = float(P.get("fine_iteration_time_s", 0.178))
        ct = int(CL["basis"]["cell_ticks"])
        dt = TICK_US

        def cellrow(arm, s, prol=None):
            return self._row(CL["rows"], arm=arm, sigma_H_us=float(s),
                             prolongation=prol or pn) \
                or self._row(CN["rows"], arm=arm, sigma_H_us=float(s),
                             prolongation=prol or pn)

        def finerow(arm, s, prol=None):
            for rows in (FN["rows"], FS["rows"]):
                r = self._row(rows, arm=arm, sigma_H_us=float(s),
                              **({"prolongation": prol} if prol else {}))
                if r is not None:
                    return r
            return None

        l1_labels = [str(a["label"]) for a in CN["arm_specs"]
                     if float(a["alpha"]) > 0]
        pos_label = [str(a["label"]) for a in CN["arm_specs"]
                     if float(a["alpha"]) == 0][0]
        lin_label = CN.get("linear_label", "cell5_minnorm")
        l1_colors = list(L1_TINTS)
        while len(l1_colors) < len(l1_labels):
            l1_colors.append(tint(C_POSL1, 0.8))

        # ---------------- C1: E_rel at the goal ---------------------------
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.2),
                               gridspec_kw={"width_ratios": [1.0, 1.9]})
        a = ax[0]
        lad = CL["ladder"]
        cs = [e["cell_ticks"] for e in lad]
        xs = np.arange(len(cs))
        for i, (prol, lab, cc) in enumerate(
                (("uniform", r"$P_0$", OI["sky"]),
                 ("corrected_hat", r"$P_1$", C_COARSE))):
            for j, s in enumerate((1.5, 2.0)):
                v = [e["representation"][f"{prol}_s{s:g}"]["E_rel"] for e in lad]
                a.bar(xs + (2 * i + j - 1.5) * 0.2, np.maximum(v, 1e-6), 0.2,
                      color=cc, alpha=1.0 if j == 0 else 0.55, edgecolor="k",
                      linewidth=0.3,
                      label=rf"{lab}, $\sigma_H={s:g}\,\mu$s")
        a.set_yscale("log")
        a.set_ylim(1e-6, 1.0)
        a.set_xticks(xs)
        a.set_xticklabels([f"c = {c}" for c in cs])
        a.set_ylabel(r"representation term $E_{\rm rel}$")
        a.set_title("(a) basis cost alone, $H(PR_c-I)x$")
        a.legend(frameon=False, fontsize=5.5, ncol=2)
        a.text(0.02, 0.02, "c = 1: exactly zero\n($R_1=I$, $P_0=P_1=I$)",
               transform=a.transAxes, fontsize=5.5, va="bottom",
               color=OI["grey"])

        a = ax[1]
        ent = [(rf"cell{ct} repr. ({'$P_0$' if pn == 'uniform' else '$P_1$'})",
                lambda s: cellrow("representation_term", s), OI["grey"]),
               (rf"cell{ct} min-norm", lambda s: cellrow(
                   f"cell{ct}_minnorm_lrel{lb:g}", s), C_FINE),
               (rf"cell{ct} pos, $\alpha=0$",
                lambda s: cellrow(pos_label, s), C_POS)]
        for i, l in enumerate(l1_labels):
            ent.append((rf"cell{ct} pos+$\ell_1$ {l.split('_')[-1]}",
                        (lambda ll: (lambda s: cellrow(ll, s)))(l),
                        l1_colors[i]))
        ent += [("fine min-norm",
                 lambda s: finerow(f"fine_lrel{fb:g}", s) or
                 finerow("fine_minnorm", s), tint(C_FINE, 0.45)),
                (r"fine pos, $\alpha=0$",
                 lambda s: finerow("fine_pos_a0", s), tint(C_POS, 0.45)),
                (r"fine pos+$\ell_1$ 0.01",
                 lambda s: finerow("fine_pos_l1_0.01", s), tint(C_POSL1, 0.45)),
                (r"coarse LS, $P_1$",
                 lambda s: finerow("LS_nopos", s, "corrected_hat"), C_COARSE),
                (r"coarse pos, $\alpha=0$",
                 lambda s: finerow("pos_a0", s, "corrected_hat"), OI["sky"])]
        xs = np.arange(len(ent))
        for j, s in enumerate((1.5, 2.0)):
            v = [(f(s) or {}).get("E_rel", np.nan) for _, f, _c in ent]
            a.bar(xs + (j - 0.5) * 0.4, v, 0.4, color=[e[2] for e in ent],
                  alpha=1.0 if j == 0 else 0.55, edgecolor="k", linewidth=0.4,
                  label=rf"$\sigma_H={s:g}\,\mu$s")
            for xi, vv in zip(xs, v):
                a.text(xi + (j - 0.5) * 0.4, vv, f"{vv:.3f}", fontsize=5.0,
                       ha="center", va="bottom", rotation=90)
        a.set_xticks(xs)
        a.set_xticklabels([e[0] for e in ent], rotation=28, ha="right",
                          fontsize=5.5)
        a.set_ylabel(r"$E_{\rm rel}$")
        a.set_ylim(0, 1.05 * np.nanmax(
            [(f(1.5) or {}).get("E_rel", 0) for _, f, _c in ent]) * 1.35)
        a.set_title(rf"(b) every arm on the $c={ct}$ basis "
                    r"(solid) beside its archived references")
        a.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, outdir, "C1_Erel_at_goal", made)

        # ---------------- C2: line-averaged profiles -----------------------
        ftk = CNA["fine_ticks"].astype(float)
        wc = CNA["stored_window_ticks"].astype(float)      # cell centres
        tp = CNA["truth_padsum_fine"]                      # per cell
        cen = float((tp * wc).sum() / max(tp.sum(), 1e-30))
        n_line = int(self._row(CN["rows"], arm=pos_label,
                               sigma_H_us=1.5)["n_line_pads"])
        fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.9))
        a = ax[0]
        ftk_f = FNA["stored_window_ticks"].astype(float)
        a.plot((ftk_f - cen) * dt,
               FNA["truth_padsum_fine"] / max(n_line, 1),
               color=C_TRUTH, lw=1.1, label=r"$x$ (fine truth, 50 ns)")
        a.plot((ftk_f - cen) * dt, FSA[f"fine_line_lrel{fb:g}"],
               color=tint(C_FINE, 0.5), lw=0.9,
               label="fine min-norm (50 ns)")
        edges = np.concatenate([wc - ct / 2.0, [wc[-1] + ct / 2.0]])
        for lab, key, cc, ls in (
                (rf"cell{ct} min-norm", lin_label, C_FINE, "-"),
                (rf"cell{ct} pos, $\alpha=0$", pos_label, C_POS, "-"),
                (rf"cell{ct} pos+$\ell_1$ {mid.split('_')[-1]}", mid,
                 C_POSL1, "--")):
            a.stairs(CNA["fine_line_raw_" + key] / ct, (edges - cen) * dt,
                     color=cc, ls=ls, lw=1.0, label=lab)
        a.axhline(0, color="k", lw=0.4)
        a.set_xlim(-9, 9)
        a.set_xlabel(r"time from the truth centroid [$\mu$s]")
        a.set_ylabel("mean over interior line pads [ke / fine tick]")
        a.set_title(r"(a) $\sigma_H = 0$ (raw)")
        a.legend(frameon=False, fontsize=5.5)

        a = ax[1]
        tus = (ftk - cen) * dt
        a.plot(tus, CNA[f"prof_{pos_label}_{pn}_s1.5_line_Hx"], color=C_HTRUTH,
               lw=1.3, label=r"$Hx$")
        a.plot(tus, FSA[f"prof_fine_lrel{fb:g}_s1.5_line_xhat"],
               color=tint(C_FINE, 0.5), lw=0.9, label="fine min-norm")
        for lab, key, cc, ls in (
                (rf"cell{ct} min-norm", lin_label, C_FINE, "-"),
                (rf"cell{ct} pos, $\alpha=0$", pos_label, C_POS, "-"),
                (rf"cell{ct} pos+$\ell_1$ {mid.split('_')[-1]}", mid,
                 C_POSL1, "--")):
            a.plot(tus, CNA[f"prof_{key}_{pn}_s1.5_line_xhat"], color=cc,
                   ls=ls, lw=1.0, label=lab)
        a.set_xlim(-9, 9)
        a.set_xlabel(r"time from the truth centroid [$\mu$s]")
        a.set_title(r"(b) $\sigma_H = 1.5\,\mu$s")
        a.legend(frameon=False, fontsize=5.5)
        fig.tight_layout()
        save(fig, outdir, "C2_line_profiles", made)

        # ---------------- C3: ring ledger ---------------------------------
        led = [(rf"cell{ct} min-norm", lambda s: cellrow(lin_label, s), C_FINE),
               (rf"cell{ct} pos $\alpha=0$",
                lambda s: cellrow(pos_label, s), C_POS)]
        for i, l in enumerate(l1_labels):
            led.append((rf"cell{ct} $\ell_1$ {l.split('_')[-1]}",
                        (lambda ll: (lambda s: cellrow(ll, s)))(l),
                        l1_colors[i]))
        led += [("fine min-norm", lambda s: finerow("fine_minnorm", s),
                 tint(C_FINE, 0.45)),
                (r"fine pos $\alpha=0$", lambda s: finerow("fine_pos_a0", s),
                 tint(C_POS, 0.45)),
                (r"coarse LS $P_1$",
                 lambda s: finerow("LS_nopos", s, "corrected_hat"), C_COARSE),
                (r"coarse pos $\alpha=0$",
                 lambda s: finerow("pos_a0", s, "corrected_hat"), OI["sky"])]
        rings = [("ring1", "ring 1"), ("ring2", "ring 2"),
                 ("ring_ge3", r"ring $\geq$3")]
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0))
        for si, s in enumerate((0.0, 1.5)):
            a = ax[si]
            xs = np.arange(len(rings))
            w = 0.8 / len(led)
            for i, (lab, f, cc) in enumerate(led):
                r = f(s)
                if r is None:
                    continue
                z = r["zero_preservation"]
                p = [z[k]["pos_per_pad_ke"] for k, _ in rings]
                n = [z[k]["neg_per_pad_ke"] for k, _ in rings]
                off = (i - (len(led) - 1) / 2) * w
                a.bar(xs + off, p, w, color=cc, edgecolor="k", linewidth=0.3,
                      label=lab if si == 0 else None)
                a.bar(xs + off, n, w, color=cc, alpha=0.45, edgecolor="k",
                      linewidth=0.3)
            a.axhline(0, color="k", lw=0.5)
            a.set_xticks(xs)
            a.set_xticklabels([t for _, t in rings])
            a.set_ylabel("charge per pad [ke]")
            a.set_title(rf"$\sigma_H = {s:g}\,\mu$s "
                        r"(solid $\Sigma^+$, pale $\Sigma^-$)")
        ax[0].legend(frameon=False, fontsize=5.0, ncol=2)
        fig.tight_layout()
        save(fig, outdir, "C3_ring_ledger", made)

        # ---------------- C4: the resolution probe -------------------------
        if cprobe is not None:
            self._c4(plt, cprobe, cprobeA, fprobe, fprobeA, outdir, made, ct,
                     pn)

        # ---------------- C5: the ladder in c ------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
        a = ax[0]
        floor = 3e-7
        for prol, lab, cc, mk in (("uniform", r"$P_0$", OI["green"], "o"),
                                  ("corrected_hat", r"$P_1$", OI["purple"],
                                   "s")):
            for s, ls in ((1.5, "-"), (2.0, "--")):
                v = [max(e["representation"][f"{prol}_s{s:g}"]["E_rel"], floor)
                     for e in lad]
                a.loglog(cs, v, color=cc, ls=ls, marker=mk, ms=3.5, lw=1.0,
                         label=rf"repr. {lab}, $\sigma_H={s:g}$")
        for s, cc, ls in ((1.5, C_FINE, "-"), (2.0, tint(C_FINE, 0.5), "--")):
            xx = [e["cell_ticks"] for e in lad if e["linear_inverse"]]
            vv = [e["linear_inverse"]["scores"][f"uniform_s{s:g}"]["E_rel"]
                  for e in lad if e["linear_inverse"]]
            r = self._row(FS["rows"], arm=f"fine_lrel{fb:g}",
                          sigma_H_us=float(s))
            if r is not None:
                xx = [1] + list(xx)
                vv = [r["E_rel"]] + list(vv)
            a.loglog(xx, vv, color=cc, ls=ls, marker="^", ms=4.5, lw=1.2,
                     label=rf"min-norm $P_0$, $\sigma_H={s:g}$")
        a.text(0.30, 0.80, "the c = 1 min-norm point is the archived fine\n"
               "result; c = 1 representation is exactly zero",
               transform=a.transAxes, fontsize=5.0, va="top",
               color=OI["grey"])
        a.set_xlabel("cell width $c$ [fine ticks]")
        a.set_ylabel(r"$E_{\rm rel}$")
        a.set_xticks(cs)
        a.set_xticklabels([str(c) for c in cs])
        a.set_title("(a) representation term and linear inverse")
        a.legend(frameon=False, fontsize=5.0, ncol=2, loc="lower right")

        a = ax[1]
        tim = CN.get("iteration_timing", [])
        xx = [1] + [t["cell_ticks"] for t in tim]
        yy = [t_fine_iter] + [t["wall_s_per_iteration"] for t in tim]
        a.loglog(xx, yy, color=C_POS, marker="o", ms=4, lw=1.2,
                 label="measured")
        a.loglog(xx, [t_fine_iter / (x / xx[0]) for x in xx], color=OI["grey"],
                 ls=":", lw=0.9, label=r"$\propto 1/c$")
        a.plot([xx[0]], [yy[0]], marker="s", ms=6, mfc="none", color=C_FINE,
               ls="none", label="fine basis (archived)")
        a.set_xlabel("cell width $c$ [fine ticks]")
        a.set_ylabel("wall time per FISTA iteration [s]")
        a.set_xticks(xx)
        a.set_xticklabels([str(x) for x in xx])
        a.set_title("(b) cost per iteration, float32, RTX 4070 Ti")
        a.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, outdir, "C5_ladder_in_c", made)

        out = {"figures": made, "cell_lambda_rel": lb,
               "fine_lambda_rel": fb, "cell_prolongation": pn,
               "inputs": {k: str(v) for k, v in P.items()
                          if str(k).endswith(("_json", "_npz"))}}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": out}, fh, indent=1,
                          default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return out

    # -- C4 ------------------------------------------------------------------
    def _c4(self, plt, cp, cA, fp, fA, outdir, made, ct, pn):
        """Probe figure: impulse responses, first moment and width vs phase,
        and the impact panel.  Impact-resolved kernels are a STRESS TEST of
        the estimator, never a correction: no arm here is given the impact."""
        from .finebasis_nonlinear_algs import C_POS, C_POSL1, tint
        lin = cp.get("linear_label", f"cell{ct}_minnorm")
        nl = [str(a["label"]) for a in cp["fine_arms"]["arms"]]
        Qm = float(cp["fine_arms"]["charges_ke"][0])
        pr = [r for r in cp["probes"] if not r.get("convergence_probe")]
        phases = sorted({r["phi"] for r in pr if r["kernel"] == "bar"})

        def g(rec, arm, phi, kn, s, prol=None):
            for r in rec["probes"]:
                if (r["arm"] == arm and r["phi"] == phi and r["kernel"] == kn
                        and r.get("convergence_probe") is not True):
                    for d in r["sigmas"]:
                        if (abs(d["sigma_H_us"] - s) < 1e-9
                                and (prol is None
                                     or d.get("prolongation", prol) == prol)):
                            return d
            return None

        fig = plt.figure(figsize=(10.5, 5.4))
        gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)
        for i, phi in enumerate(phases[:3]):
            a = fig.add_subplot(gs[0, i])
            t = cA.get(f"fine_ticks_phi{phi}_bar")
            if t is None:
                continue
            ts = [r["t_star"] for r in pr if r["phi"] == phi][0]
            tu = (t - ts) * TICK_US
            a.plot(tu, cA[f"Hdelta_phi{phi}_bar_s1.5"], color=C_TRUTH, lw=1.0,
                   label=r"$H\delta$")
            k = f"imp_phi{phi}_bar_Q{Qm:g}_{lin}_{pn}_s1.5"
            if k in cA:
                a.plot(tu, cA[k], color=C_FINE, lw=1.0,
                       label=rf"cell{ct} min-norm")
            for j, arm in enumerate(nl):
                k = f"imp_phi{phi}_bar_Q{Qm:g}_{arm}_{pn}_s1.5"
                if k in cA:
                    a.plot(tu, cA[k], color=C_POS if j == 0 else C_POSL1,
                           lw=1.0, label=rf"cell{ct} {arm.split('_', 1)[1]}")
            if fA:
                k = f"imp_phi{phi}_bar_Q{Qm:g}_fine_minnorm_s1.5"
                tf = fA.get(f"fine_ticks_phi{phi}_bar")
                if k in fA and tf is not None:
                    a.plot((tf - ts) * TICK_US, fA[k], color=tint(C_FINE, 0.5),
                           lw=0.8, ls=":", label="fine min-norm")
            a.set_xlim(-6, 6)
            a.set_xlabel(r"$t-t^*$ [$\mu$s]")
            if i == 0:
                a.set_ylabel(r"$\hat{x}/Q$ on the probed pad [1/tick]")
            a.legend(frameon=False, fontsize=5.0)
            a.set_title(rf"(a{i + 1}) $\varphi={phi}$, $Q={Qm:g}$ ke, "
                        r"$\sigma_H=1.5\,\mu$s")

        a = fig.add_subplot(gs[1, 0])
        for arm, cc, lab in ([(lin, C_FINE, f"cell{ct} min-norm")]
                             + [(nl[0], C_POS, f"cell{ct} pos $\\alpha=0$")]):
            v = [g(cp, arm, p, "bar", 0.0, pn) for p in phases]
            a.plot(phases, [x["shift_ticks"] if x else np.nan for x in v],
                   color=cc, marker="o", ms=3.5, label=lab)
        if fp is not None:
            for arm, cc, lab in (("fine_minnorm", tint(C_FINE, 0.5),
                                  "fine min-norm"),
                                 ("fine_pos_a0", tint(C_POS, 0.5),
                                  r"fine pos $\alpha=0$")):
                v = [g(fp, arm, p, "bar", 0.0) for p in phases]
                a.plot(phases, [x["shift_ticks"] if x else np.nan for x in v],
                       color=cc, marker="s", ms=3.0, ls="--", label=lab)
        a.axhline(0, color=C_TRUTH, lw=0.7, ls=":")
        a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
        a.set_ylabel("first moment [fine ticks]")
        a.set_title(r"(b) first moment, $\sigma_H=0$")
        a.legend(frameon=False, fontsize=5.5)

        a = fig.add_subplot(gs[1, 1])
        for arm, cc, lab in ([(lin, C_FINE, f"cell{ct} min-norm"),
                              (nl[0], C_POS, f"cell{ct} pos $\\alpha=0$")]):
            v = [g(cp, arm, p, "bar", 1.5, pn) for p in phases]
            a.plot(phases, [(x["width_ticks"] if x and x["width_ticks"]
                             else np.nan) for x in v], color=cc, marker="o",
                   ms=3.5, label=lab)
        if fp is not None:
            for arm, cc, lab in (("fine_minnorm", tint(C_FINE, 0.5),
                                  "fine min-norm"),
                                 ("fine_pos_a0", tint(C_POS, 0.5),
                                  r"fine pos $\alpha=0$")):
                v = [g(fp, arm, p, "bar", 1.5) for p in phases]
                a.plot(phases, [(x["width_ticks"] if x and x["width_ticks"]
                                 else np.nan) for x in v], color=cc,
                       marker="s", ms=3.0, ls="--", label=lab)
        a.axhline(1.5 / TICK_US, color=C_HTRUTH, ls="--", lw=0.9,
                  label=r"ideal $\sigma_H/\Delta t = 30$")
        a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
        a.set_ylabel("width [fine ticks]")
        a.set_title(r"(c) second moment, $\sigma_H=1.5\,\mu$s")
        a.legend(frameon=False, fontsize=5.5)

        a = fig.add_subplot(gs[1, 2])
        kns = [k for k in ("bar", "4,4", "0,0")
               if any(r["kernel"] == k and r["phi"] == 15 for r in pr)]
        styles = {"bar": ("-", "impact-averaged"), "4,4": ("--", "impact (4,4)"),
                  "0,0": (":", "impact (0,0)")}
        ts = [r["t_star"] for r in pr if r["phi"] == 15]
        if ts:
            ts = ts[0]
            for arm, cc in ((lin, C_FINE), (nl[0], C_POS)):
                for kn in kns:
                    t = cA.get(f"fine_ticks_phi15_{kn}")
                    k = f"imp_phi15_{kn}_Q{Qm:g}_{arm}_{pn}_s1.5"
                    if t is None or k not in cA:
                        continue
                    ls, lab = styles[kn]
                    a.plot((t - ts) * TICK_US, cA[k], color=cc, ls=ls, lw=1.0,
                           label=f"{arm.split('_', 1)[1]}, {lab}")
            t = cA.get("fine_ticks_phi15_bar")
            if t is not None:
                a.plot((t - ts) * TICK_US, cA["Hdelta_phi15_bar_s1.5"],
                       color=C_TRUTH, lw=1.0, label=r"$H\delta$")
        a.set_xlim(-6, 6)
        a.set_ylim(top=a.get_ylim()[1] * 1.55)
        a.set_xlabel(r"$t-t^*$ [$\mu$s]")
        a.set_ylabel(r"$\hat{x}/Q$ [1/tick]")
        a.set_title(r"(d) $\varphi=15$, impact dependence "
                    "(stress test, never a correction)", fontsize=6.5)
        a.legend(frameon=False, fontsize=5.0, ncol=2, loc="upper left")
        save(fig, outdir, "C4_resolution_probe", made)
