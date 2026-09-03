"""Nonlinear (positivity, positivity + l1) estimators on the FINE 50 ns basis.

``finebasis_algs`` measured the fine-binned operator ``A_fine`` and its
closed-form minimum-norm Tikhonov inverse -- a LINEAR map of the data.  That
estimator has an unbiased first moment (+0.045 +- 0.200 fine ticks over the
seven sub-window phases) and still scores ``E_rel`` = 0.3823 at
``sigma_H`` = 1.5 us.  This module adds the estimators that are NOT linear
maps of the data:

    fine_pos_a0      x >= 0 on the support, alpha = 0
    fine_pos_l1_<a>  x >= 0 on the support, plus alpha * sum_v x_v

and scores them with the same harness, beside the fine linear inverse and
beside the bin-integrated (1.5 us) arms ``LS_nopos``, ``pos_a0`` and
``pos_l1_0.3``.  The question is the one ``algo_plan`` D4 and ``PROBLEM.md``
Sec. 5 item 2 state on the 1.5 us basis: does positivity reduce the error at
the goal resolution, or does it rectify the zero-mean impact-averaging dipole
into one-sided charge on cells the truth says are empty?

Definitions
-----------
Everything defined in the module docstrings of
:mod:`unfoldlarpix.algs.exactrows_algs` (``Kcum``, ``phi``, the impact index),
:mod:`unfoldlarpix.algs.evalharness_algs` (``c_k``, ``C_k``, ``R``,
``P_delta``/``P_0``/``P_1``, ``H``, ``E_rel``, the decomposition, the
zero-preservation ledger, the segment sums, the probe's column metrics) and
:mod:`unfoldlarpix.algs.finebasis_algs` (``h_d(tau)``, ``A_fine``, ``G``,
``lambda``, ``P = I`` for fine candidates, ``R xhat_fine``) is used with the
same meaning and is not restated.  The names introduced HERE are:

``A_fine^real`` (the fine operator restricted to the real pads)
    ``A_fine`` carries columns on the padded ``(nxp, nyp) = (73, 188)``
    transverse grid.  The unknown of every arm in this module lives on the
    ``(nx, ny) = (49, 164)`` REAL pads only, exactly as the production
    bin-integrated unknown ``q_p[k]`` does (``op.q_shape`` is the real pads).
    ``A_fine^real`` is ``A_fine`` with the padding-pad columns deleted; its
    rows are unchanged, so the record space is still the padded
    ``(73, 188, M)`` grid on which the padding pads' records are asserted to
    be exactly zero (``finebasis_algs``, "padding pads").  The closed-form
    inverse of ``finebasis_algs`` put 6.9e-9 ke on the padding pads, so on
    this event deleting those columns is not a change of estimator; it is
    stated because for a POSITIVE estimator it could be, and because it is
    what makes the fine and the coarse unknown sets the same set of pads.

``c^fine_v = (A_fine^real)^T 1_rec`` (the fine measurement gain)
    ``1_rec`` is the indicator of the ``(pad, window)`` cells that carry an
    ACTUAL record row of the production operator -- the same ``n_data`` rows
    that ``ZSOperator.measurement_gain`` sums over -- embedded in the padded
    record grid.  So ``c^fine_v`` is the charge the recorded windows credit to
    a unit charge at fine cell ``v``, the exact fine analogue of the coarse
    ``c_v`` of ``jobs/METHODS.md`` Sec. 3.  It is NOT computed with ``1`` over
    the whole padded record grid: on a circular grid that sum is
    ``sum_{d,tau} Kbar_d(tau)`` at every fine tick by construction, which
    carries no information about the record end.

``support`` (the fine support, spec ``gain:0.5``)
    ``base_fine AND (c^fine_v > 0.5 max c^fine_v)``, where ``base_fine`` is
    the store's coarse hits support ``base[p, k]`` evaluated at the coarse
    cell ``k`` that contains the fine tick: ``base_fine[p, j] = base[p,
    k(b + j)]``, ``k(t) = coarse_index(t)``, and ``False`` where the fine tick
    falls outside every coarse cell.  This is the same spec string
    ``gain:0.5`` that every coarse arm of this campaign uses
    (``resolve_support``), read on the fine grid.  The number of fine cells it
    keeps and the truth charge it excludes are reported, not assumed.

``Sigma q / Sigma truth``
    ``sum_v xhat_v`` over the whole fine grid divided by the fine truth total
    ``sum x`` = 4211.999 ke.  Reported beside ``sum_v xhat_v / (sum_w y_w /
    sum Kbar)`` = the total the fine linear inverse is pinned to.

``nnz``
    the number of fine cells with ``xhat_v > 0``.  ``nnz_1e-3`` is the number
    with ``xhat_v > 1e-3`` ke, which is the same count with the FISTA
    round-off floor removed.

``on-line charge`` / ``off-line charge``
    ``sum_{p in T} sum_j xhat_p(j)`` and ``sum_{p not in T} sum_j xhat_p(j)``
    with ``T`` the set of pads that carry ANY truth charge (137 pads on this
    event).  The split is over pads only; no time selection is applied.  This
    is the raw (``sigma_H = 0``) fine estimate, not an ``H``-space quantity;
    the ``H``-space statement is the ring ledger of ``score_rows``.

``signal dependence``
    a nonlinear estimator has no resolution matrix: its impulse response
    depends on the injected charge ``Q``.  Every probe number in this module
    is therefore quoted with its ``Q``, and the probe is run at three charges.
    One exception is PROVED and then verified numerically: with ``alpha = 0``
    the arm is POSITIVELY HOMOGENEOUS.  ``argmin_{x >= 0, supp}
    1/2 ||A x - c y||^2 = c argmin_{x >= 0, supp} 1/2 ||A x - y||^2`` for
    ``c > 0`` (substitute ``x = c u``), and the FISTA iterates inherit it
    because ``q0 = 0``, the step is data-independent and
    ``prox(v) = max(v, 0) . supp`` is positively homogeneous.  So
    ``xhat(cy) = c xhat(y)`` EXACTLY, at any iteration count: the normalised
    impulse response ``xhat/Q`` of ``fine_pos_a0`` does not depend on ``Q``,
    and the probe solves it at one charge and checks the scaling at the others
    rather than repeating a solve whose answer is known.  ``alpha > 0``
    introduces an absolute charge scale and destroys the homogeneity, which is
    what the three-charge probe measures.

``residual floor``
    ``||A x_truth - y|| / ||y||`` with ``x_truth`` the fine effq truth.  It is
    the smallest data residual any NON-NEGATIVE estimator on this support can
    reach, because ``x_truth`` is itself non-negative and inside the support;
    it is not zero because tred deletes the current before tick 0 and the
    operator does not model that (``exactrows_isoline`` B.2).  Quoted beside
    every arm's ``||A xhat - y|| / ||y||`` so that "unconverged" and "as close
    as the model allows" are distinguishable.

Classes
-------
:class:`FineZSOperator`
    ``A_fine^real`` presented through the ``ZSOperator`` interface
    (``conv``/``sample``/``conv_adjoint``/``sample_adjoint``/``lipschitz``/
    ``d``/``q_shape``/``to_tensor``/``n_data``), so that the SHIPPED
    :class:`~unfoldlarpix.solve.engine.Fista`,
    :class:`~unfoldlarpix.terms.data.DataFidelity` and
    :class:`~unfoldlarpix.terms.base.CoordProx` run on it unmodified.  This is
    the same device as :class:`~unfoldlarpix.solve.ress.GramOperator`.

:class:`FineNonlinearArms`
    Builds the operator, the support and the arms; scores every arm and the
    references with :func:`~unfoldlarpix.algs.finebasis_algs.score_rows`.

:class:`FineNonlinearPlots`
    Figures N1-N4 and N8 from the products of the same job.

:class:`FineNonlinearProbe`
    Impulse responses of the nonlinear fine arms at three charges, the same
    exact-functional record generator as
    :class:`~unfoldlarpix.algs.finebasis_algs.FineBasisProbe`.

:class:`FineNonlinearProbePlots`
    Figures N5-N7.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..fwk.component import algorithm
from ..solve.engine import Fista
from ..terms.base import CoordProx
from ..terms.data import DataFidelity
from .evalharness_algs import (EvalHarness, TICK_US, coarse_centers,
                               coarse_index, records_from_impulse, row_lookup,
                               time_kernel)
from .exactrows_algs import _Recorder, load_impact_response
from .finebasis_algs import (C_COARSE, C_FINE, C_HTRUTH, C_TRUTH, KRAD, OI,
                             CellGrid, FineOperator, cell_xhat, coarse_xhat,
                             embed_pads, fine_xhat, ieee_style, probe_metrics,
                             save, score_rows)
from .fixedgrid_algs import block_from_rows, fit_bin_ticks, resolve_support

# two more fixed colours for the positivity arms, Okabe-Ito, so that they are
# distinguishable from the linear roles (truth black, coarse blue, fine
# vermillion, H-smoothed truth grey) already fixed by ``finebasis_algs``.
C_POS = OI["green"]          # fine positivity, alpha = 0
C_POSL1 = OI["purple"]       # fine positivity + l1


def tint(hexcol: str, f: float) -> str:
    """Blend ``hexcol`` toward white (``f > 0``) or black (``f < 0``).

    The three fine l1 arms share ONE hue (``C_POSL1``) and differ only in
    lightness, so that the figure's hue axis stays "which estimator family"
    and the lightness axis is "how strong is alpha".  Without this the third
    l1 arm would reuse the orange that the COARSE ``pos_l1_0.3`` already owns.
    """
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    t = (255, 255, 255) if f >= 0 else (0, 0, 0)
    a = abs(f)
    return "#%02x%02x%02x" % tuple(
        int(round(c * (1 - a) + t[i] * a)) for i, c in enumerate((r, g, b)))


# light -> dark purple for alpha increasing
L1_TINTS = [tint(C_POSL1, 0.55), C_POSL1, tint(C_POSL1, -0.40)]


# ---------------------------------------------------------------------------
# the ZSOperator interface around A_fine restricted to the real pads
# ---------------------------------------------------------------------------
class FineZSOperator:
    """``A_fine^real`` through the ``ZSOperator`` interface.

    ``conv`` is the identity and the whole operator lives in ``sample``, as in
    :class:`~unfoldlarpix.solve.ress.GramOperator`: that keeps
    :class:`~unfoldlarpix.terms.base.IterCtx`'s cache meaningful (one
    ``A q`` per iteration, shared by the terms) and costs nothing, because on
    the fine basis there is no separate block space -- the unknown IS on the
    fine grid.

    ``q_shape`` is ``(nx, ny, N)`` (real pads); the record space is the padded
    ``(nxp, nyp, M)`` grid, on which ``d`` is the event's records with EXACTLY
    ZERO on the padding pads and on every ``(pad, window)`` cell that carries
    no record row -- which is the same data vector the closed-form inverse of
    :class:`~unfoldlarpix.algs.finebasis_algs.FineBasisInverse` was given, so
    the linear and the nonlinear arms here solve the same system.

    ``lipschitz`` is ``F.G_max = max_nu Ghat(nu)``, which is
    ``||A_fine A_fine^T|| = ||A_fine^T A_fine||`` EXACTLY -- it is read off the
    closed-form symbol of ``A A^T`` (``finebasis_algs``, ``G``), not estimated
    by the power iteration that ``ZSOperator.lipschitz`` uses.  Deleting the
    padding-pad columns can only lower the norm, so it remains a valid FISTA
    step bound for ``A_fine^real``.

    ``forward``/``adjoint`` are written so that no more than two full fine-grid
    arrays are alive at once (the input is released before the inverse
    transform is allocated); they are TESTED against
    ``F.forward``/``F.adjoint`` on the embedded grid, they are not assumed to
    agree with them.
    """

    def __init__(self, F: FineOperator, d: torch.Tensor, nx: int, ny: int):
        self.F = F
        self.nx, self.ny = int(nx), int(ny)
        self.q_shape = (self.nx, self.ny, F.N)
        self.block_shape = self.q_shape
        self.device = F.device
        self.dtype = F.dtype
        self.d = d
        self.n_data = int(F.nxp * F.nyp * F.M)

    # -- the ZSOperator interface -------------------------------------------
    def conv(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def conv_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def sample(self, block: torch.Tensor) -> torch.Tensor:
        """``(nx, ny, N) -> (nxp, nyp, M)``: embed, convolve, decimate."""
        F = self.F
        x = torch.zeros((F.nxp, F.nyp, F.N), dtype=F.dtype, device=F.device)
        x[:self.nx, :self.ny] = block
        Xf = torch.fft.rfftn(x, dim=(0, 1, 2))
        del x
        Xf *= F.Hr
        z = torch.fft.irfftn(Xf, s=(F.nxp, F.nyp, F.N), dim=(0, 1, 2))
        del Xf
        zd = z[:, :, ::F.D].contiguous()
        del z
        return torch.roll(zd, (-F.krad, -F.krad, -1), dims=(0, 1, 2))

    def sample_adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """``(nxp, nyp, M) -> (nx, ny, N)``: upsample, correlate, crop."""
        F = self.F
        u = torch.zeros((F.nxp, F.nyp, F.N), dtype=F.dtype, device=F.device)
        u[:, :, ::F.D] = torch.roll(r, (F.krad, F.krad, 1), dims=(0, 1, 2))
        Uf = torch.fft.rfftn(u, dim=(0, 1, 2))
        del u
        Uf *= torch.conj(F.Hr)
        xx = torch.fft.irfftn(Uf, s=(F.nxp, F.nyp, F.N), dim=(0, 1, 2))
        del Uf
        out = xx[:self.nx, :self.ny].clone()
        del xx
        return out

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.sample(self.conv(q))

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return self.conv_adjoint(self.sample_adjoint(r))

    @property
    def lipschitz(self) -> float:
        return float(self.F.G_max)

    def to_tensor(self, arr, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr),
                               dtype=dtype or self.dtype, device=self.device)

    def measurement_gain(self, row_mask: torch.Tensor) -> torch.Tensor:
        """``c^fine_v = A^T 1_rec``; ``row_mask`` is ``1_rec``."""
        return self.adjoint(row_mask.to(self.dtype))


def solve_fine_arm(zop: FineZSOperator, support: torch.Tensor, alpha: float,
                   iters: int, log_every: int = 200, tag: str = "") -> torch.Tensor:
    """One nonlinear fine arm: FISTA from ``q0 = 0`` with the SHIPPED prox.

    ``CoordProx(alpha, support)`` is ``max(v - step*alpha, 0) * support``, so
    ``alpha = 0`` is positivity alone.  Nothing in the engine, the data term or
    the prox is re-implemented here.
    """
    t0 = time.time()
    hist: list = []

    def cb(k, ctx):
        if log_every and (k % log_every == 0 or k == iters - 1):
            q = ctx.q
            s = float(q.sum())
            n = int((q > 0).sum())
            hist.append({"iter": int(k), "sum_ke": s, "nnz": n,
                         "wall_s": time.time() - t0})
            print(f"    [{tag}] iter {k:5d}  sum {s:10.2f} ke  nnz {n:9d}  "
                  f"{time.time() - t0:6.1f} s", flush=True)

    prox = CoordProx(float(alpha), support)
    x = Fista(n_iter=int(iters)).minimize(zop, [DataFidelity(zop)], prox,
                                          q0=None, callback=cb)
    return x, hist, time.time() - t0


def upsample_support(base: np.ndarray, c: np.ndarray, B: float, b: int,
                     N: int, cell_ticks: int = 1) -> np.ndarray:
    """``base_cell[p, m] = base[p, k(cc_m)]``; ``False`` outside every cell.

    ``cc_m = b + cell_ticks m + (cell_ticks-1)/2`` is the unknown's own centre,
    and ``k`` is the PRODUCTION coarse cell containing it.  With
    ``cell_ticks = 1`` this is ``base[p, k(b + j)]``, the fine-grid form.
    """
    ct = int(cell_ticks)
    m = np.arange(N, dtype=np.int64)
    k = coarse_index(b + ct * m + (ct - 1) / 2.0, c[0], B)
    ok = (k >= 0) & (k < base.shape[2])
    out = np.zeros(base.shape[:2] + (N,), dtype=bool)
    out[:, :, ok] = np.asarray(base, dtype=bool)[:, :, k[ok]]
    return out


def record_row_mask(op, F: FineOperator, nx: int, ny: int,
                    device, dtype) -> torch.Tensor:
    """``1_rec`` on the padded record grid: 1 where a record row exists.

    Built from the operator's own sampling map, by pushing a vector of ones
    through ``sample_adjoint`` -- the same route ``measurement_gain`` takes, so
    the set of rows is the operator's and not a re-derivation of it.
    """
    ones = torch.ones(op.n_data, dtype=op.dtype, device=op.device)
    blk = op.sample_adjoint(ones)                 # (nx, ny, nt_block)
    m = (blk[:, :, :F.M] != 0).cpu().numpy()
    del ones, blk
    torch.cuda.empty_cache()
    return embed_pads(m.astype(np.float64), F.nxp, F.nyp, F.M, device, dtype)


# ---------------------------------------------------------------------------
# Algorithm 1: the arms
# ---------------------------------------------------------------------------
@algorithm("FineNonlinearArms")
class FineNonlinearArms(_Recorder):
    """Positivity and positivity + l1 on the fine basis, scored at the goal.

    Reference arms in the same record, none of them re-derived: the fine
    closed-form minimum-norm inverse at one ``lambda_rel``, and every
    bin-integrated arm produced by
    :class:`~unfoldlarpix.algs.evalharness_algs.LinearArms` in the same job.

    Props
    -----
    arms : list of dict
        ``{label, alpha, iters}``; positivity is always on (that is what makes
        them the arms of this module).  Default:
        ``fine_pos_a0`` at 2000 iterations plus three l1 arms.
    convergence_arm : str
        Label of the arm re-solved at ``convergence_iters`` for the
        convergence statement.  Default ``fine_pos_a0``.
    convergence_iters : int, default 1000.
    lambda_rel_ref : float, default 1e-6 -- the fine LINEAR reference.
    support : str, default ``"gain:0.5"``.
    sigma_H_us : list, default ``[0.0, 1.5, 2.0]``.
    prolongations : list, for the COARSE arms only; default
        ``["delta", "corrected_hat"]``.
    margin_windows, line_pixel_y_range, segment_pixels, segment_edge_exclude
    dtype : ``"float32"`` (default here -- the fine grid does not fit in
        float64 on an 11.7 GiB card together with a FISTA iterate).
    cell_ticks : int, default 1
        Width of the unknown's time cell in fine ticks (``1`` = the fine
        basis, unchanged in every respect).  See the "INTERMEDIATE (cell)
        time basis" section of
        :mod:`~unfoldlarpix.algs.finebasis_algs`.
    cell_model : ``"uniform"`` (default) or ``"delta"``
        The within-cell charge model of the OPERATOR.  Irrelevant at
        ``cell_ticks = 1``, where both are ``h`` itself.
    cell_prolongations : list, default ``["uniform", "corrected_hat"]``
        The evaluation prolongations ``P_0`` and ``P_1`` used for the cell
        arms.  At ``cell_ticks = 1`` they are ignored and ``P = I`` is used, as
        before.
    iteration_timing_iters : int, default 0 (off)
    iteration_timing_cell_ticks : list of int, default ``[]``
        If both are given, the wall time per FISTA iteration is measured at
        each of these cell widths with the same support and data.
    out_json, out_npz
    """

    reads = ("op", "support", "event", "readout_config", "block_offset",
             "charge_model", "arms.q")
    writes = ("finenl.result", "finenl.solutions")

    DEFAULT_ARMS = [
        {"label": "fine_pos_a0", "alpha": 0.0, "iters": 2000},
        {"label": "fine_pos_l1_0.003", "alpha": 0.003, "iters": 1000},
        {"label": "fine_pos_l1_0.01", "alpha": 0.01, "iters": 1000},
        {"label": "fine_pos_l1_0.03", "alpha": 0.03, "iters": 1000},
    ]

    def execute(self, store):
        op = store.get("op")
        arms_coarse = store.get("arms.q") if "arms.q" in store else {}
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        B = int(round(fit_bin_ticks(store)))
        dtype = (torch.float64 if str(self.props.get("dtype", "float32"))
                 == "float64" else torch.float32)
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [0.0, 1.5, 2.0])]
        pnames = [str(v) for v in self.props.get(
            "prolongations", ["delta", "corrected_hat"])]
        margin = int(self.props.get("margin_windows", 40))
        lam_rel = float(self.props.get("lambda_rel_ref", 1e-6))
        supp_spec = str(self.props.get("support", "gain:0.5"))
        specs = self.props.get("arms") or self.DEFAULT_ARMS
        conv_arm = str(self.props.get("convergence_arm", "fine_pos_a0"))
        conv_iters = int(self.props.get("convergence_iters", 1000))
        lin_label = str(self.props.get("linear_label", "fine_minnorm"))
        ct = int(self.props.get("cell_ticks", 1))
        cmodel = str(self.props.get("cell_model", "uniform"))
        cpnames = [str(v) for v in self.props.get(
            "cell_prolongations", ["uniform", "corrected_hat"])]
        time_iters = int(self.props.get("iteration_timing_iters", 0))
        time_cells = [int(v) for v in
                      self.props.get("iteration_timing_cell_ticks", [])]
        dev = op.device

        prep = self.services["detector"].prepared(B)
        fr = np.asarray(prep.full_response, dtype=np.float64)

        t0 = time.time()
        F = FineOperator(fr, op.block_shape, B, device=dev, dtype=dtype,
                         cell_ticks=ct, cell_model=cmodel)
        t_build = time.time() - t0
        H = EvalHarness(
            store, op, margin_windows=margin,
            line_pixel_y_range=self.props.get("line_pixel_y_range", (5, 131)),
            segment_pixels=int(self.props.get("segment_pixels", 7)),
            segment_edge_exclude=int(self.props.get("segment_edge_exclude", 3)))
        pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
        win_lo = int(H.fine[0]) - pad_ext
        win_hi = int(H.fine[-1]) + 1 + pad_ext
        grid = CellGrid(b, ct, F.N)
        m_lo, m_hi = grid.window(win_lo, win_hi)

        print(f"[{self.name}] operator (cell_ticks {ct}, {cmodel}) "
              f"{F.nxp}x{F.nyp}x{F.N} ({t_build:.1f} s); unknowns on the REAL "
              f"pads {H.nx}x{H.ny}x{F.N} = {H.nx * H.ny * F.N / 1e6:.1f} M; "
              f"max G {F.G_max:.6g} (= the exact Lipschitz constant of A^T A)")

        rec: dict = {
            "geometry": {
                "block_shape": [int(v) for v in op.block_shape],
                "q_shape_coarse": [int(v) for v in op.q_shape],
                "q_shape_fine_real_pads": [H.nx, H.ny, F.N],
                "n_fine_unknowns_real_pads": int(H.nx * H.ny * F.N),
                "padded_pads": [F.nxp, F.nyp],
                "block_offset": [float(v) for v in boff],
                "B_fine_ticks": B, "M_windows": F.M, "N_fine_ticks": F.N,
                "stored_fine_window": [win_lo, win_hi],
                "eval_window_cells": [H.k0, H.k1],
                "dtype": str(dtype)},
            "basis": {"cell_ticks": ct, "cell_model": cmodel,
                      "cells_per_record_window": B // ct,
                      "decimation_stride_D": F.D,
                      "n_cells_per_pad": F.N,
                      "stored_cell_window": [m_lo, m_hi],
                      "prolongations": (["identity"] if ct == 1 else cpnames)},
            "lipschitz": {
                "value": float(F.G_max),
                "source": "max_nu Ghat(nu), the closed-form symbol of A A^T",
                "exact": True,
                "note": ("this is ||A A^T|| = ||A^T A|| exactly, not a power "
                         "iteration estimate; deleting the padding-pad columns "
                         "can only lower the norm, so it stays a valid FISTA "
                         "step bound"),
                "coarse_operator_lipschitz_power_iteration": float(op.lipschitz)},
            "build_wall_s": t_build,
            "truth_total_ke": float(H.truth_total),
        }

        # ---------------- support ------------------------------------------
        rowm = record_row_mask(op, F, H.nx, H.ny, dev, dtype)
        rec["record_rows"] = {
            "n_data_rows_of_op": int(op.n_data),
            "n_record_cells_padded_grid": int(F.nxp * F.nyp * F.M),
            "n_record_cells_real_pads": int(H.nx * H.ny * F.M),
            "n_cells_with_a_row": int(rowm.sum().item()),
            "note": ("the fine operator's row space is the whole padded record "
                     "grid; cells with no record row carry data exactly zero, "
                     "which is the same data vector the closed-form inverse of "
                     "FineBasisInverse was given.  The coarse operator has no "
                     "row there at all.  c^fine_v is summed over the cells "
                     "that DO carry a row, which is the fine analogue of "
                     "ZSOperator.measurement_gain")}

        zop = FineZSOperator(F, torch.zeros((F.nxp, F.nyp, F.M), dtype=dtype,
                                            device=dev), H.nx, H.ny)
        cv = zop.measurement_gain(rowm)
        cv_max = float(cv.max())
        gain_cut = float(supp_spec.split(":", 1)[1]) if supp_spec.startswith(
            "gain:") else 0.0
        gain_mask = (cv > gain_cut * cv_max)
        del cv
        torch.cuda.empty_cache()

        base_c = np.asarray(resolve_support(store, op, "hits"))
        base_f = upsample_support(base_c, H.c, H.B, b, F.N, ct)
        supp_t = torch.as_tensor(base_f, device=dev) & gain_mask
        del gain_mask
        torch.cuda.empty_cache()
        supp_np_flat = None
        n_keep = int(supp_t.sum().item())

        # truth charge excluded by the support.  ``jj`` is the index of the
        # unknown that holds each truth deposit: the fine tick on the fine
        # basis, the cell m = floor((tick - b)/c) on the cell basis.
        jj = grid.index(H.truth_tick)
        inside = (jj >= 0) & (jj < F.N)
        keep_t = np.zeros(len(jj), dtype=bool)
        idx = (torch.as_tensor(H.truth_ix[inside], device=dev),
               torch.as_tensor(H.truth_iy[inside], device=dev),
               torch.as_tensor(jj[inside], device=dev))
        keep_t[inside] = supp_t[idx].cpu().numpy()
        q_excl = float(H.truth_q[~keep_t].sum())
        rec["support"] = {
            "spec": supp_spec,
            "c_fine_max_ke_per_unit": cv_max,
            "gain_cut": gain_cut,
            "n_fine_cells_total": int(H.nx * H.ny * F.N),
            "n_fine_cells_kept": n_keep,
            "fraction_kept": n_keep / float(H.nx * H.ny * F.N),
            "n_fine_cells_kept_by_hits_only": int(base_f.sum()),
            "truth_charge_excluded_ke": q_excl,
            "truth_charge_excluded_fraction": q_excl / max(H.truth_total, 1e-30),
            "coarse_support_gain05_cells": int(
                resolve_support(store, op, supp_spec).sum())}
        print(f"[{self.name}] support {supp_spec}: {n_keep} of "
              f"{H.nx * H.ny * F.N} fine cells ({100 * n_keep / (H.nx * H.ny * F.N):.3f} %), "
              f"hits alone {int(base_f.sum())}; truth charge excluded "
              f"{q_excl:.4g} ke ({100 * q_excl / H.truth_total:.4g} %)")
        del base_f
        torch.cuda.empty_cache()

        # ---------------- data ----------------------------------------------
        blk = block_from_rows(op)
        y_t = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
        y_norm = float(torch.linalg.vector_norm(y_t))
        zop.d = y_t
        sum_y = float(blk.sum())
        rec["data"] = {"sum_records_ke": sum_y,
                       "sum_abs_records_ke": float(np.abs(blk).sum()),
                       "sum_y_over_sum_Kbar_ke": sum_y / float(fr.sum()),
                       "sum_Kbar": float(fr.sum())}

        # the residual floor: what the TRUTH itself leaves.  x_truth is
        # non-negative and inside the support, so no non-negative estimator on
        # this support can do better than this by more than the operator's own
        # freedom; the difference from zero is tred's tick-0 truncation.
        # on the cell basis x_truth is R_c x (box coarsening onto the cells),
        # which is the non-negative candidate the basis can represent.
        xtru = torch.zeros((H.nx, H.ny, F.N), dtype=dtype, device=dev)
        xtru.index_put_((torch.as_tensor(H.truth_ix[inside], device=dev),
                         torch.as_tensor(H.truth_iy[inside], device=dev),
                         torch.as_tensor(jj[inside], device=dev)),
                        torch.as_tensor(H.truth_q[inside], dtype=dtype,
                                        device=dev), accumulate=True)
        rt = zop.forward(xtru) - y_t
        rec["data"]["truth_residual_rel"] = float(
            torch.linalg.vector_norm(rt) / y_norm)
        # split it: the cells that carry an actual record row, against the
        # cells on which the operator ASSERTS a zero record because no row
        # exists there (pads that never triggered, windows outside the
        # acquisition, and the padding ring).  Only the first part is a
        # comparison with data.
        rec["data"]["truth_residual_rel_on_record_rows"] = float(
            torch.linalg.vector_norm(rt * rowm) / y_norm)
        rec["data"]["truth_residual_rel_on_asserted_zero_rows"] = float(
            torch.linalg.vector_norm(rt * (1.0 - rowm)) / y_norm)
        rec["data"]["truth_residual_note"] = (
            "||A x_truth - y|| / ||y||: the floor a non-negative estimator on "
            "this support can reach.  It is not zero for two reasons that the "
            "split separates: tred deletes the current before tick 0 and the "
            "operator does not model that (on the record rows), and the fine "
            "operator's row space is the whole padded record grid, so it "
            "asserts a zero record on every (pad, window) cell that carries no "
            "row while the truth does induce charge there")
        del rt, xtru
        torch.cuda.empty_cache()
        print(f"[{self.name}] residual floor ||A x_truth - y||/||y|| = "
              f"{rec['data']['truth_residual_rel']:.4e} "
              f"(record rows {rec['data']['truth_residual_rel_on_record_rows']:.4e}"
              f", asserted-zero rows "
              f"{rec['data']['truth_residual_rel_on_asserted_zero_rows']:.4e})")

        # ---------------- the fine LINEAR reference --------------------------
        sol: dict = {}
        arm_rec: list = []

        def _record_solution(label, xh, meta):
            """Global statistics on the GPU, then keep only the eval window."""
            r = zop.forward(xh) - y_t
            resid = float(torch.linalg.vector_norm(r) / y_norm)
            resid_rows = float(torch.linalg.vector_norm(r * rowm) / y_norm)
            resid_zrows = float(
                torch.linalg.vector_norm(r * (1.0 - rowm)) / y_norm)
            del r
            pos = float(xh[xh > 0].sum())
            neg = float(xh[xh < 0].sum())
            tot = float(xh.sum())
            nnz = int((xh > 0).sum())
            nnz3 = int((xh > 1e-3).sum())
            padsum = xh.sum(axis=2)                      # (nx, ny)
            on_line = float(padsum.reshape(-1)[
                torch.as_tensor(H.truth_pad_rows, device=dev)].sum())
            row = {**meta, "label": label,
                   "sum_xhat_ke": tot, "sum_xhat_pos_ke": pos,
                   "sum_xhat_neg_ke": neg,
                   "sum_over_truth": tot / H.truth_total,
                   "sum_over_sum_y_over_Kbar": tot / (sum_y / float(fr.sum())),
                   "nnz": nnz, "nnz_1e-3": nnz3,
                   "residual_rel": resid,
                   "residual_rel_on_record_rows": resid_rows,
                   "residual_rel_on_asserted_zero_rows": resid_zrows,
                   "on_line_pad_charge_ke": on_line,
                   "off_line_pad_charge_ke": tot - on_line,
                   "off_line_fraction": (tot - on_line) / max(tot, 1e-30)}
            arm_rec.append(row)
            sol[label] = xh[:, :, m_lo:m_hi].cpu().numpy()
            transverse = padsum.sum(axis=1).cpu().numpy()
            del padsum
            print(f"[{self.name}] {label:22s} sum {tot:10.2f} ke "
                  f"({tot / H.truth_total:6.4f} x truth)  x+ {pos:9.2f}  "
                  f"x- {neg:10.2f}  nnz {nnz:9d}  |Ax-y|/|y| {resid:.4e}  "
                  f"off-line {tot - on_line:+9.3f} ke")
            return transverse

        arrays: dict = {}
        lam = lam_rel * F.G_max
        t0 = time.time()
        xlin = F.solve(y_t, lam)[:H.nx, :H.ny].contiguous()
        wall = time.time() - t0
        arrays["transverse_" + lin_label] = _record_solution(
            lin_label, xlin,
            {"kind": "fine_linear", "alpha": None, "iters": None,
             "lambda_rel": lam_rel, "lambda": lam, "wall_s": wall,
             "positivity": False})
        del xlin
        torch.cuda.empty_cache()

        # ---------------- the nonlinear fine arms ----------------------------
        conv_extra = []
        for spec in specs:
            lab = str(spec["label"])
            alpha = float(spec.get("alpha", 0.0))
            iters = int(spec.get("iters", 1000))
            print(f"[{self.name}] solving {lab}: positivity, alpha={alpha:g}, "
                  f"{iters} iterations", flush=True)
            xh, hist, wall = solve_fine_arm(zop, supp_t, alpha, iters, tag=lab)
            arrays["transverse_" + lab] = _record_solution(
                lab, xh, {"kind": "fine_nonlinear", "alpha": alpha,
                          "iters": iters, "wall_s": wall, "positivity": True,
                          "iteration_history": hist})
            del xh
            torch.cuda.empty_cache()
            if lab == conv_arm and iters != conv_iters:
                print(f"[{self.name}] convergence check: {lab} at {conv_iters} "
                      f"iterations", flush=True)
                xh, hist, wall = solve_fine_arm(zop, supp_t, alpha, conv_iters,
                                                tag=lab + f"@{conv_iters}")
                arrays[f"transverse_{lab}_it{conv_iters}"] = _record_solution(
                    f"{lab}_it{conv_iters}",
                    xh, {"kind": "fine_nonlinear_convergence", "alpha": alpha,
                         "iters": conv_iters, "wall_s": wall,
                         "positivity": True, "iteration_history": hist})
                conv_extra.append(f"{lab}_it{conv_iters}")
                del xh
                torch.cuda.empty_cache()
        del supp_t, y_t
        zop.d = None
        del F, zop
        torch.cuda.empty_cache()

        # ---------------- wall time per FISTA iteration vs the cell width ----
        # The cost statement of the intermediate basis.  Each entry builds the
        # operator, the record-row mask and the SAME ``gain:0.5`` support at
        # that cell width and runs ``iteration_timing_iters`` positivity
        # iterations from q0 = 0; the wall time is divided by the iteration
        # count.  Setup is outside the timed region and the final ``sum``
        # forces the CUDA queue to drain before the clock is read.
        if time_iters and time_cells:
            tim = []
            for cc in time_cells:
                Ft = FineOperator(fr, op.block_shape, B, device=dev,
                                  dtype=dtype, cell_ticks=cc,
                                  cell_model=cmodel)
                rmt = record_row_mask(op, Ft, H.nx, H.ny, dev, dtype)
                zt = FineZSOperator(
                    Ft, embed_pads(blk, Ft.nxp, Ft.nyp, Ft.M, dev, dtype),
                    H.nx, H.ny)
                cvt = zt.measurement_gain(rmt)
                mt = cvt > gain_cut * float(cvt.max())
                del cvt, rmt
                st = torch.as_tensor(
                    upsample_support(base_c, H.c, H.B, b, Ft.N, cc),
                    device=dev) & mt
                del mt
                torch.cuda.empty_cache()
                t0 = time.time()
                xt, _h, _w = solve_fine_arm(zt, st, 0.0, time_iters,
                                            log_every=0, tag=f"timing_c{cc}")
                float(xt.sum())
                wall = time.time() - t0
                tim.append({"cell_ticks": cc, "n_iterations": time_iters,
                            "n_unknowns_real_pads": int(H.nx * H.ny * Ft.N),
                            "wall_s": wall,
                            "wall_s_per_iteration": wall / time_iters,
                            "dtype": str(dtype)})
                print(f"[{self.name}] timing c = {cc:2d}: {wall / time_iters:.4f} "
                      f"s per FISTA iteration ({time_iters} iterations, "
                      f"{H.nx * H.ny * Ft.N / 1e6:.1f} M unknowns, {dtype})")
                zt.d = None
                del xt, st, zt, Ft
                torch.cuda.empty_cache()
            rec["iteration_timing"] = tim

        rec["arms"] = arm_rec
        rec["convergence_pair"] = {"arm": conv_arm, "iters_a": conv_iters,
                                   "iters_b": int(
                                       [s["iters"] for s in specs
                                        if s["label"] == conv_arm][0])
                                   if any(s["label"] == conv_arm for s in specs)
                                   else None}

        # ---------------- scoring --------------------------------------------
        rows_r1 = np.array([r for r in range(H.n_pads)
                            if H.pixel_x_of_pad[r] == 140
                            and 5 <= H.pixel_y_of_pad[r] <= 131])
        rows: list = []

        def _do(tag, xh, meta):
            m = score_rows(H, xh, meta["sigma_H_us"])
            pr = m.pop("_profiles")
            rows.append({**meta, **m})
            for k in ("line_xhat", "line_Hx", "line_e", "allpad_xhat"):
                arrays[f"prof_{tag}_{k}"] = pr[k].astype(np.float32)
            arrays[f"ring1prof_{tag}"] = xh[rows_r1].mean(axis=0).astype(np.float32)
            r = rows[-1]
            print(f"[{self.name}] s{r['sigma_H_us']:4.2f} {r['basis']:6s} "
                  f"{str(r.get('prolongation')):14s} {r['arm']:24s} "
                  f"E_rel {r['E_rel']:9.5f}  cons {r['conservation_rel']:+9.5f} "
                  f" ring1+ {r['zero_preservation']['ring1']['pos_per_pad_ke']:8.4f}"
                  f"  ring1- {r['zero_preservation']['ring1']['neg_per_pad_ke']:8.4f}")

        fine_labels = [lin_label] + [str(s["label"]) for s in specs] \
            + conv_extra
        for s in sigmas:
            for pname in pnames:
                _do(f"repr_{pname}_s{s:g}", coarse_xhat(H, H.Rx, pname, s),
                    {"arm": "representation_term", "basis": "coarse",
                     "prolongation": pname, "sigma_H_us": s})
                for lab, a in arms_coarse.items():
                    _do(f"{lab}_{pname}_s{s:g}", coarse_xhat(H, a["q"], pname, s),
                        {"arm": lab, "basis": "coarse", "prolongation": pname,
                         "sigma_H_us": s})
            for lab in fine_labels:
                if ct == 1:
                    _do(f"{lab}_s{s:g}", fine_xhat(H, sol[lab], win_lo, s),
                        {"arm": lab, "basis": "fine",
                         "prolongation": "identity", "sigma_H_us": s})
                    continue
                for pname in cpnames:
                    _do(f"{lab}_{pname}_s{s:g}",
                        cell_xhat(H, grid, sol[lab], pname, m_lo, m_hi, s,
                                  x_lo=m_lo),
                        {"arm": lab, "basis": "cell", "cell_ticks": ct,
                         "cell_model": cmodel, "prolongation": pname,
                         "sigma_H_us": s})
        rec["rows"] = rows

        # ---------------- pad-summed coarse-cell profile (N3) ----------------
        ka = int(np.argmax(np.abs(H.Rx).sum(axis=(0, 1))))
        ks = list(range(max(ka - 5, 0), min(ka + 6, H.n_coarse)))
        osc = {"cells": ks, "cell_center_ticks": [float(H.c[k]) for k in ks],
               "truth_Rbox_ke": [float(H.Rx[:, :, k].sum()) for k in ks]}
        for lab, a in arms_coarse.items():
            osc[lab + "_ke"] = [float(a["q"][:, :, k].sum()) for k in ks]
        kw = coarse_index(grid.cc[m_lo:m_hi], H.c[0], H.B)
        for lab in fine_labels:
            tot = sol[lab].sum(axis=(0, 1))
            osc[lab + "_Rxhat_ke"] = [float(tot[kw == k].sum()) for k in ks]
        rec["coarse_pad_summed_profile"] = osc

        # ---------------- arrays for the figures ------------------------------
        arrays["fine_ticks"] = H.fine.astype(np.int64)
        arrays["coarse_centers"] = H.c.astype(np.float64)
        arrays["cell_centers"] = grid.cc[m_lo:m_hi].astype(np.float64)
        arrays["cell_window"] = np.array([m_lo, m_hi])
        # the truth profile on the UNKNOWN grid: per fine tick at c = 1, per
        # cell at c > 1.  ``stored_window_ticks`` stays the abscissa of the
        # raw profiles, so it is the cell centres when c > 1.
        arrays["stored_window_ticks"] = (
            np.arange(win_lo, win_hi).astype(np.int64) if ct == 1
            else grid.cc[m_lo:m_hi])
        v = np.zeros(m_hi - m_lo)
        jt = grid.index(H.truth_tick) - m_lo
        ok = (jt >= 0) & (jt < len(v))
        np.add.at(v, jt[ok], H.truth_q[ok])
        arrays["truth_padsum_fine"] = v
        tv = np.zeros(H.nx)
        np.add.at(tv, H.truth_ix, H.truth_q)
        arrays["transverse_truth"] = tv
        for lab, a in arms_coarse.items():
            arrays["transverse_" + lab] = a["q"].sum(axis=(1, 2))
            qq = a["q"].reshape(H.n_pads, H.n_coarse)
            arrays["line_coarse_" + lab] = qq[H.line_rows].mean(axis=0)
        arrays["line_coarse_truth_Rbox"] = \
            H.Rx.reshape(H.n_pads, H.n_coarse)[H.line_rows].mean(axis=0)
        for lab in fine_labels:
            arrays["fine_line_raw_" + lab] = \
                sol[lab].reshape(H.n_pads, -1)[H.line_rows].mean(axis=0)
        arrays["ring1_row_pads"] = rows_r1
        rec["fine_labels"] = fine_labels
        rec["arm_specs"] = [dict(s) for s in specs]
        rec["sigma_H_us"] = sigmas
        rec["prolongations_coarse"] = pnames
        rec["coarse_arm_labels"] = sorted(arms_coarse.keys())
        rec["linear_label"] = lin_label
        self._emit(store, rec, arrays)
        self.put(store, self.writes[1],
                 {"x": sol, "win_lo": win_lo, "harness": H, "arrays": arrays,
                  "labels": fine_labels, "grid": grid,
                  "cell_window": [m_lo, m_hi]})


@algorithm("CellNonlinearArms")
class CellNonlinearArms(FineNonlinearArms):
    """:class:`FineNonlinearArms` on the intermediate basis, without the
    bin-integrated arms.

    Same body, same props, same solver stack; it only declares a store
    interface that does NOT require ``arms.q``, so a job can run the cell arms
    without re-solving the archived bin-integrated references, and it writes to
    its own store locations.  Set ``cell_ticks`` to the cell width; the
    defaults are those of the parent, i.e. the fine basis.
    """

    reads = ("op", "support", "event", "readout_config", "block_offset",
             "charge_model")
    writes = ("cellnl.result", "cellnl.solutions")


# ---------------------------------------------------------------------------
# Algorithm 2: figures N1-N4, N8
# ---------------------------------------------------------------------------
@algorithm("FineNonlinearPlots")
class FineNonlinearPlots(_Recorder):
    """Figures N1-N4 and N8 from the products of :class:`FineNonlinearArms`.

    Props
    -----
    figdir : str
    l1_middle : str      label of the l1 arm shown wherever one is shown.
    out_json : str
    """

    reads = ("finenl.result", "finenl.solutions")
    writes = ("finenl.figures",)

    def execute(self, store):
        self._res = store.get("finenl.result")
        self._sol = store.get("finenl.solutions")
        self.put(store, "finenl.figures", {"pending": True})

    def finalize(self):
        plt = ieee_style()
        rec, sol = self._res, self._sol
        A = sol["arrays"]
        outdir = Path(self.props.get("figdir", "figs_nonlin"))
        made: list = []
        rows = rec["rows"]
        specs = rec["arm_specs"]
        l1_labels = [str(s["label"]) for s in specs if float(s["alpha"]) > 0]
        pos_label = [str(s["label"]) for s in specs
                     if float(s["alpha"]) == 0][0]
        mid = str(self.props.get("l1_middle") or
                  (l1_labels[len(l1_labels) // 2] if l1_labels else ""))
        B = rec["geometry"]["B_fine_ticks"]
        dt = TICK_US

        def get(arm, s, pn=None):
            for r in rows:
                if (r["arm"] == arm and abs(r["sigma_H_us"] - s) < 1e-12
                        and (pn is None or r.get("prolongation") == pn)):
                    return r
            return None

        def armrec(lab):
            for r in rec["arms"]:
                if r["label"] == lab:
                    return r
            return None

        l1_colors = list(L1_TINTS)
        while len(l1_colors) < len(l1_labels):
            l1_colors.append(tint(C_POSL1, 0.8))

        # ---------------- N1: E_rel at the goal ---------------------------
        entries = [("representation ($R\\,x$)", "representation_term",
                    "corrected_hat", OI["grey"]),
                   ("LS_nopos, $P_1$", "LS_nopos", "corrected_hat", C_COARSE),
                   ("pos_a0, $P_1$", "pos_a0", "corrected_hat", OI["sky"]),
                   ("pos_l1_0.3, $P_1$", "pos_l1_0.3", "corrected_hat",
                    OI["orange"]),
                   ("fine min-norm", "fine_minnorm", None, C_FINE),
                   ("fine pos, $\\alpha=0$", pos_label, None, C_POS)]
        entries += [(rf"fine pos+$\ell_1$ {float(s['alpha']):g}",
                     str(s["label"]), None, l1_colors[i])
                    for i, s in enumerate([s for s in specs
                                           if float(s["alpha"]) > 0])]
        entries = [e for e in entries if get(e[1], 1.5, e[2]) is not None]
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
        xs = np.arange(len(entries))
        for i, s in enumerate((1.5, 2.0)):
            v = [(get(an, s, pn) or {}).get("E_rel", np.nan)
                 for _, an, pn, _c in entries]
            ax.bar(xs + (i - 0.5) * 0.4, v, 0.4, color=[e[3] for e in entries],
                   alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.4,
                   label=rf"$\sigma_H={s:g}\,\mu$s")
            for xi, vv in zip(xs, v):
                ax.text(xi + (i - 0.5) * 0.4, vv, f"{vv:.3f}", fontsize=5.5,
                        ha="center", va="bottom", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([e[0] for e in entries], rotation=25, ha="right",
                           fontsize=6)
        ax.set_ylabel(r"$E_{\rm rel}$")
        vmax = np.nanmax([(get(e[1], 1.5, e[2]) or {}).get("E_rel", 0)
                          for e in entries])
        ax.set_ylim(0, vmax * 1.45)
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, outdir, "N1_Erel_at_goal", made)

        # ---------------- N2: line-averaged profiles ----------------------
        ft = A["fine_ticks"].astype(float)
        cen = float(np.sum(A["truth_padsum_fine"] * A["stored_window_ticks"])
                    / max(A["truth_padsum_fine"].sum(), 1e-30))
        tus = (ft - cen) * dt
        wt = (A["stored_window_ticks"].astype(float) - cen) * dt
        fine_curves = [("fine min-norm", "fine_minnorm", C_FINE, "-"),
                       (r"fine pos, $\alpha=0$", pos_label, C_POS, "-")]
        fine_curves += [(rf"fine pos+$\ell_1$ {float(s['alpha']):g}",
                         str(s["label"]), l1_colors[i], "--")
                        for i, s in enumerate([s for s in specs
                                               if float(s["alpha"]) > 0])]
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.8))
        a = ax[0]
        a.plot(wt, A["truth_padsum_fine"] * 0 + np.nan)     # keep the axis
        a.plot(wt, np.zeros_like(wt), color="k", lw=0.4)
        tl = A["truth_padsum_fine"] / max(len(sol["harness"].line_rows), 1)
        a.plot(wt, tl, color=C_TRUTH, lw=1.1, label=r"$x$ (fine truth)")
        for lab, key, cc, ls in fine_curves:
            a.plot(wt, A["fine_line_raw_" + key], color=cc, ls=ls, lw=1.0,
                   label=lab)
        a.set_xlim(-9, 9)
        a.set_xlabel(r"time from the truth centroid [$\mu$s]")
        a.set_ylabel("mean over interior line pads [ke / fine tick]")
        a.set_title(r"(a) $\sigma_H = 0$ (raw)")
        a.legend(frameon=False, fontsize=6)

        a = ax[1]
        a.plot(tus, A["prof_fine_minnorm_s1.5_line_Hx"], color=C_HTRUTH,
               lw=1.2, label=r"$Hx$")
        a.plot(tus, A["prof_LS_nopos_corrected_hat_s1.5_line_xhat"],
               color=C_COARSE, lw=1.0, label=r"$HP_1$ LS_nopos")
        for lab, key, cc, ls in fine_curves:
            a.plot(tus, A[f"prof_{key}_s1.5_line_xhat"], color=cc, ls=ls,
                   lw=1.0, label=lab)
        a.set_xlim(-9, 9)
        a.set_xlabel(r"time from the truth centroid [$\mu$s]")
        a.set_title(r"(b) $\sigma_H = 1.5\,\mu$s")
        a.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, outdir, "N2_line_profiles", made)

        # ---------------- N3: pad-summed coarse-cell profile ----------------
        osc = rec["coarse_pad_summed_profile"]
        ks = osc["cells"]
        x = np.arange(len(ks))
        series = [(r"$R\,x$ (truth)", osc["truth_Rbox_ke"], C_TRUTH),
                  ("fine min-norm", osc["fine_minnorm_Rxhat_ke"], C_FINE),
                  (r"fine pos, $\alpha=0$", osc[pos_label + "_Rxhat_ke"], C_POS)]
        if mid:
            series.append((f"fine pos+$\\ell_1$ {mid.split('_')[-1]}",
                           osc[mid + "_Rxhat_ke"], C_POSL1))
        fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.8))
        w = 0.8 / len(series)
        for i, (lab, v, cc) in enumerate(series):
            ax[0].bar(x + (i - (len(series) - 1) / 2) * w, v, w, color=cc,
                      label=lab)
            if i:
                ax[1].bar(x + (i - 1 - (len(series) - 2) / 2) * w,
                          np.asarray(v) - np.asarray(osc["truth_Rbox_ke"]), w,
                          color=cc, label=lab)
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels([str(k) for k in ks], fontsize=6)
            a.set_xlabel(r"coarse cell $k$ (1.5 $\mu$s)")
            a.axhline(0, color="k", lw=0.5)
            a.legend(frameon=False, fontsize=6)
        ax[0].set_ylabel("pad-summed charge [ke]")
        ax[1].set_ylabel(r"difference to $R\,x$ [ke]")
        ax[0].set_title(r"(a) charge per 1.5 $\mu$s cell")
        ax[1].set_title(r"(b) difference to the truth cells")
        fig.tight_layout()
        save(fig, outdir, "N3_bin_space", made)

        # ---------------- N4: Q1 ring ledger --------------------------------
        led = [("LS_nopos", "LS_nopos", "corrected_hat", C_COARSE),
               ("pos_a0", "pos_a0", "corrected_hat", OI["sky"]),
               ("pos_l1_0.3", "pos_l1_0.3", "corrected_hat", OI["orange"]),
               ("fine min-norm", "fine_minnorm", None, C_FINE),
               ("fine pos $\\alpha=0$", pos_label, None, C_POS)]
        led += [(rf"fine $\ell_1$ {float(s['alpha']):g}", str(s["label"]),
                 None, l1_colors[i])
                for i, s in enumerate([s for s in specs
                                       if float(s["alpha"]) > 0])]
        led = [e for e in led if get(e[1], 0.0, e[2]) is not None]
        fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=False)
        rings = [("ring1", "ring 1"), ("ring2", "ring 2"), ("ring_ge3", r"ring $\geq$3")]
        for si, s in enumerate((0.0, 1.5)):
            a = ax[si]
            xs = np.arange(len(rings))
            w = 0.8 / len(led)
            for i, (lab, an, pn, cc) in enumerate(led):
                r = get(an, s, pn)["zero_preservation"]
                p = [r[k]["pos_per_pad_ke"] for k, _ in rings]
                n = [r[k]["neg_per_pad_ke"] for k, _ in rings]
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
        ax[0].legend(frameon=False, fontsize=5.5, ncol=2)
        fig.tight_layout()
        save(fig, outdir, "N4_ring_ledger", made)

        # ---------------- N8: the l1 scan ------------------------------------
        alphas = [0.0] + [float(s["alpha"]) for s in specs
                          if float(s["alpha"]) > 0]
        labs = [pos_label] + l1_labels
        st = [armrec(l) for l in labs]
        sq = [r["sum_over_truth"] for r in st]
        nz = [r["nnz_1e-3"] for r in st]
        er = [get(l, 1.5)["E_rel"] for l in labs]
        r1 = [get(l, 1.5)["zero_preservation"]["ring1"]["pos_per_pad_ke"]
              for l in labs]
        xax = np.array([max(a, 1e-4) for a in alphas])
        fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.5))
        for a, v, yl, ttl in ((ax[0], sq, r"$\Sigma\hat{x}\,/\,\Sigma x$",
                               "(a) charge"),
                              (ax[1], nz, "nnz ($\\hat{x}>10^{-3}$ ke)",
                               "(b) sparsity"),
                              (ax[2], er, r"$E_{\rm rel}(\sigma_H=1.5\,\mu$s)",
                               "(c) score"),
                              (ax[3], r1,
                               r"ring-1 $\Sigma^+$ per pad [ke]",
                               r"(d) ghost charge, $\sigma_H=1.5\,\mu$s")):
            a.semilogx(xax, v, color=C_POSL1, marker="o", ms=3.5, lw=1.0)
            a.set_xlabel(r"$\alpha$ [ke per fine tick]  ($10^{-4}$ = 0)")
            a.set_ylabel(yl)
            a.set_title(ttl)
        ax[0].axhline(1.0, color=C_TRUTH, lw=0.7, ls=":")
        lr = get("fine_minnorm", 1.5)
        ax[2].axhline(lr["E_rel"], color=C_FINE, lw=0.8, ls="--")
        ax[2].text(xax[0], lr["E_rel"], " fine min-norm", fontsize=5.5,
                   color=C_FINE, va="bottom")
        ax[3].axhline(lr["zero_preservation"]["ring1"]["pos_per_pad_ke"],
                      color=C_FINE, lw=0.8, ls="--")
        fig.tight_layout()
        save(fig, outdir, "N8_l1_scan", made)

        out = {"figures": made, "l1_middle": mid, "pos_label": pos_label}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": out}, fh, indent=1,
                          default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return out


# ---------------------------------------------------------------------------
# Algorithm 3: the signal-dependent probe
# ---------------------------------------------------------------------------
@algorithm("FineNonlinearProbe")
class FineNonlinearProbe(_Recorder):
    """Impulse responses of the NONLINEAR fine arms, at three charges.

    A nonlinear estimator has no resolution matrix.  What this algorithm
    measures is the response of the estimator to one unit deposit of a STATED
    charge ``Q`` at a stated sub-window phase and impact, which is a different
    object for every ``Q``; the linear fine inverse is put on the same axes and
    is the only arm whose answer scales with ``Q``.

    Records are built by the same exact-functional generator as
    :class:`~unfoldlarpix.algs.finebasis_algs.FineBasisProbe`
    (``records_from_impulse``, including tred's tick-0 current deletion) and
    mapped through ``row_meta``; the mapping is verified against the real
    event's ``op.d`` before any probe is solved.

    Props
    -----
    probe_pad, probe_phases, probe_impacts_at, probe_cell,
    current_zero_before_tick, sigma_H_us, margin_windows : as
        :class:`~unfoldlarpix.algs.finebasis_algs.FineBasisProbe`.
    cell_ticks : int, default 1;  cell_model : str, default ``"uniform"``;
    cell_prolongations : list, default ``["uniform", "corrected_hat"]``
        The unknown's time basis, as in :class:`FineNonlinearArms`.  At
        ``cell_ticks = 1`` the probe is the fine-basis probe, unchanged.
    charges_ke : list of float, default ``[5.0, 30.0, 150.0]``.
    arms : list of dict ``{label, alpha, iters}`` -- the nonlinear arms.
    lambda_rel : float, the linear reference.
    support : str, default ``"gain:0.5"``.
    convergence_iters : list of int, optional.  If given, the FIRST probe
        setting of the FIRST arm is additionally solved at each of these
        iteration counts and every metric is reported, which is the evidence
        for the iteration count used everywhere else.
    dtype, out_json, out_npz
    """

    reads = ("op", "support", "readout_config", "block_offset", "charge_model",
             "row_meta", "hits_view", "event")
    writes = ("finenl.resolution",)

    def execute(self, store):
        op = store.get("op")
        boff = np.asarray(store.get("block_offset"), dtype=float)
        b = int(boff[2])
        c, Bf, off = coarse_centers(store, op)
        B = int(round(Bf))
        charges = [float(v) for v in self.props.get("charges_ke",
                                                    [5.0, 30.0, 150.0])]
        tz = self.props.get("current_zero_before_tick", 0)
        tz = None if tz is None else int(tz)
        phases = [int(v) for v in self.props.get("probe_phases", [0, 15, 29])]
        imp_at = {int(k): [str(x) for x in v] for k, v in
                  (self.props.get("probe_impacts_at")
                   or {15: ["bar", "4,4", "0,0"]}).items()}
        ppad = [int(v) for v in self.props.get("probe_pad", [141, 68])]
        sigmas = [float(v) for v in self.props.get("sigma_H_us", [0.0, 1.5, 2.0])]
        margin = int(self.props.get("margin_windows", 40))
        lam_rel = float(self.props.get("lambda_rel", 1e-6))
        specs = self.props.get("arms") or [
            {"label": "fine_pos_a0", "alpha": 0.0, "iters": 1000}]
        conv_iters = [int(v) for v in (self.props.get("convergence_iters") or [])]
        supp_spec = str(self.props.get("support", "gain:0.5"))
        q_ref = float(self.props.get("q_ref_ke", 30.0))
        hs = self.props.get("homogeneity_setting") or [15, "bar"]
        homog_set = (int(hs[0]), str(hs[1]))
        conv_label = str(self.props.get("convergence_arm")
                         or (specs[0]["label"] if specs else ""))
        ct = int(self.props.get("cell_ticks", 1))
        cmodel = str(self.props.get("cell_model", "uniform"))
        cpnames = [str(v) for v in self.props.get(
            "cell_prolongations", ["uniform", "corrected_hat"])]
        lin_label = str(self.props.get("linear_label", "fine_minnorm"))
        dtype = (torch.float64 if str(self.props.get("dtype", "float32"))
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
        Rimp, meta = load_impact_response(str(path))
        dtus = meta["time_tick_us"]
        kcums = {}
        if "bar" in need:
            kcums["bar"] = np.cumsum(Rimp.mean(axis=(1, 3), dtype=np.float64),
                                     axis=-1) * dtus
        for nm in sorted(need - {"bar"}):
            ix, iy = (int(v) for v in nm.split(","))
            kcums[nm] = np.cumsum(Rimp[:, ix, :, iy, :].astype(np.float64),
                                  axis=-1) * dtus
        del Rimp

        prep = self.services["detector"].prepared(B)
        F = FineOperator(np.asarray(prep.full_response, dtype=np.float64),
                         op.block_shape, B, device=dev, dtype=dtype,
                         cell_ticks=ct, cell_model=cmodel)
        lam = lam_rel * F.G_max
        nx, ny = int(op.q_shape[0]), int(op.q_shape[1])
        grid = CellGrid(b, ct, F.N)
        zop = FineZSOperator(F, torch.zeros((F.nxp, F.nyp, F.M), dtype=dtype,
                                            device=dev), nx, ny)
        rec["basis"] = {"cell_ticks": ct, "cell_model": cmodel,
                        "decimation_stride_D": F.D, "n_cells_per_pad": F.N,
                        "prolongations": (["identity"] if ct == 1
                                          else cpnames)}
        rec["fine_arms"] = {"lambda_rel": lam_rel, "lambda": lam,
                            "G_max": F.G_max, "arms": [dict(s) for s in specs],
                            "charges_ke": charges, "q_ref_ke": q_ref,
                            "homogeneity_setting": list(homog_set),
                            "homogeneity_note": (
                                "alpha = 0 is positively homogeneous, so its "
                                "normalised response is charge independent by "
                                "construction; it is solved once per setting "
                                "at q_ref and at every charge only at the "
                                "homogeneity setting, where the identity is "
                                "checked numerically")}

        # -- the support, the real event's, exactly as the arms job uses ------
        rowm = record_row_mask(op, F, nx, ny, dev, dtype)
        cv = zop.measurement_gain(rowm)
        gcut = float(supp_spec.split(":", 1)[1]) if supp_spec.startswith("gain:") else 0.0
        gmask = cv > gcut * float(cv.max())
        del cv, rowm
        torch.cuda.empty_cache()
        base_c = np.asarray(resolve_support(store, op, "hits"))
        supp_t = torch.as_tensor(upsample_support(base_c, c, Bf, b, F.N, ct),
                                 device=dev) & gmask
        del gmask
        torch.cuda.empty_cache()
        rec["support"] = {"spec": supp_spec,
                          "n_fine_cells_kept": int(supp_t.sum().item())}

        # -- geometry -----------------------------------------------------------
        from .fixedgrid_algs import grid_truth
        kstar = int(self.props.get("probe_cell") or
                    np.argmax(grid_truth(store, op, mode="round").sum(axis=(0, 1))))
        px_b = ppad[0] - int(boff[0]); py_b = ppad[1] - int(boff[1])
        ring = 12
        dxy = np.array([(dx, dy) for dx in range(-ring, ring + 1)
                        for dy in range(-ring, ring + 1)], dtype=int)
        pads_abs = np.stack([dxy[:, 0] + px_b, dxy[:, 1] + py_b], axis=1)
        ok = ((pads_abs[:, 0] >= 0) & (pads_abs[:, 0] < nx)
              & (pads_abs[:, 1] >= 0) & (pads_abs[:, 1] < ny))
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

        settings = []
        for phi in phases:
            for kn in imp_at.get(phi, ["bar"]):
                settings.append((phi, kn))
        results, arrays = [], {}
        t_start = time.time()
        for phi, kn in settings:
            t_star = int(np.floor(c[kstar] - Bf / 2.0)) + int(phi)
            Hh = EvalHarness(store, op, margin_windows=margin,
                             line_pixel_y_range=(-10 ** 9, 10 ** 9),
                             truth=((np.array([px_b]), np.array([py_b]),
                                     np.array([t_star])), np.array([1.0])))
            pad_flat = px_b * Hh.ny + py_b
            pad_ext = int(np.ceil(5.0 * max(sigmas) / TICK_US)) + 2
            wlo = int(Hh.fine[0]) - pad_ext
            whi = int(Hh.fine[-1]) + 1 + pad_ext
            m_lo, m_hi = grid.window(wlo, whi)
            arrays[f"fine_ticks_phi{phi}_{kn}"] = Hh.fine.astype(np.int64)
            for s in (1.5,):
                gk = time_kernel(s / TICK_US)
                ref = np.zeros(Hh.n_fine)
                j0 = t_star - int(Hh.fine[0])
                half = (len(gk) - 1) // 2
                a0, a1 = max(0, j0 - half), min(Hh.n_fine, j0 + half + 1)
                ref[a0:a1] = gk[a0 - (j0 - half):a1 - (j0 - half)]
                arrays[f"Hdelta_phi{phi}_{kn}_s{s:g}"] = ref.astype(np.float32)

            for Q in charges:
                y1 = records_from_impulse(kcums[kn], dxy, latch_abs, t_star,
                                          Q, tz)
                blk = np.zeros(op.block_shape)
                good = rows_for >= 0
                blk[pads_abs[:, 0][:, None].repeat(Nl, 1)[good],
                    pads_abs[:, 1][:, None].repeat(Nl, 1)[good],
                    bins_for[good]] = y1[good]
                yt = embed_pads(blk, F.nxp, F.nyp, F.M, dev, dtype)
                zop.d = yt
                base = {"phi": int(phi), "kernel": kn, "t_star": int(t_star),
                        "charge_ke": Q, "sum_d_probe_ke": float(blk.sum()),
                        "sum_abs_d_probe_ke": float(np.abs(blk).sum())}

                # which arms run at this (setting, charge).  Every arm runs
                # at its own reference charge at every setting; an arm also
                # runs at the charges in its ``extra_charges`` but ONLY at the
                # designated ``homogeneity_setting``.  For alpha = 0 those
                # extra charges are the numerical verification of the
                # positive homogeneity proved in the module docstring; for an
                # l1 arm they would be a genuine charge scan.
                todo_nl = []
                for sp in specs:
                    q_arm = float(sp.get("q_ref", q_ref))
                    extra = [float(v) for v in (sp.get("extra_charges") or [])]
                    run = (abs(Q - q_arm) < 1e-9
                           or ((phi, kn) == homog_set
                               and any(abs(Q - e) < 1e-9 for e in extra)))
                    if run:
                        todo_nl.append((str(sp["label"]), float(sp["alpha"]),
                                        int(sp.get("iters", 1000))))
                if not todo_nl:
                    del yt
                    zop.d = None
                    torch.cuda.empty_cache()
                    continue
                todo = [(lin_label, None, None)] + todo_nl
                for lab, alpha, iters in todo:
                    t0 = time.time()
                    if alpha is None:
                        # the linear reference: closed form, independent of Q
                        xh = F.solve(yt, lam)[:nx, :ny].contiguous()
                        hist = []
                    else:
                        xh, hist, _ = solve_fine_arm(
                            zop, supp_t, alpha, iters, log_every=0, tag=lab)
                    wall = time.time() - t0
                    xwin = xh[:, :, m_lo:m_hi].cpu().numpy()
                    tot = float(xh.sum())
                    nnz = int((xh > 1e-3).sum())
                    del xh
                    torch.cuda.empty_cache()
                    r = {**base, "arm": lab, "alpha": alpha, "iters": iters,
                         "wall_s": wall, "sum_xhat_ke": tot,
                         "sum_over_Q": tot / Q, "nnz_1e-3": nnz, "sigmas": []}
                    for pname in (["identity"] if ct == 1 else cpnames):
                        for s in sigmas:
                            xhs = (fine_xhat(Hh, xwin, wlo, s) if ct == 1 else
                                   cell_xhat(Hh, grid, xwin, pname, m_lo,
                                             m_hi, s, x_lo=m_lo))
                            pm = probe_metrics(Hh, xhs, Q, t_star, pad_flat, s)
                            pm["prolongation"] = pname
                            r["sigmas"].append(pm)
                            if s in (0.0, 1.5):
                                tg = (f"imp_phi{phi}_{kn}_Q{Q:g}_{lab}"
                                      + ("" if ct == 1 else f"_{pname}")
                                      + f"_s{s:g}")
                                arrays[tg] = (xhs[pad_flat]
                                              / Q).astype(np.float32)
                    results.append(r)
                    m = [d for d in r["sigmas"]
                         if abs(d["sigma_H_us"] - 1.5) < 1e-9][0]
                    print(f"[{self.name}] phi {phi:2d} {kn:5s} Q {Q:6.1f} "
                          f"{lab:20s} norm {m['normalisation']:8.5f} shift "
                          f"{m['shift_ticks']:+7.2f} width "
                          f"{m['width_ticks'] if m['width_ticks'] else float('nan'):7.2f}"
                          f" ring1+ {m['ring1_pos_over_Q']:8.5f} | "
                          f"{time.time() - t_start:.0f} s", flush=True)

                    do_conv = (conv_iters and alpha is not None
                               and lab == conv_label
                               and (phi, kn) == homog_set
                               and abs(Q - q_ref) < 1e-9)
                    if do_conv:
                        for it2 in conv_iters:
                            xh2, _, w2 = solve_fine_arm(zop, supp_t, alpha, it2,
                                                        log_every=0, tag=lab)
                            xw2 = xh2[:, :, m_lo:m_hi].cpu().numpy()
                            t2 = float(xh2.sum())
                            del xh2
                            torch.cuda.empty_cache()
                            rr = {**base, "arm": lab + f"_it{it2}",
                                  "alpha": alpha, "iters": it2, "wall_s": w2,
                                  "sum_xhat_ke": t2, "sum_over_Q": t2 / Q,
                                  "convergence_probe": True, "sigmas": []}
                            for s in sigmas:
                                xh2s = (fine_xhat(Hh, xw2, wlo, s) if ct == 1
                                        else cell_xhat(Hh, grid, xw2,
                                                       cpnames[0], m_lo, m_hi,
                                                       s, x_lo=m_lo))
                                rr["sigmas"].append(probe_metrics(
                                    Hh, xh2s, Q, t_star, pad_flat, s))
                            results.append(rr)
                            mm = rr["sigmas"][1]
                            print(f"[{self.name}]   convergence {lab} "
                                  f"{it2:5d} it: norm {mm['normalisation']:8.5f} "
                                  f"shift {mm['shift_ticks']:+7.2f} width "
                                  f"{mm['width_ticks']:7.2f} sum {t2:9.4f}",
                                  flush=True)
                zop.d = None
                del yt
                torch.cuda.empty_cache()
        rec["probes"] = results
        rec["sigma_H_us"] = sigmas
        rec["linear_label"] = lin_label
        rec["settings"] = [{"phi": p, "kernel": k} for p, k in settings]
        self._emit(store, rec, arrays)


@algorithm("FineNonlinearProbePlots")
class FineNonlinearProbePlots(_Recorder):
    """Figures N5-N7 from the products of :class:`FineNonlinearProbe`.

    Props: ``figdir``, ``in_npz``, ``q_main`` (default 30.0), ``out_json``.
    """

    reads = ("finenl.resolution",)
    writes = ("finenl.probe_figures",)

    def execute(self, store):
        self._res = store.get("finenl.resolution")
        self.put(store, "finenl.probe_figures", {"pending": True})

    def finalize(self):
        plt = ieee_style()
        rec = self._res
        npz = self.props.get("in_npz")
        A = dict(np.load(npz, allow_pickle=True)) if npz else {}
        outdir = Path(self.props.get("figdir", "figs_nonlin"))
        made: list = []
        pr = [r for r in rec["probes"] if not r.get("convergence_probe")]
        Qm = float(self.props.get("q_main", 30.0))
        charges = rec["fine_arms"]["charges_ke"]
        nl = [str(s["label"]) for s in rec["fine_arms"]["arms"]]
        lin = "fine_minnorm"
        phases = sorted({r["phi"] for r in pr if r["kernel"] == "bar"})

        def g(arm, phi, kn, Q, s):
            for r in pr:
                if (r["arm"] == arm and r["phi"] == phi and r["kernel"] == kn
                        and abs(r["charge_ke"] - Q) < 1e-9):
                    for d in r["sigmas"]:
                        if abs(d["sigma_H_us"] - s) < 1e-9:
                            return d
            return None

        # ------------- N5 / N6: F6 layout, one per nonlinear arm ----------
        for fi, arm in enumerate(nl):
            cc = C_POS if fi == 0 else C_POSL1
            fig = plt.figure(figsize=(10.5, 5.2))
            gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.38)
            show = phases[:4] if len(phases) >= 4 else phases
            for i, phi in enumerate(show):
                a = fig.add_subplot(gs[0, i])
                t = A.get(f"fine_ticks_phi{phi}_bar")
                if t is None:
                    continue
                ts = [r["t_star"] for r in pr if r["phi"] == phi][0]
                tu = (t - ts) * TICK_US
                a.plot(tu, A[f"Hdelta_phi{phi}_bar_s1.5"], color=C_TRUTH,
                       lw=1.0, label=r"$H\delta$")
                k = f"imp_phi{phi}_bar_Q{Qm:g}_{lin}_s1.5"
                if k in A:
                    a.plot(tu, A[k], color=C_FINE, lw=1.0, label="min-norm")
                k = f"imp_phi{phi}_bar_Q{Qm:g}_{arm}_s1.5"
                if k in A:
                    a.plot(tu, A[k], color=cc, lw=1.0, label=arm)
                a.set_xlim(-6, 6)
                a.set_xlabel(r"$t-t^*$ [$\mu$s]")
                if i == 0:
                    a.set_ylabel(r"$\hat{x}/Q$ on the probed pad [1/tick]")
                    a.legend(frameon=False, fontsize=6)
                a.set_title(rf"(a) $\varphi={phi}$, $Q={Qm:g}$ ke")

            a = fig.add_subplot(gs[1, 0])
            for nm, c2 in ((lin, C_FINE), (arm, cc)):
                v = [g(nm, p, "bar", Qm, 0.0) for p in phases]
                a.plot(phases, [x["shift_ticks"] if x else np.nan for x in v],
                       color=c2, marker="o", ms=2.5, label=nm)
            a.plot(phases, [14.5 - p for p in phases], color=C_HTRUTH, ls="--",
                   lw=0.9, label=r"coarse-basis ideal $14.5-\varphi$")
            a.axhline(0, color=C_TRUTH, lw=0.7, ls=":")
            a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
            a.set_ylabel("time shift [fine ticks]")
            a.set_title(rf"(b) first moment, $\sigma_H=0$, $Q={Qm:g}$ ke")
            a.legend(frameon=False, fontsize=6)

            a = fig.add_subplot(gs[1, 1])
            for nm, c2 in ((lin, C_FINE), (arm, cc)):
                v = [g(nm, p, "bar", Qm, 1.5) for p in phases]
                a.plot(phases, [(x["width_ticks"] if x and x["width_ticks"]
                                 else np.nan) for x in v],
                       color=c2, marker="o", ms=2.5, label=nm)
            a.axhline(1.5 / TICK_US, color=C_HTRUTH, ls="--", lw=0.9,
                      label=r"ideal $\sigma_H/\Delta t = 30$")
            a.set_xlabel(r"arrival phase $\varphi$ [fine ticks]")
            a.set_ylabel("width [fine ticks]")
            a.set_title(rf"(c) second moment, $\sigma_H=1.5\,\mu$s, $Q={Qm:g}$ ke")
            a.legend(frameon=False, fontsize=6)

            kns = [k for k in ("bar", "4,4", "0,0")
                   if any(r["kernel"] == k and r["phi"] == 15 for r in pr)]
            styles = {"bar": ("-", "impact-averaged"),
                      "4,4": ("--", "impact (4,4)"), "0,0": (":", "impact (0,0)")}
            for i, (nm, c2, ttl) in enumerate(((lin, C_FINE, "min-norm"),
                                               (arm, cc, arm))):
                a = fig.add_subplot(gs[1, 2 + i])
                ts = [r["t_star"] for r in pr if r["phi"] == 15]
                if not ts:
                    continue
                ts = ts[0]
                for kn in kns:
                    t = A.get(f"fine_ticks_phi15_{kn}")
                    k = f"imp_phi15_{kn}_Q{Qm:g}_{nm}_s1.5"
                    if t is None or k not in A:
                        continue
                    ls, lab = styles[kn]
                    a.plot((t - ts) * TICK_US, A[k], color=c2, ls=ls, lw=1.0,
                           label=lab)
                t = A.get("fine_ticks_phi15_bar")
                if t is not None:
                    a.plot((t - ts) * TICK_US, A["Hdelta_phi15_bar_s1.5"],
                           color=C_TRUTH, lw=1.0, label=r"$H\delta$")
                a.set_xlim(-6, 6)
                a.set_xlabel(r"$t-t^*$ [$\mu$s]")
                a.set_title(rf"(d) $\varphi=15$, {ttl}, $Q={Qm:g}$ ke")
                a.legend(frameon=False, fontsize=6)
            save(fig, outdir, f"N{5 + fi}_probe_{arm}", made)

        # ------------- N7: signal dependence -------------------------------
        fig, ax = plt.subplots(1, 4, figsize=(11.0, 2.6))
        panels = [("shift_ticks", "first moment [fine ticks]",
                   "(a) first moment", 0.0),
                  ("width_ticks", "width [fine ticks]", "(b) width", 1.5),
                  ("normalisation", r"$\Sigma\hat{x}/Q$", "(c) normalisation", 1.5),
                  ("ring1_pos_over_Q", r"ring-1 $\Sigma^+/Q$",
                   "(d) ring-1 positive charge", 1.5)]
        for i, (key, yl, ttl, s) in enumerate(panels):
            a = ax[i]
            for fi, arm in enumerate(nl):
                c2 = C_POS if fi == 0 else C_POSL1
                v = [g(arm, 15, "bar", Q, s) for Q in charges]
                a.semilogx(charges, [(x[key] if x and x[key] is not None
                                      else np.nan) for x in v],
                           color=c2, marker="o", ms=3.5, lw=1.0, label=arm)
            x0 = g(lin, 15, "bar", charges[0], s)
            if x0 and x0[key] is not None:
                a.axhline(x0[key], color=C_FINE, ls="--", lw=0.9,
                          label="fine min-norm (linear)")
            a.set_xlabel(r"injected charge $Q$ [ke]")
            a.set_ylabel(yl)
            a.set_title(rf"{ttl}, $\sigma_H={s:g}\,\mu$s" if s else
                        rf"{ttl}, $\sigma_H=0$")
        ax[0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, outdir, "N7_signal_dependence", made)

        out = {"figures": made}
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": out}, fh, indent=1,
                          default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        return out
