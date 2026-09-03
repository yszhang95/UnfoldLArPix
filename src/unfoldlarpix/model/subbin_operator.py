"""Uniform-within-bin forward model — coarse unknowns, fine waveform.

The stock :class:`~unfoldlarpix.model.operator.ZSOperator` carries TWO
first-order assumptions, and they are separable:

1. **the within-bin charge model** — a fit bin's charge is released as a
   delta at the bin's LOWER edge, because that is what
   ``integrate_kernel_over_time`` builds;
2. **the window sampling** — a readout window credits its two partial fit
   bins by their overlap fraction, i.e. the current is taken as uniform
   inside a bin.

A fixed-interval readout makes (2) exact by construction (every window edge
is a fit-bin edge, one column per row at weight 1.000000), so on that sample
(1) is the whole model error.  ``ZSOperatorPhase`` attacks (2) and is a
diagnostic; this class attacks (1).

The model here: each coarse unknown ``q_j`` is a charge spread UNIFORMLY
over its fit bin, expanded onto a sub-bin grid of ``S`` cells of
``b = B/S`` fine ticks each at ``q_j/S`` per cell, convolved with the
response integrated at ``b``, and the resulting FINE waveform is then fed
to the ordinary overlap-weighted window sampling.  The unknowns never leave
the coarse grid — the solver's freedom, the support, the truth binning and
the evaluation are all unchanged — only the forward model is refined.  So
``A = Sample_b . K_b . U_S`` with ``U_S`` the flat expansion and
``U_S^T`` its adjoint (sum the S cells, divide by S).

Equivalence.  When every window edge lands on a COARSE bin boundary,
``Sample_b`` just re-sums each window's fine bins, and the whole chain
collapses to a coarse convolution with the Bartlett-combed kernel of
:func:`~unfoldlarpix.deconv_workflow.uniform_within_bin_kernel` — same
answer, ``S`` times cheaper, and it needs no new operator at all (set
``within_bin: uniform`` on the detector service).  That is the route the
production jobs use; this class is what proves the two agree and what the
general (trigger-based, unaligned-window) case would need.
"""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import LatchWindow
from .operator import ZSOperator


class ZSOperatorUniform(ZSOperator):
    """Coarse charge unknowns, sub-bin forward convolution.

    Parameters
    ----------
    fine_kernel
        Response integrated at ``adc_hold_delay // subbin`` ticks per bin
        (the DELTA form — the uniform spread is applied here, not there).
    block_shape
        The COARSE block ``(nx, ny, nt)``; the fine block is ``nt*subbin``.
    windows, adc_hold_delay
        As :class:`ZSOperator`; ``adc_hold_delay`` is the COARSE fit bin.
    subbin
        ``S``.  ``S = 1`` reduces exactly to :class:`ZSOperator`.
    """

    def __init__(
        self,
        fine_kernel: np.ndarray,
        block_shape: tuple[int, int, int],
        windows: list[LatchWindow],
        adc_hold_delay: int,
        subbin: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        row_weights: np.ndarray | None = None,
    ):
        S = int(subbin)
        B = int(adc_hold_delay)
        if S < 1 or B % S != 0:
            raise ValueError(f"subbin {S} must be >=1 and divide "
                             f"adc_hold_delay {B}")
        nx, ny, nt_c = (int(v) for v in block_shape)
        ktf = int(np.asarray(fine_kernel).shape[2])
        # the fine block covers exactly the coarse one; windows are in fine
        # ticks already, so the sampling matrix is built at bin b = B/S.
        super().__init__(fine_kernel, (nx, ny, nt_c * S), windows, B // S,
                         device=device, dtype=dtype, row_weights=row_weights)
        self.subbin = S
        self.coarse_bin = B
        self.q_shape_fine = self.q_shape            # set by ZSOperator
        # coarse unknowns: as many whole fit bins as the fine charge grid
        # holds.  With ktf = S*ktc this is nt_c - ktc, i.e. one bin short of
        # the delta operator's grid -- the same bin the uniform kernel's
        # extra tap eats, so the two routes end on the same last bin.
        qt = int(self.q_shape_fine[2]) // S
        self.q_shape = (nx, ny, qt)
        self._qt_fine_used = qt * S

    # -- flat expansion / its adjoint --------------------------------------
    def expand(self, q: torch.Tensor) -> torch.Tensor:
        """``U_S q``: a coarse bin's charge spread flat over its S cells."""
        return torch.repeat_interleave(q, self.subbin, dim=2) / self.subbin

    def reduce(self, g: torch.Tensor) -> torch.Tensor:
        """``U_S^T g``: sum each coarse bin's S cells, divided by S."""
        nx, ny, _ = self.q_shape
        return (g[:, :, :self._qt_fine_used]
                .reshape(nx, ny, self.q_shape[2], self.subbin)
                .sum(dim=3) / self.subbin)

    # -- block-space convolution, coarse in / fine block out ---------------
    def conv(self, q: torch.Tensor) -> torch.Tensor:
        pred = torch.fft.irfftn(
            torch.fft.rfftn(self.expand(q), s=self.fft_shape, dim=(0, 1, 2))
            * self._K, s=self.fft_shape, dim=(0, 1, 2))
        pred = torch.roll(pred, -self.cx, dims=0)
        pred = torch.roll(pred, -self.cy, dims=1)
        nx, ny, nt = self.block_shape
        return pred[:nx, :ny, :nt]

    def conv_adjoint(self, r_block: torch.Tensor) -> torch.Tensor:
        nx, ny, nt = self.block_shape
        padded = torch.zeros(self.fft_shape, dtype=self.dtype,
                             device=self.device)
        padded[:nx, :ny, :nt] = r_block
        padded = torch.roll(padded, self.cy, dims=1)
        padded = torch.roll(padded, self.cx, dims=0)
        out = torch.fft.irfftn(
            torch.fft.rfftn(padded, dim=(0, 1, 2)) * torch.conj(self._K),
            s=self.fft_shape, dim=(0, 1, 2))
        qx, qy, qt = self.q_shape_fine
        return self.reduce(out[:qx, :qy, :qt])
