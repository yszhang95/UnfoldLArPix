"""Phase-exact window sampling — a NEW operator, parallel to ZSOperator.

The stock operator samples window integrals from bin-integrated
predictions with fractional-overlap weights on the two partial (edge)
bins — first order in the within-bin current shape. This class computes
the edge contributions EXACTLY for the operator's own charge model
(all of a bin's charge at the bin start, matching the bin-summed
kernel): a window edge at fine tick e on pixel p contributes

    F(e) = sum_v q_v * CS(e - t_v),    row = F(t_hi) - F(t_lo),

with CS the cumulative fine response. At bin boundaries F reduces to
the running sum of the convolved block (identical to the stock full-bin
path); within a bin the increment is a 30-phase comb through CS,

    D[phi][dx, dy, k] = CS[dx, dy, phi + k*B] - CS[dx, dy, k*B],

evaluated by direct neighbourhood inner products with q. No new
unknowns, same q grid, same convolution; only the two edge weights per
row change. ``ZSOperator`` itself is untouched; this class overrides
``forward``/``adjoint`` only. NOTE: the inherited ``sample``/``conv``
pair still implements the stock (box) path — terms that consume
``block_pred`` (e.g. the stock ``DataFidelity``) will not see the edge
correction; use :class:`PhaseDataFidelity` from this module instead.
"""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import LatchWindow
from .operator import ZSOperator

CHUNK = 64


class ZSOperatorPhase(ZSOperator):
    def __init__(
        self,
        kernel: np.ndarray,
        fine_kernel: np.ndarray,
        block_shape: tuple[int, int, int],
        windows: list[LatchWindow],
        adc_hold_delay: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        row_weights: np.ndarray | None = None,
    ):
        super().__init__(kernel, block_shape, windows, adc_hold_delay,
                         device=device, dtype=dtype, row_weights=row_weights)
        B = int(adc_hold_delay)
        self._B = B
        kx, ky, kt = np.asarray(kernel).shape
        self._kt = kt
        fk = np.asarray(fine_kernel, dtype=np.float64)
        if fk.shape[0] != kx or fk.shape[1] != ky:
            raise ValueError("fine kernel pixel shape mismatch")
        need = kt * B
        if fk.shape[2] < need:
            fk = np.pad(fk, ((0, 0), (0, 0), (0, need - fk.shape[2])))
        fk = fk[:, :, :need]
        binned = fk.reshape(kx, ky, kt, B).sum(-1)
        if not np.allclose(binned, np.asarray(kernel, np.float64),
                           rtol=1e-6, atol=1e-9):
            raise ValueError("fine kernel does not bin to the coarse kernel")
        # D[phi, dx, dy, k] = CS[phi + kB] - CS[kB], phi = 0..B
        cs = np.concatenate([np.zeros((kx, ky, 1)), np.cumsum(fk, axis=2)],
                            axis=2)
        idx = (np.arange(B + 1)[:, None] + B * np.arange(kt)[None, :])
        D = cs[:, :, idx] - cs[:, :, idx[:1] * 0 + (B * np.arange(kt))[None, :]]
        # D currently (kx, ky, B+1, kt) -> (B+1, kx, ky, kt)
        D = np.moveaxis(D, 2, 0)
        # flip for the gather orientation: q[p - i + c], q[k0 - k]
        Dg = D[:, ::-1, ::-1, ::-1].copy()
        self._Dg = torch.as_tensor(Dg, dtype=self.dtype, device=self.device)

        # ---- edge decomposition, mirroring windows_to_sampling clipping
        nx, ny, nt = self.block_shape
        sw = (np.sqrt(np.asarray(row_weights, np.float64))
              if row_weights is not None else None)
        base_rows, base_cols, base_coef = [], [], []
        groups: dict[int, list] = {}
        for r, w in enumerate(windows):
            if not (0 <= w.px < nx and 0 <= w.py < ny):
                continue
            lo = max(float(w.t_lo), 0.0)
            hi = min(float(w.t_hi), float(nt * B))
            if hi <= lo:
                continue
            s0 = float(sw[r]) if sw is not None else 1.0
            for e, sgn in ((hi, +1.0), (lo, -1.0)):
                k0 = int(np.floor(e / B + 1e-9))
                phi = e - k0 * B
                if k0 >= nt:
                    k0, phi = nt, 0.0
                if k0 > 0:
                    base_rows.append(r)
                    base_cols.append((w.px * ny + w.py) * nt + (k0 - 1))
                    base_coef.append(sgn * s0)
                if phi > 1e-9 and k0 < nt:
                    ip = int(np.floor(phi))
                    f = phi - ip
                    for pphi, wt in ((ip, 1.0 - f), (ip + 1, f)):
                        if wt <= 1e-12 or pphi == 0:
                            continue
                        groups.setdefault(pphi, []).append(
                            (r, w.px, w.py, k0, sgn * s0 * wt))
        self._base_rows = torch.as_tensor(np.asarray(base_rows, np.int64),
                                          device=self.device)
        self._base_cols = torch.as_tensor(np.asarray(base_cols, np.int64),
                                          device=self.device)
        self._base_coef = torch.as_tensor(np.asarray(base_coef, np.float64),
                                          dtype=self.dtype,
                                          device=self.device)
        self._groups = []
        for phi, lst in sorted(groups.items()):
            a = np.asarray([(r, px, py, k0) for (r, px, py, k0, c) in lst],
                           np.int64)
            c = np.asarray([c for (*_, c) in lst], np.float64)
            self._groups.append((
                phi,
                torch.as_tensor(a[:, 0], device=self.device),
                torch.as_tensor(a[:, 1], device=self.device),
                torch.as_tensor(a[:, 2], device=self.device),
                torch.as_tensor(a[:, 3], device=self.device),
                torch.as_tensor(c, dtype=self.dtype, device=self.device)))

    # -- padded q frame ----------------------------------------------------
    def _pad_q(self, q: torch.Tensor) -> torch.Tensor:
        # pad time by kt-1 on BOTH sides: qp[..., k0+m] = q[..., k0-kt+1+m]
        # for m in [0, kt), valid for k0 up to nt-1 > qt-1; pixels cx/cy.
        return torch.nn.functional.pad(
            q[None], (self._kt - 1, self._kt - 1,
                      self.cy, self.cy, self.cx, self.cx)
        )[0]

    def _flat_idx(self, px, py, k0):
        """Flat indices of the (kx, ky, kt) patch around each edge in the
        padded q frame. Returns (E, kx, ky, kt) int64."""
        kx, ky2 = 2 * self.cx + 1, 2 * self.cy + 1
        qx, qy, qt = self.q_shape
        PY = qy + 2 * self.cy
        PT = qt + 2 * (self._kt - 1)
        ix = px[:, None] + torch.arange(kx, device=self.device)[None, :]
        iy = py[:, None] + torch.arange(ky2, device=self.device)[None, :]
        it = k0[:, None] + torch.arange(self._kt, device=self.device)[None, :]
        return ((ix[:, :, None, None] * PY + iy[:, None, :, None]) * PT
                + it[:, None, None, :])

    # -- exact operator ------------------------------------------------------
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        block = self.conv(q)
        cb = torch.cumsum(block, dim=2)
        out = torch.zeros(self.n_data, dtype=self.dtype, device=self.device)
        out.index_add_(0, self._base_rows,
                       self._base_coef * cb.reshape(-1)[self._base_cols])
        if self._groups:
            qp = self._pad_q(q).reshape(-1)
            for phi, rows, px, py, k0, coef in self._groups:
                for s in range(0, len(rows), CHUNK):
                    sl = slice(s, s + CHUNK)
                    fi = self._flat_idx(px[sl], py[sl], k0[sl])
                    inc = (qp[fi] * self._Dg[phi]).sum(dim=(1, 2, 3))
                    out.index_add_(0, rows[sl], coef[sl] * inc)
        return out

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        nx, ny, nt = self.block_shape
        T = torch.zeros(int(np.prod(self.block_shape)), dtype=self.dtype,
                        device=self.device)
        T.index_add_(0, self._base_cols, self._base_coef * r[self._base_rows])
        T = T.reshape(self.block_shape)
        # adjoint of cumsum-gather: suffix sum
        block_adj = torch.flip(torch.cumsum(torch.flip(T, dims=(2,)), dim=2),
                               dims=(2,))
        g = self.conv_adjoint(block_adj)
        if self._groups:
            qx, qy, qt = self.q_shape
            PT = qt + 2 * (self._kt - 1)
            gp = torch.zeros((qx + 2 * self.cx) * (qy + 2 * self.cy) * PT,
                             dtype=self.dtype, device=self.device)
            for phi, rows, px, py, k0, coef in self._groups:
                for s in range(0, len(rows), CHUNK):
                    sl = slice(s, s + CHUNK)
                    fi = self._flat_idx(px[sl], py[sl], k0[sl])
                    amp = (coef[sl] * r[rows[sl]])[:, None, None, None]
                    gp.index_add_(0, fi.reshape(-1),
                                  (amp * self._Dg[phi]).reshape(-1))
            gp = gp.reshape(qx + 2 * self.cx, qy + 2 * self.cy, PT)
            g = g + gp[self.cx:self.cx + qx, self.cy:self.cy + qy,
                       self._kt - 1:self._kt - 1 + qt]
        return g


class PhaseDataFidelity:
    """1/2 ||A_phase q - d||^2 through the exact forward/adjoint.

    Same interface as terms.data.DataFidelity but routed through
    ``op.forward``/``op.adjoint`` so the edge correction enters the
    solve. ``target`` supports the final-refit background subtraction.
    """

    def __init__(self, op: ZSOperatorPhase, target: torch.Tensor | None = None):
        self.op = op
        self.target = op.d if target is None else target

    def _resid(self, ctx) -> torch.Tensor:
        return self.op.forward(ctx.q) - self.target

    def value(self, ctx) -> torch.Tensor:
        r = self._resid(ctx)
        return 0.5 * (r * r).sum()

    def grad_into(self, ctx, out: torch.Tensor) -> None:
        out += self.op.adjoint(self._resid(ctx))

    def curvature(self) -> float:
        return self.op.lipschitz
