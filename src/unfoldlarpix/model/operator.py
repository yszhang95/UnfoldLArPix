"""The ZS measurement operator — single (torch) implementation.

A = window sampling ∘ bin-integrated response convolution.  IMMUTABLE
after construction: the measurement never changes during a solve
(mutating ``d`` for background subtraction is forbidden — express that
at the term level).  Immutability is what makes the cached Lipschitz
constant and the shared kernel FFT safe to reuse across ladder stages
and strategies.

Variants (row weighting, sub-bin split) are produced by WRAPPING, never
by mutating (see the split-operator pattern).
"""
from __future__ import annotations

import numpy as np
import torch

from ..constrained_solver import LatchWindow, windows_to_sampling


class ZSOperator:
    """Forward/adjoint operator: charge q -> recorded burst integrals."""

    def __init__(
        self,
        kernel: np.ndarray,
        block_shape: tuple[int, int, int],
        windows: list[LatchWindow],
        adc_hold_delay: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        row_weights: np.ndarray | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_shape = tuple(int(s) for s in block_shape)
        nx, ny, nt = self.block_shape
        kernel = np.asarray(kernel, dtype=np.float64)
        kx, ky, kt = kernel.shape
        self.q_shape = (nx, ny, nt - kt + 1)
        self.fft_shape = (nx + kx - 1, ny + ky - 1, nt)
        self.cx = (kx - 1) // 2
        self.cy = (ky - 1) // 2
        k_t = torch.as_tensor(kernel, dtype=self.dtype, device=self.device)
        self._K = torch.fft.rfftn(k_t, s=self.fft_shape, dim=(0, 1, 2))

        self.n_data = len(windows)
        d_np = np.array([w.value for w in windows], dtype=np.float64)
        rows, cols, weights = windows_to_sampling(
            windows, self.block_shape, adc_hold_delay)
        if row_weights is not None:
            sw = np.sqrt(np.asarray(row_weights, dtype=np.float64))
            d_np = d_np * sw
            weights = weights * sw[rows]
        self.d = torch.as_tensor(d_np, dtype=self.dtype, device=self.device)
        self._rows = torch.as_tensor(rows, device=self.device)
        self._cols = torch.as_tensor(cols, device=self.device)
        self._weights = torch.as_tensor(weights, dtype=self.dtype,
                                        device=self.device)
        self._lipschitz: float | None = None

    # -- block-space convolution ------------------------------------------
    def conv(self, q: torch.Tensor) -> torch.Tensor:
        pred = torch.fft.irfftn(
            torch.fft.rfftn(q, s=self.fft_shape, dim=(0, 1, 2)) * self._K,
            s=self.fft_shape, dim=(0, 1, 2))
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
        qx, qy, qt = self.q_shape
        return out[:qx, :qy, :qt]

    # -- sampling -----------------------------------------------------------
    def sample(self, block: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_data, dtype=self.dtype, device=self.device)
        out.index_add_(0, self._rows,
                       self._weights * block.reshape(-1)[self._cols])
        return out

    def sample_adjoint(self, r: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(int(np.prod(self.block_shape)), dtype=self.dtype,
                           device=self.device)
        flat.index_add_(0, self._cols, self._weights * r[self._rows])
        return flat.reshape(self.block_shape)

    # -- full operator ------------------------------------------------------
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.sample(self.conv(q))

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return self.conv_adjoint(self.sample_adjoint(r))

    @property
    def lipschitz(self) -> float:
        """||A^T A|| by power iteration — computed once, cached (the
        operator is immutable, so the cache is always valid)."""
        if self._lipschitz is None:
            g = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randn(self.q_shape, generator=g, dtype=self.dtype)
            x = (x / torch.linalg.vector_norm(x)).to(self.device)
            lam = 1.0
            for _ in range(12):
                y = self.adjoint(self.forward(x))
                lam = float(torch.linalg.vector_norm(y))
                if lam <= 0:
                    lam = 1.0
                    break
                x = y / lam
            self._lipschitz = lam
        return self._lipschitz

    def to_tensor(self, arr, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr),
                               dtype=dtype or self.dtype, device=self.device)
