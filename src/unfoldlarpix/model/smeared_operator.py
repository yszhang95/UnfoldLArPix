"""Band-limited forward map: A' = A G, a NEW operator beside ZSOperator.

Motivation (technote 4.4): the support carries far more unknowns than
there are rows, so A P_S has full row rank and the solver can reproduce
any d -- including the part the operator models wrongly. A Gaussian G is
invertible, so it does not change that rank; what it changes is the
PRICE: the small singular values fall by ~60x at the analysis width, so
with a penalty on the coefficients the error-absorbing directions become
expensive while the signal-carrying ones are untouched.

The unknown is the coefficient field u; the physical estimate is
q = G u, a sum of Gaussian pulses. Consequences:

* positivity is imposed on u, which implies q >= 0 because the kernel is
  non-negative;
* the l1 term is unchanged in value by the smoothing --- for u >= 0 and
  a mass-conserving G, ||G u||_1 = ||u||_1 --- so penalising u is the
  same sparsity prior, while an l2 penalty on u becomes the smoothness
  prior (Tikhonov with weight G^{-2});
* every consumer of the block prediction (censor, data fidelity) sees
  ``conv(u) = A_conv(G u)``, so the whole objective is consistent;
* the evaluation must apply the Gaussian EXACTLY ONCE. Store u as
  ``deconv_q_sharp`` and let the standard gaussian deposit supply the
  smoothing, matching the once-smeared truth.

Widths follow the project convention: frequency-domain sigmas, so the
real-space width is 1/(2 pi sigma).
"""
from __future__ import annotations

import numpy as np
import torch

from .operator import ZSOperator


class SmearedOperator(ZSOperator):
    """``A' = A G`` with G a separable Gaussian on the charge grid."""

    def __init__(self, *args, sigma_time: float = 0.005,
                 sigma_pixel: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # the time axis of the q grid is in BINS, but the project's
        # sigma_time is quoted per FINE TICK (gaussian_filter_3d passes
        # d = adc_hold_delay), so the frequency axis carries that spacing.
        self._bin_ticks = int(args[3] if len(args) > 3
                              else kwargs["adc_hold_delay"])
        self.sigma_time = float(sigma_time)
        self.sigma_pixel = float(sigma_pixel)
        self._filt = self._build_filter()
        self._lipschitz = None

    def _build_filter(self) -> torch.Tensor:
        """Frequency-domain Gaussian on the q grid (rfft over the last axis)."""
        qx, qy, qt = self.q_shape
        fx = torch.fft.fftfreq(qx, d=1.0, device=self.device)
        fy = torch.fft.fftfreq(qy, d=1.0, device=self.device)
        ft = torch.fft.rfftfreq(qt, d=float(self._bin_ticks),
                                device=self.device)
        gx = torch.exp(-0.5 * (fx / self.sigma_pixel) ** 2)
        gy = torch.exp(-0.5 * (fy / self.sigma_pixel) ** 2)
        gt = torch.exp(-0.5 * (ft / self.sigma_time) ** 2)
        f = gx[:, None, None] * gy[None, :, None] * gt[None, None, :]
        return f.to(torch.complex64 if self.dtype == torch.float32
                    else torch.complex128)

    def smear(self, u: torch.Tensor) -> torch.Tensor:
        """G u -- symmetric, mass conserving, non-negative kernel."""
        U = torch.fft.rfftn(u, dim=(0, 1, 2))
        return torch.fft.irfftn(U * self._filt, s=self.q_shape, dim=(0, 1, 2))

    # -- the operator: everything downstream sees A_conv(G u) ---------------
    def conv(self, u: torch.Tensor) -> torch.Tensor:
        return super().conv(self.smear(u))

    def conv_adjoint(self, r_block: torch.Tensor) -> torch.Tensor:
        return self.smear(super().conv_adjoint(r_block))

    def physical(self, u) -> np.ndarray:
        """The physical charge estimate q = G u (numpy, on the q grid)."""
        t = u if torch.is_tensor(u) else self.to_tensor(u)
        return self.smear(t).cpu().numpy().astype(np.float64)
