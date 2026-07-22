"""Torch/GPU backend for the constrained ZS solver.

Mirrors :mod:`unfoldlarpix.constrained_solver` — same operator geometry,
same FISTA math — with all FFTs and scatter/gather on the torch device.
Inputs and outputs at the API boundary are numpy arrays, so the driver can
swap backends without touching anything else.

float32 by default: consumer GPUs run FP64 at 1/64 rate, and FISTA
recomputes the gradient every iteration so single-precision rounding does
not accumulate.  Verify against the numpy backend once per configuration.
"""

from __future__ import annotations

import numpy as np

import torch

from .constrained_solver import LatchWindow, windows_to_sampling


class TorchZSOperator:
    """GPU forward/adjoint operator: charge q -> recorded burst integrals."""

    def __init__(
        self,
        kernel: np.ndarray,
        block_shape: tuple[int, int, int],
        windows: list[LatchWindow],
        adc_hold_delay: int,
        device: str = "cuda",
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
        self.d = torch.as_tensor(
            np.array([w.value for w in windows], dtype=np.float64),
            dtype=self.dtype, device=self.device,
        )
        rows, cols, weights = windows_to_sampling(
            windows, self.block_shape, adc_hold_delay
        )
        if row_weights is not None:
            sw = np.sqrt(np.asarray(row_weights, dtype=np.float64))
            d_np = np.array([w.value for w in windows], dtype=np.float64) * sw
            self.d = torch.as_tensor(d_np, dtype=self.dtype, device=self.device)
            weights = weights * sw[rows]
        self._rows = torch.as_tensor(rows, device=self.device)
        self._cols = torch.as_tensor(cols, device=self.device)
        self._weights = torch.as_tensor(
            weights, dtype=self.dtype, device=self.device
        )

    # -- block-space convolution ------------------------------------------
    def conv(self, q: torch.Tensor) -> torch.Tensor:
        pred = torch.fft.irfftn(
            torch.fft.rfftn(q, s=self.fft_shape, dim=(0, 1, 2)) * self._K,
            s=self.fft_shape, dim=(0, 1, 2),
        )
        pred = torch.roll(pred, -self.cx, dims=0)
        pred = torch.roll(pred, -self.cy, dims=1)
        nx, ny, nt = self.block_shape
        return pred[:nx, :ny, :nt]

    def conv_adjoint(self, r_block: torch.Tensor) -> torch.Tensor:
        nx, ny, nt = self.block_shape
        padded = torch.zeros(self.fft_shape, dtype=self.dtype, device=self.device)
        padded[:nx, :ny, :nt] = r_block
        padded = torch.roll(padded, self.cy, dims=1)
        padded = torch.roll(padded, self.cx, dims=0)
        out = torch.fft.irfftn(
            torch.fft.rfftn(padded, dim=(0, 1, 2)) * torch.conj(self._K),
            s=self.fft_shape, dim=(0, 1, 2),
        )
        qx, qy, qt = self.q_shape
        return out[:qx, :qy, :qt]

    # -- sampling -----------------------------------------------------------
    def sample(self, block: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_data, dtype=self.dtype, device=self.device)
        out.index_add_(
            0, self._rows, self._weights * block.reshape(-1)[self._cols]
        )
        return out

    def sample_adjoint(self, r: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(
            int(np.prod(self.block_shape)), dtype=self.dtype, device=self.device
        )
        flat.index_add_(0, self._cols, self._weights * r[self._rows])
        return flat.reshape(self.block_shape)

    # -- full operator ------------------------------------------------------
    def forward(self, q) -> "torch.Tensor | np.ndarray":
        """Accepts a torch tensor or numpy array; returns the same kind."""
        if isinstance(q, np.ndarray):
            out = self.sample(self.conv(self.to_tensor(q)))
            return out.cpu().numpy().astype(np.float64)
        return self.sample(self.conv(q))

    def adjoint(self, r) -> "torch.Tensor | np.ndarray":
        """Accepts a torch tensor or numpy array; returns the same kind."""
        if isinstance(r, np.ndarray):
            out = self.conv_adjoint(self.sample_adjoint(self.to_tensor(r)))
            return out.cpu().numpy().astype(np.float64)
        return self.conv_adjoint(self.sample_adjoint(r))

    def lipschitz(self, n_iter: int = 12, seed: int = 0) -> float:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(self.q_shape, generator=g, dtype=self.dtype)
        x = (x / torch.linalg.vector_norm(x)).to(self.device)
        lam = 1.0
        for _ in range(n_iter):
            y = self.adjoint(self.forward(x))
            lam = float(torch.linalg.vector_norm(y))
            if lam <= 0:
                return 1.0
            x = y / lam
        return lam

    def to_tensor(self, arr, dtype=None) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(arr), dtype=dtype or self.dtype,
            device=self.device,
        )


def _tv_gradient_torch(x: torch.Tensor, eps: float = 1e-6):
    diffs = [torch.diff(x, dim=ax) for ax in range(3)]
    norm = torch.sqrt(sum((d ** 2).sum() for d in diffs) + eps)
    grad = torch.zeros_like(x)
    for ax, d in enumerate(diffs):
        pad_hi = [0, 0, 0, 0, 0, 0]
        pad_lo = [0, 0, 0, 0, 0, 0]
        # torch.nn.functional.pad order is (last dim first): build per-axis
        idx = (2 - ax) * 2
        pad_hi[idx + 1] = 1   # append one at the end of axis
        pad_lo[idx] = 1       # prepend one at the start of axis
        grad += torch.nn.functional.pad(d, pad_hi) - torch.nn.functional.pad(d, pad_lo)
    return norm, -grad / norm


def solve_fista(
    op: TorchZSOperator,
    *,
    alpha: float = 0.0,
    beta_quiet: float = 0.0,
    quiet_mask: np.ndarray | None = None,
    quiet_threshold: float = np.inf,
    beta_censor: float = 0.0,
    censor_reset: np.ndarray | None = None,
    censor_arm: np.ndarray | None = None,
    censor_end: int | None = None,
    censor_threshold: float = np.inf,
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
    """GPU FISTA with the same semantics as the numpy backend."""
    if L is None:
        L = op.lipschitz()
    L_total = L + 2.0 * beta_quiet if (beta_quiet > 0 and quiet_mask is not None) else L
    if lam_spectral > 0 and spectral_weight is not None:
        L_total = L_total + 2.0 * lam_spectral * float(np.max(spectral_weight))
    c_ref = None
    if beta_censor > 0 and censor_reset is not None:
        nt_b = op.block_shape[2]
        r_idx = op.to_tensor(
            np.asarray(censor_reset, dtype=np.int64), torch.long)
        a_idx = (op.to_tensor(np.asarray(censor_arm, dtype=np.int64),
                              torch.long)
                 if censor_arm is not None else r_idx)
        c_end = int(censor_end) if censor_end is not None else nt_b
        t_axis = torch.arange(nt_b, device=op.device)[None, None, :]
        c_ref = t_axis >= r_idx[:, :, None]
        c_armed = (t_axis >= a_idx[:, :, None]) & (t_axis < c_end)
        # power-iterate the worst-case (full-span cumulative row)
        # linearization for a sound step size
        xc = torch.randn(op.q_shape, dtype=op.dtype, device=op.device)
        xc /= torch.linalg.vector_norm(xc)
        lam_c = 0.0
        for _ in range(6):
            b = torch.where(c_ref, op.conv(xc),
                            torch.zeros((), dtype=op.dtype, device=op.device))
            row = b.sum(dim=2)
            yc = op.conv_adjoint(
                torch.where(c_ref, row[:, :, None].expand_as(b),
                            torch.zeros((), dtype=op.dtype,
                                        device=op.device)))
            lam_c = float(torch.linalg.vector_norm(yc))
            if lam_c <= 0:
                break
            xc = yc / lam_c
        L_total = L_total + 2.0 * beta_censor * max(lam_c, 1.0)
    step = 1.0 / (L_total * 1.05)

    sw = None
    if lam_spectral > 0 and spectral_weight is not None:
        sw = op.to_tensor(np.asarray(spectral_weight, dtype=np.float64))
    if isinstance(alpha, np.ndarray):
        alpha = op.to_tensor(alpha)
    qm = op.to_tensor(quiet_mask, torch.bool) if quiet_mask is not None else None
    sm = op.to_tensor(support_mask, torch.bool) if support_mask is not None else None

    if q0 is None:
        x = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
    else:
        x = torch.clamp(op.to_tensor(q0), min=0.0)
    if sm is not None:
        x = x * sm
    y = x.clone()
    t = 1.0
    for k in range(int(n_iter)):
        block_pred = op.conv(y)
        resid = op.sample(block_pred) - op.d
        grad = op.conv_adjoint(op.sample_adjoint(resid))
        if beta_quiet > 0 and qm is not None:
            viol = torch.where(
                qm, torch.clamp(block_pred - quiet_threshold, min=0.0),
                torch.zeros((), dtype=op.dtype, device=op.device),
            )
            if bool(viol.any()):
                grad += beta_quiet * op.conv_adjoint(viol)
        if c_ref is not None:
            zero = torch.zeros((), dtype=op.dtype, device=op.device)
            C = torch.cumsum(torch.where(c_ref, block_pred, zero), dim=2)
            neg_inf = torch.tensor(float("-inf"), dtype=op.dtype,
                                   device=op.device)
            Cm = torch.where(c_armed, C, neg_inf)
            peak, arg = Cm.max(dim=2)
            cviol = torch.where(
                torch.isfinite(peak),
                torch.clamp(peak - censor_threshold, min=0.0),
                zero,
            )
            if bool(cviol.any()):
                t_axis_c = torch.arange(
                    C.shape[2], device=op.device)[None, None, :]
                upto = t_axis_c <= arg[:, :, None]
                g_c = torch.where(c_ref & upto,
                                  cviol[:, :, None].expand_as(C), zero)
                grad += beta_censor * op.conv_adjoint(g_c)
        if lam_l2 > 0:
            grad += 2.0 * lam_l2 * y
        if lam_tv > 0:
            _, g_tv = _tv_gradient_torch(y)
            grad += lam_tv * g_tv
        if sw is not None:
            grad += 2.0 * lam_spectral * torch.fft.irfft(
                sw * torch.fft.rfft(y, dim=2), n=y.shape[2], dim=2,
            )
        x_new = torch.clamp(y - step * grad - step * alpha, min=0.0)
        if sm is not None:
            x_new = x_new * sm
        t_new = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if verbose and (k % 50 == 0 or k == n_iter - 1):
            loss = 0.5 * float((resid ** 2).sum())
            print(f"  fista[torch] iter {k:4d}  data-loss {loss:.4e}  "
                  f"nnz {float((x > 0).float().mean()):.3f}")
    return x.detach().cpu().numpy().astype(np.float64)


def solve_deghost_regress(
    op: TorchZSOperator,
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
    """GPU deghost/regress alternation (see numpy backend docstring)."""
    from .constrained_solver import exponential_alpha_field

    if L is None:
        L = op.lipschitz()
    x = q0
    support = base_support
    for r in range(int(n_rounds)):
        if verbose:
            print(f"  D/R[torch] round {r}: deghost alpha={alpha_deghost}")
        x = solve_fista(
            op, alpha=alpha_deghost, n_iter=n_iter_deghost, q0=x, L=L,
            support_mask=support, verbose=verbose, **fista_kwargs,
        )
        seed = np.asarray(x) > seed_cut
        alpha_field = exponential_alpha_field(seed, alpha_regress, decay_len)
        if verbose:
            print(f"  D/R[torch] round {r}: regress alpha_min={alpha_regress}, "
                  f"skeleton {100 * float(seed.mean()):.3f}%")
        x = solve_fista(
            op, alpha=alpha_field, n_iter=n_iter_regress, q0=x, L=L,
            support_mask=base_support, verbose=verbose, **fista_kwargs,
        )
    return x


def solve_fista_ladder(
    op: TorchZSOperator,
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
    """GPU strong-charge-first homotopy (see numpy backend docstring)."""
    from .constrained_solver import _dilate_mask, exponential_alpha_field

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
            print(f"  ladder[torch] stage {k}: alpha={alpha} ({kind})  "
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
