"""SmearedOperator: A' = A G with G symmetric, mass conserving, positive."""
import numpy as np
import torch

from unfoldlarpix.constrained_solver import LatchWindow
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.model.smeared_operator import SmearedOperator

B = 6
KX = KY = 3
KT = 3
NX, NY, NTQ = 6, 7, 10
BLOCK = (NX, NY, NTQ + KT - 1)
DEV, DT = "cpu", torch.float64


def setup(seed=0):
    rng = np.random.default_rng(seed)
    kernel = rng.normal(size=(KX, KY, KT))
    ws = [LatchWindow(px=int(rng.integers(0, NX)), py=int(rng.integers(0, NY)),
                      t_lo=float(rng.uniform(0, 40)),
                      t_hi=float(rng.uniform(40, 70)),
                      value=float(rng.normal())) for _ in range(20)]
    return kernel, ws, rng


def test_smoothing_conserves_mass_and_is_positive():
    kernel, ws, rng = setup()
    op = SmearedOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT,
                         sigma_time=0.02, sigma_pixel=0.5)
    u = torch.as_tensor(rng.random((NX, NY, NTQ)), dtype=DT)
    q = op.smear(u)
    assert abs(float(q.sum()) - float(u.sum())) < 1e-8 * float(u.sum())
    assert float(q.min()) > -1e-12          # non-negative kernel


def test_adjoint_dot():
    kernel, ws, rng = setup(1)
    op = SmearedOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT,
                         sigma_time=0.01, sigma_pixel=0.5)
    u = torch.as_tensor(rng.normal(size=(NX, NY, NTQ)), dtype=DT)
    r = torch.as_tensor(rng.normal(size=(len(ws),)), dtype=DT)
    lhs = float((op.forward(u) * r).sum())
    rhs = float((u * op.adjoint(r)).sum())
    assert abs(lhs - rhs) < 1e-9 * max(abs(lhs), 1.0)


def test_matches_stock_when_smoothing_is_off():
    """A very small sigma in the frequency domain = a very wide filter in
    frequency = no smoothing, so A' -> A."""
    kernel, ws, rng = setup(2)
    a = ZSOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT)
    b = SmearedOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT,
                        sigma_time=1e6, sigma_pixel=1e6)
    u = torch.as_tensor(rng.normal(size=(NX, NY, NTQ)), dtype=DT)
    assert torch.allclose(a.forward(u), b.forward(u), atol=1e-8)


def test_smoothing_damps_high_frequencies():
    kernel, ws, rng = setup(3)
    op = SmearedOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT,
                         sigma_time=0.02, sigma_pixel=0.5)
    alt = np.zeros((NX, NY, NTQ))
    alt[NX // 2, NY // 2, :] = [(-1.0) ** k for k in range(NTQ)]
    smooth = op.smear(torch.as_tensor(alt, dtype=DT))
    assert float(smooth.abs().max()) < 0.5 * float(np.abs(alt).max())


def test_l1_is_invariant_for_positive_fields():
    kernel, ws, rng = setup(4)
    op = SmearedOperator(kernel, BLOCK, ws, B, device=DEV, dtype=DT,
                         sigma_time=0.01, sigma_pixel=0.5)
    u = torch.as_tensor(rng.random((NX, NY, NTQ)), dtype=DT)
    assert abs(float(op.smear(u).sum()) - float(u.sum())) < 1e-8 * float(u.sum())
