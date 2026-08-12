"""Correctness of the phase-exact operator (new, parallel path)."""
import numpy as np
import pytest
import torch

from unfoldlarpix.constrained_solver import LatchWindow
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.model.phase_operator import ZSOperatorPhase

B = 6
KX = KY = 5
KT = 4
NX, NY, NTQ = 7, 8, 9
BLOCK = (NX, NY, NTQ + KT - 1)
DEV, DT = "cpu", torch.float64


def make_kernels(seed=0):
    rng = np.random.default_rng(seed)
    fine = rng.normal(size=(KX, KY, KT * B))
    coarse = fine.reshape(KX, KY, KT, B).sum(-1)
    return coarse, fine


def rand_windows(rng, n=30, integer_edges=False):
    ws = []
    for _ in range(n):
        px = int(rng.integers(0, NX))
        py = int(rng.integers(0, NY))
        lo = float(rng.uniform(0, BLOCK[2] * B - 2))
        hi = float(lo + rng.uniform(1, 3 * B))
        if integer_edges:
            lo = float(int(lo // B) * B)
            hi = float(min(int(hi // B + 1) * B, BLOCK[2] * B))
        ws.append(LatchWindow(px=px, py=py, t_lo=lo, t_hi=hi,
                              value=float(rng.normal())))
    return ws


def fine_reference(fine, q, windows):
    """Brute-force: fine model current, exact window integrals."""
    T = BLOCK[2] * B
    cur = np.zeros((NX, NY, T + fine.shape[2]))
    cx, cy = (KX - 1) // 2, (KY - 1) // 2
    for x in range(NX):
        for y in range(NY):
            for t in range(NTQ):
                qv = q[x, y, t]
                if qv == 0:
                    continue
                for i in range(KX):
                    for j in range(KY):
                        # conv orientation: block[xx] = sum K[xx-x'+cx] q[x']
                        xx, yy = x - cx + i, y - cy + j
                        if 0 <= xx < NX and 0 <= yy < NY:
                            cur[xx, yy, t * B:t * B + fine.shape[2]] += \
                                qv * fine[i, j]
    cs = np.concatenate([np.zeros((NX, NY, 1)), np.cumsum(cur, -1)], -1)
    out = []
    for w in windows:
        lo, hi = int(round(w.t_lo)), int(round(min(w.t_hi, T)))
        out.append(cs[w.px, w.py, hi] - cs[w.px, w.py, lo])
    return np.array(out)


def test_integer_edges_match_stock():
    rng = np.random.default_rng(1)
    coarse, fine = make_kernels()
    ws = rand_windows(rng, integer_edges=True)
    q = torch.as_tensor(rng.normal(size=(NX, NY, NTQ)), dtype=DT)
    a = ZSOperator(coarse, BLOCK, ws, B, device=DEV, dtype=DT)
    b = ZSOperatorPhase(coarse, fine, BLOCK, ws, B, device=DEV, dtype=DT)
    assert torch.allclose(a.forward(q), b.forward(q), atol=1e-10)


def test_fractional_edges_match_fine_reference():
    rng = np.random.default_rng(2)
    coarse, fine = make_kernels()
    ws = rand_windows(rng, integer_edges=False)
    # integer fine-tick edges (the physical case)
    ws = [LatchWindow(w.px, w.py, float(int(w.t_lo)), float(int(w.t_hi)),
                      w.value) for w in ws]
    qn = rng.normal(size=(NX, NY, NTQ))
    q = torch.as_tensor(qn, dtype=DT)
    b = ZSOperatorPhase(coarse, fine, BLOCK, ws, B, device=DEV, dtype=DT)
    got = b.forward(q).numpy()
    ref = fine_reference(fine, qn, ws)
    assert np.allclose(got, ref, atol=1e-8)


def test_adjoint_dot():
    rng = np.random.default_rng(3)
    coarse, fine = make_kernels()
    ws = rand_windows(rng)
    b = ZSOperatorPhase(coarse, fine, BLOCK, ws, B, device=DEV, dtype=DT)
    q = torch.as_tensor(rng.normal(size=(NX, NY, NTQ)), dtype=DT)
    r = torch.as_tensor(rng.normal(size=(len(ws),)), dtype=DT)
    lhs = float((b.forward(q) * r).sum())
    rhs = float((q * b.adjoint(r)).sum())
    assert abs(lhs - rhs) < 1e-8 * max(abs(lhs), 1.0)


def test_row_weights_consistent():
    rng = np.random.default_rng(4)
    coarse, fine = make_kernels()
    ws = rand_windows(rng, integer_edges=True)
    rw = rng.uniform(0.5, 2.0, size=len(ws))
    q = torch.as_tensor(rng.normal(size=(NX, NY, NTQ)), dtype=DT)
    a = ZSOperator(coarse, BLOCK, ws, B, device=DEV, dtype=DT, row_weights=rw)
    b = ZSOperatorPhase(coarse, fine, BLOCK, ws, B, device=DEV, dtype=DT,
                        row_weights=rw)
    assert torch.allclose(a.forward(q), b.forward(q), atol=1e-10)
    assert torch.allclose(a.d, b.d)


def test_binning_mismatch_raises():
    coarse, fine = make_kernels()
    with pytest.raises(ValueError):
        ZSOperatorPhase(coarse + 0.1, fine, BLOCK, [], B,
                        device=DEV, dtype=DT)
