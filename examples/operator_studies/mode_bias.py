"""Condition number and norms -> bias of the solution, mode by mode.

With A = sum_k sigma_k u_k v_k^T (restricted to the support) and
d = A q_true + e + n (e = operator model error, n = readout error), the
ridge solution q_hat = (A^T A + lam I)^-1 A^T d satisfies

  q_hat - q_true = sum_k [ -lam/(sigma_k^2+lam) (v_k.q_true)
                           + sigma_k/(sigma_k^2+lam) u_k.(e+n) ] v_k

so every mode contributes a REGULARISATION BIAS (shrinks the truth,
worst where sigma_k is small) and an ERROR AMPLIFICATION (~1/sigma_k for
sigma_k^2 >> lam, bounded by the condition number). This script evaluates
both, per mode and summed, using

  v_k . q_true = (u_k . A q_true) / sigma_k
  charge-scale error = sum_k (bias_k + amp_k) * (1 . v_k)

the last line being exactly the integral bias, i.e. the error projected
on the all-ones direction. Requires a waveform sample for the exact e.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AO = f"{ROOT}/examples/analysis_output"
TOL = 1e-12
LAMS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]

from channel_coupling import gram  # noqa: E402
from filter_budget import pieces  # noqa: E402


def one(tag):
    op, supp, B, d, Aqt, e = pieces(tag)
    n = d - (Aqt - e)                 # readout error: d - d_exact
    mask = op.to_tensor(supp.astype(np.float64))
    G = gram(op, mask)                # A P A^T
    w, U = np.linalg.eigh(G)
    w = np.clip(w[::-1], 0, None)
    U = U[:, ::-1]
    keep = w > TOL * max(w.max(), 1e-30)
    w, U = w[keep], U[:, keep]
    sig = np.sqrt(w)

    # projections
    a = U.T @ Aqt                     # sigma_k * (v_k . q_true)
    eps = U.T @ e
    nu = U.T @ n
    vq = a / np.clip(sig, 1e-30, None)          # v_k . q_true
    # 1 . v_k  = (1 . A^T u_k)/sigma_k = (c . u_k)/sigma_k with c = A^T 1
    c = op.measurement_gain().cpu().numpy()
    cm = (c * supp).ravel()
    ones_proj = np.array([float(cm.sum() * 0)] * len(sig))  # placeholder
    # exact: 1.v_k = (A^T u_k restricted).sum() / sigma_k
    e_row = torch.zeros(op.n_data, dtype=op.dtype, device=op.device)
    for k in range(len(sig)):
        e_row.copy_(torch.as_tensor(U[:, k], dtype=op.dtype,
                                    device=op.device))
        v = (op.adjoint(e_row) * mask).sum().item()
        ones_proj[k] = v / max(sig[k], 1e-30)

    print(f"\n{tag}: {op.n_data} rows, modes {len(sig)}, "
          f"sigma {sig[0]:.3f} .. {sig[-1]:.4g}, cond {sig[0]/sig[-1]:.3g}",
          flush=True)
    print(f"  ||A q_true|| {np.linalg.norm(Aqt):.1f}, ||e|| "
          f"{np.linalg.norm(e):.1f}, ||n|| {np.linalg.norm(n):.1f} ke; "
          f"sum q_true {float((np.load if False else lambda x: x)(0) or 0):.0f}"
          if False else
          f"  ||A q_true|| {np.linalg.norm(Aqt):.1f}, ||e|| "
          f"{np.linalg.norm(e):.1f}, ||n|| {np.linalg.norm(n):.1f} ke",
          flush=True)
    print("%9s %11s %11s %11s %11s %11s" %
          ("lambda", "|bias|", "|amp|", "|total|", "dQ_bias", "dQ_amp"))
    rows = []
    for lam in LAMS:
        fb = -lam / (w + lam)                      # per mode, on v_k.q_true
        fa = sig / (w + lam)                       # per mode, on eps+nu
        bias = fb * vq
        amp = fa * (eps + nu)
        dq_b = float((bias * ones_proj).sum())
        dq_a = float((amp * ones_proj).sum())
        rows.append({"lambda": lam,
                     "norm_bias": float(np.linalg.norm(bias)),
                     "norm_amp": float(np.linalg.norm(amp)),
                     "norm_total": float(np.linalg.norm(bias + amp)),
                     "charge_bias": dq_b, "charge_amp": dq_a,
                     "charge_total": dq_b + dq_a})
        print("%9.4g %11.1f %11.1f %11.1f %11.1f %11.1f" %
              (lam, rows[-1]["norm_bias"], rows[-1]["norm_amp"],
               rows[-1]["norm_total"], dq_b, dq_a), flush=True)
    out = {"tag": tag, "rows": int(op.n_data), "modes": int(len(sig)),
           "sigma_max": float(sig[0]), "sigma_min": float(sig[-1]),
           "cond": float(sig[0] / sig[-1]),
           "norm_Aqtruth": float(np.linalg.norm(Aqt)),
           "norm_e": float(np.linalg.norm(e)),
           "norm_n": float(np.linalg.norm(n)),
           "sigma_deciles": [float(x) for x in
                             np.percentile(sig, [0, 10, 50, 90, 100])],
           "lambda_scan": rows}
    del op, G, U
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1"]
    res = [one(t) for t in tags]
    json.dump(res, open(f"{AO}/channel_coupling/mode_bias.json", "w"),
              indent=1)
    print("\n-> channel_coupling/mode_bias.json", flush=True)
