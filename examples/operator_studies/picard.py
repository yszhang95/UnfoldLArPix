"""Do the small-singular-value modes really carry less signal?

Shrinking small-sigma modes (ridge, truncated SVD, and implicitly any
sparsity prior) is justified only under the discrete Picard condition:
the truth's coefficients in those modes must fall at least as fast as
sigma_k, so that what is thrown away is mostly error. If instead the
truth has as much power there as anywhere, the shrinkage is a pure bias.

Measured per mode, on the support-restricted operator:

  a_k        = u_k . (A q_true)          signal in data space
  t_k        = a_k / sigma_k             = v_k . q_true, the truth's own
                                           coefficient in charge space
  eps_k      = u_k . e                   operator model error
  nu_k       = u_k . n                   readout error
  snr_k      = |a_k| / |eps_k + nu_k|
  f_ideal    = a_k^2 / (a_k^2 + (eps+nu)_k^2)     the Wiener filter that
                                           minimises the mode's error
  f_ridge    = sigma_k^2 / (sigma_k^2 + lam)

Reported in sigma deciles. If |t_k| is flat or rising towards small
sigma while f_ideal stays high, then "small sigma = little signal" is
false and any sigma-based shrinkage biases the result.
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

from channel_coupling import gram  # noqa: E402
from filter_budget import pieces  # noqa: E402


def one(tag, lam=0.1):
    op, supp, B, d, Aqt, e = pieces(tag)
    n = d - (Aqt - e)
    G = gram(op, op.to_tensor(supp.astype(np.float64)))
    w, U = np.linalg.eigh(G)
    w = np.clip(w[::-1], 0, None)
    U = U[:, ::-1]
    keep = w > TOL * max(w.max(), 1e-30)
    w, U = w[keep], U[:, keep]
    sig = np.sqrt(w)
    a = U.T @ Aqt
    eps = U.T @ e
    nu = U.T @ n
    t = a / np.clip(sig, 1e-30, None)
    err = np.abs(eps + nu)
    snr = np.abs(a) / np.clip(err, 1e-12, None)
    f_ideal = a ** 2 / (a ** 2 + (eps + nu) ** 2 + 1e-30)
    f_ridge = w / (w + lam)

    order = np.argsort(-sig)
    k = len(sig)
    print(f"\n{tag}: {k} modes, sigma {sig[0]:.3f} .. {sig[-1]:.4f}, "
          f"cond {sig[0]/sig[-1]:.2f}", flush=True)
    print("%-9s %8s %10s %10s %10s %8s %9s %9s" %
          ("decile", "sigma", "|a| sig", "|t|=|a|/s", "|eps+nu|", "SNR",
           "f_ideal", f"f_ridge({lam:g})"))
    rows = []
    for i in range(10):
        idx = order[i * k // 10:(i + 1) * k // 10]
        if not len(idx):
            continue
        r = {"decile": i + 1, "sigma": float(np.median(sig[idx])),
             "abs_a": float(np.median(np.abs(a[idx]))),
             "abs_t": float(np.median(np.abs(t[idx]))),
             "abs_err": float(np.median(err[idx])),
             "snr": float(np.median(snr[idx])),
             "f_ideal": float(np.median(f_ideal[idx])),
             "f_ridge": float(np.median(f_ridge[idx])),
             "signal_power_frac": float((a[idx] ** 2).sum() / (a ** 2).sum()),
             "truth_power_frac": float((t[idx] ** 2).sum() / (t ** 2).sum()),
             "err_power_frac": float(((eps[idx] + nu[idx]) ** 2).sum()
                                     / ((eps + nu) ** 2).sum())}
        rows.append(r)
        print("%-9d %8.3f %10.2f %10.2f %10.2f %8.2f %9.3f %9.3f" %
              (r["decile"], r["sigma"], r["abs_a"], r["abs_t"],
               r["abs_err"], r["snr"], r["f_ideal"], r["f_ridge"]),
              flush=True)
    print("  power fractions (top decile -> bottom):", flush=True)
    print("    truth  " + " ".join(f"{r['truth_power_frac']:.3f}"
                                   for r in rows), flush=True)
    print("    signal " + " ".join(f"{r['signal_power_frac']:.3f}"
                                   for r in rows), flush=True)
    print("    error  " + " ".join(f"{r['err_power_frac']:.3f}"
                                   for r in rows), flush=True)
    out = {"tag": tag, "modes": int(k), "cond": float(sig[0] / sig[-1]),
           "lambda": lam, "deciles": rows}
    del op, G, U
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["mu_a00_nb1", "mu_a50_nb1"]
    res = [one(t) for t in tags]
    json.dump(res, open(f"{AO}/channel_coupling/picard.json", "w"), indent=1)
    print("\n-> channel_coupling/picard.json", flush=True)
