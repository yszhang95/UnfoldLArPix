"""Run Wire-Cell's OWN compiled RESS on this operator's Gram system.

``LassoModel::Fit`` takes a dense ``X`` (rows x columns) and immediately
reduces it to ``ydX = X^T y`` and ``XdX = X^T X``; after that the
coordinate loop never touches ``X`` again.  So any ``X`` with the same
Gram gives their code the same problem, bit for bit.  This module hands
it one:

    G = R^T R   (Cholesky)      X := R          n x n
    y' := R^-T b               =>  X^T X = G,  X^T y' = b

which is exactly the system :class:`~unfoldlarpix.solve.ress.GramSystem`
carries -- the objective differs from ``1/2||A q - d||^2`` only by the
constant ``const - 1/2||y'||^2``, and a constant cannot move an argmin.
The real ``X`` here is 1.2565 M x 14560 (146 GB); ``R`` is n x n.

**This does not scale, and that is a finding, not a defect.**
``LassoModel::Fit`` builds ``XdX`` with a literal double loop of Eigen
column dots, ``O(n^2 * rows)`` with ``n^2`` sparse insertions -- about a
second at n = 1080 and hours at n = 14560.  Wire-Cell's ROIs are
10^2-10^3 unknowns; that is the size their solver is written for.  Use
this backend to PIN the port on a small ROI, and the port
(:func:`~unfoldlarpix.solve.ress.ress_fit`, which starts from the Gram)
for anything larger.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

WCP_SRC = Path(os.environ.get(
    "WCP_RESS_SRC", "/home/yousen/Documents/PIONEER/PIONEER-framework/ress"))
EIGEN = Path(os.environ.get("EIGEN_INCLUDE", "/usr/include/eigen3"))
_UNITS = ("LinearModel", "ElasticNetModel", "LassoModel")

SHIM = r'''
#include "WCPRess/LassoModel.h"
#include "WCPRess/ElasticNetModel.h"
#include <Eigen/Dense>
extern "C" {
static void fill(Eigen::MatrixXd& M, const double* X, int nr, int nc) {
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j) M(i, j) = X[i * nc + j];
}
void wcp_lasso(const double* X, int nr, int nc, const double* y, double lam,
               int max_iter, double tol, int nn, double* out) {
    Eigen::MatrixXd M(nr, nc); fill(M, X, nr, nc);
    Eigen::VectorXd v(nr); for (int i = 0; i < nr; ++i) v(i) = y[i];
    WCP::LassoModel m(lam, max_iter, tol, nn != 0);
    m.SetData(M, v); m.Fit();
    Eigen::VectorXd b = m.Getbeta();
    for (int j = 0; j < nc; ++j) out[j] = b(j);
}
void wcp_elnet(const double* X, int nr, int nc, const double* y, double lam,
               double alpha, int max_iter, double tol, int nn, double* out) {
    Eigen::MatrixXd M(nr, nc); fill(M, X, nr, nc);
    Eigen::VectorXd v(nr); for (int i = 0; i < nr; ++i) v(i) = y[i];
    WCP::ElasticNetModel m(lam, alpha, max_iter, tol, nn != 0);
    m.SetData(M, v); m.Fit();
    Eigen::VectorXd b = m.Getbeta();
    for (int j = 0; j < nc; ++j) out[j] = b(j);
}
}
'''


def sources_available() -> bool:
    return (all((WCP_SRC / "src" / f"{u}.cxx").exists() for u in _UNITS)
            and EIGEN.exists())


def build(dest: str | Path) -> Path:
    """Compile WCP's ``ress/src/*.cxx`` plus the shim into ``dest``.

    Rebuilds only when the library is older than any source, so a job that
    uses the backend repeatedly pays the compile once.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    srcs = [WCP_SRC / "src" / f"{u}.cxx" for u in _UNITS]
    missing = [str(p) for p in srcs if not p.exists()]
    if missing or not EIGEN.exists():
        raise FileNotFoundError(
            f"Wire-Cell/WCP RESS sources not available: {missing or EIGEN}; "
            "set WCP_RESS_SRC / EIGEN_INCLUDE")
    shim = dest.parent / "wcp_ress_shim.cxx"
    if not shim.exists() or shim.read_text() != SHIM:
        shim.write_text(SHIM)
    newest = max(p.stat().st_mtime for p in [shim, *srcs])
    if dest.exists() and dest.stat().st_mtime >= newest:
        return dest
    subprocess.run(
        ["g++", "-O2", "-fPIC", "-shared", "-o", str(dest), str(shim),
         *[str(p) for p in srcs], f"-I{WCP_SRC / 'inc'}", f"-I{EIGEN}"],
        check=True, capture_output=True)
    return dest


def _load(dest: Path):
    lib = ctypes.CDLL(str(dest))
    dp = ctypes.POINTER(ctypes.c_double)
    lib.wcp_lasso.argtypes = [dp, ctypes.c_int, ctypes.c_int, dp,
                              ctypes.c_double, ctypes.c_int, ctypes.c_double,
                              ctypes.c_int, dp]
    lib.wcp_elnet.argtypes = [dp, ctypes.c_int, ctypes.c_int, dp,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int,
                              ctypes.c_double, ctypes.c_int, dp]
    return lib


def gram_factor(G: np.ndarray, jitter: float = 0.0
                ) -> tuple[np.ndarray, float]:
    """Cholesky ``R`` with ``R^T R = G``, retrying with a relative jitter.

    A rank-deficient Gram has no Cholesky.  The retry adds
    ``jitter * max(diag G)`` to the diagonal, which is a ridge and
    therefore a DIFFERENT problem -- the amount actually used is returned
    so the caller can record it instead of hiding it.
    """
    G = np.ascontiguousarray(G, dtype=np.float64)
    scale = float(np.diag(G).max())
    eps = 0.0
    for k in range(6):
        try:
            R = np.linalg.cholesky(
                G + (eps * scale) * np.eye(G.shape[0]) if eps else G).T
            return np.ascontiguousarray(R), eps
        except np.linalg.LinAlgError:
            eps = 1e-12 if eps == 0.0 else eps * 100
    raise np.linalg.LinAlgError(
        "Gram is not positive definite even with a 1e-2 relative ridge")


def solve(sysm, lam: float = 0.0, alpha: float = 1.0, model: str = "lasso",
          max_iter: int = 100000, tol: float = 1e-3,
          non_negative: bool = True, lib: str | Path | None = None,
          jitter: float = 0.0) -> tuple[np.ndarray, dict]:
    """Solve ``sysm`` with Wire-Cell's compiled coordinate descent.

    ``lam`` is Wire-Cell's own ``lambda``: its effective L1 weight is
    ``lambda * ||X_j||^2``, the ``penalty="ress"`` convention.
    """
    if lib is None:
        raise ValueError("lib: path for the compiled WCP reference is required")
    so = build(lib)
    handle = _load(so)
    R, eps = gram_factor(sysm.G, jitter)
    # y' = R^-T b, so that X^T y' = R^T R^-T b = b
    yp = np.ascontiguousarray(
        np.linalg.solve(R.T, np.ascontiguousarray(sysm.b, dtype=np.float64)))
    n = sysm.n
    out = np.zeros(n, dtype=np.float64)
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # noqa
    if model == "lasso":
        handle.wcp_lasso(p(R), n, n, p(yp), lam, int(max_iter), float(tol),
                         int(non_negative), p(out))
    elif model == "elnet":
        handle.wcp_elnet(p(R), n, n, p(yp), lam, alpha, int(max_iter),
                         float(tol), int(non_negative), p(out))
    else:
        raise ValueError(f"model {model!r} (want 'lasso' or 'elnet')")
    return out, {"solver": f"wirecell_{model}_cxx", "library": str(so),
                 "lambda": lam, "alpha": alpha, "tol": tol,
                 "max_iter": int(max_iter), "non_negative": non_negative,
                 "cholesky_ridge_rel": eps,
                 "gram_factor_resid": float(
                     np.abs(R.T @ R - sysm.G).max()),
                 "bridge": "X := chol(G), y := R^-T b"}
