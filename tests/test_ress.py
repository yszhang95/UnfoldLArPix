"""RESS: the port must reproduce Wire-Cell's own coordinate descent.

The first test compiles WCP's ``ress/src/*.cxx`` -- the original
LassoModel/ElasticNetModel -- and drives it through a C shim, so the
agreement checked here is against THEIR code and not against a second
reading of it.  It skips when the sources, g++ or Eigen are absent; the
rest of the file is self-contained.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from unfoldlarpix.solve.ress import (GramSystem, compare, pgd_fit, ress_fit)

WCP_SRC = Path("/home/yousen/Documents/PIONEER/PIONEER-framework/ress")
EIGEN = Path("/usr/include/eigen3")

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


@pytest.fixture(scope="module")
def wcp(tmp_path_factory):
    srcs = [WCP_SRC / "src" / f"{n}.cxx"
            for n in ("LinearModel", "ElasticNetModel", "LassoModel")]
    if not all(p.exists() for p in srcs) or not EIGEN.exists():
        pytest.skip("Wire-Cell/WCP RESS sources or Eigen not available")
    d = tmp_path_factory.mktemp("wcpress")
    (d / "shim.cxx").write_text(SHIM)
    so = d / "libwcpress.so"
    try:
        subprocess.run(
            ["g++", "-O2", "-fPIC", "-shared", "-o", str(so), str(d / "shim.cxx"),
             *[str(p) for p in srcs], f"-I{WCP_SRC / 'inc'}", f"-I{EIGEN}"],
            check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"cannot build the WCP reference: {exc}")
    lib = ctypes.CDLL(str(so))
    dp = ctypes.POINTER(ctypes.c_double)
    lib.wcp_lasso.argtypes = [dp, ctypes.c_int, ctypes.c_int, dp,
                              ctypes.c_double, ctypes.c_int, ctypes.c_double,
                              ctypes.c_int, dp]
    lib.wcp_elnet.argtypes = [dp, ctypes.c_int, ctypes.c_int, dp,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int,
                              ctypes.c_double, ctypes.c_int, dp]
    return lib


def _sys(seed=7, ncell=40, nwire=28, nzero=30):
    """The system WCP's own ``test_ress.cxx`` uses: a 0/1 geometry matrix,
    fewer rows than unknowns, most cells empty."""
    rng = np.random.default_rng(seed)
    c = rng.random(ncell) * 100 + 150
    c[rng.choice(ncell, nzero, replace=False)] = 0.0
    X = np.floor(rng.random((nwire, ncell)) * 2 + 0.5)
    return X, X @ c, c


def _call(lib, fn, X, y, out_n, *args):
    X = np.ascontiguousarray(X, float); y = np.ascontiguousarray(y, float)
    out = np.zeros(out_n)
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # noqa
    fn(p(X), X.shape[0], X.shape[1], p(y), *args, p(out))
    return out


@pytest.mark.parametrize("lam,tol", [(0.0, 1e-3), (0.01, 1e-3),
                                     (0.5, 1e-4), (5.0, 1e-4)])
def test_lasso_matches_wirecell(wcp, lam, tol):
    X, y, _ = _sys()
    ref = _call(wcp, wcp.wcp_lasso, X, y, X.shape[1], lam, 100000, tol, 1)
    mine, _ = ress_fit(GramSystem.from_matrix(X, y), lam=lam, tol=tol,
                       penalty="ress")
    assert np.abs(ref - mine).max() < 1e-9


@pytest.mark.parametrize("lam,alpha", [(0.1, 0.95), (1.0, 0.5), (2.0, 0.2)])
def test_elasticnet_matches_wirecell(wcp, lam, alpha):
    X, y, _ = _sys()
    ref = _call(wcp, wcp.wcp_elnet, X, y, X.shape[1], lam, alpha, 100000,
                1e-4, 1)
    mine, _ = ress_fit(GramSystem.from_matrix(X, y), lam=lam, alpha=alpha,
                       tol=1e-4, penalty="ress")
    assert np.abs(ref - mine).max() < 1e-9


def test_two_sided_matches_wirecell(wcp):
    """``non_negtive = false`` takes the other branch of the soft threshold."""
    X, y, _ = _sys()
    ref = _call(wcp, wcp.wcp_lasso, X, y, X.shape[1], 0.5, 100000, 1e-4, 0)
    mine, _ = ress_fit(GramSystem.from_matrix(X, y), lam=0.5, tol=1e-4,
                       non_negative=False)
    assert np.abs(ref - mine).max() < 1e-9


def test_gram_objective_is_the_least_squares_objective():
    X, y, _ = _sys()
    sysm = GramSystem.from_matrix(X, y)
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = np.abs(rng.normal(size=X.shape[1]))
        assert sysm.data_term(b) == pytest.approx(
            0.5 * float(((X @ b - y) ** 2).sum()), rel=1e-10)


def test_penalty_conventions_differ_by_the_column_norm():
    """Wire-Cell's lambda is per-column; ``CoordProx``'s alpha is not."""
    X, y, _ = _sys()
    sysm = GramSystem.from_matrix(X, y)
    g_ress = sysm.gamma(0.3, penalty="ress")
    g_unif = sysm.gamma(0.3, penalty="uniform")
    assert np.allclose(g_ress, g_unif * sysm.col_norm)
    assert not np.allclose(g_ress, g_unif)


def test_nnls_coordinate_descent_and_projected_gradient_agree():
    """Both solvers, one convex problem: the same point and a clean KKT."""
    nnls = pytest.importorskip("scipy.optimize").nnls
    X, y, _ = _sys()
    sysm = GramSystem.from_matrix(X, y)
    ref, _ = nnls(X, y)
    cd, icd = ress_fit(sysm, lam=0.0, tol=1e-8, max_iter=200000)
    pg, _ = pgd_fit(sysm, n_iter=100000)
    assert icd["converged"]
    assert np.abs(cd - ref).max() < 1e-5
    assert np.abs(pg - ref).max() < 1e-5
    c = compare(sysm, cd, pg)
    assert c["l1_diff_rel"] < 1e-6
    assert c["support_jaccard"] == 1.0
    assert sysm.kkt(cd)["kkt_rel"] < 1e-8
    assert sysm.kkt(pg)["kkt_rel"] < 1e-8


def test_wirecell_default_tolerance_stops_short_of_the_minimum():
    """WCP's TOL = 1e-3 is a STEP size, not an optimality gap.

    The stopping test is ``||dbeta||^2 < TOL^2 * n``, so on a flat problem
    the fit ends measurably above the minimum -- which is the only reason
    the shipped default disagrees with a converged gradient solve.
    """
    X, y, _ = _sys()
    sysm = GramSystem.from_matrix(X, y)
    loose, _ = ress_fit(sysm, lam=0.0, tol=1e-3)
    tight, _ = ress_fit(sysm, lam=0.0, tol=1e-8, max_iter=200000)
    f_loose = sysm.objective(loose)["objective"]
    f_tight = sysm.objective(tight)["objective"]
    assert f_loose >= f_tight
    assert sysm.kkt(loose)["kkt_rel"] > sysm.kkt(tight)["kkt_rel"]


# ---------------------------------------------------------------------------
# the Cholesky bridge: Wire-Cell's own code, on a Gram it never saw an X for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_gram_bridge_reproduces_the_direct_solve(wcp, tmp_path, lam):
    """``X := chol(G)``, ``y := R^-T b`` gives their Fit the same problem.

    ``LassoModel::Fit`` uses ``X`` only through ``X^T X`` and ``X^T y``, so
    any factor with the same Gram is the same fit.  That is what lets the
    compiled solver run on an operator whose real ``X`` is 146 GB.
    """
    from unfoldlarpix.solve import ress_wirecell
    if not ress_wirecell.sources_available():
        pytest.skip("WCP RESS sources not available")
    X, y, _ = _sys()
    # a rank-deficient Gram has no Cholesky; pad to full column rank the
    # way the real ROI is (more rows than columns, independent columns)
    rng = np.random.default_rng(3)
    X = np.vstack([X, rng.normal(size=(60, X.shape[1]))])
    y = np.concatenate([y, rng.normal(size=60)])
    direct = _call(wcp, wcp.wcp_lasso, X, y, X.shape[1], lam, 100000, 1e-6, 1)
    sysm = GramSystem.from_matrix(X, y)
    bridged, info = ress_wirecell.solve(
        sysm, lam=lam, tol=1e-6, lib=tmp_path / "libwcpress.so")
    assert info["cholesky_ridge_rel"] == 0.0
    assert info["gram_factor_resid"] < 1e-8 * float(np.diag(sysm.G).max())
    assert np.abs(direct - bridged).max() < 1e-8
    # and the port, on the same system, reaches the same point
    port, _ = ress_fit(sysm, lam=lam, tol=1e-6, penalty="ress")
    assert np.abs(port - bridged).max() < 1e-8


# ---------------------------------------------------------------------------
# the repository's own solver, on the same system
# ---------------------------------------------------------------------------
def _pd_system(seed=3):
    """A system whose Gram is positive definite (the ROI Grams are)."""
    X, y, _ = _sys()
    rng = np.random.default_rng(seed)
    X = np.vstack([X, rng.normal(size=(60, X.shape[1]))])
    y = np.concatenate([y, rng.normal(size=60)])
    return X, y


def test_gram_operator_has_the_right_normal_equations():
    """``A^T A = G`` and ``A^T d = b``: the engine sees OUR problem."""
    import torch

    from unfoldlarpix.solve.ress import GramOperator
    X, y = _pd_system()
    sysm = GramSystem.from_matrix(X, y)
    op = GramOperator(sysm, device="cpu", dtype=torch.float64)
    assert op.gram_factor_resid < 1e-10 * float(np.diag(sysm.G).max())
    n = sysm.n
    AtA = np.empty((n, n))
    for j in range(n):
        e = torch.zeros(op.q_shape, dtype=torch.float64)
        e[j, 0, 0] = 1.0
        AtA[:, j] = op.adjoint(op.forward(e)).reshape(-1).numpy()
    assert np.abs(AtA - sysm.G).max() < 1e-9
    assert np.abs(op.adjoint(op.d).reshape(-1).numpy() - sysm.b).max() < 1e-9


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_repo_engine_lands_where_coordinate_descent_lands(lam):
    """``Fista`` + ``DataFidelity`` + ``CoordProx``, unmodified, vs RESS."""
    from unfoldlarpix.solve.ress import pgd_engine_fit
    X, y = _pd_system()
    sysm = GramSystem.from_matrix(X, y)
    gamma = sysm.gamma(lam, penalty="ress")
    cd, icd = ress_fit(sysm, lam=lam, tol=1e-8, max_iter=200000,
                       penalty="ress")
    eng, info = pgd_engine_fit(sysm, gamma=gamma, n_iter=50000, device="cpu")
    assert icd["converged"]
    assert info["solver"] == "pgd_repo_engine"
    c = compare(sysm, cd, eng)
    assert c["l1_diff_rel"] < 1e-6
    assert c["support_jaccard"] == 1.0
    assert sysm.objective(eng, gamma)["objective"] == pytest.approx(
        sysm.objective(cd, gamma)["objective"], rel=1e-10)
