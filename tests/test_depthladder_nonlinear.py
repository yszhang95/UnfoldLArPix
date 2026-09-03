"""Tests for the nonlinear arms of the isoline depth ladder.

CPU only, no response file, no GPU: what is testable without a solve is the
DRIVER -- that it reuses the shipped solver rather than a copy of it, that the
ring ledger is its definition, and that the fit reads the record schema the
driver writes, extends the estimate list and reproduces a known lambda.
"""
from __future__ import annotations

import json

import numpy as np

from unfoldlarpix.algs import depthladder_nonlinear_algs as dnl
from unfoldlarpix.algs import finebasis_nonlinear_algs as fnl
from unfoldlarpix.algs.depthladder_algs import DepthLadderFit, ring_masks
from unfoldlarpix.fwk.component import ALGORITHMS
from unfoldlarpix.fwk.store import EventStore

DEPTHS = [4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5]
V = 0.159645
VARIANTS = ("none", "pos_a0", "pos_l1_0.01")


def test_registration():
    for name in ("DepthLadderNonlinearEvent", "DepthLadderNonlinearFit",
                 "DepthLadderNonlinearFigures"):
        assert name in ALGORITHMS


def test_the_solver_is_the_shipped_one_not_a_copy():
    """The driver imports the FISTA arm, the operator wrapper and the support
    builder of ``finebasis_nonlinear_algs``; it defines none of them."""
    assert dnl.solve_fine_arm is fnl.solve_fine_arm
    assert dnl.FineZSOperator is fnl.FineZSOperator
    assert dnl.upsample_support is fnl.upsample_support
    assert dnl.record_row_mask is fnl.record_row_mask
    src = open(dnl.__file__).read()
    for forbidden in ("Fista(", "CoordProx(", "DataFidelity("):
        assert forbidden not in src


def test_ring_ledger_is_its_definition():
    """Per ring: total, positive part, negative part, and each per pad; the
    five rings partition the pads, so the five totals sum to the whole."""
    rng = np.random.default_rng(3)
    cheb = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 9])
    x = rng.normal(size=(len(cheb), 7))
    m = ring_masks(cheb)
    led = dnl.ring_ledger(x, m, sum_effq=100.0)
    assert set(led) == set(m)
    assert sum(led[k]["n_pads"] for k in led) == len(cheb)
    assert abs(sum(led[k]["sum_ke"] for k in led) - x.sum()) < 1e-12
    for k, msk in m.items():
        blk = x[msk]
        n = max(int(msk.sum()), 1)
        assert abs(led[k]["sum_ke"] - blk.sum()) < 1e-12
        assert abs(led[k]["sum_pos_ke"] - blk[blk > 0].sum()) < 1e-12
        assert abs(led[k]["sum_neg_ke"] - blk[blk < 0].sum()) < 1e-12
        assert abs(led[k]["sum_pos_ke"] + led[k]["sum_neg_ke"]
                   - led[k]["sum_ke"]) < 1e-12
        assert abs(led[k]["per_pad_ke"] - blk.sum() / n) < 1e-12
        assert abs(led[k]["over_sum_effq"] - blk.sum() / 100.0) < 1e-12


def _event_doc(depth: float, tau: float, mu_per_ms: dict) -> dict:
    """A per-event JSON with exactly the schema the driver writes.

    The truth follows ``exp(-t/tau)``.  Each variant's total is the truth
    times a LIFETIME-INDEPENDENT acceptance ``exp(-mu_v t)``, so the fitted
    ``lambda`` is ``1/tau + mu_v`` and the ratio to the truth is the same at
    both lifetimes -- the two facts these tests check.
    """
    t = depth / V * 1e-3
    eq = 4212.0 * np.exp(-(1.0 / tau) * t)
    var = {}
    for v in VARIANTS:
        tot = eq * np.exp(-mu_per_ms[v] * t)
        var[v] = {
            "kernel_cut": "none",
            "kernel": {"kernel_cut_tick": None, "truncation_fraction": 0.0,
                       "by_offset_ring": {k: {"deleted_over_sum_K_full": 0.0}
                                          for k in ("0", "1", "2", "3-5",
                                                    ">=6")}},
            "sum_xhat_ke": float(tot),
            "sum_xhat_line_pads_ke": float(0.9 * tot),
            "sum_xhat_off_line_ke": float(0.1 * tot),
            "by_ring": {k: {"sum_ke": float(0.2 * tot)}
                        for k in ("0", "1", "2", "3-5", ">=6")},
            "nnz": 7, "residual_rel": 0.04, "wall_s": 36.0,
            "scores": {s: {"E_rel": 0.25, "segments": {
                "rel_error_mean": -0.0003, "rel_error_rms": 0.0004}}
                for s in ("s1.5", "s2")}}
    return {"algorithm": "DepthLadderNonlinearEvent", "result": {
        "event": {"depth_cm": depth, "tau_ms": tau},
        "truth": {"sum_effq_ke": float(eq)},
        "records": {"sum_records_ke": float(eq),
                    "deficit_by_ring": {k: 0.0 for k in
                                        ("0", "1", "2", "3-5", ">=6")},
                    "by_ring": {k: {"over_sum_effq": 0.2} for k in
                                ("0", "1", "2", "3-5", ">=6")}},
        "tau_cut": {"truth": -100, "predicted_from_depth": -100.0,
                    "observed": {"first_above": {"tau_cut": -100},
                                 "shape_fit": {"tau_cut": -100}},
                    "observed_first_above_minus_truth_ticks": 0,
                    "observed_shape_fit_minus_truth_ticks": 0},
        "representation_term": {},
        "variants": var}}


def _run_fit(tmp_path, mu):
    inputs = []
    for d in DEPTHS:
        for tau in (1.0, 20.0):
            p = tmp_path / f"nl_event_{d}_{tau}.json"
            p.write_text(json.dumps(_event_doc(d, tau, mu)))
            inputs.append({"depth_cm": d, "tau_ms": tau, "json": str(p)})
    alg = ALGORITHMS["DepthLadderNonlinearFit"](
        velocity_cm_per_us=V, deep_only_min_depth_cm=16.5, inputs=inputs)
    alg.initialize({})
    store = EventStore()
    store.put("job.config", {"_meta": {"git": "test"}}, by="test")
    alg.execute(store)
    return alg.finalize()


def test_fit_reads_the_driver_schema_and_recovers_the_nonlinear_lambdas(
        tmp_path):
    """The subclass fits the two nonlinear totals beside truth, records and the
    archived linear arm, on the record schema the driver writes."""
    mu = {"none": 0.0, "pos_a0": -0.10, "pos_l1_0.01": 0.10}
    R = _run_fit(tmp_path, mu)
    for key, ltrue in (("1ms", 1.0), ("20ms", 0.05)):
        F = R["fits"][key]
        assert set(F) == {"sum_effq", "sum_records", "xhat_none",
                          "xhat_pos_a0", "xhat_pos_l1_0.01"}
        assert abs(F["sum_effq"]["lambda_per_ms"] - ltrue) < 1e-10
        for v in VARIANTS:
            # a lifetime-independent acceptance exp(-mu_v t) shifts lambda by
            # mu_v additively at BOTH lifetimes -- the behaviour the archived
            # linear ladder shows (-0.79 /ms in both samples)
            assert abs(F[f"xhat_{v}"]["lambda_per_ms"]
                       - (ltrue + mu[v])) < 1e-10
            assert F[f"xhat_{v}"]["deep_only"]["n_depths"] == 5


def test_fit_carries_the_per_event_quantities_the_figures_need(tmp_path):
    mu = {"none": 0.0, "pos_a0": -0.1, "pos_l1_0.01": 0.1}
    R = _run_fit(tmp_path, mu)
    e = R["per_event"]["1.0"]["16.5"]
    for k in ("off_line_over_sum_effq", "line_over_sum_effq",
              "by_ring_over_sum_effq", "nnz", "residual_rel", "wall_s",
              "E_rel", "segments"):
        assert k in e
        assert set(e[k]) == set(VARIANTS)
    assert abs(e["off_line_over_sum_effq"]["pos_a0"]
               - 0.1 * R["ratios"]["1ms"]["xhat_pos_a0"][4]) < 1e-12
    assert abs(e["line_over_sum_effq"]["pos_a0"]
               + e["off_line_over_sum_effq"]["pos_a0"]
               - R["ratios"]["1ms"]["xhat_pos_a0"][4]) < 1e-12


def test_ratio_lifetime_difference_is_reported_for_every_estimate(tmp_path):
    """A lifetime-independent acceptance gives zero; a nonlinear estimator need
    not, so the difference is reported rather than asserted."""
    mu = {"none": 0.0, "pos_a0": -0.1, "pos_l1_0.01": 0.1}
    R = _run_fit(tmp_path, mu)
    d = R["ratio_lifetime_difference"]["by_estimate"]
    for v in VARIANTS:
        assert f"xhat_{v}" in d
        assert len(d[f"xhat_{v}"]["per_depth"]) == len(DEPTHS)
    # these synthetic totals ARE a lifetime-independent acceptance, so every
    # difference is zero; on the real ladder the nonlinear arms need not be
    for v in VARIANTS:
        assert d[f"xhat_{v}"]["max_abs_difference"] < 1e-12


def test_estimate_list_drops_the_kernel_truncation_arms_of_the_parent():
    """The nonlinear campaign runs ONE kernel (no truncation), so the parent's
    truncated-kernel estimates must not be carried over."""
    parent = {n for n, _ in DepthLadderFit.ESTIMATES}
    assert "xhat_truth" in parent and "xhat_observed_shape_fit" in parent
    alg = ALGORITHMS["DepthLadderNonlinearFit"](inputs=[])
    alg.initialize({})
    store = EventStore()
    store.put("job.config", {}, by="test")
    alg.execute(store)
    got = {n for n, _ in alg.ESTIMATES}
    assert got == {"sum_effq", "sum_records", "xhat_none", "xhat_pos_a0",
                   "xhat_pos_l1_0.01"}


def test_copied_blocks_cover_what_the_fit_reads():
    """Every block of the archived linear per-event JSON that the fit touches
    is in ``COPIED_BLOCKS``, so the driver's record is readable by it."""
    for k in ("truth", "records", "tau_cut", "representation_term"):
        assert k in dnl.COPIED_BLOCKS


def test_figure_colours_are_the_campaign_colours():
    assert (dnl.C_POS, dnl.C_POSL1, dnl.C_LIN, dnl.C_TRUTH) == \
        ("#009E73", "#CC79A7", "#D55E00", "#000000")
    arms = dnl.DepthLadderNonlinearFigures.ARMS
    assert [a[1] for a in arms] == ["none", "pos_a0", "pos_l1_0.01"]
    assert [a[2] for a in arms] == [dnl.C_LIN, dnl.C_POS, dnl.C_POSL1]
