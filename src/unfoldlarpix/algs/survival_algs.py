"""Which warm-start voxels the l1 ladder sets to zero, resolved by c_v.

One algorithm: :class:`GainSurvival`.  It answers a single question --
of the voxels the FFT warm start hands to the solver, what fraction is
still nonzero after each stage of the l1 ladder, as a function of the
per-voxel measurement gain ``c_v = (A^T 1)_v``.

Two distinct mechanisms set a warm-start voxel to zero and they are
reported separately, because they have nothing to do with each other:

* the SUPPORT.  ``CoordProx`` multiplies by the support mask, so every
  seed outside it is zero at the first iteration whatever alpha is.
* the l1 PROX.  Inside the support a coordinate stays at zero unless
  ``|(A^T r)_v|`` exceeds its l1 weight; ``A^T r`` itself scales with
  ``c_v``, so a uniform alpha is not a uniform threshold on charge.

Two denominators are reported per bin and they answer different questions.
The warm start is a RECONSTRUCTION: its charge on a voxel is what the
unregularised inverse claimed, ringing included, so a fraction weighted by
it says only how much of that claim survived.  The decision-relevant
denominator is TWO-SIDED -- the union of the warm-start seed set and the
voxels that actually hold truth charge -- weighted by the truth.  A voxel
holding truth charge that the warm start never populated is invisible to a
reco-only denominator and is exactly the failure a survival study is
looking for, so it is in the union set by construction.

CAVEAT on the truth side: ``grid_truth`` deposits effq into the fit bin
whose CENTRE is nearest, and where the field response places a charge in
time relative to effq is an ASSUMPTION (``truth_deposit``,
``truth_shift_ticks``).  A registration error of half a bin moves truth
charge to a neighbouring time bin, which is enough to make a surviving
voxel look killed.  ``truth_killed_far_ke`` is the guard: it counts only
the killed truth charge with NO surviving voxel within one voxel in any
direction.

The ladder is the shipped one (:class:`unfoldlarpix.solve.strategy.Ladder`)
run one stage at a time from the same ``q0 = max(warm.deconv_q, 0)`` the
:class:`Solve` algorithm uses, with the same smooth terms built by
:func:`unfoldlarpix.algs.reco_algs.build_terms`.  Running stage by stage
is equivalent to one multi-stage call: each stage's alpha field depends
only on the previous stage's skeleton, which ``Ladder.run`` sets at the
end of every stage.
"""
from __future__ import annotations

import numpy as np

from ..fwk.component import algorithm
from ..solve.engine import Fista
from ..solve.strategy import FinalRefit, Ladder, SolveState
from .fixedgrid_algs import _grow, _JsonRecorder, grid_truth
from .reco_algs import build_terms

# Default c_v bin edges.  Logarithmic below 0.1 and linear above it,
# because that is how the on-support c_v distribution is shaped: on the
# nb1 samples the median support voxel has c_v ~ 0.03-0.2 while the fully
# covered voxels sit near 1.  ``-inf .. 0`` is kept as its own bin: the
# kernel is bipolar across neighbour pixels, so c_v <= 0 voxels are a
# different object (adding charge there LOWERS the prediction), not the
# small-c_v end of a continuum.
DEFAULT_EDGES = (float("-inf"), 0.0, 1e-3, 1e-2, 0.03, 0.1, 0.3, 0.5,
                 0.7, 0.9, 1.05, 1.2, float("inf"))
DEFAULT_SEED_EPS = (0.0, 0.01, 0.1, 0.3, 1.0)


def _bin_label(lo: float, hi: float) -> str:
    if lo == float("-inf"):
        return f"<= {hi:g}"
    if hi == float("inf"):
        return f"> {lo:g}"
    return f"{lo:g}-{hi:g}"


@algorithm("GainSurvival")
class GainSurvival(_JsonRecorder):
    """Survival of the warm-start voxels through the l1 ladder, by ``c_v``.

    Props
    -----
    strategy : dict
        The ladder, same schema as :class:`Solve` (``alphas``, ``seed_cut``,
        ``soft_len``, ``soft_exponent``, ``soft_axis_cost``).  Copy it from
        the solve job whose behaviour is being explained.
    terms : list
        Smooth terms other than the data fidelity, same schema as
        :class:`Solve`.  A censor term changes A^T r and therefore which
        coordinates activate, so it belongs here whenever the solve of
        record carries one.
    engine : dict
        ``iters`` per stage (default 600).
    refit : dict, optional
        Append a :class:`FinalRefit` stage (``eps``, ``alpha``).
    edges : list, optional
        c_v bin edges; ``DEFAULT_EDGES`` otherwise.
    seed_eps : list, optional
        Warm-start charge thresholds [ke] defining the seed sets.  ``0.0``
        (any positive warm-start charge) is the literal reading of "start
        from the FFTWarmStart output"; the larger ones separate the FFT
        ringing from the charge the warm start actually claims.
    truth_deposit : str
        ``round`` (bin centre, the adopted protocol) or ``floor``.
    truth_shift_ticks : float
        Shift applied to the effq arrival before binning (default 0).
    truth_cut : float
        A voxel is on the truth side when ``truth > truth_cut`` [ke]
        (default 0).
    alive_cut : float
        A voxel counts as surviving when ``q > alive_cut`` [ke] (default
        0.0, i.e. the prox has not set it exactly to zero).
    out : str, optional
        JSON path.
    """

    reads = ("op", "support", "warm.deconv_q", "event", "hits_view",
             "block_offset", "readout_config", "time_subbin")
    writes = ("gain.survival",)

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        S = int(store.get("time_subbin") or 1)

        cv = op.measurement_gain().cpu().numpy()
        support = np.asarray(store.get("support"), dtype=bool)
        q0 = np.clip(store.get("warm.deconv_q"), 0.0, None)
        if S > 1:                       # same lift Solve applies (conserve)
            q0 = np.repeat(q0, S, axis=2) / S
        q0 = q0[:, :, : op.q_shape[2]]

        qt = grid_truth(store, op,
                        mode=str(self.props.get("truth_deposit", "round")),
                        shift_ticks=float(self.props.get(
                            "truth_shift_ticks", 0.0)))
        tcut = float(self.props.get("truth_cut", 0.0))

        terms = build_terms(self.props.get("terms", []), store, op, rc)
        supp_t = op.to_tensor(support.astype(np.float64))
        engine = Fista(n_iter=int(self.props.get("engine", {})
                                  .get("iters", 600)))
        scfg = dict(self.props.get("strategy", {}))
        stype = scfg.pop("type", "ladder")
        if stype != "ladder":
            raise ValueError(f"unknown strategy: {stype}")
        alphas = [float(a) for a in scfg.pop("alphas")]

        # one stage at a time, so the intermediate q is observable
        state = SolveState(q=op.to_tensor(q0))
        snaps = []
        for k, a in enumerate(alphas):
            lad = Ladder(alphas=[a], n_iter=engine.n_iter, **scfg)
            state = lad.run(engine, op, terms, supp_t, state)
            q = state.q.cpu().numpy().astype(np.float64)
            snaps.append((f"ladder[{k}]", a, q))
            rec = state.history[-1]
            print(f"[{self.name}] ladder[{k}] alpha={a} q_sum={rec.q_sum:.1f} "
                  f"nnz={rec.nnz}")
        if "refit" in self.props:
            rcfg = self.props["refit"]
            state = FinalRefit(eps=float(rcfg.get("eps", 0.5)),
                               alpha=float(rcfg.get("alpha", 0.0)),
                               n_iter=engine.n_iter).run(
                engine, op, terms, supp_t, state)
            snaps.append(("refit", float(rcfg.get("alpha", 0.0)),
                          state.q.cpu().numpy().astype(np.float64)))
            print(f"[{self.name}] refit q_sum={state.history[-1].q_sum:.1f} "
                  f"nnz={state.history[-1].nnz}")

        edges = [float(e) for e in self.props.get("edges", DEFAULT_EDGES)]
        eps_list = [float(e) for e in self.props.get("seed_eps",
                                                     DEFAULT_SEED_EPS)]
        cut = float(self.props.get("alive_cut", 0.0))
        pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

        rec = {
            "q_shape": [int(n) for n in op.q_shape],
            "n_data": int(op.n_data),
            "alive_cut_ke": cut,
            "cv": {"min": float(cv.min()), "max": float(cv.max()),
                   "n_voxels": int(cv.size),
                   "frac_le_zero": float((cv <= 0).mean()),
                   "percentiles": {str(p): float(v) for p, v in
                                   zip(pcts, np.percentile(cv, pcts))},
                   "percentiles_on_support": {
                       str(p): float(v) for p, v in
                       zip(pcts, np.percentile(cv[support], pcts))}},
            "support": {"n": int(support.sum()),
                        "frac": float(support.mean())},
            "truth": {"deposit": str(self.props.get("truth_deposit",
                                                     "round")),
                      "shift_ticks": float(self.props.get(
                          "truth_shift_ticks", 0.0)),
                      "cut_ke": tcut,
                      "sum_ke": float(qt.sum()),
                      "sum_on_support_ke": float(qt[support].sum()),
                      "n_above_cut": int((qt > tcut).sum())},
            "warm": {"sum_ke": float(q0.sum()),
                     "sum_on_support_ke": float(q0[support].sum()),
                     "n_positive": int((q0 > 0).sum())},
            "stages": [{"label": lab, "alpha": a, "q_sum": float(q.sum()),
                        "n_positive": int((q > cut).sum()),
                        "sum_on_support": float(q[support].sum())}
                       for lab, a, q in snaps],
            "edges": edges,
            "bins": [],
        }

        # per-stage grid-wide masks: nonzero, and within one voxel of a
        # nonzero one (the registration guard for the truth side)
        stage_masks = [(lab, q > cut, _grow(q > cut)) for lab, a, q in snaps]
        truth_on = qt > tcut

        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (cv > lo) & (cv <= hi)
            b = {"lo": lo, "hi": hi, "label": _bin_label(lo, hi),
                 "n_grid": int(sel.sum()),
                 "n_support": int((sel & support).sum()),
                 "seeds": []}
            for eps in eps_list:
                seed = sel & (q0 > eps)
                on = seed & support
                entry = {"eps_ke": eps,
                         "n_seed": int(seed.sum()),
                         "q0_seed_ke": float(q0[seed].sum()),
                         "n_seed_on_support": int(on.sum()),
                         "q0_seed_on_support_ke": float(q0[on].sum()),
                         "n_killed_by_support": int((seed & ~support).sum()),
                         "q0_killed_by_support_ke": float(
                             q0[seed & ~support].sum()),
                         # TWO-SIDED evaluation set: the warm-start seeds
                         # UNION the voxels holding truth charge.  A truth
                         # voxel the warm start never populated is a failure
                         # a reco-only denominator cannot see.
                         "union": {
                             "n": int((seed | (sel & truth_on)).sum()),
                             "n_truth_only": int(
                                 (sel & truth_on & ~seed).sum()),
                             "truth_ke": float(
                                 qt[seed | (sel & truth_on)].sum()),
                             "truth_only_ke": float(
                                 qt[sel & truth_on & ~seed].sum()),
                             "q0_ke": float(q0[seed | (sel & truth_on)].sum()),
                             "n_off_support": int(
                                 ((seed | (sel & truth_on)) & ~support).sum()),
                             "truth_off_support_ke": float(
                                 qt[(seed | (sel & truth_on))
                                    & ~support].sum())},
                         "stages": []}
                uni = seed | (sel & truth_on)
                w = qt + q0            # two-sided weight
                for (lab, a, q), (_, alive_all, near_all) in zip(snaps,
                                                                 stage_masks):
                    alive = on & (q > cut)
                    ua = uni & alive_all
                    ukilled = uni & ~alive_all
                    far = ukilled & ~near_all
                    two = {
                        "n_union": int(uni.sum()),
                        "n_alive_union": int(ua.sum()),
                        "surv_n_union": (float(ua.sum() / uni.sum())
                                         if uni.sum() else float("nan")),
                        # THE truth-weighted fraction: of the truth charge in
                        # this c_v bin, how much sits on a voxel the solution
                        # still holds nonzero
                        "truth_union_ke": float(qt[uni].sum()),
                        "truth_alive_ke": float(qt[ua].sum()),
                        "truth_killed_ke": float(qt[ukilled].sum()),
                        "truth_killed_far_ke": float(qt[far].sum()),
                        "surv_truth": (float(qt[ua].sum() / qt[uni].sum())
                                       if qt[uni].sum() > 0 else float("nan")),
                        # the same fraction under the warm start's own weight
                        # and under the two-sided weight truth + warm start
                        "surv_q0_union": (float(q0[ua].sum() / q0[uni].sum())
                                          if q0[uni].sum() > 0
                                          else float("nan")),
                        "surv_w_union": (float(w[ua].sum() / w[uni].sum())
                                         if w[uni].sum() > 0
                                         else float("nan")),
                        "q_alive_union_ke": float(q[ua].sum()),
                        "q_bin_ke": float(q[sel].sum())}
                    entry["stages"].append({**two,
                        "label": lab, "alpha": a,
                        "n_alive": int(alive.sum()),
                        "n_killed_by_l1": int((on & ~(q > cut)).sum()),
                        "q0_alive_ke": float(q0[alive].sum()),
                        "q0_killed_by_l1_ke": float(q0[on & ~(q > cut)].sum()),
                        "q_alive_ke": float(q[alive].sum()),
                        # survival among the seeds the support kept: the
                        # l1 ladder's own effect
                        "surv_n": (float(alive.sum() / on.sum())
                                   if on.sum() else float("nan")),
                        "surv_q0": (float(q0[alive].sum() / q0[on].sum())
                                    if q0[on].sum() > 0 else float("nan")),
                        # survival among ALL seeds in the bin: support and
                        # l1 together
                        "surv_n_all": (float(alive.sum() / seed.sum())
                                       if seed.sum() else float("nan")),
                        "surv_q0_all": (float(q0[alive].sum() / q0[seed].sum())
                                        if q0[seed].sum() > 0
                                        else float("nan"))})
                b["seeds"].append(entry)
            # voxels the ladder ACTIVATES: zero in the warm start, nonzero
            # after the stage.  The complement of survival, and the reason
            # a survival fraction alone does not conserve voxel count.
            born = sel & support & ~(q0 > 0)
            b["born"] = [{"label": lab,
                          "n_born": int((born & (q > cut)).sum()),
                          "q_born_ke": float(q[born & (q > cut)].sum())}
                         for lab, a, q in snaps]
            rec["bins"].append(b)

        eps0 = eps_list[0]
        tot_on = sum(b["seeds"][0]["n_seed_on_support"] for b in rec["bins"])
        last = snaps[-1][0]
        tot_alive = sum(s["n_alive"] for b in rec["bins"]
                        for s in b["seeds"][0]["stages"] if s["label"] == last)
        print(f"[{self.name}] seeds q0>{eps0:g} on support {tot_on}, "
              f"alive after {last}: {tot_alive} "
              f"({100.0 * tot_alive / max(tot_on, 1):.1f}%)")
        tk = sum(st["truth_killed_ke"] for b in rec["bins"]
                 for st in b["seeds"][0]["stages"] if st["label"] == last)
        tf = sum(st["truth_killed_far_ke"] for b in rec["bins"]
                 for st in b["seeds"][0]["stages"] if st["label"] == last)
        print(f"[{self.name}] truth {qt.sum():.1f} ke; on voxels zero after "
              f"{last}: {tk:.1f} ke ({100.0 * tk / max(qt.sum(), 1e-9):.1f}%), "
              f"of which {tf:.1f} ke with no nonzero voxel within 1")
        self._emit(store, rec)
