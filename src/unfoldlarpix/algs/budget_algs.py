"""Charge and step-size budgets: where the total goes, and what it costs.

Ports ``noiseless_closure_round/event_total.py`` (and the ``gain_from_colsum``
/ ``gain_profile`` family it belongs to) and ``_drivers/step_probe.py``.
"""
from __future__ import annotations

import numpy as np

from ..fwk.component import Algorithm, algorithm
from .reco_algs import build_terms


@algorithm("ChargeBudget")
class ChargeBudget(Algorithm):
    """Event-level charge accounting: what is booked, against what exists.

    All sums over the whole fit grid, in ke:

        Q_truth   the point charges assigned to the grid (BuildTruth)
        Q_off     charge that fell outside the grid -- NOT in Q_truth, and
                  reported because the old scripts masked it silently
        Q_hat     the solution's total
        Q_data    the recorded total, sum over rows of d

    The two ratios that matter are ``Q_hat / Q_truth`` (did the solve book the
    charge that is there) and ``Q_hat / (Q_truth + Q_off)`` (did it book the
    charge the event *had*).  Quoting the first alone credits the solve for
    charge the grid never offered it, which is how a 0.3% off-grid loss became
    invisible.

    Props: ``truth_prefix``, ``by_pixel`` (default False; adds the per-pixel
    booked/true ratio quantiles, which is the ``gain_profile`` question).
    """

    reads = ("op", "solve.q")
    writes = ("budget.summary",)

    def __init__(self, **props):
        super().__init__(**props)
        self.prefix = str(props.get("truth_prefix", "truth"))
        self.reads = tuple(self.reads) + (f"{self.prefix}.q",
                                          f"{self.prefix}.meta")

    def execute(self, store):
        op = store.get("op")
        q = np.asarray(store.get("solve.q"), dtype=np.float64)
        t = np.asarray(store.get(f"{self.prefix}.q"), dtype=np.float64)
        tm = store.get(f"{self.prefix}.meta")
        d = np.asarray(op.d.detach().cpu().numpy(), np.float64)

        Q_t = float(t.sum())
        Q_off = float(tm.get("off_grid_ke", 0.0))
        Q_h = float(q.sum())
        rec = {
            "truth_convention": tm["convention"],
            "Q_truth_ke": Q_t, "Q_off_grid_ke": Q_off,
            "Q_hat_ke": Q_h, "Q_data_ke": float(d.sum()),
            "hat_over_truth": (Q_h / Q_t) if Q_t else None,
            "hat_over_event": (Q_h / (Q_t + Q_off)) if (Q_t + Q_off) else None,
            "off_grid_frac": (Q_off / (Q_t + Q_off)) if (Q_t + Q_off) else None,
            "nnz_hat": int((q > 0.01).sum()), "nnz_truth": int((t > 0.01).sum()),
        }
        if bool(self.props.get("by_pixel", False)):
            # per-pixel booked/true: the gain_profile question -- is the
            # over-book spread evenly or concentrated on some pixels?
            pt = t.sum(axis=2)
            ph = q.sum(axis=2)
            m = pt > float(self.props.get("pixel_floor", 1.0))
            if m.any():
                r = ph[m] / pt[m]
                rec["by_pixel"] = {
                    "n_pixels": int(m.sum()),
                    "median": float(np.median(r)),
                    "q10": float(np.percentile(r, 10)),
                    "q90": float(np.percentile(r, 90)),
                    "frac_over_1": float((r > 1.0).mean())}
        print("[ChargeBudget] Q_hat/Q_truth %.4f  Q_hat/Q_event %.4f  "
              "(off grid %.3f%%)"
              % (rec["hat_over_truth"] or float("nan"),
                 rec["hat_over_event"] or float("nan"),
                 100 * (rec["off_grid_frac"] or 0.0)))
        self.put(store, "budget.summary", rec)


@algorithm("StepBudget")
class StepBudget(Algorithm):
    """Each objective term's curvature, and the FISTA step it allows.

    Ports ``_drivers/step_probe.py``.  The censor term's curvature is linear in
    ``beta``, so raising ``beta`` shrinks the step -- which means a ``beta``
    scan changes two things at once unless the iteration count is matched.
    That confound is why ``beta_scan/`` must not be read as a beta scan.

    Builds the terms through the SAME ``build_terms`` the solver uses, so a
    probe cannot drift from what is solved.

    Props: ``terms`` (same schema as ``Solve``), ``betas`` (optional list; the
    censor term is rebuilt at each and its curvature reported).
    """

    reads = ("op", "readout_config", "hits_view", "block_offset")
    writes = ("step.summary",)

    def execute(self, store):
        op = store.get("op")
        rc = store.get("readout_config")
        cfgs = list(self.props.get("terms", []))

        def curvatures(term_cfgs):
            terms = build_terms(term_cfgs, store, op, rc)
            out = {}
            for t in terms:
                out[type(t).__name__] = out.get(type(t).__name__, 0.0) + \
                    float(t.curvature())
            return out

        base = curvatures(cfgs)
        L = sum(base.values())
        rec = {"terms": base, "L_total": L,
               "step": (1.0 / L) if L else None,
               "lipschitz_data_only": float(op.lipschitz)}
        betas = self.props.get("betas")
        if betas and cfgs:
            scan = {}
            for b in betas:
                cs = [dict(c) for c in cfgs]
                for c in cs:
                    if c.get("type") in ("censor", "censor_pre"):
                        c["beta"] = float(b)
                cv = curvatures(cs)
                tot = sum(cv.values())
                scan[str(b)] = {"terms": cv, "L_total": tot,
                                "step": (1.0 / tot) if tot else None,
                                "step_ratio_to_base": (L / tot) if tot else None}
            rec["beta_scan"] = scan
        print("[StepBudget] L = %.4g (%s), step %.4g"
              % (L, ", ".join("%s %.4g" % kv for kv in base.items()),
                 rec["step"] or float("nan")))
        self.put(store, "step.summary", rec)
