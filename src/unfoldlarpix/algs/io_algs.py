"""Source and sink algorithms: event input, charges output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constrained_solver import gaussian_post_smooth, split_deposit
from ..fwk.component import Algorithm, algorithm
from ..io.hits import HitsView
from ..model.conventions import TIME_CONVENTION


@algorithm("LoadEvent")
class LoadEvent(Algorithm):
    """Event source: iterates events of one tred NPZ file."""

    writes = ("event", "hits_view", "readout_config")

    def initialize(self, services):
        super().initialize(services)
        from ..data_loader import DataLoader
        loader = DataLoader(self.props["input"])
        self.rc = loader.get_readout_config()
        tpc = self.props.get("tpc")
        self.events = [e for e in loader.iter_events()
                       if e.hits and (tpc is None or e.tpc_id == tpc)]
        if "max_events" in self.props:
            self.events = self.events[: int(self.props["max_events"])]
        self._cursor = 0

    def n_events(self) -> int:
        return len(self.events)

    def execute(self, store):
        ev = self.events[self._cursor]
        self._cursor += 1
        self.put(store, "event", ev)
        self.put(store, "readout_config", self.rc)
        self.put(store, "hits_view", HitsView(
            np.asarray(ev.hits.location), np.asarray(ev.hits.data),
            self.rc.adc_hold_delay))


@algorithm("LoadSolution")
class LoadSolution(Algorithm):
    """Publish a SHIPPED solution as ``solve.q``, without re-solving.

    Every post-hoc study of a delivered solve needs this, and until now each
    one loaded the NPZ itself and the provenance of ``solve.q`` was whatever
    the driver chose.  Here it is a store product with its source recorded, so
    a consumer cannot confuse a re-solve with an archived one.

    Props: ``path`` (the NPZ), ``field`` (default ``deconv_q_sharp``;
    ``deconv_q`` is the pre-threshold field).  Validates the shape against the
    operator when one is present.
    """

    reads = ()
    writes = ("solve.q", "solve.provenance")

    def execute(self, store):
        path = Path(str(self.props["path"]))
        field = str(self.props.get("field", "deconv_q_sharp"))
        z = np.load(path, allow_pickle=True)
        if field not in z.files:
            raise KeyError(f"{path.name} has no field {field!r}; "
                           f"available: {sorted(z.files)}")
        q = np.asarray(z[field], dtype=np.float64)
        prov = {"path": str(path), "field": field, "sum_ke": float(q.sum()),
                "shape": list(q.shape)}
        if "provenance" in z.files:
            try:
                prov["npz_provenance"] = z["provenance"].item()
            except Exception:
                pass
        if "op" in store:
            want = tuple(store.get("op").q_shape)
            if tuple(q.shape) != want:
                raise ValueError(
                    f"{path.name}:{field} has shape {q.shape} against the "
                    f"operator's q_shape {want} -- different block or grid")
        print(f"[LoadSolution] {path.name}:{field} sum {q.sum():.1f} ke, "
              f"nnz(>0.01) {(q > 0.01).sum()}")
        self.put(store, "solve.q", q)
        self.put(store, "solve.provenance", prov)


@algorithm("WriteCharges")
class WriteCharges(Algorithm):
    """Write the solver-schema NPZ (self-describing: config + provenance)."""

    reads = ("event", "readout_config", "hits_view", "solve.q",
             "block_offset")
    writes = ("output.path",)

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        hv = store.get("hits_view")
        q_hat = store.get("solve.q")
        raw_off = np.asarray(store.get("block_offset"), dtype=float)
        B = rc.adc_hold_delay
        u = store.get("offsets.u") if "offsets.u" in store else None

        q_dep = split_deposit(q_hat, u) if u is not None else q_hat
        sigma = float(self.props.get("sigma_time", 0.005))
        sigma_pxl = float(self.props.get("sigma_pixel", 0.2))
        q_smooth = gaussian_post_smooth(q_dep, B, sigma, sigma_pxl)

        # The written grid is declared at the operator's own release point:
        # bin k means charge released at boffset_raw + k*B.  The legacy pair
        # (declare half a bin early, deposit at the bin centre) cancelled to
        # the same instant; it is recorded in the file only as the ABSENCE of
        # `time_convention`.
        boffset = raw_off.copy()

        ci, cj, ck = np.where(q_hat > 0.01)
        t_centers = raw_off[2] + ck * float(B)
        if u is not None:
            t_centers = t_centers + u[ci, cj, ck] * float(B)
        charges = np.stack([raw_off[0] + ci, raw_off[1] + cj, t_centers,
                            q_hat[ci, cj, ck],
                            (q_hat[ci, cj, ck] > 0.5).astype(float)], axis=1)

        payload = {
            "deconv_q": q_smooth,
            "deconv_q_sharp": q_hat.astype(np.float32),
            "boffset": boffset,
            "boffset_raw": raw_off,
            "time_convention": TIME_CONVENTION,
            "adc_hold_delay": B,
            "readout_nburst": hv.nburst,
            "readout_threshold": float(rc.threshold),
            "lean_output": True,
            "charges": charges,
            "charges_columns":
                "pixel_x pixel_y t_center_tick charge_ke on_skeleton",
            "job_config": json.dumps(store.get("job.config"), default=str),
            "loss_components": json.dumps(
                store.get("solve.loss")
                if "solve.loss" in store else None),
            "loss_trace": json.dumps(
                store.get("solve.trace")
                if "solve.trace" in store else None),
            "provenance": json.dumps(store.provenance()),
        }
        if u is not None:
            payload["deconv_q_offsets"] = (u * float(B)).astype(np.float32)

        # the FIT-GRID truth and the row residual, if BuildTruth/RowResidual
        # ran.  Distinct from `smeared_true` below: that one is the smeared
        # evaluation truth on the universal grid, this one is the operator's
        # own grid under a declared binning convention, and it is what
        # d - A q_truth is built from.
        for key, name in (("truth.q", "truth_q_fitgrid"),
                          ("truth.meta", "truth_meta"),
                          ("resid.rows", "resid_rows"),
                          ("resid.summary", "resid_summary"),
                          ("resid.solution", "resid_solution"),
                          ("resid.solution_summary", "resid_solution_summary")):
            if key in store:
                v = store.get(key)
                if v is not None:
                    payload[name] = (np.asarray(v) if not isinstance(v, dict)
                                     else np.array(v, dtype=object))
        if bool(self.props.get("embed_truth", False)):
            # self-contained output: eval/plots need no external truth ref
            from ..deconv_workflow import smear_effective_charge
            smear_offset, smeared = smear_effective_charge(
                ev, sigma_time=sigma, sigma_pixel=sigma_pxl)
            payload["smeared_true"] = smeared
            payload["smear_offset"] = np.array(smear_offset)

        out_dir = Path(self.props["out_dir"]).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.props.get("prefix", "unfold")
        path = out_dir / f"{prefix}_event_{ev.tpc_id}_{ev.event_id}.npz"
        np.savez(path, **payload)
        self.put(store, "output.path", str(path))
        print(f"[WriteCharges] {path}")
