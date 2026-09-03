"""Exact-functional rows: the impact-resolved kernel and its closure on data.

Two algorithms, both for the fixed-interval readout, which is the reference
case: the accumulator is sampled on a fixed grid of stride ``B`` fine ticks,
there is no trigger, no reset and no zero suppression, so every record is an
exact linear functional of the fine charge (``FORWARD_MODEL_revised.md``
Sec. 4.1, ``algo_plan.md`` Sec. 4.0).

:class:`KernelStructure`
    Kernel-only (``reads = ()``).  Measures every kernel quantity the plan
    marks ``[arith]`` or ``[est]``: window weights against arrival phase, the
    impact-averaging error and what grouping windows does to it, the
    neighbour-column residual against group length, the arrival-window
    fraction against phase, and the leading-part fraction.

:class:`FineTruthClosure`
    Asks whether the fine-tick pad-level truth is a solution of the
    exact-functional forward model, on a real event, and what the residual is
    made of.

Definitions
-----------
All of these are used below and none of them is standard terminology except
where said so; each is defined here before it is used.

``Delta_t``
    the response file's ``time_tick``, 0.05 us.  ``B`` is
    ``adc_hold_delay`` in fine ticks (30 = 1.5 us in every campaign).

``R_{i,d}[m]``
    the field response as produced by ``tred.response.ndlarsim`` and reshaped
    to ``(pixel_x, impact_x, pixel_y, impact_y, tick)``; entry
    ``[12 + dx, ix, 12 + dy, iy, m]``.  ``d = (dx, dy)`` is the offset of the
    SENSING pixel from the pixel that collects the electron and
    ``i = (ix, iy)`` is the impact cell of the electron inside its own pixel,
    cell ``ix`` covering transverse offsets ``(ix - 5 .. ix - 4) * pitch/10``
    from the pixel's lower edge, i.e. centred at ``(ix - 4.5) * pitch/10``
    from the pixel centre, POSITIVE along the +pixel_x axis (and likewise
    ``iy`` along +pixel_y).  This sign convention is verified in
    ``tests/test_exactrows.py``: at impact ``ix = 0`` (electron near the low
    edge of its pad) the large induced excursion is at ``dx = -1``.
    Unit: ``R`` is a current in electron charges per microsecond, so
    ``R[m] * Delta_t`` is the charge induced during tick ``m`` in units of
    the drifting electron's charge.

``Kcum_{i,d}(tau)``
    ``sum_{m=0}^{tau} R_{i,d}[m] * Delta_t`` for ``0 <= tau < 3900``; ``0``
    for ``tau < 0``; ``Kcum(3899)`` for ``tau >= 3900``.  The upper limit is
    INCLUSIVE, which is what makes ``Kcum`` the object tred samples: tred
    forms ``Xacc = I.cumsum(t)`` and reads ``Xacc[offset + s*B]``
    (``tred/readout.py:fixed_interval_readout``), an inclusive running sum.

``Kbar_d``, ``Kcumbar_d``
    the impact average ``(1/100) sum_i K_{i,d}`` and its cumulative.  This is
    what ``FieldResponseProcessor.process_response()`` returns (elementwise,
    with no further flip: ``ndlarsim`` and ``_flip_kernel_for_convolution``
    apply the SAME per-pixel spatial flip, so the two agree directly).

``phi``, arrival phase
    the electron crosses the response plane at absolute tick
    ``t0 = 30 m + phi``, ``phi`` in ``0..29``.  The readout samples the
    accumulator at absolute ticks ``30 s``; the charge assigned to window
    ``W_s = [30 s, 30 (s+1))`` is

        w_{i,d}(s; phi) = Kcum_{i,d}(30 (s+1) - t0) - Kcum_{i,d}(30 s - t0)

    which counts response ticks ``30 s - t0 + 1 .. 30 (s+1) - t0``.

``s_a``, arrival window
    the window containing the anode arrival, ``30 s_a <= t0 + 3812 <
    30 (s_a + 1)``.  Windows are reported by ``n = s - s_a``.

``d_{i,d}(s; phi) = w_{i,d}(s; phi) - wbar_d(s; phi)``
    the impact-averaging error, called a "time dipole" in
    ``FORWARD_MODEL_revised.md`` Sec. 3.6(b).  ``sum_i d = 0`` by
    construction; ``sum_s d`` over the whole support is the difference of two
    time integrals, ``0`` on the collecting pixel to float precision.

``D_{g,e}(i, phi)``, grouped impact-averaging error
    partition the windows into blocks of ``g`` consecutive windows such that
    the arrival window is the ``(e+1)``-th member of its block:
    ``G_m = {s : s_a - e + m g <= s < s_a - e + (m+1) g}``, ``e`` in
    ``0..g-1``.  Then

        D_{g,e}(i, phi) = max_m | sum_{s in G_m} d_{i,(0,0)}(s; phi) |

    over the blocks that intersect the response support.  Two summaries:
    ``max`` over ``i, phi, e`` is the FIXED-COMMON-GRID worst case (the only
    case available under fixed-interval readout, where the grid is fixed in
    absolute time and the arrival phase is a property of the signal);
    ``max_phi min_e max_i`` is the PAD-ANCHORED best case, where the group
    edges may be chosen knowing the phase but not the impact.

``Nbar_g(d; phi) = sum_{s = s_a - g + 1}^{s_a} wbar_d(s; phi)``
    the neighbour-column residual: what an operator that keeps ``g`` windows
    ending at the arrival window still owes the neighbour whose electron it
    is.  Zero only in the limit of the full transient support.

``L(n) = Kcumbar_{(0,0)}(30 (s_a - n) - t0)``
    the leading-part fraction: the fraction of one electron's charge already
    registered on the collecting pad more than ``n`` windows before the
    arrival window.

``x_p(j)``
    the fine truth: ``effq`` charge [ke] on pad ``p`` at fine tick ``j``.  The
    tick is the time the charge crosses the RESPONSE PLANE (30.431 cm out),
    which is ``tau = 0`` of the kernel.

``q_p[k]``, ``y_p[k]``, ``l_k``
    the records, the window charges and the latch times:
    ``l_k = trigger + k B`` (``l_0 = trigger``), ``q_p[k] = A_p(l_k) -
    A_p(l_0)`` with ``A_p`` the CSA accumulator, and
    ``y_p[k] = q_p[k] - q_p[k-1]``, ``q_p[0] := 0``.

``yhat_p[k]``, ``eps_p[k]``
    the prediction and the residual,

        yhat_p[k] = sum_{p'} sum_j x_{p'}(j)
                    [ Kcumbar_{p-p'}(l_k - j) - Kcumbar_{p-p'}(l_{k-1} - j) ]
        eps_p[k]  = y_p[k] - yhat_p[k] .

``current_zero_before_tick``
    tred deletes every current sample before the event time reference:
    ``graph_effq.py:148-159``, ``concatenate_waveforms(..., event_t =
    global_tref[1] // tspace)``, which is tick 0 in every campaign.  This is
    part of the realised measurement functional and not of the field
    response, so it is carried as an explicit, separately reported term:
    with the deletion, ``A_p(t) = 0`` for ``t < t_z`` and
    ``A_p(t) = [that sum] - sum_j x(j) Kcumbar(t_z - 1 - j)`` for
    ``t >= t_z``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fwk.component import Algorithm, algorithm

# ---------------------------------------------------------------------------
# kernel loading -- shared library code, NOT inter-algorithm communication
# ---------------------------------------------------------------------------
COLLECTION_PIXEL = 12          # index of the collecting pixel in the 25x25 grid
ARRIVAL_TICK = 3812            # drift_length / velocity / time_tick, drtoa 30.431 cm
N_IMPACT = 10                  # npath per axis


def load_impact_response(path: str):
    """``(R, meta)`` with ``R`` shaped (25, 10, 25, 10, Nt), float32.

    Uses ``tred.response.ndlarsim`` -- the code that produced the data -- so
    the expansion is not re-implemented here.  ``R[12+dx, ix, 12+dy, iy, m]``
    is the current on pixel offset ``d`` for an electron at impact ``i``; see
    the module docstring for the sign convention.
    """
    from tred.response import ndlarsim
    f = np.load(path)
    raw = f["response"]
    npath = int(f["npath"])
    dt = float(np.squeeze(f["time_tick"]))
    R = ndlarsim(raw, nd_response_shape=raw.shape[:2], nd_nimp=npath).numpy()
    npix = R.shape[0] // npath
    meta = {"response_path": str(path), "time_tick_us": dt,
            "npath": npath, "n_pixel": int(npix),
            "n_tick": int(R.shape[-1]),
            "bin_size_cm": float(np.squeeze(f["bin_size"])),
            "drift_length_cm": float(np.squeeze(f["drift_length"]))}
    return R.reshape(npix, npath, npix, npath, R.shape[-1]), meta


def cumulative(arr: np.ndarray, dt: float) -> np.ndarray:
    """``Kcum`` from a current array: inclusive running sum times ``Delta_t``."""
    return np.cumsum(np.asarray(arr, dtype=np.float64), axis=-1) * dt


def kcum_at(kc: np.ndarray, tau) -> np.ndarray:
    """``Kcum`` evaluated at integer arguments, 0 below and saturated above.

    ``kc`` has the tick axis last; ``tau`` broadcasts against its leading
    axes' trailing shape.
    """
    t = np.asarray(tau)
    nt = kc.shape[-1]
    return np.where(t < 0, 0.0, kc[..., np.clip(t, 0, nt - 1)])


def window_weights(kc: np.ndarray, phi: int, n: np.ndarray, B: int,
                   t_arrive: int = ARRIVAL_TICK) -> np.ndarray:
    """``w(s_a + n; phi)`` for one cumulative kernel ``kc`` (tick axis last).

    ``n`` is the window index relative to the arrival window ``s_a``.
    """
    na = (t_arrive + phi) // B
    s = np.asarray(na) + np.asarray(n)
    return (kcum_at(kc, B * (s + 1) - phi) - kcum_at(kc, B * s - phi))


def arrival_window_index(phi: int, B: int, t_arrive: int = ARRIVAL_TICK) -> int:
    """``s_a - m`` for an electron crossing the plane at ``t0 = mB + phi``."""
    return int((t_arrive + phi) // B)


def gaussian_cell_weights(center_cell: float, sigma_cells: float,
                          n_cells: int = N_IMPACT):
    """Impact-cell probabilities of a Gaussian transverse profile.

    ``pi_i = int_i^{i+1} N(u; center_cell, sigma_cells) du`` for
    ``i = 0..n_cells-1``, renormalised to sum to one inside the pad.
    Returns ``(pi, outside_fraction)``; ``outside_fraction`` is the mass that
    falls beyond the pad edges and is dropped by the renormalisation.  It is
    dropped rather than moved to the neighbouring pad because the quantity
    being distributed is the charge the truth says pad ``p`` COLLECTED --
    charge that landed on a neighbour is already the neighbour's own truth
    row.
    """
    from math import erf, sqrt
    edges = np.arange(n_cells + 1, dtype=np.float64)
    cdf = np.array([0.5 * (1.0 + erf((e - center_cell)
                                     / (sigma_cells * sqrt(2.0))))
                    for e in edges])
    p = np.diff(cdf)
    inside = float(p.sum())
    return p / inside, float(1.0 - inside)


class _Recorder(Algorithm):
    """Collect per-event records, write JSON (and NPZ) at ``finalize``.

    Every output carries ``job_config`` (the resolved YAML with its
    ``_meta.git``) and ``provenance`` (the store's write log), so the file
    replays without the shell history around it.
    """

    def initialize(self, services):
        super().initialize(services)
        self._records: list[dict] = []
        self._arrays: dict = {}
        self._recipe: dict = {}
        self.out_json = self.props.get("out_json")
        self.out_npz = self.props.get("out_npz")

    def _emit(self, store, rec: dict, arrays: dict | None = None):
        self.put(store, self.writes[0], rec)
        self._records.append(rec)
        if arrays:
            self._arrays.update(arrays)
        if not self._recipe:
            self._recipe = {"job_config": store.get("job.config"),
                            "provenance": store.provenance()}

    def finalize(self):
        if not self._records:
            return {}
        body = (self._records[0] if len(self._records) == 1
                else {"events": self._records})
        if self.out_json:
            Path(self.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_json, "w") as fh:
                json.dump({"algorithm": self.name, "result": body,
                           **self._recipe}, fh, indent=1, default=str)
            print(f"[{self.name}] wrote {self.out_json}")
        if self.out_npz and self._arrays:
            Path(self.out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.out_npz, **self._arrays,
                job_config=np.array(json.dumps(self._recipe.get("job_config"),
                                               default=str)),
                provenance=np.array(json.dumps(
                    self._recipe.get("provenance"), default=str)))
            print(f"[{self.name}] wrote {self.out_npz}")
        return body


# ---------------------------------------------------------------------------
@algorithm("KernelStructure")
class KernelStructure(_Recorder):
    """Impact-resolved kernel structure: every table ``algo_plan`` Sec. 9 wants.

    A source algorithm (``reads = ()``): it needs no event, only the response
    file.  See the module docstring for every definition used below.

    Props
    -----
    response : str, optional
        Response NPZ.  Defaults to the ``detector`` service's
        ``response_path`` when the service is configured.
    adc_hold_delay : int
        ``B`` in fine ticks (default 30).
    ring : int
        Largest Chebyshev pixel offset stored impact-resolved in the NPZ
        (default 2).  All pixel-level tables use the full 25x25 grid
        regardless.
    window_range : [lo, hi]
        Windows reported relative to the arrival window (default -5..2).
    group_sizes : list
        ``g`` values for the grouped impact-averaging error (default 1..8).
    neighbour_group_max : int
        Largest ``g`` in the neighbour-column residual table (default 60).
    leading_windows : int
        Largest ``n`` in the leading-part table (default 60).
    out_json, out_npz : str
    """

    reads = ()
    writes = ("kernel.structure",)

    def execute(self, store):
        path = self.props.get("response")
        if path is None:
            path = self.services["detector"].response_path
        B = int(self.props.get("adc_hold_delay", 30))
        ring = int(self.props.get("ring", 2))
        wlo, whi = [int(v) for v in self.props.get("window_range", [-5, 2])]
        gsizes = [int(v) for v in self.props.get("group_sizes", range(1, 9))]
        gmax = int(self.props.get("neighbour_group_max", 60))
        nlead = int(self.props.get("leading_windows", 60))

        R, meta = load_impact_response(str(path))
        dt = meta["time_tick_us"]
        npix, nimp, nt = meta["n_pixel"], meta["npath"], meta["n_tick"]
        c = COLLECTION_PIXEL
        print(f"[{self.name}] response {Path(str(path)).name}: "
              f"{npix}x{nimp} x {npix}x{nimp} x {nt} ticks, "
              f"Delta_t {dt} us, B {B} ticks")

        # ---- time integrals per ring (Sec. 3.2) ---------------------------
        integ = R.sum(axis=-1, dtype=np.float64) * dt        # (25,10,25,10)
        rings = {}
        for r in range(0, min(npix - c, c + 1)):
            vals = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    vals.append(integ[c + dx, :, c + dy, :])
            v = np.stack(vals)
            rings[r] = {"n_pixels": int(v.shape[0]),
                        "min": float(v.min()), "max": float(v.max()),
                        "max_abs": float(np.abs(v).max())}
        total_per_impact = integ.sum(axis=(0, 2))            # (10,10)

        # ---- impact-averaged kernel --------------------------------------
        Kbar = R.mean(axis=(1, 3), dtype=np.float64)         # (25,25,nt)
        Kcum_bar = np.cumsum(Kbar, axis=-1) * dt

        # ---- impact-resolved cumulative in the stored ring ----------------
        sl = slice(c - ring, c + ring + 1)
        Kcum_imp = cumulative(R[sl, :, sl, :, :], dt)        # (2r+1,10,2r+1,10,nt)
        own = Kcum_imp[ring, :, ring, :, :]                  # (10,10,nt)
        own_cur = R[c, :, c, :, :]                           # (10,10,nt)

        # ---- Sec. 3.3 percentile ticks, Sec. 3.4 peak current -------------
        fracs = [0.01, 0.10, 0.50, 0.90, 0.99]
        pct = np.empty((nimp, nimp, len(fracs)), dtype=np.int64)
        for a in range(nimp):
            for b in range(nimp):
                cc = own[a, b]
                for q, fr in enumerate(fracs):
                    pct[a, b, q] = int(np.searchsorted(cc, fr)) - ARRIVAL_TICK
        peak = own_cur.max(axis=-1)
        peak_tick = own_cur.argmax(axis=-1) - ARRIVAL_TICK

        # ---- window tables ------------------------------------------------
        ns = np.arange(wlo, whi + 1)
        phis = np.arange(B)
        offs = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 2)]
        wbar = np.zeros((len(offs), B, len(ns)))
        wimp = np.zeros((len(offs), B, nimp, nimp, len(ns)))
        for io, (dx, dy) in enumerate(offs):
            kb = Kcum_bar[c + dx, c + dy]
            ki = Kcum_imp[ring + dx, :, ring + dy, :, :]
            for phi in phis:
                wbar[io, phi] = window_weights(kb, int(phi), ns, B)
                wimp[io, phi] = window_weights(ki, int(phi), ns, B)
        dip = wimp - wbar[:, :, None, None, :]

        # sum_i d = 0 exactly by construction; sum_s d over the FULL support
        # is the difference of two time integrals
        sum_i_resid = float(np.abs(dip.mean(axis=(2, 3))).max())
        full_n = np.arange(-(nt // B) - 2, 3)
        sum_s_resid = {}
        for io, (dx, dy) in enumerate(offs):
            kb = Kcum_bar[c + dx, c + dy]
            ki = Kcum_imp[ring + dx, :, ring + dy, :, :]
            wb = window_weights(kb, 0, full_n, B).sum()
            wi = window_weights(ki, 0, full_n, B).sum(axis=-1)
            sum_s_resid[f"({dx},{dy})"] = float(np.abs(wi - wb).max())

        # ---- grouped impact-averaging error -------------------------------
        grouping = {}
        for io, (dx, dy) in enumerate(offs[:3]):
            kb = Kcum_bar[c + dx, c + dy]
            ki = Kcum_imp[ring + dx, :, ring + dy, :, :]
            dfull = np.zeros((B, nimp, nimp, len(full_n)))
            for phi in phis:
                dfull[phi] = (window_weights(ki, int(phi), full_n, B)
                              - window_weights(kb, int(phi), full_n, B))
            tab = {}
            for g in gsizes:
                for e in range(g):
                    # block index of window n is floor((n + e) / g)
                    blk = (full_n + e) // g
                    ub = np.unique(blk)
                    acc = np.zeros((B, nimp, nimp, len(ub)))
                    for q, m in enumerate(ub):
                        acc[..., q] = dfull[..., blk == m].sum(axis=-1)
                    Dge = np.abs(acc).max(axis=-1)            # (B,10,10)
                    tab[f"g{g}_e{e}"] = {
                        "max_over_impact_phase": float(Dge.max()),
                        "max_over_impact_at_phi0": float(Dge[0].max()),
                        "mean_over_impact_phase": float(Dge.mean())}
                per_phi_best = np.array(
                    [min(tab[f"g{g}_e{e}"]["max_over_impact_phase"]
                         for e in range(g))])
                # pad-anchored: e may be chosen knowing phi, not the impact
                Dall = np.stack([
                    np.abs(np.stack([
                        dfull[..., ((full_n + e) // g) == m].sum(axis=-1)
                        for m in np.unique((full_n + e) // g)], axis=-1)
                    ).max(axis=-1) for e in range(g)])         # (g,B,10,10)
                fixed_worst = float(Dall.max())
                pad_anchored = float(Dall.max(axis=(2, 3)).min(axis=0).max())
                tab[f"g{g}_summary"] = {
                    "fixed_common_grid_worst": fixed_worst,
                    "pad_anchored_best": pad_anchored,
                    "best_e_ignoring_phase": float(per_phi_best[0])}
            grouping[f"({dx},{dy})"] = tab

        # ---- neighbour-column residual ------------------------------------
        neigh = {}
        for dx, dy in offs[1:]:
            kb = Kcum_bar[c + dx, c + dy]
            rows0, rows_min, rows_max = [], [], []
            for g in range(1, gmax + 1):
                nn = np.arange(-g + 1, 1)
                vals = np.array([window_weights(kb, int(p), nn, B).sum()
                                 for p in phis])
                rows0.append(float(vals[0]))
                rows_min.append(float(vals.min()))
                rows_max.append(float(vals.max()))
            neigh[f"({dx},{dy})"] = {"g": list(range(1, gmax + 1)),
                                     "phi0": rows0, "min_over_phi": rows_min,
                                     "max_over_phi": rows_max}
        # The plan (Sec. 9) asks for the "zero crossing" of the ring-1
        # cumulative.  Measured: the impact-AVERAGED ring-1 cumulative has no
        # sign change -- it rises to a peak and decays back to its (zero)
        # integral from above.  What exists instead is a decay time, reported
        # as the ticks at which it falls to 10 / 1 / 0.1 % of its peak.
        kb10 = Kcum_bar[c + 1, c]
        pk = int(np.argmax(kb10))
        decay = {}
        for fr in (0.10, 0.01, 0.001):
            tail = np.flatnonzero(kb10[pk:] < fr * kb10[pk])
            decay[f"to_{fr:g}_of_peak_tick_rel_arrival"] = (
                int(tail[0] + pk) - ARRIVAL_TICK if tail.size else None)
        ring1_shape = {
            "peak_value": float(kb10[pk]),
            "peak_tick_rel_arrival": pk - ARRIVAL_TICK,
            "min_value_over_ticks": float(kb10.min()),
            "sign_change": bool(kb10.min() < -1e-9 * kb10[pk]),
            "impact_resolved_min_over_impacts": float(
                Kcum_imp[ring + 1, :, ring, :, :].min()),
            **decay}

        # ---- arrival-window fraction vs phase -----------------------------
        kb00 = Kcum_bar[c, c]
        ki00 = Kcum_imp[ring, :, ring, :, :]
        f0 = np.array([float(window_weights(kb00, int(p), np.array([0]), B)[0])
                       for p in phis])
        f1 = np.array([float(window_weights(kb00, int(p), np.array([-1]), B)[0])
                       for p in phis])
        f0i = np.stack([window_weights(ki00, int(p), np.array([0]), B)[..., 0]
                        for p in phis])
        f1i = np.stack([window_weights(ki00, int(p), np.array([-1]), B)[..., 0]
                        for p in phis])
        # phi is periodic: at phi >= B - (ARRIVAL_TICK mod B) the arrival
        # window index steps by one and w(s_a) restarts.  The slope is only
        # meaningful on the branch where s_a - m is constant.
        na_of_phi = np.array([arrival_window_index(int(p), B) for p in phis])
        branch = na_of_phi == na_of_phi[0]
        pb = phis[branch]
        slope = float(np.polyfit(pb, f0[branch], 1)[0])

        # ---- leading-part fraction ----------------------------------------
        na0 = arrival_window_index(0, B)
        lead = [float(kcum_at(kb00, B * (na0 - n))) for n in range(1, nlead + 1)]
        after = float(1.0 - kcum_at(kb00, B * (na0 + 1)))

        rec = {
            "response": meta, "adc_hold_delay": B,
            "arrival_tick": ARRIVAL_TICK, "collection_pixel": c,
            "kcum_convention": "inclusive upper limit, matches tred cumsum",
            "ring_stored_impact_resolved": ring,
            "time_integral_by_ring": rings,
            "total_integral_over_25x25_per_impact": {
                "min": float(total_per_impact.min()),
                "max": float(total_per_impact.max()),
                "impact_mean": float(total_per_impact.mean()),
                "matrix": total_per_impact.round(6).tolist()},
            "own_percentile_ticks_rel_arrival": {
                "fractions": fracs,
                "impact_4_4": pct[4, 4].tolist(),
                "impact_0_0": pct[0, 0].tolist(),
                "min_over_impacts": pct.min(axis=(0, 1)).tolist(),
                "median_over_impacts": np.median(
                    pct.reshape(-1, len(fracs)), axis=0).tolist(),
                "max_over_impacts": pct.max(axis=(0, 1)).tolist()},
            "own_peak_current": {
                "unit": "electron charge per microsecond",
                "matrix": peak.round(4).tolist(),
                "tick_rel_arrival": peak_tick.tolist(),
                "min": float(peak.min()), "max": float(peak.max())},
            "window_range_rel_arrival": [wlo, whi],
            "window_weights_impact_averaged": {
                f"({dx},{dy})": {f"phi{p}": wbar[io, p].round(6).tolist()
                                 for p in (0, 1, 15, 29)}
                for io, (dx, dy) in enumerate(offs)},
            "window_weights_selected_impacts_phi1": {
                f"({dx},{dy})": {
                    f"impact({a},{b})": wimp[io, 1, a, b].round(6).tolist()
                    for (a, b) in [(4, 4), (2, 2), (0, 4), (0, 0), (9, 4)]}
                for io, (dx, dy) in enumerate(offs)},
            "dipole_checks": {
                "max_abs_mean_over_impacts": sum_i_resid,
                "max_abs_sum_over_windows_full_support": sum_s_resid},
            "grouped_impact_averaging_error": grouping,
            "neighbour_column_residual": neigh,
            "ring1_side_cumulative_shape": ring1_shape,
            "arrival_window_fraction_vs_phase": {
                "phi": phis.tolist(),
                "wbar_arrival": f0.round(6).tolist(),
                "wbar_arrival_minus_1": f1.round(6).tolist(),
                "impact_min_arrival": f0i.min(axis=(1, 2)).round(6).tolist(),
                "impact_max_arrival": f0i.max(axis=(1, 2)).round(6).tolist(),
                "impact_min_arrival_minus_1": f1i.min(axis=(1, 2)).round(6).tolist(),
                "impact_max_arrival_minus_1": f1i.max(axis=(1, 2)).round(6).tolist(),
                "arrival_window_index_vs_phi": na_of_phi.tolist(),
                "monotone_branch_phi": pb.tolist(),
                "slope_per_phase_tick_on_branch": slope,
                "branch_range": [float(f0[branch].min()),
                                 float(f0[branch].max())]},
            "leading_part_fraction_phi0": {
                "n_windows": list(range(1, nlead + 1)), "L": lead,
                "fraction_after_arrival_window": after},
        }
        self._print_tables(rec, offs, wbar, wimp)
        self._emit(store, rec, {
            "Kcum_imp": Kcum_imp.astype(np.float32),
            "Kcum_bar": Kcum_bar,
            "w_bar": wbar, "w_imp": wimp.astype(np.float32),
            "dipole": dip.astype(np.float32),
            "window_offsets": np.array(offs),
            "window_n": ns, "phi": phis,
            "peak_current": peak, "peak_tick": peak_tick,
            "percentile_ticks": pct,
            "total_integral_per_impact": total_per_impact,
        })

    # -- printing -----------------------------------------------------------
    def _print_tables(self, rec, offs, wbar, wimp):
        n = self.name
        doc35 = {"impact-averaged": [0.043, 0.123, 0.605, 0.181],
                 "impact (2,2)": [0.045, 0.132, 0.687, 0.087],
                 "impact (0,4)": [0.041, 0.111, 0.511, 0.291],
                 "impact (0,0)": [0.033, 0.073, 0.229, 0.622]}
        wlo = rec["window_range_rel_arrival"][0]
        i3 = slice(-3 - wlo, 1 - wlo)
        print(f"[{n}] FORWARD_MODEL Sec. 3.5, collecting pad, windows -3..0, "
              f"arrival phase phi=1")
        mine = {"impact-averaged": wbar[0, 1][i3],
                "impact (2,2)": wimp[0, 1, 2, 2][i3],
                "impact (0,4)": wimp[0, 1, 0, 4][i3],
                "impact (0,0)": wimp[0, 1, 0, 0][i3]}
        for k, v in doc35.items():
            print(f"    {k:16s} doc {v}  measured "
                  f"{[round(float(z), 4) for z in mine[k]]}")
        doc35b = {"impact-averaged": [0.017, 0.023, -0.037, -0.037],
                  "impact (4,4)": [0.015, 0.012, -0.054, -0.006],
                  "impact (9,4)": [0.037, 0.088, 0.003, -0.172]}
        mineb = {"impact-averaged": wbar[1, 1][i3],
                 "impact (4,4)": wimp[1, 1, 4, 4][i3],
                 "impact (9,4)": wimp[1, 1, 9, 4][i3]}
        print(f"[{n}] FORWARD_MODEL Sec. 3.5, ring-1 side (+1,0)")
        for k, v in doc35b.items():
            print(f"    {k:16s} doc {v}  measured "
                  f"{[round(float(z), 4) for z in mineb[k]]}")
        print(f"[{n}] FORWARD_MODEL Sec. 3.6(c) grouped error, own pad "
              f"(doc, 3 tabulated impacts: g1 0.376, g2 aligned 0.065, "
              f"g2 misaligned 0.426, g3 0.015, g4 0.005)")
        tab = rec["grouped_impact_averaging_error"]["(0,0)"]
        for g in sorted({int(k[1:].split('_')[0])
                         for k in tab if k.endswith("summary")}):
            s = tab[f"g{g}_summary"]
            print(f"    g={g}: fixed-common-grid worst "
                  f"{s['fixed_common_grid_worst']:.4f}   pad-anchored best "
                  f"{s['pad_anchored_best']:.4f}")
        nb = rec["neighbour_column_residual"]["(1,0)"]
        print(f"[{n}] neighbour-column residual (1,0), doc -0.034 at g=4: "
              f"g=1 {nb['phi0'][0]:+.4f}  g=4 {nb['phi0'][3]:+.4f}  "
              f"g=10 {nb['phi0'][9]:+.4f}  g=60 {nb['phi0'][59]:+.4f}  "
              f"(range over phi at g=4 "
              f"{nb['min_over_phi'][3]:+.4f}..{nb['max_over_phi'][3]:+.4f})")
        r1 = rec["ring1_side_cumulative_shape"]
        print(f"[{n}] ring-1 side cumulative: peak {r1['peak_value']:.4f} at "
              f"tick {r1['peak_tick_rel_arrival']:+d} rel arrival; sign change "
              f"{r1['sign_change']}; min over ticks "
              f"{r1['min_value_over_ticks']:+.3e}; falls to 10/1/0.1% of peak "
              f"at {r1['to_0.1_of_peak_tick_rel_arrival']}/"
              f"{r1['to_0.01_of_peak_tick_rel_arrival']}/"
              f"{r1['to_0.001_of_peak_tick_rel_arrival']} ticks rel arrival")
        aw = rec["arrival_window_fraction_vs_phase"]
        print(f"[{n}] arrival-window fraction wbar(s_a): phi=0 "
              f"{aw['wbar_arrival'][0]:.4f} -> phi="
              f"{aw['monotone_branch_phi'][-1]} "
              f"{aw['wbar_arrival'][aw['monotone_branch_phi'][-1]]:.4f} "
              f"(the branch on which s_a does not step), slope "
              f"{aw['slope_per_phase_tick_on_branch']:.5f}/tick; "
              f"impact band at phi=0 "
              f"{aw['impact_min_arrival'][0]:.4f}.."
              f"{aw['impact_max_arrival'][0]:.4f}")
        lp = rec["leading_part_fraction_phi0"]
        print(f"[{n}] leading part L(n): L(1) {lp['L'][0]:.4f}  L(4) "
              f"{lp['L'][3]:.5f}  L(10) {lp['L'][9]:.6f}  L(60) "
              f"{lp['L'][59]:.3e}; after the arrival window "
              f"{lp['fraction_after_arrival_window']:.3e}")


# ---------------------------------------------------------------------------
@algorithm("FineTruthClosure")
class FineTruthClosure(_Recorder):
    """Is the fine-tick pad-level truth a solution of the exact-functional model?

    Builds every record's row at the record's own sample times from the fine
    truth (module docstring, ``yhat_p[k]``) and reports the residual
    ``eps_p[k] = y_p[k] - yhat_p[k]`` structurally: by pad ring, by window,
    grouped on the common record grid, and against a scan of the truth's tick
    alignment.  Four forward models are evaluated:

    ``bar``        impact-averaged kernel, no waveform truncation -- the model
                   of ``FORWARD_MODEL_revised.md`` Sec. 4.1.
    ``bar_trunc``  the same with tred's ``current_zero_before_tick`` deletion.
    ``mix``        the impact-resolved kernel weighted by the derived
                   transverse impact distribution of this sample.
    ``mix_trunc``  the same with the deletion.

    Props
    -----
    out_json, out_npz : str
    shift_scan : list of int
        Truth tick shifts for the alignment scan (default -15..15).
    ring_predict : int
        Largest Chebyshev pixel offset kept in the prediction (default 12,
        the full 25x25).
    group_sizes : list
        ``g`` for the grouped residual (default 1..8).
    transverse_model : {'none', 'isoline'}
        ``isoline`` adds the impact-resolved models.  The impact distribution
        is DERIVED, not measured: a Gaussian across the pad in ``pixel_x``
        with ``y_center_cell`` and ``sigma_cells``, uniform (1/10) in
        ``pixel_y``.
    y_center_cell, sigma_cells : float
        The derived transverse profile, in impact cells.  Defaults 5.789 and
        1.18 for the isoline sample at 16.5 cm drift.
    current_zero_before_tick : int
        The absolute fine tick before which tred deletes the current
        (``graph_effq.py:148-159``); default 0.
    line_pixel_y_range : [lo, hi]
        Pads used for the along-the-line residual tables, chosen to exclude
        the line ends where the uniform-in-``pixel_y`` impact model fails
        (default [5, 131]).
    """

    reads = ("event", "readout_config", "hits_view")
    writes = ("closure.record",)

    def execute(self, store):
        ev = store.get("event")
        rc = store.get("readout_config")
        hv = store.get("hits_view")
        B = int(rc.adc_hold_delay)
        N = int(hv.nburst)
        shifts = [int(v) for v in self.props.get(
            "shift_scan", range(-15, 16))]
        rpred = int(self.props.get("ring_predict", 12))
        gsizes = [int(v) for v in self.props.get("group_sizes", range(1, 9))]
        tmodel = str(self.props.get("transverse_model", "none"))
        ycen = float(self.props.get("y_center_cell", 5.789))
        sigc = float(self.props.get("sigma_cells", 1.18))
        tz = self.props.get("current_zero_before_tick", 0)
        tz = None if tz is None else int(tz)
        ylo, yhi = [int(v) for v in self.props.get(
            "line_pixel_y_range", [5, 131])]

        path = self.props.get("response")
        if path is None:
            path = self.services["detector"].response_path
        R, meta = load_impact_response(str(path))
        dt = meta["time_tick_us"]
        npix, nimp, nt = meta["n_pixel"], meta["npath"], meta["n_tick"]
        c = COLLECTION_PIXEL
        Kbar = np.cumsum(R.mean(axis=(1, 3), dtype=np.float64), axis=-1) * dt

        pi_x, outside = gaussian_cell_weights(ycen, sigc, nimp)
        pi_y = np.full(nimp, 1.0 / nimp)
        Kmix = None
        if tmodel == "isoline":
            Kmix = np.cumsum(np.einsum("aibjt,i,j->abt",
                                       R.astype(np.float64), pi_x, pi_y),
                             axis=-1) * dt
        del R

        # ---- records ------------------------------------------------------
        loc = np.asarray(hv.location)
        trig = np.unique(hv.trigger)
        if trig.size != 1:
            raise ValueError(f"expected one common trigger tick, got {trig}")
        trig = int(trig[0])
        Cq = np.asarray(hv.cumulative_charges, dtype=np.float64)
        y = np.diff(np.concatenate([np.zeros((len(Cq), 1)), Cq], axis=1),
                    axis=1)
        latch = trig + np.arange(N + 1) * B

        # ---- fine truth ---------------------------------------------------
        el = np.asarray(ev.effq.location).astype(np.int64)
        eq = np.asarray(ev.effq.data, dtype=np.float64)[:, -1]
        ticks = np.unique(el[:, 2])
        tix = np.searchsorted(ticks, el[:, 2])
        px_lo = int(min(loc[:, 0].min(), el[:, 0].min() - rpred))
        px_hi = int(max(loc[:, 0].max(), el[:, 0].max() + rpred))
        py_lo = int(min(loc[:, 1].min(), el[:, 1].min() - rpred))
        py_hi = int(max(loc[:, 1].max(), el[:, 1].max() + rpred))
        nx, ny = px_hi - px_lo + 1, py_hi - py_lo + 1
        X = np.zeros((nx, ny, len(ticks)))
        np.add.at(X, (el[:, 0] - px_lo, el[:, 1] - py_lo, tix), eq)
        truth_by_pad = X.sum(axis=2)

        print(f"[{self.name}] {len(loc)} recorded pads, {len(el)} truth cells "
              f"on {int((truth_by_pad > 0).sum())} pads, {len(ticks)} distinct "
              f"ticks {ticks.min()}..{ticks.max()}, trigger {trig}, B {B}, "
              f"nburst {N}")

        def accumulator(K, shift=0):
            """Predicted CSA accumulator ``A_p(t)`` on the latch grid.

            Returns ``(nx, ny, N+2)``: columns ``0..N`` are the latches
            ``l_0..l_N``, column ``N+1`` is the instant ``t_z - 1`` used by
            the truncated model.
            """
            times = np.concatenate([latch, [(-10 ** 9) if tz is None
                                            else tz - 1]])
            arg = times[None, :] - (ticks[:, None] + shift)
            ok = arg >= 0
            ac = np.clip(arg, 0, nt - 1)
            A = np.zeros((nx, ny, len(times)))
            for dx in range(-rpred, rpred + 1):
                for dy in range(-rpred, rpred + 1):
                    kc = np.where(ok, K[c + dx, c + dy][ac], 0.0)
                    if np.abs(kc).max() < 1e-16:
                        continue
                    x0, x1 = max(0, dx), nx + min(0, dx)
                    y0, y1 = max(0, dy), ny + min(0, dy)
                    src = X[x0 - dx:x1 - dx, y0 - dy:y1 - dy, :]
                    A[x0:x1, y0:y1, :] += (
                        src.reshape(-1, len(ticks)) @ kc
                    ).reshape(x1 - x0, y1 - y0, len(times))
            return A

        def windows_from(A, truncate):
            """``yhat`` from a predicted accumulator, with or without the cut."""
            Acc = A[:, :, :N + 1].copy()
            if truncate:
                if tz is None:
                    raise ValueError("current_zero_before_tick is null")
                ped = A[:, :, N + 1]
                below = latch < tz
                Acc = Acc - ped[:, :, None]
                Acc[:, :, below] = 0.0
            return np.diff(Acc, axis=2)

        pads = (loc[:, 0] - px_lo, loc[:, 1] - py_lo)
        models = {}
        A_bar = accumulator(Kbar)
        models["bar"] = windows_from(A_bar, False)[pads]
        if tz is not None:
            models["bar_trunc"] = windows_from(A_bar, True)[pads]
        if Kmix is not None:
            A_mix = accumulator(Kmix)
            models["mix"] = windows_from(A_mix, False)[pads]
            if tz is not None:
                models["mix_trunc"] = windows_from(A_mix, True)[pads]

        # ---- pad classification -------------------------------------------
        # Chebyshev and Manhattan distance from each recorded pad to the
        # NEAREST pad carrying truth charge.  ring-1 side = Chebyshev 1 and
        # Manhattan 1; ring-1 diagonal = Chebyshev 1 and Manhattan 2.
        tpads = np.argwhere(truth_by_pad > 0)
        delta = np.abs(tpads[None, :, :] - np.stack(pads, axis=1)[:, None, :])
        chid = delta.max(axis=2).min(axis=1)
        manh = delta.sum(axis=2).min(axis=1)
        inspan = (loc[:, 1] >= ylo) & (loc[:, 1] <= yhi)
        line = chid == 0
        side_all = (chid == 1) & (manh == 1)
        diag_all = (chid == 1) & (manh == 2)
        r2_all = chid == 2
        # residual tables use the same classes restricted to the pixel_y span
        # away from the line ends, where the uniform-in-pixel_y part of the
        # derived impact model holds.  Note that for a straight line running
        # along pixel_y the ring-1 DIAGONAL class is empty inside the span:
        # every pad diagonally adjacent to one line pad is side-adjacent to
        # the next one, so its Chebyshev/Manhattan pair is (1, 1).
        span = line & inspan
        side = side_all & inspan
        diag = diag_all & inspan
        r2 = r2_all & inspan

        # ---- 1. pre-signal zero -------------------------------------------
        first_nz = np.array([int(np.argmax(np.abs(r) > 0)) if np.any(r) else -1
                             for r in Cq])
        rec: dict = {
            "pre_signal": {
                "max_abs_q1_ke": float(np.abs(Cq[:, 0]).max()),
                "first_nonzero_latch_k": sorted(
                    {int(v) + 1 for v in first_nz}),
                "first_nonzero_latch_tick": sorted(
                    {int(trig + (v + 1) * B) for v in first_nz})},
        }

        # ---- 2. charge closure --------------------------------------------
        fin = Cq[:, -1]
        tr_pad = truth_by_pad[pads]
        line_close = fin[line] - tr_pad[line]
        ring_sums = {}
        for r in range(0, 6):
            m = chid == r
            ring_sums[f"ring{r}"] = {"n_pads": int(m.sum()),
                                     "final_accumulator_ke": float(fin[m].sum())}
        m = chid >= 6
        ring_sums["ring_ge6"] = {"n_pads": int(m.sum()),
                                 "final_accumulator_ke": float(fin[m].sum())}
        pred_ring = {}
        for name, yh in models.items():
            pred_ring[name] = {f"ring{r}": float(yh[chid == r].sum())
                               for r in range(0, 6)}
            pred_ring[name]["ring_ge6"] = float(yh[chid >= 6].sum())
            pred_ring[name]["total"] = float(yh.sum())
        rec["charge_closure"] = {
            "truth_total_ke": float(eq.sum()),
            "record_total_ke": float(y.sum()),
            "difference_ke": float(y.sum() - eq.sum()),
            "difference_frac": float((y.sum() - eq.sum()) / eq.sum()),
            "truth_pads_missing_from_records": int(
                (truth_by_pad > 0).sum() - line.sum()),
            "line_pads": {
                "n": int(line.sum()),
                "sum_final_minus_truth_ke": float(line_close.sum()),
                "mean_final_minus_truth_ke": float(line_close.mean()),
                "mean_relative": float(
                    (line_close / np.maximum(tr_pad[line], 1e-12)).mean())},
            "measured_final_accumulator_by_ring": ring_sums,
            "predicted_window_sum_by_ring": pred_ring,
        }

        # ---- 3. monotonicity ----------------------------------------------
        neg = y < 0
        mono = {}
        for lab, m in (("line", line), ("ring1_side", side_all),
                       ("ring1_diagonal", diag_all), ("ring2", r2_all)):
            mono[lab] = {"n_pads": int(m.sum()),
                         "n_negative_windows": int(neg[m].sum()),
                         "n_windows": int(m.sum() * N),
                         "most_negative_ke": float(y[m].min()) if m.any()
                         else 0.0}
        worst = int(np.argmin(y.min(axis=1)))
        wk = int(np.argmin(y[worst]))
        nb_charge = float(np.abs(y[:, wk]).max())
        mono["most_negative_overall"] = {
            "pixel_x": int(loc[worst, 0]), "pixel_y": int(loc[worst, 1]),
            "k": wk, "latch_tick": int(latch[wk + 1]),
            "value_ke": float(y[worst, wk]),
            "largest_window_charge_same_k_ke": nb_charge,
            "fraction_of_that": float(y[worst, wk] / max(nb_charge, 1e-12))}
        rec["monotonicity"] = mono

        # ---- 4. residual tables -------------------------------------------
        ka = int(np.argmax(np.abs(y).sum(axis=0)))
        kwin = [k for k in range(max(ka - 5, 0), min(ka + 3, N))]
        kearly = list(range(0, 9))
        # charge scale used to normalise the residual tables: the median pad
        # charge among pads carrying more than 1% of the largest pad's charge
        lc = tr_pad[line]
        big = lc[lc > 0.01 * max(lc.max(), 1e-12)]
        pad_scale = float(np.median(big)) if big.size else 1.0
        resid = {}
        for name, yh in models.items():
            eps = y - yh
            entry = {
                "total_abs_eps_ke": float(np.abs(eps).sum()),
                "total_abs_y_ke": float(np.abs(y).sum()),
                "abs_eps_over_abs_y": float(np.abs(eps).sum()
                                            / np.abs(y).sum()),
                "sum_eps_ke": float(eps.sum()),
                "max_abs_eps_ke": float(np.abs(eps).max()),
                "windows_around_arrival": [int(k) for k in kwin],
                "arrival_window_k": ka,
            }
            for lab, m in (("line_span", span), ("ring1_side", side),
                           ("ring1_diagonal", diag), ("ring2", r2)):
                if not m.any():
                    continue
                e = eps[m]
                entry[lab] = {
                    "n_pads": int(m.sum()),
                    "mean_ke": [float(v) for v in e[:, kwin].mean(axis=0)],
                    "rms_ke": [float(v) for v in
                               np.sqrt((e[:, kwin] ** 2).mean(axis=0))],
                    "mean_over_pad_charge": [float(v / pad_scale)
                                             for v in e[:, kwin].mean(axis=0)],
                    "rms_over_pad_charge": [
                        float(v / pad_scale) for v in
                        np.sqrt((e[:, kwin] ** 2).mean(axis=0))],
                    "mean_ke_early_k0_8": [float(v) for v in
                                           e[:, kearly].mean(axis=0)],
                    "rms_all_windows_ke": float(np.sqrt((e ** 2).mean())),
                    "max_abs_ke": float(np.abs(e).max())}
            resid[name] = entry
        rec["pad_charge_scale_ke"] = pad_scale
        rec["residual"] = resid

        # ---- 5. grouped residual on the common grid ------------------------
        grouped = {}
        for name, yh in models.items():
            eps = y - yh
            tab = {}
            for g in gsizes:
                for e0 in range(g):
                    blk = (np.arange(N) - e0) // g
                    ub = np.unique(blk)
                    acc = np.stack([eps[:, blk == m].sum(axis=1) for m in ub],
                                   axis=1)
                    a = np.abs(acc)
                    tab[f"g{g}_e{e0}"] = {
                        "max_over_line_pads_ke": float(a[span].max()),
                        "mean_over_line_pads_ke": float(
                            a[span].max(axis=1).mean())}
                tab[f"g{g}_summary"] = {
                    "worst_over_e_ke": max(tab[f"g{g}_e{e0}"]
                                           ["max_over_line_pads_ke"]
                                           for e0 in range(g)),
                    "best_over_e_ke": min(tab[f"g{g}_e{e0}"]
                                          ["max_over_line_pads_ke"]
                                          for e0 in range(g))}
            grouped[name] = tab
        rec["grouped_residual"] = grouped

        # ---- 6. alignment scan ---------------------------------------------
        scan = []
        kernels = [("bar", Kbar)] + ([("mix", Kmix)] if Kmix is not None
                                     else [])
        for s in shifts:
            row = {"shift_ticks": int(s)}
            for base, K in kernels:
                A = accumulator(K, shift=s)
                for trunc, suffix in ((False, ""), (True, "_trunc")):
                    if trunc and tz is None:
                        continue
                    e = y - windows_from(A, trunc)[pads]
                    row[base + suffix] = {
                        "total_abs_eps_ke": float(np.abs(e).sum()),
                        "rms_eps_ke": float(np.sqrt((e ** 2).mean())),
                        "line_span_rms_eps_ke": float(
                            np.sqrt((e[span] ** 2).mean()))}
            scan.append(row)
        labels = [k for k in scan[0] if k != "shift_ticks"]
        best = {lab: min(scan, key=lambda r: r[lab]["total_abs_eps_ke"])
                ["shift_ticks"] for lab in labels}
        rec["alignment_scan"] = {"rows": scan, "argmin_shift_ticks": best}

        # ---- 7. impact-resolved reduction -----------------------------------
        rec["transverse_model"] = {
            "model": tmodel, "y_center_cell": ycen, "sigma_cells": sigc,
            "pi_x": pi_x.round(8).tolist(),
            "outside_pad_fraction_dropped": outside,
            "pi_y": "uniform 1/10",
            "status": "DERIVED from the sample geometry, not measured"}
        if "mix" in models:
            for a, b in (("bar", "mix"), ("bar_trunc", "mix_trunc")):
                if a in resid and b in resid:
                    rec["residual"][b]["fraction_of_" + a
                                       + "_residual_removed"] = float(
                        1.0 - resid[b]["total_abs_eps_ke"]
                        / resid[a]["total_abs_eps_ke"])

        self._print(rec, models, y)
        arrays = {"y": y, "loc": loc, "latch": latch,
                  "chebyshev_to_truth": chid,
                  "truth_by_pad": truth_by_pad,
                  "truth_ticks": ticks,
                  "pad_index_offset": np.array([px_lo, py_lo])}
        for name, yh in models.items():
            arrays["yhat_" + name] = yh.astype(np.float32)
            arrays["eps_" + name] = (y - yh).astype(np.float32)
        self._emit(store, rec, arrays)

    def _print(self, rec, models, y):
        n = self.name
        cc = rec["charge_closure"]
        print(f"[{n}] pre-signal max|q_1| {rec['pre_signal']['max_abs_q1_ke']:.3e}"
              f" ke; first non-zero latch k "
              f"{rec['pre_signal']['first_nonzero_latch_k']} "
              f"(tick {rec['pre_signal']['first_nonzero_latch_tick']})")
        print(f"[{n}] charge: truth {cc['truth_total_ke']:.3f} ke, records "
              f"{cc['record_total_ke']:.3f} ke, difference "
              f"{cc['difference_ke']:+.3f} ke = "
              f"{100 * cc['difference_frac']:+.3f}% | line pads carry "
              f"{cc['line_pads']['sum_final_minus_truth_ke']:+.3f} ke of it, "
              f"non-line pads "
              f"{cc['difference_ke'] - cc['line_pads']['sum_final_minus_truth_ke']:+.3f}")
        for k, v in cc["measured_final_accumulator_by_ring"].items():
            print(f"    {k:10s} n={v['n_pads']:5d} measured final "
                  f"{v['final_accumulator_ke']:+9.4f} ke   predicted "
                  + "  ".join(
                      f"{m}={rec['charge_closure']['predicted_window_sum_by_ring'][m].get(k, float('nan')):+8.4f}"
                      for m in models))
        mono = rec["monotonicity"]
        print(f"[{n}] negative window charges: "
              + ", ".join(f"{k} {mono[k]['n_negative_windows']}/"
                          f"{mono[k]['n_windows']}"
                          for k in ("line", "ring1_side", "ring1_diagonal",
                                    "ring2"))
              + f" | most negative {mono['most_negative_overall']['value_ke']:.4f} ke"
                f" = {mono['most_negative_overall']['fraction_of_that']:.3f} of the"
                f" largest window charge at the same latch")
        for name in models:
            r = rec["residual"][name]
            print(f"[{n}] model {name:10s} sum|eps| {r['total_abs_eps_ke']:9.3f} ke"
                  f" = {100 * r['abs_eps_over_abs_y']:7.4f}% of sum|y|; "
                  f"sum eps {r['sum_eps_ke']:+8.3f} ke; max|eps| "
                  f"{r['max_abs_eps_ke']:.4f} ke; line-span RMS "
                  f"{r['line_span']['rms_all_windows_ke']:.5f} ke")
        print(f"[{n}] alignment scan argmin "
              f"{rec['alignment_scan']['argmin_shift_ticks']} ticks "
              f"(a minimum away from 0 is reported, not corrected)")
