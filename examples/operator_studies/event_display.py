"""Pixel-vs-time event display: smeared truth (blue) over reco (red).

Built from the universal-grid rebin, so the truth is the once-smeared
truth and the reco is the deposited solution -- the same pair the
acceptance metrics use, no second smearing anywhere.

The main panel composites the two fields additively in RGB: truth-only
cells go blue, reco-only cells go red, cells where both agree go dark.
Raw hits are overlaid as an INDICATOR only, and their placement is a
FORCED alignment, not a measurement: a hit is a window integral, so
collapsing it to its trigger time (or spreading it over its window) is a
convention, and the two choices give fall/rise HWHM 0.117 and 0.299 on
the same event.  Only the peak position survives the choice (-1.7 and
-1.5 us against the truth), so the hits are shifted by whole bins to put
their peak on the truth peak and the legend states both the lag that
does it and the nominal response lag.  No asymmetry number may be quoted
for the raw hits.

The marginals are the charge profiles along each axis, and the time
marginal carries the asymmetry numbers for truth and reco only, since
that profile is what the positron-shower argument rests on.

Time is the response-plane clock that effq and deconv_q share, so charge
times are negative.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/src")
AO = f"{ROOT}/examples/analysis_output"

from unfoldlarpix.eval.universal import universal_rebin  # noqa: E402

PITCH_CM = 0.4434
TBIN_US = 1.5
BLUE = "#1a52a8"
RED = "#c2270a"
GREEN = "#1a7f37"


def universal_offsets(npz_path):
    """Map a q-block index onto the universal grid used by the metrics.

    Replicates the origin arithmetic of eval.universal.universal_rebin:
    q bin k lands at universal index ``k + dt``, pixel (px, py) of the
    block at ``(px + dx, py + dy)``.
    """
    f = np.load(npz_path, allow_pickle=True)
    B = int(f["adc_hold_delay"])
    s_off = np.asarray(f["smear_offset"], dtype=np.int64)
    b_off = np.asarray(f["boffset"], dtype=np.float64)
    ntq = np.asarray(f["deconv_q"]).shape[2]
    i0 = np.floor((b_off[2] + (np.arange(ntq) + 0.5) * B) / B
                  - 0.5).astype(np.int64)
    u_min = int(min(i0.min(), int(s_off[2] // B)))
    p_min = (min(int(b_off[0]), int(s_off[0])),
             min(int(b_off[1]), int(s_off[1])))
    return (int(b_off[0]) - p_min[0], int(b_off[1]) - p_min[1],
            int(i0[0]) - u_min, B, u_min)


def response_lag(op):
    """Bins between a charge's q-time and where the operator puts it.

    The q grid's time coordinate is the DRIFT-START time, and the 25x25
    field response covers the whole drift path, so a delta at q bin k
    produces data from k+48 to k+127 bins later (the long-range induction
    turns on well before collection).  Two conventions are returned: the
    lag of the response peak, and its charge-weighted centroid.  Neither
    is 'the' arrival time -- a hit is a window integral over a response
    80 bins long, so placing it at a single bin is a convention and the
    raw-hit curve below is an indicator, not an estimator.
    """
    import torch
    q = torch.zeros(op.q_shape, dtype=op.dtype, device=op.device)
    k0 = op.q_shape[2] // 2
    q[op.q_shape[0] // 2, op.q_shape[1] // 2, k0] = 1.0
    prof = np.abs(op.conv(q).detach().cpu().numpy()).sum(axis=(0, 1))
    j = np.arange(len(prof))
    peak = int(prof.argmax()) - k0
    cen = float((prof * j).sum() / max(prof.sum(), 1e-30)) - k0
    return peak, cen


def raw_hits_universal(tag, jobdir, npz_path):
    """Raw hits placed on the universal grid the metrics live on.

    ``hits_location[:, :3]`` is (px, py, trigger tick) in global
    coordinates and ``hits_data[:, 3:]`` the cumulative latched charges,
    so for nburst=1 the charge is column 3.  The trigger tick is on the
    DATA clock; subtracting the response lag brings it onto the charge
    clock the truth and the reco share.
    """
    import yaml
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from channel_coupling import replay
    cfg = yaml.safe_load(open(f"{jobdir}/job_{tag}.yaml"))
    cfg["sequence"] = [e for e in cfg["sequence"]
                       if list(e)[0] in ("LoadEvent", "FFTWarmStart",
                                         "BuildMeasurement", "BuildSupport")]
    store, _ = replay(cfg)
    ev, op = store.get("event"), store.get("op")
    boff = np.asarray(store.get("block_offset"), float)
    loc = np.asarray(ev.hits.location)
    q = np.asarray(ev.hits.data, dtype=float)[:, 3]
    dx, dy, dt, B, _ = universal_offsets(npz_path)
    lag_peak, lag_cen = response_lag(op)
    kq = np.floor((loc[:, 2] - boff[2]) / B).astype(int) - lag_peak
    return (loc[:, 0].astype(int) - int(boff[0]) + dx,
            loc[:, 1].astype(int) - int(boff[1]) + dy,
            kq + dt, q, lag_peak, lag_cen)


def asymmetry(p, x):
    """Charge-weighted moments plus a rise/fall width ratio."""
    s = p.sum()
    if s <= 0:
        return {}
    mu = float((p * x).sum() / s)
    var = float((p * (x - mu) ** 2).sum() / s)
    sd = np.sqrt(max(var, 1e-12))
    skew = float((p * (x - mu) ** 3).sum() / s / sd ** 3)
    k = int(np.argmax(p))
    half = p.max() / 2.0

    def cross(idx_range):
        """First position on this side where the profile drops below half
        the peak, linearly interpolated between the bracketing bins."""
        prev = k
        for i in idx_range:
            if p[i] < half:
                dp = p[prev] - p[i]
                f = (p[prev] - half) / dp if dp > 0 else 0.0
                return x[prev] + f * (x[i] - x[prev])
            prev = i
        return x[prev]
    lo = cross(range(k - 1, -1, -1))
    hi = cross(range(k + 1, len(p)))
    rise = abs(x[k] - lo)
    fall = abs(hi - x[k])
    return {"mean": mu, "rms": sd, "skew": skew, "peak": float(x[k]),
            "rise_hwhm": float(rise), "fall_hwhm": float(fall),
            "fall_over_rise": float(fall / max(rise, 1e-9)),
            "frac_after_peak": float(p[k:].sum() / s)}


def composite(T, R):
    """Additive RGB: blue = truth only, red = reco only, dark = both."""
    def norm(a):
        m = a.max()
        if m <= 0:
            return np.zeros_like(a)
        lo = max(m * 1e-3, 1e-9)
        z = np.log10(np.clip(a, lo, None) / lo) / np.log10(m / lo)
        return np.clip(z, 0.0, 1.0)
    at, ar = norm(T), norm(R)
    img = np.ones((*T.shape, 3))
    img[..., 0] = 1.0 - at                      # truth removes red
    img[..., 1] = 1.0 - np.maximum(at, ar)
    img[..., 2] = 1.0 - ar                      # reco removes blue
    return img


def main_component(truth, reco, qcut):
    """Largest connected component of (truth + reco) above ``qcut``.

    Isolates the main track from the isolated satellite deposits without
    a hand-drawn box: 26-connectivity on the universal grid, component
    chosen by total charge.  Returns a boolean mask.
    """
    from scipy import ndimage
    lab, n = ndimage.label((truth + reco) > qcut, structure=np.ones((3,) * 3))
    if n == 0:
        return np.ones_like(truth, dtype=bool)
    tot = ndimage.sum(truth + reco, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(tot)) + 1)


def display(tag, jobdir=f"{AO}/nb1_fraccensor/B", out=None, pad=3,
            qcut=0.0, main_only=False):
    p = f"{jobdir}/{tag}/{tag}_event_0_0.npz"
    truth, reco = universal_rebin(p)
    if qcut > 0:
        truth = np.where(truth > qcut, truth, 0.0)
        reco = np.where(reco > qcut, reco, 0.0)
    if main_only:
        m = main_component(truth, reco, max(qcut, 0.5))
        truth, reco = truth * m, reco * m
    live = (truth + reco) > 0.01
    ix, iy, it = (np.nonzero(live.any(axis=(1, 2)))[0],
                  np.nonzero(live.any(axis=(0, 2)))[0],
                  np.nonzero(live.any(axis=(0, 1)))[0])
    sl = (slice(max(ix.min() - pad, 0), ix.max() + 1 + pad),
          slice(max(iy.min() - pad, 0), iy.max() + 1 + pad),
          slice(max(it.min() - pad, 0), it.max() + 1 + pad))
    T, R = truth[sl], reco[sl]
    # project out the short pixel axis; keep the long one against time
    ax_keep = 0 if T.shape[0] >= T.shape[1] else 1
    ax_drop = 1 - ax_keep
    T2, R2 = T.sum(axis=ax_drop), R.sum(axis=ax_drop)

    t0 = sl[2].start
    p0 = sl[ax_keep].start
    # PHYSICAL time on the response-plane clock that effq and deconv_q
    # share: universal bin m spans absolute ticks [m*B, (m+1)*B) and the
    # array starts at m = u_min, so the charge times are negative -- the
    # charge crosses the response plane before the acquisition epoch.
    u_min = universal_offsets(p)[4]
    tvec = (np.arange(T2.shape[1]) + t0 + u_min + 0.5) * TBIN_US
    pvec = (np.arange(T2.shape[0]) + p0) * PITCH_CM
    tprof_t, tprof_r = T2.sum(axis=0), R2.sum(axis=0)
    pprof_t, pprof_r = T2.sum(axis=1), R2.sum(axis=1)
    at = asymmetry(tprof_t, tvec)
    ar = asymmetry(tprof_r, tvec)

    # raw hits, de-lagged onto the same charge-time axis
    px_u, py_u, ui, hq, lag_peak, lag_cen = raw_hits_universal(
        tag, jobdir, p)
    keep = ((ui >= sl[2].start) & (ui < sl[2].stop)
            & (px_u >= sl[0].start) & (px_u < sl[0].stop)
            & (py_u >= sl[1].start) & (py_u < sl[1].stop))
    hit_pix = (px_u if ax_keep == 0 else py_u)[keep]
    hit_t, hit_q = ui[keep], hq[keep]
    hprof = np.zeros_like(tprof_t)
    np.add.at(hprof, hit_t - sl[2].start, hit_q)
    # hard-align: the response peak lag is a nominal number, and only the
    # raw peak position is stable across placement conventions, so put the
    # raw peak onto the truth peak and report the lag that does it.
    shift = int(tprof_t.argmax()) - int(hprof.argmax())
    if shift:
        hit_t = hit_t + shift
        keep2 = ((hit_t >= sl[2].start) & (hit_t < sl[2].stop))
        hit_t, hit_pix, hit_q = hit_t[keep2], hit_pix[keep2], hit_q[keep2]
        hprof = np.zeros_like(tprof_t)
        np.add.at(hprof, hit_t - sl[2].start, hit_q)
    lag_used = lag_peak - shift
    ah = asymmetry(hprof, tvec)
    print(f"  raw hits: {keep.sum()}/{len(hq)} inside the window, "
          f"{hit_q.sum():.0f} ke (truth {truth.sum():.0f}, "
          f"reco {reco.sum():.0f}); nominal lag {lag_peak} (centroid "
          f"{lag_cen:.1f}), peak-aligned lag {lag_used}", flush=True)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15],
                          height_ratios=[1.15, 4], hspace=0.05, wspace=0.05)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axp = fig.add_subplot(gs[1, 1], sharey=axm)

    ext = [tvec[0], tvec[-1] + TBIN_US, pvec[0], pvec[-1] + PITCH_CM]
    axm.imshow(composite(T2, R2), origin="lower", extent=ext,
               aspect="auto", interpolation="nearest")
    axm.scatter((hit_t + u_min + 0.5) * TBIN_US,
                (hit_pix + 0.5) * PITCH_CM,
                s=np.clip(hit_q * 0.55, 1.0, 45.0), marker="o",
                facecolors="none", edgecolors=GREEN, linewidths=0.55,
                alpha=0.75, label=f"raw hits, FORCED peak alignment\n(shifted to lag {lag_used}, nominal {lag_peak})")
    axm.legend(loc="upper left", fontsize=9, framealpha=0.85)
    axm.set_xlabel(r"charge time at the response plane [$\mu$s]")
    axm.set_ylabel(f"pixel {'x' if ax_keep == 0 else 'y'} [cm]")

    axt.plot(tvec, tprof_t, color=BLUE, lw=1.6, label="smeared truth")
    axt.plot(tvec, tprof_r, color=RED, lw=1.6, label="reco")
    axt.plot(tvec, hprof, color=GREEN, lw=1.2, ls="--",
             label=f"raw hits, FORCED peak alignment\n(shifted to lag {lag_used}, nominal {lag_peak})")
    axt.set_ylabel("charge [ke]")
    axt.legend(loc="upper right", fontsize=9, frameon=False)
    axt.tick_params(labelbottom=False)
    axt.grid(alpha=0.15)
    axt.text(0.01, 0.97,
             "truth / reco\n"
             f"  fall/rise HWHM  {at['fall_over_rise']:.2f} / "
             f"{ar['fall_over_rise']:.2f}\n"
             f"  frac after peak {at['frac_after_peak']:.3f} / "
             f"{ar['frac_after_peak']:.3f}\n"
             f"  peak [us]       {at['peak']:.1f} / {ar['peak']:.1f}\n"
             f"raw hit peak      {ah['peak']:.1f}",
             transform=axt.transAxes, va="top", ha="left", fontsize=8,
             family="monospace")

    axp.plot(pprof_t, pvec, color=BLUE, lw=1.6)
    axp.plot(pprof_r, pvec, color=RED, lw=1.6)
    hpp = np.zeros_like(pprof_t)
    np.add.at(hpp, hit_pix - sl[ax_keep].start, hit_q)
    axp.plot(hpp, pvec, color=GREEN, lw=1.0, ls="--")
    axp.set_xlabel("charge [ke]")
    axp.tick_params(labelleft=False)
    axp.grid(alpha=0.15)

    fig.suptitle(f"{tag}  --  smeared truth (blue) over reco (red), "
                 f"universal grid, projected along "
                 f"pixel {'y' if ax_keep == 0 else 'x'}   "
                 f"[truth {truth.sum():.0f} ke, reco {reco.sum():.0f} ke]",
                 y=0.965, fontsize=11)
    suf = ("_main" if main_only else "") + (f"_q{qcut:g}" if qcut else "")
    out = out or f"{AO}/event_display/{tag}_pixel_time{suf}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    # companion figure: the time profile is where the asymmetry lives, and
    # a linear axis hides the tails that carry it
    f2, ax2 = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1.2],
                                        "hspace": 0.08})
    for a, log in ((ax2[0], False), (ax2[1], True)):
        a.plot(tvec, tprof_t, color=BLUE, lw=1.7, label="smeared truth")
        a.plot(tvec, tprof_r, color=RED, lw=1.7, label="reco")
        a.plot(tvec, hprof, color=GREEN, lw=1.3, ls="--",
               label=f"raw hits, FORCED peak alignment\n(shifted to lag {lag_used}, nominal {lag_peak})")
        a.axvline(at["peak"], color="0.6", lw=0.8, ls="--")
        a.grid(alpha=0.15)
        a.set_ylabel("charge [ke]")
        if log:
            a.set_yscale("log")
            a.set_ylim(max(tprof_t.max() * 1e-4, 1e-2), None)
    ax2[0].legend(loc="upper left", fontsize=10, frameon=False)
    ax2[0].set_title(
        f"{tag}: time profile summed over pixels   "
        f"fall/rise HWHM {at['fall_over_rise']:.2f} (truth) -> "
        f"{ar['fall_over_rise']:.2f} (reco)", fontsize=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tprof_t > tprof_t.max() * 1e-3,
                         tprof_r / tprof_t, np.nan)
    ax2[2].plot(tvec, ratio, color="#333333", lw=1.4)
    ax2[2].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax2[2].axvline(at["peak"], color="0.6", lw=0.8, ls="--")
    ax2[2].set_ylim(0.5, 1.5)
    ax2[2].set_ylabel("reco / truth")
    ax2[2].set_xlabel(r"charge time at the response plane [$\mu$s]")
    ax2[2].grid(alpha=0.15)
    f2.savefig(f"{out}_timeprofile.png", dpi=150, bbox_inches="tight")
    f2.savefig(f"{out}_timeprofile.pdf", bbox_inches="tight")
    plt.close(f2)

    rec = {"tag": tag, "truth": at, "reco": ar, "raw": ah,
           "sum_truth": float(truth.sum()), "sum_reco": float(reco.sum()),
           "lag_nominal": int(lag_peak), "lag_used": int(lag_used),
           "lag_centroid": float(lag_cen)}
    print(f"{tag}: skew truth {at['skew']:+.3f} reco {ar['skew']:+.3f} raw {ah['skew']:+.3f}; "
          f"fall/rise {at['fall_over_rise']:.2f} / "
          f"{ar['fall_over_rise']:.2f}; -> {out}.png", flush=True)
    return rec


if __name__ == "__main__":
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or \
        ["pos_a00_nb1"]
    main_only = "--main" in sys.argv
    qcut = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--qcut="):
            qcut = float(a.split("=", 1)[1])
    res = [display(t, qcut=qcut, main_only=main_only) for t in tags]
    suf = ("_main" if main_only else "") + (f"_q{qcut:g}" if qcut else "")
    json.dump(res, open(f"{AO}/event_display/asymmetry{suf}.json", "w"),
              indent=1)
