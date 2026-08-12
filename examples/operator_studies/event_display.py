"""Pixel-vs-time event display: smeared truth (blue) over reco (red).

Built from the universal-grid rebin, so the truth is the once-smeared
truth and the reco is the deposited solution -- the same pair the
acceptance metrics use, no second smearing anywhere.

The main panel composites the two fields additively in RGB: truth-only
cells go blue, reco-only cells go red, cells where both agree go dark.
The marginals are the charge profiles along each axis, and the time
marginal carries the asymmetry numbers, since that profile is what the
positron-shower argument rests on.
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


def display(tag, jobdir=f"{AO}/nb1_fraccensor/B", out=None, pad=3):
    p = f"{jobdir}/{tag}/{tag}_event_0_0.npz"
    truth, reco = universal_rebin(p)
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
    tvec = (np.arange(T2.shape[1]) + t0) * TBIN_US
    pvec = (np.arange(T2.shape[0]) + p0) * PITCH_CM
    tprof_t, tprof_r = T2.sum(axis=0), R2.sum(axis=0)
    pprof_t, pprof_r = T2.sum(axis=1), R2.sum(axis=1)
    at = asymmetry(tprof_t, tvec)
    ar = asymmetry(tprof_r, tvec)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15],
                          height_ratios=[1.15, 4], hspace=0.05, wspace=0.05)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axp = fig.add_subplot(gs[1, 1], sharey=axm)

    ext = [tvec[0], tvec[-1] + TBIN_US, pvec[0], pvec[-1] + PITCH_CM]
    axm.imshow(composite(T2, R2), origin="lower", extent=ext,
               aspect="auto", interpolation="nearest")
    axm.set_xlabel(r"drift time [$\mu$s]")
    axm.set_ylabel(f"pixel {'x' if ax_keep == 0 else 'y'} [cm]")

    axt.plot(tvec, tprof_t, color=BLUE, lw=1.6, label="smeared truth")
    axt.plot(tvec, tprof_r, color=RED, lw=1.6, label="reco")
    axt.set_ylabel("charge [ke]")
    axt.legend(loc="upper right", fontsize=9, frameon=False)
    axt.tick_params(labelbottom=False)
    axt.grid(alpha=0.15)
    axt.text(0.01, 0.97,
             "time-profile asymmetry (blue / red)\n"
             f"  skew          {at['skew']:+.3f} / {ar['skew']:+.3f}\n"
             f"  fall/rise HWHM {at['fall_over_rise']:.2f} / "
             f"{ar['fall_over_rise']:.2f}\n"
             f"  frac after peak {at['frac_after_peak']:.3f} / "
             f"{ar['frac_after_peak']:.3f}",
             transform=axt.transAxes, va="top", ha="left", fontsize=8,
             family="monospace")

    axp.plot(pprof_t, pvec, color=BLUE, lw=1.6)
    axp.plot(pprof_r, pvec, color=RED, lw=1.6)
    axp.set_xlabel("charge [ke]")
    axp.tick_params(labelleft=False)
    axp.grid(alpha=0.15)

    fig.suptitle(f"{tag}  --  smeared truth (blue) over reco (red), "
                 f"universal grid, projected along "
                 f"pixel {'y' if ax_keep == 0 else 'x'}   "
                 f"[truth {truth.sum():.0f} ke, reco {reco.sum():.0f} ke]",
                 y=0.965, fontsize=11)
    out = out or f"{AO}/event_display/{tag}_pixel_time"
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
        a.axvline(at["peak"], color="0.6", lw=0.8, ls="--")
        a.grid(alpha=0.15)
        a.set_ylabel("charge [ke]")
        if log:
            a.set_yscale("log")
            a.set_ylim(max(tprof_t.max() * 1e-4, 1e-2), None)
    ax2[0].legend(loc="upper left", fontsize=10, frameon=False)
    ax2[0].set_title(
        f"{tag}: time profile summed over pixels   "
        f"skew {at['skew']:+.3f} -> {ar['skew']:+.3f}   "
        f"fall/rise HWHM {at['fall_over_rise']:.2f} -> "
        f"{ar['fall_over_rise']:.2f}", fontsize=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tprof_t > tprof_t.max() * 1e-3,
                         tprof_r / tprof_t, np.nan)
    ax2[2].plot(tvec, ratio, color="#333333", lw=1.4)
    ax2[2].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax2[2].axvline(at["peak"], color="0.6", lw=0.8, ls="--")
    ax2[2].set_ylim(0.5, 1.5)
    ax2[2].set_ylabel("reco / truth")
    ax2[2].set_xlabel(r"drift time [$\mu$s]")
    ax2[2].grid(alpha=0.15)
    f2.savefig(f"{out}_timeprofile.png", dpi=150, bbox_inches="tight")
    f2.savefig(f"{out}_timeprofile.pdf", bbox_inches="tight")
    plt.close(f2)

    rec = {"tag": tag, "truth": at, "reco": ar,
           "sum_truth": float(truth.sum()), "sum_reco": float(reco.sum())}
    print(f"{tag}: skew truth {at['skew']:+.3f} reco {ar['skew']:+.3f}; "
          f"fall/rise {at['fall_over_rise']:.2f} / "
          f"{ar['fall_over_rise']:.2f}; -> {out}.png", flush=True)
    return rec


if __name__ == "__main__":
    tags = sys.argv[1:] or ["pos_a00_nb1"]
    res = [display(t) for t in tags]
    json.dump(res, open(f"{AO}/event_display/asymmetry.json", "w"), indent=1)
