"""dQ/dx per pixel row for an isochronous track, from three estimators.

Geometry: the anode is a plane of constant x, so drift is along x and the
pixel plane spans (y, z).  An isochronous track runs along +z at fixed
x and y=0, hence it lights up ONE row of pixels in the transverse
coordinate and many along the beam coordinate.

Per pixel row along the track we form dQ/dx = Q / pitch, where Q sums the
charge inside a TRIM TUBE of +-TUBE_PIX pixels transversally: this keeps
the primary trunk and rejects the delta rays whose TPC-boundary clipping
differs between depths (they are the same shower translated in x, so the
clipping -- not the physics -- would otherwise vary).

The per-depth summary is the Landau/Moyal MPV of that dQ/dx sample, not
the mean: the mean is pulled by the delta tail, the peak is not.
"""
import numpy as np
from scipy.optimize import curve_fit

PITCH_CM = 0.4434          # pixel pitch [cm] (tab:sim-params)
VDRIFT = 1.59645           # mm/us
ANODE_X = 3.069            # cm, TPC0
TUBE_PIX = 3               # +-3 pixels (~1.3 cm) around the trunk
Z_LO, Z_HI = 12.0, 58.0    # cm, well inside the TPC z range


def moyal_pdf(x, mpv, sigma, amp):
    lam = np.clip((x - mpv) / np.abs(sigma), -30.0, 30.0)   # guard exp overflow
    return amp * np.exp(-0.5 * (lam + np.exp(-lam))) / np.abs(sigma)


def mpv_err(vals, nboot=200, seed=0):
    """Bootstrap uncertainty of the MPV estimator on this sample."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 8:
        return float('nan')
    rng = np.random.default_rng(seed)
    b = [mpv_of(rng.choice(vals, vals.size, replace=True))[0]
         for _ in range(nboot)]
    b = np.array(b, float)
    b = b[np.isfinite(b)]
    return float(b.std()) if b.size > 10 else float('nan')


def mpv_of(vals, nbins=28):
    """Landau(Moyal) MPV of a dQ/dx sample, with the median as fallback."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 8:
        return float('nan'), float('nan')
    lo, hi = np.percentile(vals, [1, 97])
    h, e = np.histogram(vals, bins=nbins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    p0 = [c[np.argmax(h)], 0.15 * np.median(vals), h.max()]
    try:
        popt, _ = curve_fit(moyal_pdf, c, h, p0=p0, maxfev=20000)
        mpv = float(popt[0])
        if not (lo <= mpv <= hi):
            raise RuntimeError
    except Exception:
        mpv = float(c[np.argmax(h)])
    return mpv, float(np.median(vals))


def rows_from_pixel_charge(px_a, px_b, q, pitch=PITCH_CM):
    """Given per-pixel (idx_a, idx_b, charge), pick the along-track axis as
    the one with the larger extent, trim transversally, and return the
    per-row dQ/dx."""
    ea, eb = px_a.max() - px_a.min(), px_b.max() - px_b.min()
    along, trans = (px_a, px_b) if ea >= eb else (px_b, px_a)
    # trunk = charge-weighted transverse position, evaluated robustly
    order = np.argsort(q)[::-1][:max(20, q.size // 20)]
    t0 = int(np.round(np.average(trans[order], weights=q[order])))
    keep = np.abs(trans - t0) <= TUBE_PIX
    along, q = along[keep], q[keep]
    if along.size == 0:
        return np.array([]), np.array([]), t0
    lo, hi = along.min(), along.max()
    nrow = hi - lo + 1
    Q = np.zeros(nrow)
    np.add.at(Q, along - lo, q)
    return np.arange(lo, hi + 1), Q / pitch, t0


def drift_time_us(depth_cm):
    return depth_cm * 10.0 / VDRIFT
