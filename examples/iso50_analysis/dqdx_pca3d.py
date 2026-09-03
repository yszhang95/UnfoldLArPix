"""dQ/dx with the 3-D iterative track fit of the data analysis, reco only.

The estimator of record (track_dqdx.py) fits the axis in the PIXEL PLANE
only, after summing each pixel over time.  The analysis this feeds fits in
3-D and iterates (2x2_ql, filter_mc_events/preprocessing/track_selection.py):

  1. PCA on all cluster points
  2. PCA again on the central 60% of the projection
     (``range_cut = (proj_max - proj_min) * cut_fraction``, cut_fraction=0.15)
  3. PCA again on the points within ``dmax`` of that axis (dmax = 2.0 cm)

and its PCA is on COORDINATES, unweighted -- with hits as points, the charge
enters through point density.  This script applies that recipe to the
reconstruction's voxels, so the third coordinate is the drift one: a voxel's
time bin becomes x through the drift velocity (0.2395 cm per 30-tick bin,
tab:sim-params), which is why the fit can be 3-D at all.

Three conventions, reco only:
  raw       fit-grid voxels, no cut          (as the estimator of record)
  cut       + the 500 e- per-voxel cut
  gaus_cut  + the full universal-grid Gaussian deposit first (time Gaussian
            at sigma_time = 0.005 -> 31.8 ticks, then the spatial Gaussian at
            sigma_pxl = 0.2 -> 0.796 px), i.e. the analysis convention

Also reported: sin(theta) = |dir_x| / |dir|, which the analysis cuts at 0.05
to select isochronous tracks -- a direct check that these samples pass their
own selection.

Usage:  python dqdx_pca3d.py [ARM...]
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L                                  # noqa: E402
from track_dqdx import LL, PITCH                      # noqa: E402

NFS = ('/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/'
       'tests/pgun_farfield')
AO = ('/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/'
      'examples/analysis_output')
DEPTHS = ['01p5', '04p5', '07p5', '10p5', '13p5', '16p5', '19p5', '22p5',
          '25p5', '28p5']
CM_PER_TICK = L.VDRIFT * 0.05 / 10.0     # 1.59645 mm/us * 50 ns -> cm
DMAX_CM = 2.0                            # their --dmax
CUT_FRACTION = 0.15                      # their central-60% trim
BIN_CM = 3.0
QCUT_KE = 0.5
SIGMA_TIME, SIGMA_PXL = 0.005, 0.2
NVALID = 5


def voxels(path):
    """Sharp reco voxels -> (points [cm], charge [ke]); x from drift time."""
    z = np.load(path, allow_pickle=True)
    q = np.asarray(z['deconv_q_sharp'], float)
    off = np.asarray(z['boffset'], float)
    B = int(z['adc_hold_delay'])
    a, b, k = np.nonzero(q > 0)
    y = LL[0] + (a + int(off[0]) + 0.5) * PITCH
    zc = LL[1] + (b + int(off[1]) + 0.5) * PITCH
    x = (off[2] + (k + 0.5) * B) * CM_PER_TICK
    return np.stack([x, y, zc], 1), q[a, b, k]


def unigaus_voxels(path):
    """Same, after the universal-grid Gaussian deposit (time then pixels)."""
    z = np.load(path, allow_pickle=True)
    qs = np.asarray(z['deconv_q_sharp'], float)
    off = np.asarray(z['boffset'], float)
    B = int(z['adc_hold_delay'])
    nx, ny, ntq = qs.shape
    sig = 1.0 / (2.0 * np.pi * SIGMA_TIME)          # ticks
    centers = off[2] + (np.arange(ntq) + 0.5) * B
    reach = int(np.ceil(6.0 * sig / B)) + 1
    m0 = int(np.floor(centers.min() / B)) - reach
    m1 = int(np.floor(centers.max() / B)) + reach + 1
    edges = (np.arange(m0, m1 + 2) * B).astype(float)
    erf = np.vectorize(math.erf)
    zz = (edges[None, :] - centers[:, None]) / (np.sqrt(2.0) * sig)
    cdf = 0.5 * (1.0 + erf(zz))
    W = cdf[:, 1:] - cdf[:, :-1]                    # (ntq, ntu)
    blk = np.einsum('xyk,km->xym', qs, W)
    fx, fy = np.fft.fftfreq(nx), np.fft.fftfreq(ny)
    gx = np.exp(-0.5 * fx ** 2 / SIGMA_PXL ** 2)
    gy = np.exp(-0.5 * fy ** 2 / SIGMA_PXL ** 2)
    blk = np.real(np.fft.ifftn(np.fft.fftn(blk, axes=(0, 1))
                               * gx[:, None, None] * gy[None, :, None],
                               axes=(0, 1)))
    a, b, m = np.nonzero(blk > 1e-9)
    y = LL[0] + (a + int(off[0]) + 0.5) * PITCH
    zc = LL[1] + (b + int(off[1]) + 0.5) * PITCH
    x = ((m + m0) + 0.5) * B * CM_PER_TICK
    return np.stack([x, y, zc], 1), blk[a, b, m]


def _pca(pts):
    c = pts.mean(axis=0)
    d = PCA(n_components=1).fit(pts - c).components_[0]
    return c, (d if d[2] >= 0 else -d)              # orient along +z


def fit_axis_3d(pts):
    """Their three-pass unweighted PCA; returns (centroid, direction)."""
    if len(pts) < NVALID:
        return None, None
    c, d = _pca(pts)
    proj = (pts - c) @ d                            # pass 2: central 60%
    lo, hi = proj.min(), proj.max()
    rc = (hi - lo) * CUT_FRACTION
    m = (proj > lo + rc) & (proj < hi - rc)
    if m.sum() >= NVALID:
        c, d = _pca(pts[m])
    rel = pts - c                                   # pass 3: inside dmax
    perp = np.linalg.norm(np.cross(rel, d[None, :]), axis=1)
    m = perp < DMAX_CM
    if m.sum() >= NVALID:
        c, d = _pca(pts[m])
    return c, d


def dqdx(pts, q):
    c, d = fit_axis_3d(pts)
    if c is None:
        return np.array([]), float('nan')
    rel = pts - c
    perp = np.linalg.norm(np.cross(rel, d[None, :]), axis=1)
    keep = perp < DMAX_CM
    if keep.sum() < NVALID:
        return np.array([]), float('nan')
    proj, qk = rel[keep] @ d, q[keep]
    edges = np.arange(proj.min(), proj.max() + BIN_CM, BIN_CM)
    if len(edges) < 4:
        return np.array([]), float('nan')
    h, _ = np.histogram(proj, bins=edges, weights=qk)
    ne = np.nonzero(h > 0)[0]
    if len(ne) < 3:
        return np.array([]), float('nan')
    h = h[ne[0] + 1: ne[-1]]
    sin_t = abs(d[0]) / np.linalg.norm(d)
    return h[h > 0] / BIN_CM, float(sin_t)


def fit_lambda(t_us, mpvs):
    m = np.isfinite(mpvs) & (mpvs > 0)
    A = np.vstack([np.asarray(t_us)[m], np.ones(m.sum())]).T
    sol, *_ = np.linalg.lstsq(A, np.log(np.asarray(mpvs)[m]), rcond=None)
    return -float(sol[0]) * 1000.0


if __name__ == '__main__':
    arms = sys.argv[1:] or ['C', 'B']
    res = {}
    for arm in arms:
        pool = {v: {d: [] for d in DEPTHS} for v in ('raw', 'cut', 'gaus_cut')}
        sins = {v: [] for v in pool}
        t_us = []
        for dep in DEPTHS:
            tag = f'pgun_mu_3gev_iso50_d{dep}'
            t_us.append(L.drift_time_us(float(dep.replace('p', '.'))))
            for ev in range(50):
                p = f'{AO}/iso50/{arm}/{tag}/{tag}_event_0_{ev}.npz'
                if not os.path.exists(p):
                    continue
                P, Q = voxels(p)
                sets = {'raw': (P, Q)}
                m = Q > QCUT_KE
                sets['cut'] = (P[m], Q[m])
                Pg, Qg = unigaus_voxels(p)
                mg = Qg > QCUT_KE
                sets['gaus_cut'] = (Pg[mg], Qg[mg])
                for v, (pp, qq) in sets.items():
                    if len(pp) < NVALID:
                        continue
                    s, st = dqdx(pp, qq)
                    if len(s):
                        pool[v][dep].append(s)
                        sins[v].append(st)
            print(f'{arm} {tag}: ' + '  '.join(
                f'{v} n={len(pool[v][dep])}' for v in pool), flush=True)
        res[arm] = {'lambda': {}, 'mpv': {}, 'sin_theta': {}}
        print(f'\narm {arm}: 3-D iterative PCA (dmax={DMAX_CM} cm, '
              f'central {100*(1-2*CUT_FRACTION):.0f}%)')
        print(f'{"variant":9s} ' + ' '.join(f'd{d}' for d in DEPTHS)
              + '   lambda[1/ms]  <sin(theta)>')
        for v in pool:
            mpvs = np.array([L.mpv_of(np.concatenate(pool[v][d]))[0]
                             if pool[v][d] else np.nan for d in DEPTHS])
            lam = fit_lambda(t_us[1:], mpvs[1:])
            st = float(np.nanmean(sins[v])) if sins[v] else float('nan')
            res[arm]['lambda'][v] = lam
            res[arm]['mpv'][v] = [None if not np.isfinite(x) else float(x)
                                  for x in mpvs]
            res[arm]['sin_theta'][v] = st
            print(f'{v:9s} ' + ' '.join(f'{x:5.1f}' for x in mpvs)
                  + f'   {lam:6.3f}       {st:.4f}')
    json.dump(res, open(f'{AO}/iso50_pca3d.json', 'w'), indent=1)
    print(f'\n-> {AO}/iso50_pca3d.json')
