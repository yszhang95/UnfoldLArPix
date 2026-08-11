"""Is decC's extra dQ/dx width just solver resolution?

Test: smear the TIME-SUMMED effq pixel map with a 2D Gaussian (sigma in
pixel units) and redo the 4-cm segment dQ/dx.  Time smearing is irrelevant
here because segment charge is time-integrated per pixel; only in-plane
charge migration along the track widens dQ/dx.  Scan sigma and see which
(if any) reproduces the decC width.

Width metrics per pooled depth sample: Moyal-fit sigma, and the 16-84
percentile half-spread (robust, fit-free).
"""
import numpy as np, sys, json
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from rebin4 import seg_dqdx_bw
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

BW = 4.0
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]


def pixmap(pa, pb, q):
    """Aggregate duplicate (pa,pb) entries into per-pixel charge."""
    pa = np.asarray(pa, int); pb = np.asarray(pb, int)
    q = np.asarray(q, float)
    key = pa.astype(np.int64) * 100000 + pb
    uk, inv = np.unique(key, return_inverse=True)
    qs = np.zeros(uk.size)
    np.add.at(qs, inv, q)
    return uk // 100000, uk % 100000, qs


def smear(pa, pb, q, sigma, pad=8, qmin=1e-3):
    if sigma == 0.0:
        return pa, pb, q
    a0, b0 = pa.min() - pad, pb.min() - pad
    grid = np.zeros((pa.max() - a0 + pad + 1, pb.max() - b0 + pad + 1))
    grid[pa - a0, pb - b0] = q
    grid = gaussian_filter(grid, sigma, mode='constant')
    ai, bi = np.nonzero(grid > qmin)
    return ai + a0, bi + b0, grid[ai, bi]


def moyal_fit(vals, nbins=30):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lo, hi = np.percentile(vals, [1, 97])
    h, e = np.histogram(vals, bins=nbins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    p0 = [c[np.argmax(h)], 0.15 * np.median(vals), float(h.max())]
    try:
        popt, _ = curve_fit(L.moyal_pdf, c, h, p0=p0, maxfev=20000)
        return float(popt[0]), abs(float(popt[1]))
    except Exception:
        return float(c[np.argmax(h)]), float('nan')


def widths(vals):
    mpv, sig = moyal_fit(vals)
    p16, p84 = np.percentile(vals, [16, 84])
    return {'mpv': mpv, 'moyal_sig': sig, 'hw6884': 0.5 * (p84 - p16),
            'n': int(len(vals))}


if __name__ == '__main__':
    keys = [f'sm{sg:g}' for sg in SIGMAS] + ['hits', 'decC']
    pool = {k: {} for k in keys}
    for tag in A50.TAGS:
        f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for k in keys:
            pool[k][tag] = []
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            pa, pb, q = pixmap(el[:, 0], el[:, 1], eq)
            for sg in SIGMAS:
                sa, sb, sq = smear(pa, pb, q, sg)
                pool[f'sm{sg:g}'][tag].append(seg_dqdx_bw(sa, sb, sq, BW))
            hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
            hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
            pool['hits'][tag].append(seg_dqdx_bw(hl[:, 0], hl[:, 1], hq, BW))
            p = A50.find_solved('C', tag, ev)
            pool['decC'][tag].append(
                seg_dqdx_bw(*A50.deconv_pix(p), BW) if p else np.array([]))
        print(f'{tag} done', flush=True)

    out = {}
    print('\n%-28s %8s %10s %10s %6s' %
          ('depth/estimator', 'MPV', 'moyal_sig', 'hw(16-84)', 'n'))
    for tag in A50.TAGS:
        print(f'-- {tag} --')
        out[tag] = {}
        for k in keys:
            allv = np.concatenate([v for v in pool[k][tag] if len(v)])
            w = widths(allv)
            out[tag][k] = w
            print('%-28s %8.2f %10.3f %10.3f %6d' %
                  (k, w['mpv'], w['moyal_sig'], w['hw6884'], w['n']))
    json.dump(out, open(f'{A50.AO}/iso50_smear_scan.json', 'w'), indent=1)
    print('-> iso50_smear_scan.json', flush=True)

    # depth-averaged width summary
    print('\n== depth-averaged widths ==')
    print('%-8s %10s %10s' % ('est', '<moyal_s>', '<hw6884>'))
    for k in keys:
        ms = np.array([out[t][k]['moyal_sig'] for t in A50.TAGS])
        hw = np.array([out[t][k]['hw6884'] for t in A50.TAGS])
        print('%-8s %10.3f %10.3f' %
              (k, np.nanmean(ms), np.nanmean(hw)))
