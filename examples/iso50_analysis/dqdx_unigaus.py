"""dQ/dx and the fitted lifetime under the ANALYSIS conventions.

The dQ/dx of record (iso50_analyse.py) works on the raw fit grid with no
charge cut.  The data analysis this feeds does neither: it puts charge on the
universal grid with the analysis Gaussian and applies a per-pixel reporting
cut.  Matching the MC study to the analysis convention is not optional even
where it is rate-neutral, because the smearing moves the PCA axis and hence
which charge lands in the tube.

Equivalence used here (so the whole 3-D universal rebin is not needed): the
dQ/dx estimator is the per-pixel TIME-INTEGRATED charge, and the universal
grid's time deposit is charge conserving, so time-integrating commutes with
it.  What survives is the SPATIAL analysis Gaussian on the pixel axes, and
the universal pixel axis IS the global hardware index -- the same index the
fit-grid estimator already uses.  The convolution here is line-for-line the
one in eval/universal.py (fftfreq, exp(-f^2/2 sigma_pxl^2), sigma_pxl = 0.2
in FREQUENCY units = 0.796 px in real space).

Four variants per estimator, so the two conventions can be attributed
separately:  raw | cut | gaus | gaus+cut.

Reported: the fitted decay rate lambda per variant (9 depths, d = 1.5 cm
excluded per sec:iso:result), and the PCA axis rotation the smearing causes,
which is the reason the user asked for it.

Usage:  python dqdx_unigaus.py [ARM] [--out JSON]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L                                        # noqa: E402
from track_dqdx import fit_direction, px_to_cm, segment_dqdx  # noqa: E402

NFS = ('/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/'
       'tests/pgun_farfield')
AO = ('/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/'
      'examples/analysis_output')
DEPTHS = ['01p5', '04p5', '07p5', '10p5', '13p5', '16p5', '19p5', '22p5',
          '25p5', '28p5']
SIGMA_PXL = 0.2          # frequency-domain, as shipped (= 0.796 px real)
QCUT_KE = 0.5            # the 500 e- per-pixel reporting cut
VARIANTS = ('raw', 'cut', 'gaus', 'gaus_cut')


def smear_pixels(pa, pb, q, sigma_pxl=SIGMA_PXL, pad=8):
    """Apply the analysis spatial Gaussian to a sparse per-pixel charge map.

    Same convention as eval/universal.py: full-array FFT on the pixel axes
    with exp(-f^2 / 2 sigma^2) on fftfreq.  Padded so the periodic wrap does
    not fold the track onto itself.
    """
    a0, b0 = int(pa.min()) - pad, int(pb.min()) - pad
    na = int(pa.max()) - a0 + 1 + pad
    nb = int(pb.max()) - b0 + 1 + pad
    grid = np.zeros((na, nb))
    np.add.at(grid, (pa.astype(int) - a0, pb.astype(int) - b0), q)
    fx = np.fft.fftfreq(na)
    fy = np.fft.fftfreq(nb)
    gx = np.exp(-0.5 * fx ** 2 / float(sigma_pxl) ** 2)
    gy = np.exp(-0.5 * fy ** 2 / float(sigma_pxl) ** 2)
    G = np.fft.fftn(grid, axes=(0, 1)) * gx[:, None] * gy[None, :]
    out = np.real(np.fft.ifftn(G, axes=(0, 1)))
    ia, ib = np.nonzero(out > 1e-9)
    return ia + a0, ib + b0, out[ia, ib]


def variants_of(pa, pb, q):
    """The four convention combinations, as (pa, pb, q) triples."""
    out = {'raw': (pa, pb, q)}
    m = q > QCUT_KE
    out['cut'] = (pa[m], pb[m], q[m])
    ga, gb, gq = smear_pixels(pa, pb, q)
    out['gaus'] = (ga, gb, gq)
    mg = gq > QCUT_KE
    out['gaus_cut'] = (ga[mg], gb[mg], gq[mg])
    return out


def axis_angle_deg(pa, pb, q):
    """PCA axis of the high-Q points, in degrees, or nan."""
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    c, d = fit_direction(np.stack([y, z], 1), np.asarray(q, float))
    return float('nan') if c is None else float(
        np.degrees(np.arctan2(d[0], d[1])))


def per_pixel_from_solved(path):
    z = np.load(path, allow_pickle=True)
    per = np.asarray(z['deconv_q_sharp'], float).sum(axis=2)
    off = np.asarray(z['boffset'], float)
    a, b = np.nonzero(per > 0)
    return a + int(off[0]), b + int(off[1]), per[a, b]


def per_pixel_from_effq(f, ev):
    el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
    eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
    key = el[:, 0].astype(np.int64) * 100000 + el[:, 1]
    u, inv = np.unique(key, return_inverse=True)
    return (u // 100000).astype(int), (u % 100000).astype(int), \
        np.bincount(inv, weights=eq)


def per_pixel_from_hits(f, ev):
    loc = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
    dat = np.asarray(f[f'hits_tpc0_batch{ev}'], float)
    if len(loc) == 0:
        return np.array([]), np.array([]), np.array([])
    key = loc[:, 0].astype(np.int64) * 100000 + loc[:, 1]
    u, inv = np.unique(key, return_inverse=True)
    return (u // 100000).astype(int), (u % 100000).astype(int), \
        np.bincount(inv, weights=dat[:, 3])


def fit_lambda(t_us, mpvs):
    """Decay rate [1/ms] from ln(MPV) against drift time."""
    m = np.isfinite(mpvs) & (mpvs > 0)
    A = np.vstack([np.asarray(t_us)[m], np.ones(m.sum())]).T
    sol, *_ = np.linalg.lstsq(A, np.log(np.asarray(mpvs)[m]), rcond=None)
    return -float(sol[0]) * 1000.0


if __name__ == '__main__':
    arm = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') \
        else 'C'
    segs = {est: {v: {} for v in VARIANTS} for est in ('effq', 'hits', 'dec')}
    ang = {}
    t_us = []
    for dep in DEPTHS:
        tag = f'pgun_mu_3gev_iso50_d{dep}'
        depth = float(dep.replace('p', '.'))
        t_us.append(L.drift_time_us(depth))
        f = np.load(f'{NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for est in segs:
            for v in VARIANTS:
                segs[est][v][dep] = []
        da = []
        for ev in range(50):
            src = {'effq': lambda: per_pixel_from_effq(f, ev),
                   'hits': lambda: per_pixel_from_hits(f, ev)}
            p = f'{AO}/iso50/{arm}/{tag}/{tag}_event_0_{ev}.npz'
            if os.path.exists(p):
                src['dec'] = lambda: per_pixel_from_solved(p)
            for est, get in src.items():
                pa, pb, q = get()
                if len(pa) == 0:
                    continue
                vs = variants_of(pa, pb, q)
                for v, (a, b, qq) in vs.items():
                    if len(a) == 0:
                        continue
                    s = segment_dqdx(a, b, qq)
                    if len(s):
                        segs[est][v][dep].append(s)
                if est == 'dec':
                    a0 = axis_angle_deg(*vs['raw'])
                    a1 = axis_angle_deg(*vs['gaus_cut'])
                    if np.isfinite(a0) and np.isfinite(a1):
                        da.append(a1 - a0)
        ang[dep] = {'n': len(da),
                    'mean_deg': float(np.mean(da)) if da else float('nan'),
                    'rms_deg': float(np.std(da)) if da else float('nan'),
                    'max_abs_deg': float(np.max(np.abs(da))) if da else 0.0}
        print(f'{tag}  PCA axis rotation (gaus+cut - raw): '
              f'mean {ang[dep]["mean_deg"]:+.4f} deg, rms '
              f'{ang[dep]["rms_deg"]:.4f}, max|.| {ang[dep]["max_abs_deg"]:.4f}'
              f'  ({ang[dep]["n"]} events)', flush=True)

    res = {'arm': arm, 'sigma_pxl': SIGMA_PXL, 'qcut_ke': QCUT_KE,
           'axis_rotation': ang, 'mpv': {}, 'lambda': {}}
    print(f'\n{"estimator":10s} {"variant":9s} ' +
          ' '.join(f'd{d}' for d in DEPTHS) + '   lambda[1/ms]')
    for est in segs:
        res['mpv'][est] = {}
        for v in VARIANTS:
            mpvs = np.array([L.mpv_of(np.concatenate(segs[est][v][d]))[0]
                             if segs[est][v][d] else np.nan for d in DEPTHS])
            res['mpv'][est][v] = [None if not np.isfinite(x) else float(x)
                                  for x in mpvs]
            lam9 = fit_lambda(t_us[1:], mpvs[1:])     # d = 1.5 cm excluded
            res['lambda'][f'{est}|{v}'] = lam9
            print(f'{est:10s} {v:9s} ' +
                  ' '.join(f'{x:5.1f}' for x in mpvs) +
                  f'   {lam9:6.3f}')
    out = '--out' in sys.argv and sys.argv[sys.argv.index('--out') + 1]
    json.dump(res, open(out or f'{AO}/iso50_unigaus_{arm}.json', 'w'), indent=1)
    print(f"\n-> {out or f'{AO}/iso50_unigaus_{arm}.json'}")
