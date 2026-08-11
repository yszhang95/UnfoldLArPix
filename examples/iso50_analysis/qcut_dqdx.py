"""Does a minimum pixel-q cut rescue the dQ/dx lifetime?

Apply q > CUT to the time-summed pixel charges of decC (and effq, same
treatment) BEFORE the track-fit + 3-cm segment pipeline. tau(MPV) per cut.
"""
import numpy as np, sys
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap

CUTS = [0.0, 1.0, 2.0, 5.0]


def cutseg(pa, pb, q, cut):
    m = q > cut
    if m.sum() < 8:
        return np.array([])
    return segment_dqdx(pa[m], pb[m], q[m])


pool = {(k, c): {} for k in ('effq', 'decC') for c in CUTS}
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for key in pool:
        pool[key][tag] = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        pe = pixmap(el[:, 0], el[:, 1], eq)
        fp = A50.find_solved('C', tag, ev)
        pd = A50.deconv_pix(fp) if fp else None
        for c in CUTS:
            pool[('effq', c)][tag].append(cutseg(*pe, c))
            pool[('decC', c)][tag].append(
                cutseg(*pd, c) if pd else np.array([]))
    print(f'{tag} done', flush=True)

t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]
print('\n== tau(MPV), 3-cm segments, minimum pixel-q cut ==')
print('%-8s %10s %10s %14s' % ('cut[ke]', 'effq', 'decC', 'n_seg(d28,dec)'))
for c in CUTS:
    taus = {}
    for k in ('effq', 'decC'):
        pe = [pool[(k, c)][tag] for tag in A50.TAGS]
        taus[k] = A50.boot_tau(t_us, pe, mpvfn)
    nlast = sum(len(v) for v in pool[('decC', c)][A50.TAGS[-1]])
    print('%-8g %5.3f+-%.3f %5.3f+-%.3f %10d' %
          (c, *taus['effq'], *taus['decC'], nlast), flush=True)
