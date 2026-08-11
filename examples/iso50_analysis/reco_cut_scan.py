"""Reco-cut scan on the fig-11 symmetric 4x4 comparison.

Selection: group enters iff decC_group > CUT (reco-side cut, like the
note's q_reco > 500e), on top of the union floor. Reports per-depth
r / OLS slope / through-origin slope / sum ratio for each cut, and how
the drift-slide of the ratio responds.
"""
import numpy as np, sys
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from smear_scan import pixmap, smear

SIG = 1.0 / (2.0 * np.pi * 0.5)
CUTS = [0.0, 0.5, 2.0, 5.0]
QFLOOR = 1.0
G = 4


def groups(pa, pb, q):
    key = (np.asarray(pa, np.int64) // G) * 100000 + (np.asarray(pb, np.int64) // G)
    uk, inv = np.unique(key, return_inverse=True)
    qs = np.zeros(uk.size)
    np.add.at(qs, inv, np.asarray(q, float))
    return dict(zip(uk.tolist(), qs.tolist()))

pairs = {}   # tag -> (x, y) arrays, no reco cut yet
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    xs, ys = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        gt = groups(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG))
        fp = A50.find_solved('C', tag, ev)
        if fp is None:
            continue
        gd = groups(*smear(*A50.deconv_pix(fp), SIG))
        for k in set(gt) | set(gd):
            a, b = gt.get(k, 0.0), gd.get(k, 0.0)
            if max(a, b) > QFLOOR:
                xs.append(a); ys.append(b)
    pairs[tag] = (np.array(xs), np.array(ys))
    print(f'{tag} done', flush=True)

for cut in CUTS:
    print(f'\n== reco cut: decC group > {cut:g} ke ==')
    print('%-6s %8s %8s %8s %8s %8s %6s' %
          ('d[cm]', 'r', 'ols', 'origin', 'ratio', 'kept%', 'N'))
    rows = []
    for tag in A50.TAGS:
        x0, y0 = pairs[tag]
        m = y0 > cut
        x, y = x0[m], y0[m]
        d = float(tag.split('_d')[1].replace('p', '.'))
        ratio = y.sum() / x.sum()
        rows.append((L.drift_time_us(d), ratio))
        print('%-6.1f %8.4f %8.3f %8.4f %8.4f %7.1f%% %6d' %
              (d, np.corrcoef(x, y)[0, 1], np.polyfit(x, y, 1)[0],
               (x*y).sum()/(x*x).sum(), ratio, 100*m.mean(), len(x)),
              flush=True)
    tr = np.array(rows)
    b, a = np.polyfit(tr[:, 0], np.log(tr[:, 1]), 1)
    print('   ratio drift-slide: %+0.2f%% over 170 us' % (100*b*170))
