"""What transport CAN do to the dQ/dx MPV -- a bound that needs no model.

Rather than infer the transport level from residual correlations (whose
smooth component is not modelled reliably), inject transport of KNOWN
fraction and KNOWN range into the truth and push it through the real
dQ/dx pipeline.  A 3 cm segment is 6.77 pixels, so short-range transport
is invisible to it by construction; this quantifies how far and how much
charge would have to move before the MPV notices.

  scatter(f, R): every pixel sends fraction f of its charge to a
                 uniformly random pixel in b +- [1..R] along the track
                 axis.  Charge exact; for R=1 it is nearest-neighbour
                 exchange, the case the residual anomaly points at.

Also the DIRECT question for the MPV: does charge cross 3 cm SEGMENT
boundaries?  Measured as the lag-1 correlation of the segment-level
residual under a common (truth) segmentation, with `hits` -- per-pixel by
construction -- as the null.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import track_dqdx as T
import dqdx_lib as L

FRACS = [0.10, 0.20, 0.40]
RANGES = [1, 2, 3, 5, 10]


def scatter_pixels(pa, pb, q, f, R, rng):
    step = rng.integers(1, R + 1, q.size) * rng.choice([-1, 1], q.size)
    return (np.concatenate([pa, pa]),
            np.concatenate([pb, pb + step]),
            np.concatenate([q * (1 - f), q * f]))


def lam_of(points):
    t = np.array([p[0] for p in points]); y = np.log([p[1] for p in points])
    m = t > 20
    return -np.polyfit(t[m], y[m], 1)[0] * 1000


if __name__ == '__main__':
    rng = np.random.default_rng(11)

    # ---------------------------------------------------------------- bound
    print('=' * 90)
    print('MPV LEVERAGE OF TRANSPORT: pooled MPV ratio to the untouched truth')
    print('(3 cm dQ/dx segment = 6.77 px; truth lambda with no transport = 1/tau = 1.0)')
    print('=' * 90)
    cfg = [(0.0, 0)] + [(f, R) for f in FRACS for R in RANGES]
    curves = {c: [] for c in cfg}
    hdr = '%-7s' % 'depth' + ''.join('%9s' % f'{f:.1f}/{R}' for f, R in cfg[1:])
    print(hdr)
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        ev_truth = [PS.truth_pix(fz, ev) for ev in range(50)]
        mp = {}
        for (f, R) in cfg:
            pool = []
            for (ta, tb, tq) in ev_truth:
                if f:
                    ta2, tb2, tq2 = scatter_pixels(ta, tb, tq, f, R, rng)
                else:
                    ta2, tb2, tq2 = ta, tb, tq
                pool.append(T.segment_dqdx(ta2.astype(float), tb2.astype(float), tq2))
            v = np.concatenate([x for x in pool if len(x)])
            mp[(f, R)] = L.mpv_of(v)[0]
            curves[(f, R)].append((L.drift_time_us(d_cm), mp[(f, R)]))
        print('%-7.1f' % d_cm + ''.join('%9.4f' % (mp[c] / mp[(0.0, 0)])
                                        for c in cfg[1:]))
    print()
    print('fitted decay rate lambda [/ms] over the 9 fitted depths:')
    print('   no transport      %.4f' % lam_of(curves[(0.0, 0)]))
    for f in FRACS:
        print('   f=%.1f  ' % f + '  '.join(
            f'R={R}: {lam_of(curves[(f, R)]):.4f}' for R in RANGES))
    print('\n   for scale: measured deconv C lambda = 1.555, truth control 0.970')

    # ------------------------------------------------- segment-level crossing
    print()
    print('=' * 90)
    print('DOES CHARGE CROSS 3 cm SEGMENT BOUNDARIES?')
    print('lag-1 correlation of the segment residual, common (truth) segmentation')
    print('`hits` is per-pixel by construction -> the null')
    print('=' * 90)
    print('%-7s %10s %10s %10s' % ('depth', 'hits', 'decC', 'decB'))
    seg_out = {}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        pr = {k: [] for k in ['hits', 'decC', 'decB']}
        for ev in range(50):
            ta, tb, tq = PS.truth_pix(fz, ev)
            ax = PS.truth_axis_edges(ta, tb, tq)
            if ax is None:
                continue
            c, dv, edges, lo, hi = ax
            ht, _ = PS.seg_common(ta, tb, tq, c, dv, edges)
            if not ht.size:
                continue
            ht = ht[lo:hi]
            est = {'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for k, (pa, pb, q) in est.items():
                hr, _ = PS.seg_common(pa, pb, q, c, dv, edges)
                if hr.size:
                    pr[k].append((ht, hr[lo:hi]))
        row = {}
        for k in pr:
            if not pr[k]:
                continue
            x = np.concatenate([p[0] for p in pr[k]])
            y = np.concatenate([p[1] for p in pr[k]])
            a, cc = np.polyfit(x, y, 1)
            num = den = 0.0
            for t_, r_ in pr[k]:
                r = r_ - a * t_ - cc
                r = r - r.mean()
                den += float(r @ r)
                num += float(r[1:] @ r[:-1])
            row[k] = num / den
        seg_out[tag] = {'depth_cm': d_cm, **row}
        print('%-7.1f %10.4f %10.4f %10.4f'
              % (d_cm, row.get('hits', np.nan), row.get('decC', np.nan),
                 row.get('decB', np.nan)))
    json.dump({'segment_lag1': seg_out}, open(f'{PS.OUT}/transport_bound.json', 'w'),
              indent=1)
    print('\n->', f'{PS.OUT}/transport_bound.json')
