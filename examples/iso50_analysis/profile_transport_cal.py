"""Calibrate the transport deficit Delta into a charge fraction, and measure
what that fraction is worth on the dQ/dx MPV.

One operator throughout, used for BOTH the calibration and the leverage:

    scatter(f): every row sends fraction f of its charge to a uniformly
                random neighbour (b+1 or b-1).  Charge exact, centroid and
                global width untouched -- pure transport.

Calibration: inject scatter(f) into `hits` (Delta_0 = 0 by construction,
the transport-free baseline) and into `decC` (local slope at the operating
point); read off Delta(f).
Leverage:    inject scatter(f) into the TRUTH pixel charges and push it
             through the real segment_dqdx pipeline -> pooled MPV.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import track_dqdx as T
import dqdx_lib as L

FR = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


def scatter_profile(P, f, rng):
    s = rng.choice([-1, 1], P.size)
    out = P * (1 - f)
    mv = P * f
    idx = np.arange(P.size) + s
    ok = (idx >= 0) & (idx < P.size)
    np.add.at(out, idx[ok], mv[ok])
    return out


def scatter_pixels(pa, pb, q, f, rng):
    s = rng.choice([-1, 1], q.size)
    return (np.concatenate([pa, pa]),
            np.concatenate([pb, pb + s]),
            np.concatenate([q * (1 - f), q * f]))


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    print('=' * 78)
    print('CALIBRATION: Delta vs injected scatter fraction f (pooled, 3 depths)')
    print('=' * 78)
    base = {'hits': [], 'decC': []}
    for gi in [3, 5, 7]:
        tag = PS.TAGS[gi]
        f = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for ev in range(50):
            pr = PT.profiles(tag, ev, f)
            if pr is None:
                continue
            base['hits'].append((pr['truth'], pr['hits']))
            base['decC'].append((pr['truth'], pr['decC']))
    print('%-8s %14s %14s' % ('f', 'Delta(hits)', 'Delta(decC)'))
    cal = {}
    for q in FR:
        row = []
        for k in ['hits', 'decC']:
            pr = [(t, scatter_profile(r, q, rng) if q else r) for t, r in base[k]]
            rho, _, _ = PT.rho_of(pr)
            row.append(PT.delta_of(rho))
        cal[q] = row
        print('%-8.2f %14.4f %14.4f' % (q, row[0], row[1]))

    dh = np.array([cal[q][0] for q in FR]) - cal[0.0][0]
    dc = np.array([cal[q][1] for q in FR]) - cal[0.0][1]
    fr = np.array(FR)
    sh = np.polyfit(fr[:4], dh[:4], 1)[0]
    sc = np.polyfit(fr[:4], dc[:4], 1)[0]
    print(f'\nlocal slope dDelta/df:  hits {sh:.4f}   decC {sc:.4f}')

    print()
    print('=' * 78)
    print('IMPLIED TRANSPORT FRACTION per depth (Delta / slope), arm C')
    print('=' * 78)
    J = json.load(open(f'{PS.OUT}/transport.json'))
    tags = list(J)
    tt, ff, ee = [], [], []
    print('%-7s %10s %10s %12s %12s' % ('depth', 'Delta', 'err', 'f(vs truth)', 'f(vs hits)'))
    for g in tags:
        r = J[g]
        d, de = r['decC']['delta'], r['decC']['delta_err']
        dh2 = r['decC|hits']['delta']
        tt.append(r['t_us']); ff.append(d / sh); ee.append(de / sh)
        print('%-7.1f %+10.4f %10.4f %12.4f %12.4f'
              % (r['depth_cm'], d, de, d / sh, dh2 / sh))
    tt, ff, ee = np.array(tt), np.array(ff), np.array(ee)
    m = tt > 20
    w = 1 / ee[m] ** 2
    A = np.vstack([tt[m], np.ones(m.sum())]).T
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    p = cov @ (A.T @ np.diag(w) @ ff[m])
    span = tt[m].max() - tt[m].min()
    print(f'\nmean f (9 depths, vs truth) = {np.average(ff[m], weights=w):.4f}'
          f' +- {1/np.sqrt(w.sum()):.4f}')
    print(f'depth trend of f: {p[0]:+.3e} +- {np.sqrt(cov[0,0]):.3e} per us'
          f'  ->  {p[0]*span:+.4f} +- {np.sqrt(cov[0,0])*span:.4f} over 28-179 us')

    print()
    print('=' * 78)
    print('LEVERAGE: scatter(f) applied to the TRUTH, through segment_dqdx')
    print('=' * 78)
    print('%-7s' % 'depth' + ''.join('%11s' % f'f={q:.2f}' for q in FR))
    slide = {q: [] for q in FR}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        mp = {}
        for q in FR:
            pool = []
            for ev in range(50):
                ta, tb, tq = PS.truth_pix(fz, ev)
                if q:
                    ta, tb, tq = scatter_pixels(ta, tb, tq, q, rng)
                pool.append(T.segment_dqdx(ta.astype(float), tb.astype(float), tq))
            v = np.concatenate([x for x in pool if len(x)])
            mp[q] = L.mpv_of(v)[0]
            slide[q].append((L.drift_time_us(d_cm), mp[q]))
        print('%-7.1f' % d_cm + ''.join('%11.4f' % (mp[q] / mp[0.0]) for q in FR))
    print()
    print('effect on the FITTED decay rate (9 depths, d>3 cm):')
    for q in FR:
        s = [x for x in slide[q] if x[0] > 20]
        t_ = np.array([x[0] for x in s]); y = np.log([x[1] for x in s])
        lam = -np.polyfit(t_, y, 1)[0] * 1000
        print(f'   f={q:.2f}  lambda = {lam:.4f} /ms')
