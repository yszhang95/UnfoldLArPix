"""Transverse (across-pixel) profile vs depth: width, and tube leakage.

The one positional mechanism that COULD give a depth-dependent dQ/dx bias:
transverse diffusion widens the track with drift, so charge leaks out of
the +-2 cm dQ/dx tube by a depth-dependent amount.  Measured for truth and
for both reconstructions.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import dqdx_lib as L

TUBE_PX = int(round(2.0 / 0.4434))     # +-2 cm dQ/dx tube, in pixels
rows = []
print('%-6s %-6s %8s %8s %8s %8s' % ('depth', 'est', 'rms[px]', 'f|<=3|',
                                     'f|<=4.5cm|', 'cen_a'))
for tag in PS.TAGS:
    d_cm = float(tag.split('_d')[1].replace('p', '.'))
    f = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    acc = {k: {'rms': [], 'f3': [], 'ftube': [], 'cen': []}
           for k in ['effq', 'hits', 'decC', 'decB']}
    for ev in range(50):
        ta, tb, tq = PS.truth_pix(f, ev)
        est = {'effq': (ta, tb, tq), 'hits': PS.hits_pix(f, ev)}
        for arm in ['C', 'B']:
            p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
            if os.path.exists(p):
                est[f'dec{arm}'] = PS.deconv_pix(p)
        # trunk row from truth, and a longitudinal window well inside
        blo, bhi = np.percentile(tb, [10, 90])
        w = (tb >= blo) & (tb <= bhi)
        ac = float(np.average(ta[w], weights=tq[w]))
        for k, (pa, pb, q) in est.items():
            m = (pb >= blo) & (pb <= bhi)
            if not m.any() or q[m].sum() <= 0:
                continue
            a, qq = pa[m].astype(float), q[m]
            s = qq.sum()
            cen = float((a * qq).sum() / s)
            rms = float(np.sqrt(((a - cen) ** 2 * qq).sum() / s))
            acc[k]['rms'].append(rms)
            acc[k]['cen'].append(cen - ac)
            acc[k]['f3'].append(float(qq[np.abs(a - ac) <= 3].sum() / s))
            acc[k]['ftube'].append(float(qq[np.abs(a - ac) <= TUBE_PX].sum() / s))
    r = {'depth_cm': d_cm, 't_us': L.drift_time_us(d_cm)}
    for k in acc:
        if not acc[k]['rms']:
            continue
        r[k] = {kk: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))]
                for kk, v in acc[k].items()}
        print('%-6.1f %-6s %8.4f %8.5f %8.5f %+8.4f'
              % (d_cm, k, r[k]['rms'][0], r[k]['f3'][0], r[k]['ftube'][0],
                 r[k]['cen'][0]))
    rows.append(r)
    print()
json.dump(rows, open(f'{PS.OUT}/transverse.json', 'w'), indent=1)
print('->', f'{PS.OUT}/transverse.json')
