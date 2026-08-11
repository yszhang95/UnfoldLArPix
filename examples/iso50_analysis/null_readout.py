"""Null-test, readout side: paired hits-vs-effq segment slope at
lifetime = 1 ms (amplitude falls 16%) vs 20 ms (amplitude ~constant).
If the depth slide of the slope vanishes at 20 ms, the readout capture
tilt is driven by AMPLITUDE (threshold coupling), not by depth per se."""
import numpy as np, sys
sys.path.insert(0, '.')
import iso50_analyse as A50
from track_dqdx import px_to_cm, fit_direction, BIN_CM, TUBE_CM

def pts(pa, pb, q):
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    return np.stack([y, z], axis=1), np.asarray(q, float)

def slope_of(f):
    Xs, Ys = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
        hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
        yz_e, q_e = pts(el[:,0], el[:,1], eq)
        c, d = fit_direction(yz_e, q_e)
        if c is None: continue
        yz_h, q_h = pts(hl[:,0], hl[:,1], hq)
        pr_e = (yz_e - c) @ d
        pp_e = np.linalg.norm((yz_e - c) - np.outer(pr_e, d), axis=1)
        pr_h = (yz_h - c) @ d
        pp_h = np.linalg.norm((yz_h - c) - np.outer(pr_h, d), axis=1)
        ke, kh = pp_e < TUBE_CM, pp_h < TUBE_CM
        edges = np.arange(pr_e[ke].min(), pr_e[ke].max() + BIN_CM, BIN_CM)
        if len(edges) < 5: continue
        he, _ = np.histogram(pr_e[ke], bins=edges, weights=q_e[ke])
        hh, _ = np.histogram(pr_h[kh], bins=edges, weights=q_h[kh])
        ne = np.nonzero(he > 0)[0]
        sl = slice(ne[0]+1, ne[-1])
        Xs.append(he[sl]); Ys.append(hh[sl])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    m = (X > 0) & (Y > 0)
    return float((X[m]*Y[m]).sum()/(X[m]*X[m]).sum())

print('%-8s %12s %12s' % ('d[cm]', 'slope@1ms', 'slope@20ms'))
for dd in ['d04p5', 'd16p5', 'd28p5']:
    tag = f'pgun_mu_3gev_iso50_{dd}'
    s1 = slope_of(np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True))
    try:
        s20 = slope_of(np.load(f'{A50.NFS}/{tag}_tred20ms_nb1.npz', allow_pickle=True))
        s20s = f'{s20:12.4f}'
    except FileNotFoundError:
        s20s = '     pending'
    print('%-8s %12.4f %s' % (dd, s1, s20s), flush=True)
