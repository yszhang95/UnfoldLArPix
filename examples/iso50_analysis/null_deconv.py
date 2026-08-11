"""Deconv-side null test: paired decC/effq slope at 20 ms vs 1 ms."""
import numpy as np, sys, os
sys.path.insert(0, '.')
import iso50_analyse as A50
from track_dqdx import px_to_cm, fit_direction, BIN_CM, TUBE_CM

AO = A50.AO
def pts(pa, pb, q):
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    return np.stack([y, z], axis=1), np.asarray(q, float)

def slope_pair(f, dec_dir, tag):
    Xs, Ys = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        yz_e, q_e = pts(el[:,0], el[:,1], eq)
        c, d = fit_direction(yz_e, q_e)
        if c is None: continue
        if dec_dir == 'ARM_C_1MS':
            p = A50.find_solved('C', tag, ev)
            if p is None: continue
        else:
            p = f'{dec_dir}/{tag}/{tag}_event_0_{ev}.npz'
            if not os.path.exists(p): continue
        pa, pb, qd = A50.deconv_pix(p)
        yz_d, q_d = pts(pa, pb, qd)
        pr_e = (yz_e - c) @ d
        pp_e = np.linalg.norm((yz_e - c) - np.outer(pr_e, d), axis=1)
        pr_d = (yz_d - c) @ d
        pp_d = np.linalg.norm((yz_d - c) - np.outer(pr_d, d), axis=1)
        ke, kd = pp_e < TUBE_CM, pp_d < TUBE_CM
        edges = np.arange(pr_e[ke].min(), pr_e[ke].max() + BIN_CM, BIN_CM)
        if len(edges) < 5: continue
        he, _ = np.histogram(pr_e[ke], bins=edges, weights=q_e[ke])
        hd, _ = np.histogram(pr_d[kd], bins=edges, weights=q_d[kd])
        ne = np.nonzero(he > 0)[0]
        sl = slice(ne[0]+1, ne[-1])
        Xs.append(he[sl]); Ys.append(hd[sl])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    m = (X > 0) & (Y > 0)
    return float((X[m]*Y[m]).sum()/(X[m]*X[m]).sum())

print('%-8s %12s %12s %12s %12s %12s' % ('d[cm]', 'std-C', '20ms', 'hitsupp', 'half-a', 'no-censor'))
for dd in ['d04p5', 'd16p5', 'd28p5']:
    tag = f'pgun_mu_3gev_iso50_{dd}'
    f1 = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    s1 = slope_pair(f1, 'ARM_C_1MS', tag)
    f20 = np.load(f'{A50.NFS}/{tag}_tred20ms_nb1.npz', allow_pickle=True)
    s20 = slope_pair(f20, f'{AO}/iso50_null20ms/C', tag)
    sh = slope_pair(f1, f'{AO}/iso50_hitsupp/C', tag)
    ha = slope_pair(f1, f'{AO}/iso50_halfalpha/C', tag)
    nc = slope_pair(f1, f'{AO}/iso50_nocensor/C', tag)
    print('%-8s %12.4f %12.4f %12.4f %12.4f %12.4f' % (dd, s1, s20, sh, ha, nc), flush=True)
