"""THE reconciliation test: per-depth segment-level regression between
truth and deconv on a COMMON axis and COMMON bins."""
import numpy as np, sys
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import px_to_cm, fit_direction, BIN_CM, TUBE_CM

def pts(pa, pb, q):
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    return np.stack([y, z], axis=1), np.asarray(q, float)

print('%-6s %8s %9s %9s %7s' % ('d[cm]','r','slope','slope_0','n_seg'))
res = []
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    X, Y = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        yz_e, q_e = pts(el[:,0], el[:,1], eq)
        c, d = fit_direction(yz_e, q_e)
        if c is None: continue
        pdec = A50.find_solved('C', tag, ev)
        pa, pb, qd = A50.deconv_pix(pdec)
        yz_d, q_d = pts(pa, pb, qd)
        # common bins from the truth's projection
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
        X.append(he[sl]); Y.append(hd[sl])
    X = np.concatenate(X)/BIN_CM; Y = np.concatenate(Y)/BIN_CM
    m = (X > 0) & (Y > 0)
    X, Y = X[m], Y[m]
    r = float(np.corrcoef(X, Y)[0,1])
    slope = float(np.polyfit(X, Y, 1)[0])          # free intercept
    slope0 = float((X*Y).sum()/(X*X).sum())        # through origin
    d_cm = float(tag.split('_d')[1].replace('p','.'))
    res.append((L.drift_time_us(d_cm), slope0))
    print('%-6.1f %8.4f %9.4f %9.4f %7d' % (d_cm, r, slope, slope0, len(X)), flush=True)
tr = np.array(res)
b, a = np.polyfit(tr[1:,0], tr[1:,1], 1)
print(f'\nthrough-origin slope trend (drop d=1.5): {100*b*169/a:+.2f}% over the drift')
