"""Segment length 4 cm (user request): shapes + the tau table, side by
side with 3 cm."""
import numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import iso50_analyse as A50
import dqdx_lib as L
import track_dqdx as T
from track_dqdx import px_to_cm, fit_direction, TUBE_CM

def seg_dqdx_bw(pa, pb, q, bw):
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1); q = np.asarray(q, float)
    c, d = fit_direction(yz, q)
    if c is None: return np.array([])
    rel = yz - c; proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    k = perp < TUBE_CM
    if k.sum() < 5: return np.array([])
    edges = np.arange(proj[k].min(), proj[k].max() + bw, bw)
    if len(edges) < 4: return np.array([])
    h, _ = np.histogram(proj[k], bins=edges, weights=q[k])
    ne = np.nonzero(h > 0)[0]
    if len(ne) < 3: return np.array([])
    h = h[ne[0]+1:ne[-1]]
    return h[h > 0] / bw

COL = {'effq': '#2e8b57', 'hits': '#a6304a', 'decC': '#d1701a'}
res = {}
for BW in [3.0, 4.0]:
    per = {k: {} for k in COL}
    for tag in A50.TAGS:
        f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for k in per: per[k][tag] = []
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
            hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
            per['effq'][tag].append(seg_dqdx_bw(el[:,0], el[:,1], eq, BW))
            per['hits'][tag].append(seg_dqdx_bw(hl[:,0], hl[:,1], hq, BW))
            per['decC'][tag].append(seg_dqdx_bw(*A50.deconv_pix(A50.find_solved('C', tag, ev)), BW))
    res[BW] = per
    t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p','.'))) for t in A50.TAGS])
    print(f'-- {BW:.0f} cm segments --')
    for k in COL:
        pe = [per[k][tag] for tag in A50.TAGS]
        tau, err = A50.boot_tau(t_us, pe, lambda v: L.mpv_of(v)[0], nboot=120)
        n = sum(len(v) for v in per[k][A50.TAGS[3]])
        print(f'  {k:6s} tau(MPV) = {tau:6.3f} +- {err:5.3f} ms   (~{n} seg/depth)', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
for ax, BW in zip(axes, [3.0, 4.0]):
    tag = 'pgun_mu_3gev_iso50_d16p5'
    for k, c in COL.items():
        v = np.concatenate([x for x in res[BW][k][tag] if len(x)])
        v = v[(v > 20) & (v < 120)]
        h, e = np.histogram(v, bins=30, range=(20, 120))
        ctr = 0.5*(e[1:]+e[:-1])
        ax.errorbar(ctr, h, yerr=np.sqrt(h), drawstyle='steps-mid', color=c,
                    lw=1.1, elinewidth=0.6, label=k)
    ax.set_xlabel('dQ/dx [ke/cm]'); ax.set_title(f'{BW:.0f}-cm segments, d=16.5', fontsize=8.5)
    ax.legend(fontsize=7)
axes[0].set_ylabel('segments/bin')
fig.tight_layout(pad=0.4)
fig.savefig('rebin4.png', dpi=145)
print('wrote rebin4.png')
