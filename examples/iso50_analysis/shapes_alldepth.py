"""dQ/dx shapes at every depth: 4-cm segments, 950->700/depth, all three
estimators, one panel per depth."""
import numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import iso50_analyse as A50
from rebin4 import seg_dqdx_bw

COL = {'effq': '#2e8b57', 'hits': '#a6304a', 'decC': '#d1701a'}
BW = 4.0
plt.rcParams.update({'font.size': 7.5, 'axes.grid': True, 'grid.alpha': 0.25,
                     'legend.frameon': False, 'figure.dpi': 150})
fig, axes = plt.subplots(5, 2, figsize=(9.6, 12.0), sharex=True)
axes = axes.ravel()
for i, tag in enumerate(A50.TAGS):
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    ax = axes[i]
    for k, c in COL.items():
        segs = []
        for ev in range(50):
            if k == 'effq':
                l = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
                q = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
                segs.append(seg_dqdx_bw(l[:,0], l[:,1], q, BW))
            elif k == 'hits':
                l = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
                q = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
                segs.append(seg_dqdx_bw(l[:,0], l[:,1], q, BW))
            else:
                segs.append(seg_dqdx_bw(*A50.deconv_pix(A50.find_solved('C', tag, ev)), BW))
        v = np.concatenate([x for x in segs if len(x)])
        v = v[(v > 20) & (v < 120)]
        h, e = np.histogram(v, bins=30, range=(20, 120))
        ctr = 0.5*(e[1:]+e[:-1])
        ax.errorbar(ctr, h, yerr=np.sqrt(h), drawstyle='steps-mid', color=c,
                    lw=1.0, elinewidth=0.5, label=k if i == 0 else None)
    d = tag.split('_d')[1].replace('p', '.')
    t = float(d) * 10 / 1.59645
    ax.set_title(f'd = {d} cm   (t = {t:.0f} µs)', fontsize=8)
    if i % 2 == 0:
        ax.set_ylabel('segments/bin')
    if i == 0:
        ax.legend(fontsize=7)
for ax in axes[-2:]:
    ax.set_xlabel('dQ/dx [ke/cm]')
fig.suptitle('4-cm-segment dQ/dx at every depth (700 segments per estimator per panel)',
             fontsize=9, y=0.995)
fig.tight_layout(pad=0.5)
fig.savefig('shapes_alldepth.png', dpi=140)
print('wrote shapes_alldepth.png')
