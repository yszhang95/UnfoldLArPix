"""Overlay: truth vs best-smeared truth vs decC at two depths, plus the
depth trend of the Moyal core width from iso50_smear_scan.json."""
import numpy as np, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from rebin4 import seg_dqdx_bw
from smear_scan import pixmap, smear

BW = 4.0
SHOW = ['pgun_mu_3gev_iso50_d16p5', 'pgun_mu_3gev_iso50_d28p5']
COL = {'effq': '#2e8b57', 'sm1': '#4062bb', 'decC': '#d1701a', 'hits': '#a6304a'}
LBL = {'effq': 'effq (truth)', 'sm1': 'effq smeared 1 px',
       'decC': 'decC', 'hits': 'hits'}

pools = {}
for tag in SHOW:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    p = {'effq': [], 'sm1': [], 'decC': []}
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        pa, pb, q = pixmap(el[:, 0], el[:, 1], eq)
        p['effq'].append(seg_dqdx_bw(pa, pb, q, BW))
        p['sm1'].append(seg_dqdx_bw(*smear(pa, pb, q, 1.0), BW))
        fp = A50.find_solved('C', tag, ev)
        if fp:
            p['decC'].append(seg_dqdx_bw(*A50.deconv_pix(fp), BW))
    pools[tag] = {k: np.concatenate([v for v in p[k] if len(v)])
                  for k in p}

scan = json.load(open(f'{A50.AO}/iso50_smear_scan.json'))
tags = A50.TAGS
t_us = [L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
        for t in tags]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
bins = np.linspace(30, 90, 31)
for ax, tag in zip(axes[:2], SHOW):
    d = tag.split('_d')[1].replace('p', '.')
    for k in ['effq', 'sm1', 'decC']:
        v = pools[tag][k]
        h, e = np.histogram(v, bins=bins)
        c = 0.5 * (e[1:] + e[:-1])
        ax.step(c, h / h.sum(), where='mid', color=COL[k],
                label=f"{LBL[k]}  (n={len(v)})", lw=1.6)
    ax.set_title(f'd = {d} cm, 4-cm segments (normalized)')
    ax.set_xlabel('dQ/dx [ke/cm]')
    ax.set_ylabel('fraction/bin')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

ax = axes[2]
for k, sk in [('effq', 'sm0'), ('sm1', 'sm1'), ('hits', 'hits'),
              ('decC', 'decC')]:
    ax.plot(t_us, [scan[t][sk]['moyal_sig'] for t in tags], 'o-',
            color=COL[k], label=LBL[k], ms=4)
ax.set_xlabel('drift time [µs]')
ax.set_ylabel('Moyal core width σ [ke/cm]')
ax.set_title('core width vs drift')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('./smear_check.png', dpi=130)
print('wrote smear_check.png')
