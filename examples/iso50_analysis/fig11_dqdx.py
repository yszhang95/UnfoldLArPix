"""Figure-11-style symmetric treatment, applied to the dQ/dx pipeline.

Fig 11 (app_corr2d) smears BOTH axes with the same 3D Gaussian:
freq-domain sigma_pixel=0.5 -> real-space 1/(2*pi*0.5) = 0.318 px, plus a
time kernel.  For segment dQ/dx the time kernel drops out exactly (charge
per pixel is time-integrated), so the recipe reduces to smearing BOTH the
effq pixel map and the decC pixel map with the same 0.318-px Gaussian.

Outputs: per-depth MPV + width for the symmetric pair, tau(MPV) at 3 and
4 cm, and a two-depth shape overlay against the unsmeared baseline.
"""
import numpy as np, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from rebin4 import seg_dqdx_bw
from smear_scan import pixmap, smear, widths

SIG = 1.0 / (2.0 * np.pi * 0.5)     # 0.318 px, fig-11 pixel kernel
SHOW = ['pgun_mu_3gev_iso50_d16p5', 'pgun_mu_3gev_iso50_d28p5']

ESTS = ['effq_f11', 'decC_f11', 'effq', 'decC']
pool = {bw: {k: {} for k in ESTS} for bw in (3.0, 4.0)}
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for bw in pool:
        for k in ESTS:
            pool[bw][k][tag] = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        pe = pixmap(el[:, 0], el[:, 1], eq)
        fp = A50.find_solved('C', tag, ev)
        pd = A50.deconv_pix(fp) if fp else None
        se = smear(*pe, SIG)
        sd = smear(*pd, SIG) if pd else None
        for bw in pool:
            pool[bw]['effq'][tag].append(seg_dqdx_bw(*pe, bw))
            pool[bw]['effq_f11'][tag].append(seg_dqdx_bw(*se, bw))
            pool[bw]['decC'][tag].append(
                seg_dqdx_bw(*pd, bw) if pd else np.array([]))
            pool[bw]['decC_f11'][tag].append(
                seg_dqdx_bw(*sd, bw) if sd else np.array([]))
    print(f'{tag} done', flush=True)

t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])

print('\n== per-depth MPV / Moyal core width (4 cm, fig-11 symmetric) ==')
print('%-28s %12s %12s %12s %12s' %
      ('depth', 'effq_f11', 'decC_f11', 'effq', 'decC'))
wtab = {}
for tag in A50.TAGS:
    row = {}
    for k in ESTS:
        allv = np.concatenate([v for v in pool[4.0][k][tag] if len(v)])
        row[k] = widths(allv)
    wtab[tag] = row
    print('%-28s' % tag.split('iso50_')[1] +
          ''.join(' %5.1f/%5.2f' % (row[k]['mpv'], row[k]['moyal_sig'])
                  for k in ESTS))
print('\ndepth-averaged Moyal core width:')
for k in ESTS:
    print('  %-9s %6.3f' %
          (k, np.nanmean([wtab[t][k]['moyal_sig'] for t in A50.TAGS])))

print('\n== tau(MPV) ==')
mpvfn = lambda v: L.mpv_of(v)[0]
out = {}
for bw in (3.0, 4.0):
    for k in ESTS:
        pe = [pool[bw][k][tag] for tag in A50.TAGS]
        tau, err = A50.boot_tau(t_us, pe, mpvfn)
        out[f'{k}_{bw:g}cm'] = {'tau': tau, 'err': err}
        print('  %-9s %gcm  tau = %6.3f +- %5.3f ms' % (k, bw, tau, err),
              flush=True)
json.dump(out, open(f'{A50.AO}/iso50_fig11_dqdx.json', 'w'), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
bins = np.linspace(30, 90, 31)
CS = {'effq_f11': '#2e8b57', 'decC_f11': '#d1701a'}
for ax, tag in zip(axes, SHOW):
    d = tag.split('_d')[1].replace('p', '.')
    for k, c in CS.items():
        v = np.concatenate([x for x in pool[4.0][k][tag] if len(x)])
        h, e = np.histogram(v, bins=bins)
        ax.step(0.5*(e[1:]+e[:-1]), h/h.sum(), where='mid', color=c,
                lw=1.7, label=f'{k} (n={len(v)})')
    for k, c in CS.items():
        v = np.concatenate([x for x in pool[4.0][k.split("_")[0]][tag]
                            if len(x)])
        h, e = np.histogram(v, bins=bins)
        ax.step(0.5*(e[1:]+e[:-1]), h/h.sum(), where='mid', color=c,
                lw=1.0, ls='--', alpha=0.6,
                label=f'{k.split("_")[0]} unsmeared')
    ax.set_title(f'd = {d} cm, 4-cm segments, fig-11 symmetric 0.318 px')
    ax.set_xlabel('dQ/dx [ke/cm]')
    ax.set_ylabel('fraction/bin')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('./fig11_dqdx.png', dpi=130)
print('wrote fig11_dqdx.png')
