"""Figures of record for the technote iso50 section.

Outputs (PDF) into analysis_output/iso50_report/:
  tau_fit.pdf     ln(MPV) vs drift time + exponential fits, 3-cm segments
  shapes_grid.pdf pooled 4-cm dQ/dx shapes at every depth
  tilt_width.pdf  (a) per-event integral ratio vs depth  (b) Moyal core
                  width vs depth (truth / hits / decC)
"""
import numpy as np, sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from rebin4 import seg_dqdx_bw

OUT = f'{A50.AO}/iso50_report'
os.makedirs(OUT, exist_ok=True)
COL = {'effq': '#2e8b57', 'hits': '#a6304a', 'decC': '#d1701a'}
LBL = {'effq': 'effq (truth)', 'hits': 'raw hits', 'decC': 'deconv (C)'}

# ---- gather segments + per-event sums --------------------------------
seg3 = {k: {} for k in COL}
seg4 = {k: {} for k in COL}
sums = {k: {} for k in COL}
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for k in COL:
        seg3[k][tag] = []; seg4[k][tag] = []; sums[k][tag] = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
        hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
        fp = A50.find_solved('C', tag, ev)
        dd = A50.deconv_pix(fp) if fp else None
        src = {'effq': (el[:, 0], el[:, 1], eq),
               'hits': (hl[:, 0], hl[:, 1], hq), 'decC': dd}
        for k, s in src.items():
            if s is None:
                continue
            seg3[k][tag].append(segment_dqdx(*s))
            seg4[k][tag].append(seg_dqdx_bw(*s, 4.0))
            sums[k][tag].append(float(np.sum(s[2])))
    print(f'{tag} done', flush=True)

t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

# ---- tau_fit.pdf ------------------------------------------------------
# Fits exclude the shallowest depth (d = 1.5 cm): the acquisition-edge
# response truncation (the Ramo prompt deficit, 1.3% there) puts the
# point off-trend for every reconstructed estimator. It is plotted open.
FIT = slice(1, None)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
res = {}
for k in COL:
    m = np.array([mpvfn(np.concatenate(seg3[k][tag])) for tag in A50.TAGS])
    tau, err = A50.boot_tau(t_us[FIT],
                            [seg3[k][tag] for tag in A50.TAGS[FIT]], mpvfn)
    res[k] = {'tau3': tau, 'err3': err}
    b, a = np.polyfit(t_us[FIT], np.log(m[FIT]), 1)
    ax.plot(t_us[FIT], m[FIT], 'o', color=COL[k], ms=5)
    ax.plot(t_us[:1], m[:1], 'o', mfc='none', color=COL[k], ms=6)
    tt = np.linspace(0, 185, 50)
    lam, lerr = 1.0/tau, err/tau**2
    ax.plot(tt, np.exp(a + b*tt), '-', color=COL[k], lw=1.2,
            label=rf'{LBL[k]}: $\lambda = {lam:.3f} \pm {lerr:.3f}$ ms$^{{-1}}$')
ax.plot(tt, np.exp(np.log(57.7) - tt/1000.), 'k--', lw=1,
        label=r'true $\lambda = 1$ ms$^{-1}$ (slope)')
ax.set_yscale('log')
ax.set_xlabel(r'drift time [$\mu$s]')
ax.set_ylabel('dQ/dx MPV [ke/cm] (3-cm segments)')
ax.set_yticks([45, 50, 55, 60], ['45', '50', '55', '60'])
ax.set_title('open marker: d = 1.5 cm, excluded from fits', fontsize=9)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/tau_fit.pdf'); plt.close(fig)

for k in COL:
    tau, err = A50.boot_tau(t_us[FIT],
                            [seg4[k][tag] for tag in A50.TAGS[FIT]], mpvfn)
    res[k].update(tau4=tau, err4=err)
json.dump(res, open(f'{OUT}/tau_report.json', 'w'), indent=1)
print('tau_fit.pdf', {k: round(v['tau3'], 3) for k, v in res.items()},
      flush=True)

# ---- shapes_grid.pdf --------------------------------------------------
fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
bins = np.linspace(20, 120, 31)
for ax, tag in zip(axes.ravel(), A50.TAGS):
    d = float(tag.split('_d')[1].replace('p', '.'))
    for k in COL:
        v = np.concatenate([x for x in seg4[k][tag] if len(x)])
        h, e = np.histogram(v, bins=bins)
        c = 0.5*(e[1:]+e[:-1])
        ax.step(c, h, where='mid', color=COL[k], lw=1.3, label=LBL[k])
        ax.errorbar(c, h, yerr=np.sqrt(h), fmt='none', ecolor=COL[k],
                    elinewidth=0.6, alpha=0.55)
    ax.set_title(rf'$d = {d:g}$ cm   ($t = {L.drift_time_us(d):.0f}\,\mu$s)',
                 fontsize=10)
    ax.grid(alpha=0.25)
    if ax is axes.ravel()[0]:
        ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('dQ/dx [ke/cm]')
for row in axes:
    row[0].set_ylabel('segments/bin')
fig.tight_layout(); fig.savefig(f'{OUT}/shapes_grid.pdf'); plt.close(fig)
print('shapes_grid.pdf', flush=True)

# ---- tilt_width.pdf ---------------------------------------------------
scan = json.load(open(f'{A50.AO}/iso50_smear_scan.json'))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for k in ['hits', 'decC']:
    r = np.array([[s/e for s, e in zip(sums[k][tag], sums['effq'][tag])]
                  for tag in A50.TAGS])
    a1.errorbar(t_us, r.mean(axis=1), yerr=r.std(axis=1)/np.sqrt(r.shape[1]),
                fmt='o-', color=COL[k], ms=4, lw=1.2, label=LBL[k])
a1.axhline(1.0, color='k', ls='--', lw=1)
a1.set_xlabel(r'drift time [$\mu$s]')
a1.set_ylabel(r'per-event $\Sigma q_\mathrm{reco} / \Sigma q_\mathrm{effq}$')
a1.set_title('integral capture vs depth (mean $\\pm$ SEM, 50 events)')
a1.legend(fontsize=9); a1.grid(alpha=0.3)
for k, sk in [('effq', 'sm0'), ('hits', 'hits'), ('decC', 'decC')]:
    a2.plot(t_us, [scan[t][sk]['moyal_sig'] for t in A50.TAGS], 'o-',
            color=COL[k], ms=4, lw=1.2, label=LBL[k])
a2.set_xlabel(r'drift time [$\mu$s]')
a2.set_ylabel(r'Moyal core width $\sigma$ [ke/cm] (4-cm segments)')
a2.set_title('dQ/dx core width vs depth')
a2.legend(fontsize=9); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/tilt_width.pdf'); plt.close(fig)
print('tilt_width.pdf', flush=True)
