"""dQ/dx computation shown CONCRETELY on one event, every step visible,
plus the 10-depth bias curve with real error bars (mean +- SEM of 50
independent events, raw sums)."""
import numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import iso50_analyse as A50
import dqdx_lib as L
from track_dqdx import px_to_cm, fit_direction, BIN_CM, TUBE_CM

tag, EV = 'pgun_mu_3gev_iso50_d04p5', 7
f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
el = np.asarray(f[f'effq_tpc0_batch{EV}_location'])
eq = np.asarray(f[f'effq_tpc0_batch{EV}'], float)[:, 3]
pa, pb, qd = A50.deconv_pix(A50.find_solved('C', tag, EV))

ye, ze = px_to_cm(el[:,0].astype(float), el[:,1].astype(float))
yd, zd = px_to_cm(pa.astype(float), pb.astype(float))
c, d = fit_direction(np.stack([ye, ze], axis=1), eq)

plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.25,
                     'legend.frameon': False, 'figure.dpi': 150})
fig = plt.figure(figsize=(10, 8.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1])

# --- step 1: pixel plane, track fit, tube
ax = fig.add_subplot(gs[0, :])
sc = ax.scatter(ze, ye, c=eq, s=14, cmap='Greens', vmax=30, label='truth (effq)',
                marker='s', alpha=0.8)
ax.scatter(zd, yd, c=qd, s=7, cmap='Oranges', vmax=30, label='deconv$_C$',
           marker='o', alpha=0.7)
tt = np.array([-40, 40])
p0 = c + tt[0]*d; p1 = c + tt[1]*d
ax.plot([p0[1], p1[1]], [p0[0], p1[0]], 'k-', lw=1.2, label='PCA track fit')
n = np.array([-d[1], d[0]])
for s_ in (+TUBE_CM, -TUBE_CM):
    q0 = c + tt[0]*d + s_*n; q1 = c + tt[1]*d + s_*n
    ax.plot([q0[1], q1[1]], [q0[0], q1[0]], 'k--', lw=0.8)
# segment boundaries
rel = np.stack([ye, ze], 1) - c
pr = rel @ d
edges = np.arange(pr.min(), pr.max() + BIN_CM, BIN_CM)
for e_ in edges:
    q0 = c + e_*d + TUBE_CM*n; q1 = c + e_*d - TUBE_CM*n
    ax.plot([q0[1], q1[1]], [q0[0], q1[0]], color='0.6', lw=0.4)
ax.set_xlabel('z [cm]'); ax.set_ylabel('y [cm]')
ax.set_title(f'STEP 1: pixel plane, one event ({tag}, event {EV}) -- track fit, '
             f'$\\pm${TUBE_CM:.0f} cm tube, 3-cm segment boundaries', fontsize=8.5)
ax.legend(fontsize=7, loc='upper left'); ax.set_ylim(c[0]-8, c[0]+8)

# --- step 2: projection histogram
def proj_hist(y, z, q):
    rel = np.stack([y, z], 1) - c
    p = rel @ d
    perp = np.linalg.norm(rel - np.outer(p, d), axis=1)
    k = perp < TUBE_CM
    h, _ = np.histogram(p[k], bins=edges, weights=q[k])
    return h
he, hd = proj_hist(ye, ze, eq), proj_hist(yd, zd, qd)
ne = np.nonzero(he > 0)[0]
keepmask = np.zeros(len(he), bool); keepmask[ne[0]+1:ne[-1]] = True
ctr = 0.5*(edges[1:]+edges[:-1])
ax = fig.add_subplot(gs[1, :])
ax.stairs(he, edges, color='#2e8b57', lw=1.4, label='truth')
ax.stairs(hd, edges, color='#d1701a', lw=1.4, label='deconv$_C$')
for i in np.nonzero(~keepmask)[0]:
    ax.axvspan(edges[i], edges[i+1], color='0.85', alpha=0.6)
ax.set_xlabel('projection along track s [cm]'); ax.set_ylabel('charge / 3 cm [ke]')
ax.set_title('STEP 2: project every charge in the tube onto the track, histogram '
             'in 3-cm bins; grey = dropped end bins', fontsize=8.5)
ax.legend(fontsize=7)

# --- step 3: per-segment dQ/dx + ratio
ax = fig.add_subplot(gs[2, 0])
ax.step(ctr[keepmask], he[keepmask]/BIN_CM, where='mid', color='#2e8b57', label='truth')
ax.step(ctr[keepmask], hd[keepmask]/BIN_CM, where='mid', color='#d1701a', label='deconv$_C$')
ax.set_xlabel('s [cm]'); ax.set_ylabel('dQ/dx [ke/cm]')
ax.set_title('STEP 3: dQ/dx = bin content / 3 cm', fontsize=8.5)
ax.legend(fontsize=7)
ax = fig.add_subplot(gs[2, 1])
r = hd[keepmask]/np.maximum(he[keepmask], 1e-9)
ax.step(ctr[keepmask], r, where='mid', color='0.2')
ax.axhline(1, color='0.5', ls=':')
ax.axhline(np.median(r), color='#d1701a', ls='--',
           label=f'median ratio {np.median(r):.3f}')
ax.set_xlabel('s [cm]'); ax.set_ylabel('deconv / truth per segment')
ax.set_title('same segments, ratio', fontsize=8.5)
ax.legend(fontsize=7)
fig.tight_layout(pad=0.5)
fig.savefig('walkthrough.png', dpi=140)
print('median per-seg ratio this event:', round(float(np.median(r)),4))

# --- bias curve with SEM over all 10 depths (raw whole-event sums)
means, sems, ts = [], [], []
for tg in A50.TAGS:
    ff = np.load(f'{A50.NFS}/{tg}_tred_nb1.npz', allow_pickle=True)
    rr = []
    for ev in range(50):
        et = np.asarray(ff[f'effq_tpc0_batch{ev}'], float)[:,3].sum()
        _,_,qq = A50.deconv_pix(A50.find_solved('C', tg, ev))
        rr.append(qq.sum()/et)
    rr = np.array(rr)
    means.append(rr.mean()); sems.append(rr.std()/np.sqrt(len(rr)))
    ts.append(L.drift_time_us(float(tg.split('_d')[1].replace('p','.'))))
fig2, ax = plt.subplots(figsize=(5.6, 3.2))
ax.errorbar(ts, means, yerr=sems, fmt='o', color='#d1701a', ms=4, capsize=2.5,
            label='mean of 50 independent events $\\pm$ SEM')
ax.axhline(1, color='0.5', ls=':')
ax.set_xlabel('drift time [$\\mu$s]')
ax.set_ylabel('whole-event $\\Sigma$deconv$_C$ / $\\Sigma$truth')
ax.set_title('raw-sum bias vs depth; no fits, no histograms, no dQ/dx', fontsize=8.5)
ax.legend(fontsize=7)
fig2.tight_layout(pad=0.4)
fig2.savefig('bias_curve.png', dpi=140)
print('bias curve:', ' '.join(f'{m:.3f}±{s:.3f}' for m, s in zip(means, sems)))
