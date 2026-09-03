"""iso50, arm C, d >= 16.5 cm: the dQ/dx distributions and the lifetime fit.

ONE sample, ONE reconstruction arm.  Nothing is re-simulated or re-solved;
every input is read from the archive.  The pipeline is the one of record --
track_dqdx.segment_dqdx + dqdx_lib.mpv_of + iso50_analyse.boot_tau -- used
verbatim, not re-implemented.
"""
import numpy as np, os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import track_dqdx as T

NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield'
AO = '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output'
OUT = f'{AO}/iso50_deep'
TAGS = ['pgun_mu_3gev_iso50_d16p5', 'pgun_mu_3gev_iso50_d19p5',
        'pgun_mu_3gev_iso50_d22p5', 'pgun_mu_3gev_iso50_d25p5',
        'pgun_mu_3gev_iso50_d28p5']
NEV = 50
CO = {'effq': '#000000', 'hits': '#E69F00', 'decC': '#0072B2'}
LB = {'effq': r'$\mathrm{eff}Q$  (TRUTH, input sample)',
      'hits': 'raw hits  (input sample)',
      'decC': 'deconv arm C  (output sample)'}


def style(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    ax.grid(False)


def deconv_pix(path):
    z = np.load(path, allow_pickle=True)
    per = np.asarray(z['deconv_q_sharp'], float).sum(axis=2)
    off = np.asarray(z['boffset'], float)
    a_, b_ = np.nonzero(per > 0)
    return a_ + int(off[0]), b_ + int(off[1]), per[a_, b_]


def collect():
    per = {k: [] for k in CO}
    t_us, d_cm = [], []
    for tag in TAGS:
        d = float(tag.split('_d')[1].replace('p', '.'))
        d_cm.append(d); t_us.append(L.drift_time_us(d))
        f = np.load(f'{NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        ev_seg = {k: [] for k in CO}
        for ev in range(NEV):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
            hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
            ev_seg['effq'].append(T.segment_dqdx(el[:, 0], el[:, 1], eq))
            ev_seg['hits'].append(T.segment_dqdx(hl[:, 0], hl[:, 1], hq))
            p = f'{AO}/iso50/C/{tag}/{tag}_event_0_{ev}.npz'
            ev_seg['decC'].append(T.segment_dqdx(*deconv_pix(p))
                                  if os.path.exists(p) else np.array([]))
        for k in CO:
            per[k].append(ev_seg[k])
    return np.array(d_cm), np.array(t_us), per


def boot_lambda(t_us, per_depth, nb=300, seed=1):
    """iso50_analyse.boot_tau, resampling the 50 EVENTS at each depth."""
    rng = np.random.default_rng(seed)
    lam = []
    A = np.vstack([t_us, np.ones_like(t_us)]).T
    for _ in range(nb):
        m = []
        for evs in per_depth:
            pk = rng.integers(0, len(evs), len(evs))
            v = np.concatenate([evs[i] for i in pk if len(evs[i])])
            m.append(L.mpv_of(v)[0])
        lam.append(-np.linalg.lstsq(A, np.log(m), rcond=None)[0][0] * 1000)
    lam = np.array(lam)
    return float(np.median(lam)), float(lam.std(ddof=1))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    d_cm, t_us, per = collect()
    mpv = {k: np.array([L.mpv_of(np.concatenate([x for x in per[k][i] if len(x)]))[0]
                        for i in range(len(TAGS))]) for k in CO}
    nseg = {k: [int(sum(len(x) for x in per[k][i])) for i in range(len(TAGS))] for k in CO}
    fit = {k: boot_lambda(t_us, per[k]) for k in CO}

    # ---------------------------------------------------- Fig A: distributions
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    bins = np.linspace(0, 200, 51)
    for i, tag in enumerate(TAGS):
        ax = axes[i]; style(ax)
        for k in ['effq', 'hits', 'decC']:
            v = np.concatenate([x for x in per[k][i] if len(x)])
            h, e = np.histogram(v[(v > 0) & (v < 200)], bins=bins, density=True)
            c = 0.5 * (e[1:] + e[:-1])
            ax.step(c, h, where='mid', color=CO[k], lw=1.4,
                    label=LB[k] if i == 0 else None)
            ax.axvline(mpv[k][i], color=CO[k], ls=':', lw=1.1)
        ax.set_title(f'$d$ = {d_cm[i]:.1f} cm   ($t$ = {t_us[i]:.0f} $\\mu$s)\n'
                     f'{nseg["effq"][i]} segments', fontsize=9)
        ax.set_xlabel(r'd$Q$/d$x$  [ke/cm]', fontsize=9)
        ax.tick_params(labelsize=8); ax.set_xlim(0, 200)
    axes[0].set_ylabel('segments (normalised)', fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('iso50, arm C  --  pooled d$Q$/d$x$, 3 cm segments, '
                 '50 events per depth   (dotted: Moyal MPV, the fitted statistic)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/deep_dqdx_shapes.pdf'); fig.savefig(f'{OUT}/deep_dqdx_shapes.png', dpi=150)

    # ------------------------------------------------------- Fig B: the fit
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for a in axes:
        style(a)
    tt = np.linspace(t_us.min() - 5, t_us.max() + 5, 50)
    for k in ['effq', 'hits', 'decC']:
        A = np.vstack([t_us, np.ones_like(t_us)]).T
        sl, ic = np.linalg.lstsq(A, np.log(mpv[k]), rcond=None)[0]
        axes[0].plot(t_us, mpv[k], 'o', color=CO[k], ms=6,
                     label=f'{LB[k]}\n   $\\lambda$ = {fit[k][0]:.3f} $\\pm$ {fit[k][1]:.3f} ms$^{{-1}}$')
        axes[0].plot(tt, np.exp(ic + sl * tt), '-', color=CO[k], lw=1.2)
        axes[1].plot(t_us, mpv[k] / mpv['effq'], 'o-', color=CO[k], ms=6, lw=1.2)
    axes[0].plot(tt, mpv['effq'][0] * np.exp(-(tt - t_us[0]) / 1000.0), 'k--', lw=1.0,
                 label=r'true slope, $\lambda$ = 1 ms$^{-1}$')
    axes[0].set_xlabel(r'drift time  [$\mu$s]'); axes[0].set_ylabel('pooled MPV  [ke/cm]')
    axes[0].set_title('unweighted OLS of $\\ln$MPV on $t$  (5 depths, 16.5-28.5 cm)', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(1.0, color='0.6', lw=0.8)
    axes[1].set_xlabel(r'drift time  [$\mu$s]')
    axes[1].set_ylabel(r'MPV / MPV$_{\rm truth}$')
    axes[1].set_title('capture: what the fit is actually a slope of', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/deep_lifetime_fit.pdf'); fig.savefig(f'{OUT}/deep_lifetime_fit.png', dpi=150)

    print('=' * 78)
    print('iso50, arm C, d >= 16.5 cm   (5 depths, 50 events each)')
    print('=' * 78)
    print('%8s %8s %11s %11s %11s %10s' % ('d[cm]', 't[us]', 'MPV effq', 'MPV hits',
                                           'MPV decC', 'decC/effq'))
    for i in range(len(TAGS)):
        print('%8.1f %8.1f %11.2f %11.2f %11.2f %10.4f'
              % (d_cm[i], t_us[i], mpv['effq'][i], mpv['hits'][i], mpv['decC'][i],
                 mpv['decC'][i] / mpv['effq'][i]))
    print()
    print('fitted lambda [1/ms], bootstrap over the 50 events, 300 resamples:')
    for k in ['effq', 'hits', 'decC']:
        print('   %-6s %.4f +- %.4f' % (k, *fit[k]))
    print('   truth value: 1.0000')
    json.dump({'d_cm': d_cm.tolist(), 't_us': t_us.tolist(),
               'mpv': {k: mpv[k].tolist() for k in mpv},
               'n_segments': nseg, 'lambda': {k: fit[k] for k in fit}},
              open(f'{OUT}/deep.json', 'w'), indent=1)
    print('\n->', OUT)
