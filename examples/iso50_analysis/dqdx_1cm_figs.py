"""dQ/dx at 1 cm segmentation: pooled shapes, segment-length dependence, and
the along-track profile.

1 cm = 2.26 px, so unlike the 3 cm segments of the note this granularity is
NOT blind to the +-1 px sharpening measured in transport_kernel.py.
"""
import numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

OUT = f'{PS.OUT}/figs'
os.makedirs(OUT, exist_ok=True)
CO = {'effq': '#000000', 'hits': '#E69F00', 'decC': '#0072B2', 'decB': '#D55E00'}
LB = {'effq': r'$\mathrm{eff}Q$ (truth)', 'hits': 'raw hits',
      'decC': 'deconv C', 'decB': 'deconv B'}
SEGLEN = [1.0, 2.0, 3.0, 4.0]


def style(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')
    ax.grid(False)


def seg_at(pa, pb, q, binw):
    old = T.BIN_CM
    T.BIN_CM = binw
    try:
        return T.segment_dqdx(np.asarray(pa, float), np.asarray(pb, float), q)
    finally:
        T.BIN_CM = old


def profile_common(pa, pb, q, c, d, edges):
    """dQ/dx per bin on a GIVEN axis and GIVEN edges (position resolved)."""
    y, z = T.px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1)
    q = np.asarray(q, float)
    rel = yz - c
    proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    keep = perp < T.TUBE_CM
    h, _ = np.histogram(proj[keep], bins=edges, weights=q[keep])
    return h / (edges[1] - edges[0])


def load_all():
    """Per depth, per estimator: pooled segments at each segment length,
    plus one common-segmentation profile per event at 1 cm."""
    data = {}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        pools = {b: {k: [] for k in CO} for b in SEGLEN}
        prof = []
        for ev in range(50):
            ta, tb, tq = PS.truth_pix(fz, ev)
            est = {'effq': (ta, tb, tq), 'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for b in SEGLEN:
                for k, (a_, b_, q_) in est.items():
                    pools[b][k].append(seg_at(a_, b_, q_, b))
            if ev < 6:
                oldb = T.BIN_CM
                T.BIN_CM = 1.0
                ax = PS.truth_axis_edges(ta, tb, tq)
                T.BIN_CM = oldb
                if ax is not None:
                    c, dv, edges, lo, hi = ax
                    pr = {k: profile_common(*v, c, dv, edges)[lo:hi]
                          for k, v in est.items()}
                    pr['x'] = (0.5 * (edges[1:] + edges[:-1]))[lo:hi]
                    prof.append(pr)
        data[tag] = {'d_cm': d_cm, 't_us': L.drift_time_us(d_cm),
                     'pool': {b: {k: np.concatenate([x for x in v if len(x)])
                                  for k, v in pools[b].items()} for b in SEGLEN},
                     'prof': prof}
        print('loaded', tag, flush=True)
    return data


if __name__ == '__main__':
    D = load_all()
    tags = PS.TAGS

    # ---------------------------------------------- Fig A: pooled 1 cm shapes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.0), sharex=True)
    for i, tag in enumerate(tags):
        ax = axes.flat[i]
        style(ax)
        pool = D[tag]['pool'][1.0]
        bins = np.linspace(0, 220, 56)
        for k in ['effq', 'hits', 'decC']:
            v = pool[k]
            h, e = np.histogram(v[(v > 0) & (v < 220)], bins=bins, density=True)
            cen = 0.5 * (e[1:] + e[:-1])
            ax.step(cen, h, where='mid', color=CO[k], lw=1.3,
                    label=LB[k] if i == 0 else None)
            m = L.mpv_of(v)[0]
            ax.axvline(m, color=CO[k], ls=':', lw=1.0)
        ax.set_title(f"$d$ = {D[tag]['d_cm']:.1f} cm  "
                     f"($t$ = {D[tag]['t_us']:.0f} $\\mu$s)", fontsize=9)
        ax.set_xlim(0, 220)
        if i >= 5:
            ax.set_xlabel(r'd$Q$/d$x$  [ke/cm]', fontsize=9)
        if i % 5 == 0:
            ax.set_ylabel('segments (normalised)', fontsize=9)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Pooled d$Q$/d$x$, 1 cm segments  (dotted: Moyal MPV)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/dqdx_1cm_shapes.pdf')
    fig.savefig(f'{OUT}/dqdx_1cm_shapes.png', dpi=150)

    # ------------------------------------- Fig B: segment-length dependence
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for a in axes:
        style(a)
    t = np.array([D[g]['t_us'] for g in tags])
    mk = {1.0: 'o', 2.0: 's', 3.0: '^', 4.0: 'D'}
    lam = {k: [] for k in ['effq', 'decC', 'hits']}
    for b in SEGLEN:
        mv = {k: np.array([L.mpv_of(D[g]['pool'][b][k])[0] for g in tags])
              for k in ['effq', 'hits', 'decC']}
        axes[0].plot(t, mv['effq'], mk[b] + '-', color=CO['effq'], ms=4, lw=0.9,
                     label=f'{b:.0f} cm')
        axes[0].plot(t, mv['decC'], mk[b] + '--', color=CO['decC'], ms=4, lw=0.9)
        axes[1].plot(t, mv['decC'] / mv['effq'], mk[b] + '-', ms=4, lw=1.0,
                     label=f'{b:.0f} cm segments')
        for k in ['effq', 'decC', 'hits']:
            m = t > 20
            lam[k].append(-np.polyfit(t[m], np.log(mv[k][m]), 1)[0] * 1000)
    axes[0].set_xlabel(r'drift time  [$\mu$s]'); axes[0].set_ylabel('MPV  [ke/cm]')
    axes[0].set_title('MPV vs depth: truth (solid) / deconv C (dashed)', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, title='segment', title_fontsize=8)
    axes[1].axhline(1.0, color='0.6', lw=0.8)
    axes[1].set_xlabel(r'drift time  [$\mu$s]')
    axes[1].set_ylabel(r'MPV$_{\rm dec}$ / MPV$_{\rm true}$')
    axes[1].set_title('MPV capture', fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    for k in ['effq', 'decC', 'hits']:
        axes[2].plot(SEGLEN, lam[k], 'o-', color=CO[k], ms=5, label=LB[k])
    axes[2].axhline(1.0, color='0.6', lw=0.8)
    axes[2].set_xlabel('segment length  [cm]')
    axes[2].set_ylabel(r'fitted $\lambda$  [ms$^{-1}$]')
    axes[2].set_title(r'fitted decay rate (truth $\lambda$ = 1)', fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/dqdx_seglen.pdf')
    fig.savefig(f'{OUT}/dqdx_seglen.png', dpi=150)
    print('\nfitted lambda vs segment length:')
    for i, b in enumerate(SEGLEN):
        print('  %.0f cm   effq %.4f   hits %.4f   decC %.4f'
              % (b, lam['effq'][i], lam['hits'][i], lam['decC'][i]))

    # ------------------------------------ Fig C: along-track profile at 1 cm
    show = [1, 5, 9]
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), sharex='col')
    for r, gi in enumerate(show):
        tag = tags[gi]
        for cix, ev in enumerate([0, 1]):
            ax = axes[r, cix]; style(ax)
            pr = D[tag]['prof'][ev]
            for k in ['effq', 'hits', 'decC']:
                ax.step(pr['x'], pr[k], where='mid', color=CO[k], lw=1.1,
                        label=LB[k] if (r == 0 and cix == 0) else None)
            ax.set_ylabel('d$Q$/d$x$ [ke/cm]', fontsize=9)
            ax.tick_params(labelsize=8)
            if r == 2:
                ax.set_xlabel('position along the fitted track  [cm]', fontsize=9)
            ax.set_title(f"$d$ = {D[tag]['d_cm']:.1f} cm, event {ev}", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=3)
    fig.suptitle('Along-track d$Q$/d$x$ profile, 1 cm bins, common (truth) '
                 'segmentation', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/dqdx_1cm_profile.pdf')
    fig.savefig(f'{OUT}/dqdx_1cm_profile.png', dpi=150)
    print('\n->', OUT)
