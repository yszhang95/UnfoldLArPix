"""Why do raw hits sit within 2% of the truth MPV, while their lambda is 1.48?

Per depth, on the COMMON (truth) segmentation so segment i of the reco is the
same physical 3 cm as segment i of the truth, fit

    Q_reco = a * Q_true + c

and decompose the MPV ratio:  MPV_reco / MPV_true  ~=  a + c / MPV_true .
The two terms are the multiplicative gain and the additive offset expressed
as a fraction of the charge at the Landau peak.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

TAGS = ['pgun_mu_3gev_iso50_d16p5', 'pgun_mu_3gev_iso50_d19p5',
        'pgun_mu_3gev_iso50_d22p5', 'pgun_mu_3gev_iso50_d25p5',
        'pgun_mu_3gev_iso50_d28p5']
NB = 300

print('=' * 100)
print('SEGMENT-LEVEL AFFINE  Q_reco = a*Q_true + c   (3 cm, common truth segmentation)')
print('=' * 100)
print('%7s %-6s %9s %11s %11s %11s %11s %11s'
      % ('d[cm]', 'est', 'a', 'c [ke/cm]', 'MPV_true', 'c/MPV_true', 'a+c/MPV', 'measured'))
store = {}
for tag in TAGS:
    d = float(tag.split('_d')[1].replace('p', '.'))
    fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    pr = {'hits': [], 'decC': []}
    for ev in range(50):
        ta, tb, tq = PS.truth_pix(fz, ev)
        ax = PS.truth_axis_edges(ta, tb, tq)
        if ax is None:
            continue
        c_, dv, edges, lo, hi = ax
        ht, _ = PS.seg_common(ta, tb, tq, c_, dv, edges)
        if not ht.size:
            continue
        ht = ht[lo:hi] / T.BIN_CM
        est = {'hits': PS.hits_pix(fz, ev)}
        p = f'{PS.DIRS["C"][0]}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            est['decC'] = PS.deconv_pix(p)
        for k, v in est.items():
            hr, _ = PS.seg_common(*v, c_, dv, edges)
            if hr.size:
                m = ht > 0
                pr[k].append((ht[m], hr[lo:hi][m] / T.BIN_CM))
    for k in ['hits', 'decC']:
        x = np.concatenate([p[0] for p in pr[k]])
        y = np.concatenate([p[1] for p in pr[k]])
        a, c = np.polyfit(x, y, 1)
        mt = L.mpv_of(x)[0]; mr = L.mpv_of(y)[0]
        store.setdefault(k, []).append((d, a, c, mt, mr))
        print('%7.1f %-6s %9.4f %+11.2f %11.2f %+11.4f %11.4f %11.4f'
              % (d, k, a, c, mt, c / mt, a + c / mt, mr / mt))
    print()

print('=' * 100)
print('THE CANCELLATION, and how it drifts')
print('=' * 100)
V = 1.59645
for k in ['hits', 'decC']:
    r = np.array(store[k])
    d, a, c, mt, mr = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]
    t = d * 10.0 / V
    print(f'--- {k}')
    print('   gain term a          %.4f -> %.4f   (%+.2f%% over the span)'
          % (a[0], a[-1], 100 * (a[-1] / a[0] - 1)))
    print('   offset c [ke/cm]     %+.2f -> %+.2f  (%+.2f%%)'
          % (c[0], c[-1], 100 * (c[-1] / c[0] - 1)))
    print('   offset as a fraction c/MPV_true   %+.4f -> %+.4f  (moves %+.4f)'
          % (c[0] / mt[0], c[-1] / mt[-1], c[-1] / mt[-1] - c[0] / mt[0]))
    print('   |a-1| + |c/MPV| = how much has to cancel:  %.3f at %.1f cm, %.3f at %.1f cm'
          % (abs(a[0] - 1) + abs(c[0] / mt[0]), d[0],
             abs(a[-1] - 1) + abs(c[-1] / mt[-1]), d[-1]))
    A = np.vstack([t, np.ones_like(t)]).T
    for nm, series in [('gain a alone', a),
                       ('offset c/MPV_true alone', 1 + c / mt),
                       ('full ratio a + c/MPV', a + c / mt)]:
        sl = np.polyfit(t, np.log(series), 1)[0] * 1000
        print('   lambda offset from %-24s %+7.4f /ms' % (nm, -sl))
    print()
print('READING: the MPV ratio is a DIFFERENCE of two large terms for hits and')
print('a small residual for decC.  What sets lambda is how the two terms DRIFT,')
print('not how well they cancel at any one depth.')
