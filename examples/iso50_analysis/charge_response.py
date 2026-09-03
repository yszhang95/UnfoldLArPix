"""Is the response CHARGE DEPENDENT -- weak regions reconstructed weaker and
strong regions stronger -- and does that bias the dQ/dx MPV?

Measured non-parametrically (no affine assumption) at the two levels that
matter:

  ROW level     bin by the truth row charge T(b), report <R>/<T> per bin.
  SEGMENT level bin by the truth 3 cm segment charge under a COMMON (truth)
                segmentation -- this is exactly what the dQ/dx Landau sees.

Then the closure that decides it: apply the MEASURED segment response
g(Q) to the truth segments, pool, take the Moyal MPV per depth, and
compare with the reconstruction's own MPV and fitted decay rate.  If g
reproduces them, the dQ/dx bias is the charge-dependent response and
nothing else.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import track_dqdx as T
import dqdx_lib as L

QEDGES_SEG = np.array([0, 30, 40, 50, 60, 70, 85, 105, 140, 200, 1e9])
QEDGES_ROW = np.array([0, 5, 10, 15, 20, 25, 30, 40, 55, 80, 1e9])
ESTS = ['hits', 'decC', 'decB']


def binned_response(x, y, edges):
    """<y>/<x> in bins of x, plus the bin population."""
    i = np.digitize(x, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = i == b
        if m.sum() < 30:
            out.append((np.nan, np.nan, int(m.sum())))
            continue
        out.append((float(x[m].mean()), float(y[m].mean() / x[m].mean()),
                    int(m.sum())))
    return out


if __name__ == '__main__':
    seg_pairs, row_pairs = {}, {}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        sp = {k: [[], []] for k in ESTS}
        rp = {k: [[], []] for k in ESTS}
        for ev in range(50):
            pr = PT.profiles(tag, ev, fz)
            if pr is not None:
                for k in ESTS:
                    rp[k][0].append(pr['truth']); rp[k][1].append(pr[k])
            ta, tb, tq = PS.truth_pix(fz, ev)
            ax = PS.truth_axis_edges(ta, tb, tq)
            if ax is None:
                continue
            c, dv, edges, lo, hi = ax
            ht, _ = PS.seg_common(ta, tb, tq, c, dv, edges)
            if not ht.size:
                continue
            ht = ht[lo:hi] / T.BIN_CM
            est = {'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for k, (pa, pb, q) in est.items():
                hr, _ = PS.seg_common(pa, pb, q, c, dv, edges)
                if hr.size:
                    m = ht > 0
                    sp[k][0].append(ht[m]); sp[k][1].append(hr[lo:hi][m] / T.BIN_CM)
        seg_pairs[tag] = {k: (np.concatenate(v[0]), np.concatenate(v[1]))
                          for k, v in sp.items() if v[0]}
        row_pairs[tag] = {k: (np.concatenate(v[0]), np.concatenate(v[1]))
                          for k, v in rp.items() if v[0]}

    # ------------------------------------------------------------- row level
    print('=' * 88)
    print('ROW-LEVEL RESPONSE <R>/<T> vs truth row charge  (arm C)')
    print('=' * 88)
    print('%-7s' % 'depth' + ''.join('%9.0f' % q for q in QEDGES_ROW[1:-1]) + '%9s' % 'hi')
    for tag in PS.TAGS:
        x, y = row_pairs[tag]['decC']
        b = binned_response(x, y, QEDGES_ROW)
        print('%-7.1f' % float(tag.split('_d')[1].replace('p', '.'))
              + ''.join('%9.4f' % r for _, r, _ in b))

    # --------------------------------------------------------- segment level
    print()
    print('=' * 88)
    print('SEGMENT-LEVEL RESPONSE <R>/<T> vs truth 3 cm segment dQ/dx [ke/cm]')
    print('(this is what the Landau sees;  >1 = boosted, <1 = suppressed)')
    print('=' * 88)
    for k in ['decC', 'hits']:
        print(f'--- {k}')
        print('%-7s' % 'depth' + ''.join('%9.0f' % q for q in QEDGES_SEG[1:-1]) + '%9s' % 'hi')
        for tag in PS.TAGS:
            x, y = seg_pairs[tag][k]
            b = binned_response(x, y, QEDGES_SEG)
            print('%-7.1f' % float(tag.split('_d')[1].replace('p', '.'))
                  + ''.join('%9.4f' % r for _, r, _ in b))
        print()

    # ------------------------------------------------------------- closure
    print('=' * 88)
    print('CLOSURE: apply the measured segment response g(Q) to the TRUTH')
    print('=' * 88)
    print('%-7s %10s %10s %10s %10s %10s' %
          ('depth', 'MPV_true', 'MPV_reco', 'MPV_g(T)', 'reco/true', 'g/true'))
    cur = {'reco': [], 'gT': [], 'true': []}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        x, y = seg_pairs[tag]['decC']
        # g from the binned response of THIS depth, linearly interpolated in Q
        b = [v for v in binned_response(x, y, QEDGES_SEG) if np.isfinite(v[0])]
        qc = np.array([v[0] for v in b]); rr = np.array([v[1] for v in b])
        g = np.interp(x, qc, rr)
        mt, mr, mg = (L.mpv_of(x)[0], L.mpv_of(y)[0], L.mpv_of(x * g)[0])
        t_us = L.drift_time_us(d_cm)
        cur['true'].append((t_us, mt)); cur['reco'].append((t_us, mr))
        cur['gT'].append((t_us, mg))
        print('%-7.1f %10.2f %10.2f %10.2f %10.4f %10.4f'
              % (d_cm, mt, mr, mg, mr / mt, mg / mt))

    def lam(pts):
        t = np.array([p[0] for p in pts]); yy = np.log([p[1] for p in pts])
        m = t > 20
        return -np.polyfit(t[m], yy[m], 1)[0] * 1000
    print(f'\nfitted decay rate lambda [/ms]:  truth {lam(cur["true"]):.4f}'
          f'   deconv C {lam(cur["reco"]):.4f}   g(truth) {lam(cur["gT"]):.4f}')
    print('   (truth tau = 1 ms; the note quotes deconv C = 1.555 with its own segmentation)')

    json.dump({'seg_edges': QEDGES_SEG.tolist(),
               'seg_response': {t: {k: binned_response(*seg_pairs[t][k], QEDGES_SEG)
                                    for k in seg_pairs[t]} for t in seg_pairs}},
              open(f'{PS.OUT}/charge_response.json', 'w'), indent=1)
    print('\n->', f'{PS.OUT}/charge_response.json')

# ---------------------------------------------------------------------------
# Decomposition appended: is it the LEVEL of g or its TILT that bends lambda?
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    print('=' * 88)
    print('DECOMPOSITION of the charge response: flat level vs charge tilt')
    print('=' * 88)
    cur2 = {'true': [], 'full': [], 'flat': [], 'tilt': []}
    print('%-7s %9s %9s %9s %9s %11s' % ('depth', 'MPV_true', 'g full',
                                         'flat only', 'tilt only', 'g(50)/g(140)'))
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        x, y = seg_pairs[tag]['decC']
        b = [v for v in binned_response(x, y, QEDGES_SEG) if np.isfinite(v[0])]
        qc = np.array([v[0] for v in b]); rr = np.array([v[1] for v in b])
        g = np.interp(x, qc, rr)
        flat = y.sum() / x.sum()                 # charge-weighted mean gain
        t_us = L.drift_time_us(d_cm)
        cur2['true'].append((t_us, L.mpv_of(x)[0]))
        cur2['full'].append((t_us, L.mpv_of(x * g)[0]))
        cur2['flat'].append((t_us, L.mpv_of(x * flat)[0]))
        cur2['tilt'].append((t_us, L.mpv_of(x * g / flat)[0]))
        lo = np.interp(50.0, qc, rr); hi = np.interp(140.0, qc, rr)
        print('%-7.1f %9.2f %9.2f %9.2f %9.2f %11.4f'
              % (d_cm, cur2['true'][-1][1], cur2['full'][-1][1],
                 cur2['flat'][-1][1], cur2['tilt'][-1][1], lo / hi))

    def lam2(pts):
        t = np.array([p[0] for p in pts]); yy = np.log([p[1] for p in pts])
        m = t > 20
        return -np.polyfit(t[m], yy[m], 1)[0] * 1000
    lt, lf, lfl, lti = (lam2(cur2['true']), lam2(cur2['full']),
                        lam2(cur2['flat']), lam2(cur2['tilt']))
    print(f'\nlambda [/ms]  truth {lt:.4f}   full g {lf:.4f}'
          f'   flat level only {lfl:.4f}   tilt only {lti:.4f}')
    print(f'   error budget:  flat level {lfl-lt:+.4f}   charge tilt {lti-lt:+.4f}'
          f'   (sum {lfl+lti-2*lt:+.4f} vs full {lf-lt:+.4f})')
