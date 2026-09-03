"""Bootstrap errors on the segment-level affine fit and on the flat/tilt
decomposition -- both quoted bare earlier.

Pairing is the COMMON (truth) segmentation: truth PCA axis + truth bin
edges, so segment i of the reco is the same physical 3 cm of track as
segment i of the truth.  (errors_redo.py paired each estimator's OWN
segmentation index by index -- wrong, they have different axes and bin
phases; its section 3 is superseded by this.)
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

NB = 300
QEDGES = np.array([0, 30, 40, 50, 60, 70, 85, 105, 140, 200, 1e9])


def collect(tag, arm='C'):
    fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    pairs = []
    for ev in range(50):
        ta, tb, tq = PS.truth_pix(fz, ev)
        ax = PS.truth_axis_edges(ta, tb, tq)
        if ax is None:
            continue
        c, dv, edges, lo, hi = ax
        ht, _ = PS.seg_common(ta, tb, tq, c, dv, edges)
        if not ht.size:
            continue
        ht = ht[lo:hi] / T.BIN_CM
        p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
        if not os.path.exists(p):
            continue
        hr, _ = PS.seg_common(*PS.deconv_pix(p), c, dv, edges)
        if hr.size:
            m = ht > 0
            pairs.append((ht[m], hr[lo:hi][m] / T.BIN_CM))
    return pairs


def g_of(x, y):
    i = np.digitize(x, QEDGES) - 1
    qc, rr = [], []
    for b in range(len(QEDGES) - 1):
        m = i == b
        if m.sum() >= 30:
            qc.append(x[m].mean()); rr.append(y[m].mean() / x[m].mean())
    return np.array(qc), np.array(rr)


if __name__ == '__main__':
    t_us, data = [], []
    for tag in PS.TAGS:
        d = float(tag.split('_d')[1].replace('p', '.'))
        t_us.append(L.drift_time_us(d))
        data.append(collect(tag))
        print('loaded', tag, len(data[-1]), flush=True)
    t_us = np.array(t_us)

    print()
    print('=' * 72)
    print('SEGMENT-LEVEL AFFINE (common truth segmentation), bootstrap over events')
    print('=' * 72)
    print('%-7s %18s %18s' % ('depth', 'a', 'c [ke/cm]'))
    A, C = [], []
    for di, tag in enumerate(PS.TAGS):
        pr = data[di]
        rg = np.random.default_rng(4)
        bs = []
        for _ in range(NB):
            pk = rg.integers(0, len(pr), len(pr))
            X = np.concatenate([pr[i][0] for i in pk])
            Y = np.concatenate([pr[i][1] for i in pk])
            bs.append(np.polyfit(X, Y, 1))
        bs = np.array(bs)
        A.append((bs[:, 0].mean(), bs[:, 0].std(ddof=1)))
        C.append((bs[:, 1].mean(), bs[:, 1].std(ddof=1)))
        print('%-7.1f %18s %18s'
              % (float(tag.split('_d')[1].replace('p', '.')),
                 f'{A[-1][0]:.4f} +- {A[-1][1]:.4f}',
                 f'{C[-1][0]:+.2f} +- {C[-1][1]:.2f}'))

    m = t_us > 20
    for nm, V in [('a', A), ('c', C)]:
        y = np.array([v[0] for v in V]); e = np.array([v[1] for v in V])
        w = 1 / e[m] ** 2
        X = np.vstack([t_us[m], np.ones(m.sum())]).T
        cov = np.linalg.inv(X.T @ np.diag(w) @ X)
        p = cov @ (X.T @ np.diag(w) @ y[m])
        span = t_us[m].max() - t_us[m].min()
        print(f'  depth trend of {nm}: {p[0]*span:+.4f} +- {np.sqrt(cov[0,0])*span:.4f}'
              f' over 28-179 us   ({abs(p[0])/np.sqrt(cov[0,0]):.1f} sigma)')

    print()
    print('=' * 72)
    print('FLAT / TILT DECOMPOSITION OF lambda, bootstrap over events')
    print('=' * 72)
    rg = np.random.default_rng(12)
    out = {'true': [], 'full': [], 'flat': [], 'tilt': []}
    for _ in range(NB):
        picks = [rg.integers(0, len(data[di]), len(data[di]))
                 for di in range(len(PS.TAGS))]
        mv = {k: [] for k in out}
        for di in range(len(PS.TAGS)):
            pr = data[di]; pk = picks[di]
            x = np.concatenate([pr[i][0] for i in pk])
            y = np.concatenate([pr[i][1] for i in pk])
            qc, rr = g_of(x, y)
            g = np.interp(x, qc, rr)
            flat = y.sum() / x.sum()
            mv['true'].append(L.mpv_of(x)[0]); mv['full'].append(L.mpv_of(x * g)[0])
            mv['flat'].append(L.mpv_of(x * flat)[0])
            mv['tilt'].append(L.mpv_of(x * g / flat)[0])
        for k in out:
            v = np.array(mv[k]); kk = np.isfinite(v) & m
            X = np.vstack([t_us[kk], np.ones(kk.sum())]).T
            out[k].append(-np.linalg.lstsq(X, np.log(v[kk]), rcond=None)[0][0] * 1000)
    lam = {k: (np.median(out[k]), np.std(out[k], ddof=1)) for k in out}
    for k in ['true', 'full', 'flat', 'tilt']:
        print('  %-6s lambda = %.4f +- %.4f' % (k, *lam[k]))
    dfl = np.array(out['flat']) - np.array(out['true'])
    dti = np.array(out['tilt']) - np.array(out['true'])
    print('  error budget (paired, so these errors are the right ones):')
    print('     flat level  %+.4f +- %.4f /ms' % (np.median(dfl), dfl.std(ddof=1)))
    print('     charge tilt %+.4f +- %.4f /ms' % (np.median(dti), dti.std(ddof=1)))
