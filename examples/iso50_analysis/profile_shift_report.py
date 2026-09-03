"""Depth trends, injection closure and MPV-leverage bound for profile_shift."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import dqdx_lib as L

AO = PS.AO
OUT = PS.OUT
J = json.load(open(f'{OUT}/profile_shift.json'))
D = J['depths']
tags = list(D)
t = np.array([D[k]['t_us'] for k in tags])
d = np.array([D[k]['depth_cm'] for k in tags])


def wfit(y, e):
    """Weighted OLS slope of y vs t, with error."""
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    A = np.vstack([t, np.ones_like(t)]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    p = cov @ (A.T @ W @ y)
    return p[0], np.sqrt(cov[0, 0]), p[1]


print('=' * 78)
print('DEPTH TRENDS  (slope vs drift time; span = 28-179 us = 151 us)')
print('=' * 78)
print('%-12s %-9s %12s %12s %10s' % ('estimator', 'metric', 'slope/us', 'err', 'over span'))
rows = {}
for k in ['hits', 'decC', 'decB']:
    for fld, lab in [('d_shift', 'shift[px]'), ('d_cen', 'centroid[px]'),
                     ('d_cen_a', 'trans[px]'), ('d_wid', 'width'),
                     ('cap', 'capture')]:
        y = np.array([D[g][f'{k}.{fld}'][0] for g in tags])
        e = np.array([D[g][f'{k}.{fld}'][1] for g in tags])
        m = d > 3.0                                   # drop d=1.5 as the note does
        yy, ee, tt = y[m], e[m], t[m]
        w = 1 / ee ** 2
        A = np.vstack([tt, np.ones_like(tt)]).T
        cov = np.linalg.inv(A.T @ np.diag(w) @ A)
        p = cov @ (A.T @ np.diag(w) @ yy)
        sl, se = p[0], np.sqrt(cov[0, 0])
        span = tt.max() - tt.min()
        rows[(k, fld)] = (sl, se, sl * span)
        print('%-12s %-9s %12.3e %12.3e %10.4f' % (k, lab, sl, se, sl * span))

print()
print('=' * 78)
print('MPV RATIO vs effq, and its depth slide (same 9 depths)')
print('=' * 78)
for k in ['hits', 'decC', 'decB']:
    r = np.array([D[g]['mpv_own'][k] / D[g]['mpv_own']['effq'] for g in tags])
    m = d > 3.0
    A = np.vstack([t[m], np.ones_like(t[m])]).T
    sl, ic = np.linalg.lstsq(A, r[m], rcond=None)[0]
    span = t[m].max() - t[m].min()
    cap = np.array([D[g][f'{k}.cap'][0] for g in tags])
    slc, icc = np.linalg.lstsq(A, cap[m], rcond=None)[0]
    print(f'{k:6s} MPV ratio {r[m][0]:.4f} -> {r[m][-1]:.4f}   slide {sl*span*100:+6.2f}%'
          f'   | window capture {cap[m][0]:.4f} -> {cap[m][-1]:.4f}  slide {slc*span*100:+6.2f}%')

print()
print('=' * 78)
print('INJECTION CLOSURE: does the estimator see a shift that IS there?')
print('=' * 78)
INJ = [0.0, 0.02, 0.05, 0.10, 0.25]
for gi in [1, 5, 9]:
    tag = tags[gi]
    f = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    got = {q: [] for q in INJ}
    for ev in range(20):
        ta, tb, tq = PS.truth_pix(f, ev)
        p = f'{PS.DIRS["C"][0]}/{tag}/{tag}_event_0_{ev}.npz'
        if not os.path.exists(p):
            continue
        ra, rb, rq = PS.deconv_pix(p)
        a0, a1 = min(ta.min(), ra.min()), max(ta.max(), ra.max())
        b0, b1 = min(tb.min(), rb.min()), max(tb.max(), rb.max())
        Qt = PS.dense(ta, tb, tq, a0, a1, b0, b1)
        ac = int(np.argmax(Qt.sum(axis=1)))
        rows_ = slice(max(ac - PS.TUBE, 0), ac + PS.TUBE + 1)
        Ptf = Qt[rows_].sum(axis=0)
        lit = np.nonzero(Ptf > 0)[0]
        w = slice(lit[0] + PS.TRIM, lit[-1] - PS.TRIM + 1)
        Pt = Ptf[w]
        for q in INJ:
            pa2, pb2, q2 = PS.shift_pixels(ra, rb, rq, q)
            Qr = PS.dense(np.asarray(pa2, int), np.asarray(pb2, int), q2, a0, a1, b0, b1)
            got[q].append(PS.reg_shift(Pt, Qr[rows_].sum(axis=0)[w]))
    print(f'{tag}  (n={len(got[0.0])})')
    for q in INJ:
        v = np.array(got[q])
        print(f'   injected {q:+.3f} px -> measured {v.mean():+.4f} +- {v.std(ddof=1)/np.sqrt(v.size):.4f} px'
              f'   (recovered {v.mean()-np.array(got[0.0]).mean():+.4f})')

print()
print('=' * 78)
print('LEVERAGE BOUND: MPV change per pixel of rigid shift (truth, pooled)')
print('=' * 78)
for g in tags:
    m = D[g]['mpv_truth_shift']
    b = m['0.0']
    print('%5.1f cm  ' % D[g]['depth_cm'] + '  '.join(
        f'{float(dd):+.2f}px {m[dd]/b-1:+.4f}' for dd in ['0.25', '0.5', '1.0']))
allr = []
for g in tags:
    m = D[g]['mpv_truth_shift']; b = m['0.0']
    allr.append(abs(m['0.5'] / b - 1))
print(f'\nmax |dMPV| for a 0.50 px rigid shift: {max(allr)*100:.2f}%')
