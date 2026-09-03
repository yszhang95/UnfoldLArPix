"""The transport that IS there: the solve is SHARPER than the truth.

The residual anomaly rho(1) < rho(2) seen in both deconv arms (and absent
in `hits`, which is per-pixel by construction) is reproduced by the
symmetric sharing operator

    blur(f):  every pixel gives fraction f of its charge to EACH of its two
              along-track neighbours   (charge exact, first moment exact,
              so the centroid / registration tests are blind to it)

applied with the OPPOSITE sign: blurring the deconv by f ~ 0.05 nulls its
anomaly, i.e. the deconv has pulled roughly that much charge IN from its
neighbours relative to the truth.  This script

  1. calibrates Delta(f) on `hits` (Delta_0 = 0, the transport-free null),
  2. reads off f per depth for decC / decB / decC-vs-its-own-input,
  3. fits the depth trend of f,
  4. measures what blur/sharpen at that f is worth on the dQ/dx MPV and on
     the fitted decay rate, through the real segment_dqdx pipeline.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import track_dqdx as T
import dqdx_lib as L

CAL_F = [0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10]
CAL_DEPTHS = [1, 3, 5, 7, 9]


def blur(P, f):
    o = P * (1 - 2 * f)
    o[:-1] += P[1:] * f
    o[1:] += P[:-1] * f
    return o


def blur_pixels(pa, pb, q, f):
    """Same operator on the pixel list, for the dQ/dx pipeline."""
    return (np.concatenate([pa, pa, pa]),
            np.concatenate([pb, pb - 1, pb + 1]),
            np.concatenate([q * (1 - 2 * f), q * f, q * f]))


def lam_of(pts):
    t = np.array([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    m = t > 20
    return -np.polyfit(t[m], y[m], 1)[0] * 1000


if __name__ == '__main__':
    # ---------------------------------------------------------- calibration
    print('=' * 74)
    print('CALIBRATION of Delta vs blur fraction f, on `hits` (the null)')
    print('=' * 74)
    cal = []
    pool = []
    for gi in CAL_DEPTHS:
        tag = PS.TAGS[gi]
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for ev in range(50):
            pr = PT.profiles(tag, ev, fz)
            if pr is not None:
                pool.append((pr['truth'], pr['hits']))
    print('%-8s %10s' % ('f', 'Delta'))
    for q in CAL_F:
        rho, _, _ = PT.rho_of([(t, blur(r, q) if q else r) for t, r in pool])
        cal.append(PT.delta_of(rho))
        print('%-8.3f %10.4f' % (q, cal[-1]))
    cal = np.array(cal) - cal[0]
    slope = np.polyfit(CAL_F[:5], cal[:5], 1)[0]
    print(f'\nlinear response dDelta/df = {slope:.3f} over f <= 0.05')

    # ------------------------------------------------------- read off per depth
    print()
    print('=' * 74)
    print('SHARPENING f = Delta / (dDelta/df), per depth   [f>0 = solve is SHARPER]')
    print('=' * 74)
    J = json.load(open(f'{PS.OUT}/transport.json'))
    print('%-7s %18s %18s %18s' % ('depth', 'decC (vs truth)', 'decB (vs truth)',
                                   'decC (vs its input)'))
    tt, ff, ee = [], [], []
    for g in J:
        r = J[g]
        out = []
        for k in ['decC', 'decB', 'decC|hits']:
            out.append((r[k]['delta'] / slope, r[k]['delta_err'] / slope))
        tt.append(r['t_us']); ff.append(out[0][0]); ee.append(out[0][1])
        print('%-7.1f' % r['depth_cm'] + ''.join('%12.4f +-%5.4f' % o for o in out))
    tt, ff, ee = np.array(tt), np.array(ff), np.array(ee)
    m = tt > 20
    w = 1 / ee[m] ** 2
    A = np.vstack([tt[m], np.ones(m.sum())]).T
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    p = cov @ (A.T @ np.diag(w) @ ff[m])
    span = tt[m].max() - tt[m].min()
    print(f'\ndecC, 9 fitted depths: mean f = {np.average(ff[m], weights=w):.4f}'
          f' +- {1/np.sqrt(w.sum()):.4f}')
    print(f'depth trend: {p[0]*span:+.4f} +- {np.sqrt(cov[0,0])*span:.4f} over 28-179 us'
          f'  ({p[0]/np.sqrt(cov[0,0]):+.1f} sigma)')

    # ------------------------------------------------------------- leverage
    print()
    print('=' * 74)
    print('MPV LEVERAGE of blur/sharpen applied to the TRUTH, through segment_dqdx')
    print('=' * 74)
    FS = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
    curves = {q: [] for q in FS + [0.0]}
    print('%-7s' % 'depth' + ''.join('%10s' % f'{q:+.2f}' for q in FS))
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        evs = [PS.truth_pix(fz, ev) for ev in range(50)]
        mp = {}
        for q in [0.0] + FS:
            seg = []
            for (ta, tb, tq) in evs:
                if q:
                    a2, b2, q2 = blur_pixels(ta, tb, tq, q)
                else:
                    a2, b2, q2 = ta, tb, tq
                seg.append(T.segment_dqdx(a2.astype(float), b2.astype(float), q2))
            v = np.concatenate([x for x in seg if len(x)])
            mp[q] = L.mpv_of(v)[0]
            curves[q].append((L.drift_time_us(d_cm), mp[q]))
        print('%-7.1f' % d_cm + ''.join('%10.4f' % (mp[q] / mp[0.0]) for q in FS))
    print('\nfitted decay rate lambda [/ms]:  no transport %.4f' % lam_of(curves[0.0]))
    for q in FS:
        print('   f=%+.2f  lambda = %.4f' % (q, lam_of(curves[q])))
    print('\n   for scale: measured deconv C lambda = 1.555, truth control 0.970')
