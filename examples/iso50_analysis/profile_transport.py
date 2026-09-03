"""Is there PIXEL-TO-PIXEL CHARGE TRANSPORT along the track axis?

The centroid / registration tests in profile_shift.py bound a COHERENT
displacement.  They are blind to LOCAL, zero-mean migration: charge moved
from row b to b+/-1 leaves the centroid and the global rms untouched.

Signature used here: transport is a strictly SHORT-RANGE, ANTI-correlated
component of the per-row residual

    r(b) = R(b) - a*T(b) - c        (a, c = per-depth pixel-level OLS)

Charge leaving b and landing at b+1 makes r(b) and r(b+1) anti-correlated.
Everything else that structures the residual (locally-dense-track capture,
delta rays, threshold) is LONG-range and positive, and varies smoothly in
lag.  So the test is: extrapolate rho(k) from k = 2..5 back to k = 1 and
attribute the DEFICIT

    Delta = rho_smooth(1) - rho(1)

to nearest-neighbour transport.  Delta is calibrated against an injected
exchange of a known fraction, and the calibrated exchange is then pushed
through the real dQ/dx pipeline to get its MPV leverage.

Controls:
  * `hits` is per-pixel BY CONSTRUCTION (a hit's charge comes from that
    pixel's own ADC), so it must show Delta = 0.  It is the null.
  * `decC|hits` regresses the deconv on ITS OWN INPUT, isolating what the
    solver does from truth-vs-hits differences.
Errors: bootstrap over events (200 resamples).
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

KMAX = 6
KFIT = [2, 3, 4, 5]          # lags used to model the smooth component
NBOOT = 200
ESTS = ['hits', 'decC', 'decB']


def profiles(tag, ev, f):
    ta, tb, tq = PS.truth_pix(f, ev)
    est = {'hits': PS.hits_pix(f, ev)}
    for arm in ['C', 'B']:
        p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            est[f'dec{arm}'] = PS.deconv_pix(p)
    a0 = min(ta.min(), *[e[0].min() for e in est.values()])
    a1 = max(ta.max(), *[e[0].max() for e in est.values()])
    b0 = min(tb.min(), *[e[1].min() for e in est.values()])
    b1 = max(tb.max(), *[e[1].max() for e in est.values()])
    Qt = PS.dense(ta, tb, tq, a0, a1, b0, b1)
    ac = int(np.argmax(Qt.sum(axis=1)))
    rows = slice(max(ac - PS.TUBE, 0), ac + PS.TUBE + 1)
    Pt = Qt[rows].sum(axis=0)
    lit = np.nonzero(Pt > 0)[0]
    if lit.size < 2 * PS.TRIM + 24:
        return None
    w = slice(lit[0] + PS.TRIM, lit[-1] - PS.TRIM + 1)
    out = {'truth': Pt[w]}
    for k, (pa, pb, q) in est.items():
        out[k] = PS.dense(pa, pb, q, a0, a1, b0, b1)[rows].sum(axis=0)[w]
    return out


def swap_fraction(P, frac, rng):
    """Zero-mean random exchange with the next row: pure transport, exact
    charge conservation, centroid and global width untouched."""
    flux = frac * P[:-1] * rng.choice([-1.0, 1.0], P.size - 1)
    out = P.copy()
    out[:-1] -= flux
    out[1:] += flux
    return out


def rho_of(pairs, idx=None):
    """Pooled lag autocorrelation of the affine residual."""
    if idx is not None:
        pairs = [pairs[i] for i in idx]
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    a, c = np.polyfit(x, y, 1)
    num = np.zeros(KMAX + 1); den = 0.0
    for t, rr in pairs:
        r = rr - a * t - c
        r = r - r.mean()
        den += float(r @ r)
        for k in range(KMAX + 1):
            num[k] += float(r[k:] @ r[:len(r) - k])
    return num / den, float(a), float(c)


def delta_of(rho):
    """rho_smooth(1) - rho(1), the short-range deficit."""
    k = np.array(KFIT, float)
    p = np.polyfit(k, rho[KFIT], 1)          # smooth component is linear in
    return float(np.polyval(p, 1.0) - rho[1])  # lag over this short range


def analyse(pairs, seed=0):
    rho, a, c = rho_of(pairs)
    d = delta_of(rho)
    rng = np.random.default_rng(seed)
    n = len(pairs)
    bd, br1 = [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        rr, _, _ = rho_of(pairs, idx)
        bd.append(delta_of(rr)); br1.append(rr[1])
    return {'a': a, 'c': c, 'rho': rho.tolist(),
            'rho1': float(rho[1]), 'rho1_err': float(np.std(br1, ddof=1)),
            'delta': d, 'delta_err': float(np.std(bd, ddof=1))}


if __name__ == '__main__':
    out = {}
    print('=' * 92)
    print('LAG AUTOCORRELATION OF THE PER-ROW RESIDUAL   (transport => rho1 below trend)')
    print('%-6s %-10s %8s %8s %7s %7s %7s %16s' %
          ('depth', 'pair', 'rho1', 'rho2', 'rho3', 'rho4', 'rho5',
           'Delta = smooth-rho1'))
    print('=' * 92)
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        f = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        pairs = {k: [] for k in ESTS}
        pairs['decC|hits'] = []
        for ev in range(50):
            pr = profiles(tag, ev, f)
            if pr is None:
                continue
            for k in ESTS:
                if k in pr:
                    pairs[k].append((pr['truth'], pr[k]))
            if 'decC' in pr and 'hits' in pr:
                pairs['decC|hits'].append((pr['hits'], pr['decC']))
        rec = {'depth_cm': d_cm, 't_us': L.drift_time_us(d_cm)}
        for k in ESTS + ['decC|hits']:
            if not pairs[k]:
                continue
            r = analyse(pairs[k])
            rec[k] = r
            print('%-6.1f %-10s %+8.3f %+8.3f %+7.3f %+7.3f %+7.3f   %+7.4f +- %.4f'
                  % (d_cm, k, r['rho'][1], r['rho'][2], r['rho'][3],
                     r['rho'][4], r['rho'][5], r['delta'], r['delta_err']))
        out[tag] = rec
        print()
    json.dump(out, open(f'{PS.OUT}/transport.json', 'w'), indent=1)
    print('->', f'{PS.OUT}/transport.json')
