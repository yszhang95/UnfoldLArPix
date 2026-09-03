"""Put errors on the numbers that were quoted bare.

Three things were reported this session without uncertainties:
  1. lambda vs segment length (1/2/3/4 cm)  -- bare polyfit, no weights
  2. the MPV-leverage bounds of transport   -- is "<= 1%" a bound or a floor?
  3. the segment-level affine a, c

All errors here are BOOTSTRAP OVER EVENTS (the independent unit: 50 seeded
copies per depth), following iso50_analyse.boot_tau.  For the leverage the
bootstrap is PAIRED -- the operator is applied to the same resampled events,
so the ratio MPV(f)/MPV(0) keeps its correlation and gets its proper (much
smaller) error.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

NB = 300
SEGLEN = [1.0, 2.0, 3.0, 4.0]


def seg_at(pa, pb, q, b):
    old = T.BIN_CM
    T.BIN_CM = b
    try:
        return T.segment_dqdx(np.asarray(pa, float), np.asarray(pb, float), q)
    finally:
        T.BIN_CM = old


def blur_pixels(pa, pb, q, f):
    return (np.concatenate([pa, pa, pa]),
            np.concatenate([pb, pb - 1, pb + 1]),
            np.concatenate([q * (1 - 2 * f), q * f, q * f]))


def scatter_pixels(pa, pb, q, f, R, rng):
    step = rng.integers(1, R + 1, q.size) * rng.choice([-1, 1], q.size)
    return (np.concatenate([pa, pa]), np.concatenate([pb, pb + step]),
            np.concatenate([q * (1 - f), q * f]))


def mpv_pool(per_ev, pick):
    v = [per_ev[i] for i in pick if len(per_ev[i])]
    if not v:
        return np.nan
    return L.mpv_of(np.concatenate(v))[0]


def boot_lambda(t_us, per_depth, nb=NB, seed=1):
    """per_depth: list over depths of (list over events of segment arrays)."""
    rng = np.random.default_rng(seed)
    lam = []
    A = None
    for _ in range(nb):
        m = []
        for evs in per_depth:
            m.append(mpv_pool(evs, rng.integers(0, len(evs), len(evs))))
        m = np.array(m)
        k = np.isfinite(m) & (t_us > 20)
        if A is None or A.shape[0] != k.sum():
            A = np.vstack([t_us[k], np.ones(k.sum())]).T
        lam.append(-np.linalg.lstsq(A, np.log(m[k]), rcond=None)[0][0] * 1000)
    lam = np.array(lam)
    return float(np.median(lam)), float(lam.std(ddof=1))


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                     for t in PS.TAGS])

    # ------------------------------------------ collect per-event segments
    store = {b: {k: [] for k in ['effq', 'hits', 'decC', 'decB']} for b in SEGLEN}
    lev = {}          # leverage configs, 3 cm only
    LEVCFG = [('blur +0.05', lambda a, b_, q: blur_pixels(a, b_, q, 0.05)),
              ('blur +0.20', lambda a, b_, q: blur_pixels(a, b_, q, 0.20)),
              ('sharp -0.05', lambda a, b_, q: blur_pixels(a, b_, q, -0.05)),
              ('scatter f.10 R1', lambda a, b_, q: scatter_pixels(a, b_, q, 0.10, 1, rng)),
              ('scatter f.40 R1', lambda a, b_, q: scatter_pixels(a, b_, q, 0.40, 1, rng)),
              ('scatter f.40 R10', lambda a, b_, q: scatter_pixels(a, b_, q, 0.40, 10, rng)),
              ('shift +0.50px', None)]
    for lab, _ in LEVCFG:
        lev[lab] = []
    lev['truth'] = []
    aff = {}
    for tag in PS.TAGS:
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        ev_t, ev_lev = [], {lab: [] for lab, _ in LEVCFG}
        per = {b: {k: [] for k in store[b]} for b in SEGLEN}
        for ev in range(50):
            ta, tb, tq = PS.truth_pix(fz, ev)
            est = {'effq': (ta, tb, tq), 'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for b in SEGLEN:
                for k, (a_, b2_, q_) in est.items():
                    per[b][k].append(seg_at(a_, b2_, q_, b))
            ev_t.append(per[3.0]['effq'][-1])
            for lab, fn in LEVCFG:
                if fn is None:
                    a2, b2, q2 = PS.shift_pixels(ta, tb, tq, 0.5)
                else:
                    a2, b2, q2 = fn(ta, tb, tq)
                ev_lev[lab].append(seg_at(a2, b2, q2, 3.0))
        for b in SEGLEN:
            for k in store[b]:
                store[b][k].append(per[b][k])
        lev['truth'].append(ev_t)
        for lab, _ in LEVCFG:
            lev[lab].append(ev_lev[lab])
        print('loaded', tag, flush=True)

    # ---------------------------------------------- 1. lambda vs seg length
    print()
    print('=' * 72)
    print('1. FITTED lambda WITH BOOTSTRAP ERRORS (300 resamples over events)')
    print('=' * 72)
    print('%-8s %16s %16s %16s' % ('segment', 'effq (truth)', 'raw hits', 'deconv C'))
    for b in SEGLEN:
        row = []
        for k in ['effq', 'hits', 'decC']:
            m, e = boot_lambda(t_us, store[b][k])
            row.append(f'{m:.3f} +- {e:.3f}')
        print('%-8s %16s %16s %16s' % (f'{b:.0f} cm', *row))

    # ------------------------------------------------- 2. leverage, paired
    print()
    print('=' * 72)
    print('2. TRANSPORT LEVERAGE, PAIRED BOOTSTRAP  (3 cm, ratio to untouched')
    print('   truth on the SAME resample; also lambda)')
    print('=' * 72)
    print('%-18s %22s %18s' % ('operator on truth', 'max |MPV ratio - 1|',
                               'lambda'))
    lam0, e0 = boot_lambda(t_us, lev['truth'])
    print('%-18s %22s %18s' % ('none (truth)', '--', f'{lam0:.3f} +- {e0:.3f}'))
    for lab, _ in LEVCFG:
        rat = []
        rg = np.random.default_rng(9)
        for _ in range(NB):
            worst = 0.0
            for di in range(len(PS.TAGS)):
                n = len(lev['truth'][di])
                pick = rg.integers(0, n, n)
                a = mpv_pool(lev['truth'][di], pick)
                c = mpv_pool(lev[lab][di], pick)
                if np.isfinite(a) and np.isfinite(c):
                    worst = max(worst, abs(c / a - 1))
            rat.append(worst)
        lm, le = boot_lambda(t_us, lev[lab])
        print('%-18s %22s %18s'
              % (lab, f'{np.median(rat)*100:.2f} +- {np.std(rat, ddof=1)*100:.2f} %',
                 f'{lm:.3f} +- {le:.3f}'))

    # ------------------------------------------------- 3. affine a, c errors
    print()
    print('=' * 72)
    print('3. SEGMENT-LEVEL AFFINE  Q_dec = a*Q_true + c,  bootstrap errors')
    print('=' * 72)
    print('%-7s %18s %18s' % ('depth', 'a', 'c [ke/cm]'))
    for di, tag in enumerate(PS.TAGS):
        xs = store[3.0]['effq'][di]
        ys = store[3.0]['decC'][di]
        n = min(len(xs), len(ys))
        pairs = [(xs[i], ys[i]) for i in range(n)
                 if len(xs[i]) and len(ys[i]) and len(xs[i]) == len(ys[i])]
        if len(pairs) < 10:
            print('%-7.1f  (segment counts differ; use profile_shift_affine.py)'
                  % float(tag.split('_d')[1].replace('p', '.')))
            continue
        rg = np.random.default_rng(4)
        bs = []
        for _ in range(200):
            pk = rg.integers(0, len(pairs), len(pairs))
            X = np.concatenate([pairs[i][0] for i in pk])
            Y = np.concatenate([pairs[i][1] for i in pk])
            bs.append(np.polyfit(X, Y, 1))
        bs = np.array(bs)
        print('%-7.1f %18s %18s'
              % (float(tag.split('_d')[1].replace('p', '.')),
                 f'{bs[:,0].mean():.4f} +- {bs[:,0].std(ddof=1):.4f}',
                 f'{bs[:,1].mean():+.2f} +- {bs[:,1].std(ddof=1):.2f}'))
