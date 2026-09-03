"""How far does charge actually move?  Fit the effective transport kernel.

Regress the reconstructed row charge on a WHOLE NEIGHBOURHOOD of truth rows

    R(b) = sum_{k=-K..K} w_k * T(b+k) + c

The fitted weights ARE the effective transport kernel: w_0 is the local
gain, w_k (k != 0) is how much of the truth charge k rows away ends up
here.  K = 16 px = 7.1 cm, so the fit sees past every distance asked about
(1 cm = 2.26 px, 2 cm = 4.51, 3 cm = 6.77, 5 cm = 11.28).

The isochronous track fluctuates strongly row to row (Landau), so the
design matrix is well conditioned -- reported as the truth autocorrelation.

Validation: inject a KNOWN kernel into the truth and check it is recovered.
Controls: `hits` (per-pixel charge, but its ADC still sees neighbour
induction) and the deconv against its own input.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import dqdx_lib as L

K = 16
PITCH = 0.4434
NBOOT = 100
CM = {1.0: 1.0 / PITCH, 2.0: 2.0 / PITCH, 3.0: 3.0 / PITCH, 5.0: 5.0 / PITCH}


def design(pairs, idx=None):
    """Stack (T-neighbourhood, R) rows from a list of (T, R) profiles."""
    if idx is not None:
        pairs = [pairs[i] for i in idx]
    X, Y = [], []
    for t, r in pairs:
        n = t.size
        if n < 2 * K + 20:
            continue
        rows = np.arange(K, n - K)
        X.append(np.stack([t[rows + k] for k in range(-K, K + 1)], axis=1))
        Y.append(r[rows])
    if not X:
        return None, None
    X = np.vstack(X); Y = np.concatenate(Y)
    return np.hstack([X, np.ones((X.shape[0], 1))]), Y


def fit_kernel(pairs, idx=None):
    X, Y = design(pairs, idx)
    if X is None:
        return None
    w, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return w


def summarise(w):
    k = np.arange(-K, K + 1)
    ker = w[:-1]
    tot = ker.sum()
    out = {'w0': float(ker[K]), 'total': float(tot), 'c': float(w[-1]),
           'w_pm1': float(ker[K - 1] + ker[K + 1])}
    for cm, px in CM.items():
        m = np.abs(k) > px
        out[f'beyond_{cm:g}cm'] = float(ker[m].sum() / tot)
        out[f'absbeyond_{cm:g}cm'] = float(np.abs(ker[m]).sum() / tot)
    return out, ker


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    ESTS = ['hits', 'decC', 'decB']
    res = {}

    # ---------------------------------------------------------- validation
    print('=' * 78)
    print('VALIDATION: inject a known kernel into the truth, recover it')
    print('=' * 78)
    tag = PS.TAGS[5]
    fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    base = []
    for ev in range(50):
        pr = PT.profiles(tag, ev, fz)
        if pr is not None:
            base.append(pr['truth'])

    def apply_kernel(T, ker, off):
        out = np.zeros_like(T)
        for w_, k_ in zip(ker, off):
            if k_ == 0:
                out += w_ * T
            elif k_ > 0:
                out[k_:] += w_ * T[:-k_]
            else:
                out[:k_] += w_ * T[-k_:]
        return out

    tests = [('gain 0.95 only', [0.95], [0]),
             ('blur 5% each side', [0.90, 0.05, 0.05], [0, -1, 1]),
             ('move 10% to +7px (3.1cm)', [0.90, 0.10], [0, 7]),
             ('share 8% at +-12px (5.3cm)', [0.84, 0.08, 0.08], [0, -12, 12])]
    for lab, ker, off in tests:
        pr = [(t, apply_kernel(t, ker, off)) for t in base]
        w = fit_kernel(pr)
        s, kk = summarise(w)
        rec = {f'{o:+d}': round(float(kk[K + o]), 4) for o in off}
        print(f'  {lab:28s} recovered {rec}  total {s["total"]:.4f}')

    # ------------------------------------------------------------ the data
    print()
    print('=' * 78)
    print('FITTED KERNEL, arm C (bootstrap over events)')
    print('=' * 78)
    print('%-6s %8s %8s %9s %9s %9s %9s %9s' %
          ('depth', 'w0', 'w(+-1)', 'total', '>1cm', '>2cm', '>3cm', '>5cm'))
    kernels = {}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        pairs = {k: [] for k in ESTS}
        pairs['decC|hits'] = []
        for ev in range(50):
            pr = PT.profiles(tag, ev, fz)
            if pr is None:
                continue
            for k in ESTS:
                pairs[k].append((pr['truth'], pr[k]))
            pairs['decC|hits'].append((pr['hits'], pr['decC']))
        rec = {'depth_cm': d_cm, 't_us': L.drift_time_us(d_cm)}
        for k in ESTS + ['decC|hits']:
            w = fit_kernel(pairs[k])
            if w is None:
                continue
            s, ker = summarise(w)
            bs = []
            n = len(pairs[k])
            for _ in range(NBOOT):
                wb = fit_kernel(pairs[k], rng.integers(0, n, n))
                if wb is not None:
                    bs.append(summarise(wb)[0])
            for fld in list(s):
                s[fld + '_err'] = float(np.std([b[fld] for b in bs], ddof=1))
            rec[k] = s
            kernels.setdefault(k, []).append(ker.tolist())
        res[tag] = rec
        c = rec['decC']
        print('%-6.1f %8.4f %8.4f %9.4f %+9.4f %+9.4f %+9.4f %+9.4f'
              % (d_cm, c['w0'], c['w_pm1'], c['total'], c['beyond_1cm'],
                 c['beyond_2cm'], c['beyond_3cm'], c['beyond_5cm']))

    print()
    print('same, with bootstrap errors, pooled over the 9 fitted depths:')
    for k in ESTS + ['decC|hits']:
        for fld in ['w0', 'w_pm1', 'total', 'beyond_1cm', 'beyond_2cm',
                    'beyond_3cm', 'beyond_5cm', 'absbeyond_3cm']:
            v = np.array([res[g][k][fld] for g in res if res[g]['depth_cm'] > 3])
            e = np.array([res[g][k][fld + '_err'] for g in res
                          if res[g]['depth_cm'] > 3])
            w_ = 1 / e ** 2
            print('  %-10s %-14s %+9.5f +- %.5f'
                  % (k, fld, np.average(v, weights=w_), 1 / np.sqrt(w_.sum())))
        print()

    print('MEAN KERNEL, arm C, 9 fitted depths (w_k vs distance):')
    kc = np.array(kernels['decC'][1:]).mean(axis=0)
    for o in range(-K, K + 1):
        bar = '#' * int(abs(kc[K + o]) * 300)
        print('  k=%+3d  %+7.1f mm  %+9.5f %s' % (o, o * PITCH * 10, kc[K + o], bar))
    json.dump({'K': K, 'depths': res, 'mean_kernel_decC': kc.tolist()},
              open(f'{PS.OUT}/transport_kernel.json', 'w'), indent=1)
    print('\n->', f'{PS.OUT}/transport_kernel.json')
