"""Same fit as deep_figs.py, but with the per-depth MPV uncertainty measured,
plotted, and used: unweighted vs weighted OLS, and chi2/dof as an honesty check
on the quoted lambda error."""
import numpy as np, os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import track_dqdx as T
from deep_figs import collect, CO, LB, style, TAGS, OUT

NB = 400

if __name__ == '__main__':
    d_cm, t_us, per = collect()
    n = len(TAGS)
    rng = np.random.default_rng(7)
    # bootstrap over EVENTS: per-depth MPV distribution, kept per resample so
    # the line fit inherits the correlations
    draws = {k: np.zeros((NB, n)) for k in CO}
    for b in range(NB):
        for k in CO:
            for i in range(n):
                evs = per[k][i]
                pk = rng.integers(0, len(evs), len(evs))
                v = np.concatenate([evs[j] for j in pk if len(evs[j])])
                draws[k][b, i] = L.mpv_of(v)[0]
    mpv = {k: np.array([L.mpv_of(np.concatenate([x for x in per[k][i] if len(x)]))[0]
                        for i in range(n)]) for k in CO}
    err = {k: draws[k].std(axis=0, ddof=1) for k in CO}

    A = np.vstack([t_us, np.ones(n)]).T
    out = {}
    print('=' * 92)
    print('PER-DEPTH MPV WITH ITS BOOTSTRAP ERROR (50 events, %d resamples)' % NB)
    print('=' * 92)
    print('%8s %8s' % ('d[cm]', 't[us]') + ''.join('%20s' % LB[k].split('(')[0].strip()
                                                   for k in ['effq', 'hits', 'decC']))
    for i in range(n):
        print('%8.1f %8.1f' % (d_cm[i], t_us[i]) + ''.join(
            '%13.2f +-%5.2f' % (mpv[k][i], err[k][i]) for k in ['effq', 'hits', 'decC']))
    print()
    print('%-8s %20s %20s %12s %12s' % ('est', 'unweighted OLS', 'weighted OLS',
                                        'chi2/dof', 'scaled err'))
    for k in ['effq', 'hits', 'decC']:
        y = np.log(mpv[k]); sy = err[k] / mpv[k]
        lam_u = np.array([-np.linalg.lstsq(A, np.log(draws[k][b]), rcond=None)[0][0] * 1000
                          for b in range(NB)])
        W = np.diag(1 / sy ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        p = cov @ (A.T @ W @ y)
        lam_w, sig_w = -p[0] * 1000, np.sqrt(cov[0, 0]) * 1000
        resid = y - A @ p
        chi2 = float((resid ** 2 / sy ** 2).sum()) / (n - 2)
        out[k] = {'mpv': mpv[k].tolist(), 'err': err[k].tolist(),
                  'lam_unw': float(np.median(lam_u)), 'lam_unw_err': float(lam_u.std(ddof=1)),
                  'lam_w': float(lam_w), 'lam_w_err': float(sig_w), 'chi2_dof': chi2}
        print('%-8s %13.4f +-%6.4f %13.4f +-%6.4f %12.2f %12.4f'
              % (k, out[k]['lam_unw'], out[k]['lam_unw_err'], lam_w, sig_w,
                 chi2, sig_w * np.sqrt(max(chi2, 1.0))))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for a_ in axes:
        style(a_)
    tt = np.linspace(t_us.min() - 6, t_us.max() + 6, 60)
    for k in ['effq', 'hits', 'decC']:
        sl, ic = np.linalg.lstsq(A, np.log(mpv[k]), rcond=None)[0]
        axes[0].errorbar(t_us, mpv[k], yerr=err[k], fmt='o', color=CO[k], ms=6,
                         capsize=3, lw=1.2,
                         label=f'{LB[k]}\n   $\\lambda$ = {out[k]["lam_unw"]:.3f} $\\pm$ '
                               f'{out[k]["lam_unw_err"]:.3f} ms$^{{-1}}$  '
                               f'($\\chi^2$/dof = {out[k]["chi2_dof"]:.2f})')
        axes[0].plot(tt, np.exp(ic + sl * tt), '-', color=CO[k], lw=1.2)
        r = mpv[k] / mpv['effq']
        re = r * np.sqrt((err[k] / mpv[k]) ** 2 + (err['effq'] / mpv['effq']) ** 2)
        axes[1].errorbar(t_us, r, yerr=re, fmt='o-', color=CO[k], ms=6, capsize=3, lw=1.2)
    axes[0].plot(tt, mpv['effq'][0] * np.exp(-(tt - t_us[0]) / 1000.0), 'k--', lw=1.0,
                 label=r'true slope, $\lambda$ = 1 ms$^{-1}$')
    axes[0].set_xlabel(r'drift time  [$\mu$s]'); axes[0].set_ylabel('pooled MPV  [ke/cm]')
    axes[0].set_title('unweighted OLS of $\\ln$MPV on $t$; errors bootstrapped over the 50 events',
                      fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].axhline(1.0, color='0.6', lw=0.8)
    axes[1].set_xlabel(r'drift time  [$\mu$s]')
    axes[1].set_ylabel(r'MPV / MPV$_{\rm truth}$')
    axes[1].set_title('capture, with the same errors (correlated: same events)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/deep_lifetime_fit_err.pdf')
    fig.savefig(f'{OUT}/deep_lifetime_fit_err.png', dpi=150)
    json.dump(out, open(f'{OUT}/deep_err.json', 'w'), indent=1)
    print('\n->', OUT)
