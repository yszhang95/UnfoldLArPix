"""Error budget of the 3 ms closure + the smoothed-calibration variant.

sigma_tau = tau^2 * sigma_rate. Compare: raw, calibrated with the 10
per-depth ratio points (noise of the 1 ms sample injected), calibrated
with a LINEAR fit R(t) = a + b t to those points (1 ms noise averaged),
and a joint bootstrap where the 1 ms calibration sample is resampled
together with the 3 ms sample (honest total error).
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from iso3ms_calib import collect, find_3ms

t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

print('collect 1 ms', flush=True)
p1 = collect('', lambda tag, ev: A50.find_solved('C', tag, ev))
print('collect 3 ms', flush=True)
p3 = collect('_3ms', find_3ms)

mpv1 = {k: np.array([mpvfn(np.concatenate(p1[k][t])) for t in A50.TAGS])
        for k in ['effq', 'decC']}
ratio = mpv1['decC'] / mpv1['effq']
b, a = np.polyfit(t_us, ratio, 1)
Rsmooth = a + b * t_us
print('\nlinear calibration curve: R(t) = %.4f %+．3e t' % (a, b)
      if False else
      f'\nlinear calibration curve: R(t) = {a:.4f} {b:+.3e} * t')
print('point-vs-smooth spread: rms %.4f' % np.std(ratio - Rsmooth))


def tau_of(segs_by_depth, div):
    m = np.array([mpvfn(np.concatenate(s)) / d
                  for s, d in zip(segs_by_depth, div)])
    sl, _ = np.polyfit(t_us, np.log(m), 1)
    return -1.0 / sl / 1000.0


def boot(div_fixed=None, joint=False, nboot=200, seed=1):
    """Bootstrap over events; joint=True also resamples the 1 ms sample
    and rebuilds the (smooth) calibration each replica."""
    rng = np.random.default_rng(seed)
    taus = []
    d3 = [p3['decC'][t] for t in A50.TAGS]
    for _ in range(nboot):
        if joint:
            r = []
            for tag in A50.TAGS:
                e1d, e1e = p1['decC'][tag], p1['effq'][tag]
                pick = rng.integers(0, 50, 50)
                r.append(mpvfn(np.concatenate([e1d[i] for i in pick]))
                         / mpvfn(np.concatenate([e1e[i] for i in pick])))
            bb, aa = np.polyfit(t_us, r, 1)
            div = aa + bb * t_us
        else:
            div = div_fixed
        m = []
        for i, segs in enumerate(d3):
            pick = rng.integers(0, len(segs), len(segs))
            m.append(mpvfn(np.concatenate([segs[j] for j in pick]))
                     / div[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        taus.append(-1.0 / sl / 1000.0)
    taus = np.array(taus)
    return float(np.median(taus)), float(taus.std())


ones = np.ones_like(t_us)
for name, kw in [('raw (div=1)', dict(div_fixed=ones)),
                 ('cal: 10-point curve', dict(div_fixed=ratio)),
                 ('cal: linear-fit curve', dict(div_fixed=Rsmooth)),
                 ('cal: linear, joint boot', dict(joint=True))]:
    tau, err = boot(**kw)
    print('%-26s tau = %6.2f +- %5.2f   (sigma_rate %.2e /us)' %
          (name, tau, err, err / tau**2 * 1e-3))
