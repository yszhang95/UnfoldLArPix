"""The 3 ms closure with the HYBRID calibration sample.

Compute-budget salvage of the 10x plan: two depths carry 500 independent
events (iso500, seed family disjoint from every test sample) --
d = 1.5 cm (t = 9 us) and d = 16.5 cm (t = 103 us) -- and the remaining
eight keep the 50-event iso50 statistics. The joint bootstrap resamples
each depth from its own sample, so the mixed precision is handled
honestly.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear
from iso3ms_calib import collect, find_3ms, SIG

I500 = {'pgun_mu_3gev_iso50_d01p5': 'pgun_mu_3gev_iso500_d01p5',
        'pgun_mu_3gev_iso50_d16p5': 'pgun_mu_3gev_iso500_d16p5'}
DIRS500 = [f'{A50.AO}/iso500cal/C', '/home/yousen/iso500_staging/C']
OUT = f'{A50.AO}/iso50_3ms_report'


def find_500(tag, ev):
    for base in DIRS500:
        p = f'{base}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            return p
    return None


t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

print('collect 1 ms iso50 (8 depths keep 50 ev)', flush=True)
p1 = collect('', lambda tag, ev: A50.find_solved('C', tag, ev))
cal = {'effq': {t: p1['effq'][t] for t in A50.TAGS},
       'decC': {t: p1['decC'][t] for t in A50.TAGS}}

print('collect iso500 anchors (500 ev)', flush=True)
for t50, t500 in I500.items():
    f = np.load(f'{A50.NFS}/{t500}_tred_nb1.npz', allow_pickle=True)
    ee, dd = [], []
    for ev in range(500):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        ee.append(segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
        fp = find_500(t500, ev)
        dd.append(segment_dqdx(*smear(*A50.deconv_pix(fp), SIG))
                  if fp else np.array([]))
    cal['effq'][t50] = ee
    cal['decC'][t50] = dd
    print(f'  {t500} done ({sum(len(x) > 0 for x in dd)} solved)',
          flush=True)

print('collect 3 ms (test sample)', flush=True)
p3 = collect('_3ms', find_3ms)

ratio = np.array([mpvfn(np.concatenate(cal['decC'][t]))
                  / mpvfn(np.concatenate(cal['effq'][t]))
                  for t in A50.TAGS])
print('hybrid calibration curve:', np.round(ratio, 4).tolist())


def boot(joint, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    taus = []
    d3 = [p3['decC'][t] for t in A50.TAGS]
    for _ in range(nboot):
        if joint:
            div = []
            for tag in A50.TAGS:
                cd, ce = cal['decC'][tag], cal['effq'][tag]
                n = len(cd)
                pick = rng.integers(0, n, n)
                div.append(mpvfn(np.concatenate([cd[i] for i in pick]))
                           / mpvfn(np.concatenate([ce[i] for i in pick])))
        else:
            div = ratio
        m = []
        for i, segs in enumerate(d3):
            pick = rng.integers(0, len(segs), len(segs))
            m.append(mpvfn(np.concatenate([segs[j] for j in pick]))
                     / div[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        taus.append(-1.0 / sl / 1000.0)
    taus = np.array(taus)
    return float(np.median(taus)), float(taus.std())


res = {}
for name, joint in [('fixed', False), ('joint', True)]:
    tau, err = boot(joint)
    res[name] = {'tau': tau, 'err': err}
    print('cal(hybrid) %-6s  tau = %6.2f +- %5.2f  (sigma_rate %.2e /us)'
          % (name, tau, err, err / tau**2 * 1e-3), flush=True)
json.dump({'ratio_hybrid': ratio.tolist(),
           'anchors_500ev': list(I500.values()), 'tau': res},
          open(f'{OUT}/hybrid_calib.json', 'w'), indent=1)
print(f'-> {OUT}/hybrid_calib.json', flush=True)
