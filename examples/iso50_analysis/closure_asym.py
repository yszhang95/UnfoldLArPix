"""Asymmetric (percentile) errors for every entry of the closure table.

The fitted quantity (rate) is Gaussian; tau = 1/rate is right-skewed, so
tau is quoted as median with 16-84-percentile asymmetric errors from
2000 bootstrap replicas. Also rechecks the 1 ms table, where the
relative rate error is small and the asymmetry should be negligible.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear
from iso3ms_calib import SIG, find_3ms
from iso3ms_B_calib import find_b3

TAGS = A50.TAGS[1:]
t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]
NBOOT = 2000


def collect(suffix, kind, find_dec=None):
    pool = {}
    for tag in TAGS:
        f = np.load(f'{A50.NFS}/{tag}_tred_nb1{suffix}.npz',
                    allow_pickle=True)
        pool[tag] = []
        for ev in range(50):
            if kind == 'effq':
                el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
                eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
                pool[tag].append(segment_dqdx(
                    *smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
            elif kind == 'effq_raw':
                el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
                eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
                pool[tag].append(segment_dqdx(el[:, 0], el[:, 1], eq))
            elif kind == 'hits':
                hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
                hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
                pool[tag].append(segment_dqdx(hl[:, 0], hl[:, 1], hq))
            else:
                fp = find_dec(tag, ev)
                pool[tag].append(segment_dqdx(
                    *smear(*A50.deconv_pix(fp), SIG))
                    if fp else np.array([]))
        print(f'  {tag}', flush=True)
    return pool


def taus_of(pool, div=None, seed=1):
    rng = np.random.default_rng(seed)
    dv = np.ones(len(TAGS)) if div is None else div
    out = []
    for _ in range(NBOOT):
        m = []
        for i, tag in enumerate(TAGS):
            segs = pool[tag]
            pick = rng.integers(0, len(segs), len(segs))
            m.append(mpvfn(np.concatenate([segs[j] for j in pick])) / dv[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        out.append(-1.0/sl/1000.0)
    return np.array(out)


def quote(name, taus):
    med = np.median(taus)
    lo, hi = np.percentile(taus, [15.87, 84.13])
    print('%-22s tau = %6.2f  +%.2f -%.2f   (std %.2f)'
          % (name, med, hi-med, med-lo, taus.std()), flush=True)
    return {'med': float(med), 'up': float(hi-med), 'dn': float(med-lo),
            'std': float(taus.std())}


print('pools: 1 ms', flush=True)
e1 = collect('', 'effq')
h1r = collect('', 'hits')
c1 = collect('', 'dec', lambda t, e: A50.find_solved('C', t, e))
b1 = collect('', 'dec', lambda t, e: A50.find_solved('B', t, e))
print('pools: 3 ms', flush=True)
e3 = collect('_3ms', 'effq')
h3 = collect('_3ms', 'hits')
c3 = collect('_3ms', 'dec', find_3ms)
b3 = collect('_3ms', 'dec', find_b3)

curve = {}
e1m = np.array([mpvfn(np.concatenate(e1[t])) for t in TAGS])
e1r = collect('', 'effq_raw')
e1rm = np.array([mpvfn(np.concatenate(e1r[t])) for t in TAGS])
curve['C'] = np.array([mpvfn(np.concatenate(c1[t])) for t in TAGS]) / e1m
curve['B'] = np.array([mpvfn(np.concatenate(b1[t])) for t in TAGS]) / e1m
curve['h'] = np.array([mpvfn(np.concatenate(h1r[t])) for t in TAGS]) / e1rm

out = {}
print('\n== 3 ms closure with asymmetric errors ==')
out['truth3'] = quote('truth control', taus_of(e3))
out['hits3_raw'] = quote('hits raw', taus_of(h3))
out['hits3_cal'] = quote('hits calibrated', taus_of(h3, curve['h']))
out['B3_raw'] = quote('dec B raw', taus_of(b3))
out['B3_cal'] = quote('dec B calibrated', taus_of(b3, curve['B']))
out['C3_raw'] = quote('dec C raw', taus_of(c3))
out['C3_cal'] = quote('dec C calibrated', taus_of(c3, curve['C']))

print('\n== 1 ms table asymmetry check ==')
out['effq1'] = quote('effq 1ms', taus_of(e1))
out['hits1'] = quote('hits 1ms', taus_of(h1r))
out['decB1'] = quote('decB 1ms', taus_of(b1))
out['decC1'] = quote('decC 1ms', taus_of(c1))

json.dump(out, open(f'{A50.AO}/iso50_3ms_report/closure_asym.json', 'w'),
          indent=1)
print('-> closure_asym.json', flush=True)
