"""The reverse closure: calibration derived at 3 ms, applied to the
1 ms sample (strong-signal test side, 15% attenuation over the span).

If the capture curve is lifetime-independent the deconv arms must
recover lambda = 1; the hits curve should now UNDER-correct (the 3 ms
slide is shallower), leaving lambda high by the same ~0.09/ms the
forward test showed as over-correction. Nine depths, 2000 replicas,
rates quoted.
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
                pool[tag].append(segment_dqdx(
                    *pixmap(el[:, 0], el[:, 1], eq)))
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


def rate_quote(name, pool, div=None, seed=1):
    rng = np.random.default_rng(seed)
    dv = np.ones(len(TAGS)) if div is None else div
    rates = []
    for _ in range(NBOOT):
        m = []
        for i, tag in enumerate(TAGS):
            segs = pool[tag]
            pick = rng.integers(0, len(segs), len(segs))
            m.append(mpvfn(np.concatenate([segs[j] for j in pick])) / dv[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        rates.append(-sl*1000.0)
    r = np.array(rates)
    med, std = float(np.median(r)), float(r.std())
    print('%-24s lambda = %6.3f +- %5.3f /ms' % (name, med, std),
          flush=True)
    return [med, std]


print('pools: 3 ms (calibration source)', flush=True)
e3 = collect('_3ms', 'effq')
h3 = collect('_3ms', 'hits')
e3r = collect('_3ms', 'effq_raw')
c3 = collect('_3ms', 'dec', find_3ms)
b3 = collect('_3ms', 'dec', find_b3)
print('pools: 1 ms (test sample)', flush=True)
e1 = collect('', 'effq')
h1 = collect('', 'hits')
e1r = collect('', 'effq_raw')
c1 = collect('', 'dec', lambda t, e: A50.find_solved('C', t, e))
b1 = collect('', 'dec', lambda t, e: A50.find_solved('B', t, e))

e3m = np.array([mpvfn(np.concatenate(e3[t])) for t in TAGS])
e3rm = np.array([mpvfn(np.concatenate(e3r[t])) for t in TAGS])
curve = {
    'C': np.array([mpvfn(np.concatenate(c3[t])) for t in TAGS]) / e3m,
    'B': np.array([mpvfn(np.concatenate(b3[t])) for t in TAGS]) / e3m,
    'h': np.array([mpvfn(np.concatenate(h3[t])) for t in TAGS]) / e3rm}

out = {}
print('\n== reverse closure: 3 ms curve applied to 1 ms (truth: 1.0) ==')
out['truth1'] = rate_quote('truth control', e1)
out['hits_raw'] = rate_quote('hits raw', h1)
out['hits_cal'] = rate_quote('hits cal (3ms curve)', h1, curve['h'])
out['B_raw'] = rate_quote('dec B raw', b1)
out['B_cal'] = rate_quote('dec B cal (3ms curve)', b1, curve['B'])
out['C_raw'] = rate_quote('dec C raw', c1)
out['C_cal'] = rate_quote('dec C cal (3ms curve)', c1, curve['C'])
json.dump(out, open(f'{A50.AO}/iso50_3ms_report/reverse_closure.json', 'w'),
          indent=1)
print('-> reverse_closure.json', flush=True)
