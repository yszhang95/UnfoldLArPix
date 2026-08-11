"""Remaining record numbers under the d=1.5-excluded (9-depth) convention.

Computes what exclude_first.py did not: the tmean/median cross-check
columns (1 ms), the 3 ms truth control and hits closure rows, rate-space
errors for the joint bootstraps, and the pixel-q-cut table.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear
from iso3ms_calib import SIG, find_3ms
from iso3ms_B_calib import find_b3
from qcut_dqdx import cutseg

TAGS = A50.TAGS[1:]
t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]


def tmean(v, keep=0.7):
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    return float(v[:max(int(len(v)*keep), 3)].mean())


def boot9(per_event, statfn, div=None, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    taus, rates = [], []
    dv = np.ones(len(TAGS)) if div is None else div
    for _ in range(nboot):
        m = []
        for i, evs in enumerate(per_event):
            pick = rng.integers(0, len(evs), len(evs))
            m.append(statfn(np.concatenate([evs[j] for j in pick])) / dv[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        rates.append(-sl*1000.0)
        taus.append(-1.0/sl/1000.0)
    taus, rates = np.array(taus), np.array(rates)
    return (float(np.median(taus)), float(taus.std()),
            float(np.median(rates)), float(rates.std()))


out = {}
# ---------- 1 ms raw pools: cross-check statistics ----------------------
print('1 ms pools', flush=True)
raw1 = {k: {} for k in ['effq', 'hits', 'decB', 'decC']}
q1 = {'effq': {}, 'decC': {}}
for tag in TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for k in raw1:
        raw1[k][tag] = []
    for k in q1:
        q1[k][tag] = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
        hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
        raw1['effq'][tag].append(segment_dqdx(el[:, 0], el[:, 1], eq))
        raw1['hits'][tag].append(segment_dqdx(hl[:, 0], hl[:, 1], hq))
        pe = pixmap(el[:, 0], el[:, 1], eq)
        q1['effq'][tag].append(pe)
        for arm, kk in [('B', 'decB'), ('C', 'decC')]:
            p = A50.find_solved(arm, tag, ev)
            pa, pb, q = A50.deconv_pix(p)
            raw1[kk][tag].append(segment_dqdx(pa, pb, q))
            if kk == 'decC':
                q1['decC'][tag].append((pa, pb, q))
    print(f'  {tag}', flush=True)

print('\n== 1 ms cross-check statistics (9 depths) ==')
out['xcheck'] = {}
for k in raw1:
    pe = [raw1[k][t] for t in TAGS]
    row = {}
    for name, fn in [('mpv', mpvfn), ('tmean', tmean),
                     ('median', np.median)]:
        tau, err, _, _ = boot9(pe, fn)
        row[name] = [tau, err]
        print('  %-5s %-7s %.3f +- %.3f' % (k, name, tau, err), flush=True)
    out['xcheck'][k] = row

print('\n== pixel-q-cut table (9 depths) ==')
out['qcut'] = {}
for cut in [0.0, 1.0, 2.0, 5.0]:
    row = {}
    for k in ('effq', 'decC'):
        pe = [[cutseg(*s, cut) for s in q1[k][t]] for t in TAGS]
        tau, err, _, _ = boot9(pe, mpvfn)
        row[k] = [tau, err]
    out['qcut'][f'{cut:g}'] = row
    print('  cut>%g: effq %.3f+-%.3f  decC %.3f+-%.3f'
          % (cut, *row['effq'], *row['decC']), flush=True)

# ---------- 3 ms: truth control + hits row -------------------------------
print('\n3 ms pools (effq sym, hits raw)', flush=True)
e3, h3 = {}, {}
for tag in TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1_3ms.npz', allow_pickle=True)
    e3[tag], h3[tag] = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        e3[tag].append(
            segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
        hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
        hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
        h3[tag].append(segment_dqdx(hl[:, 0], hl[:, 1], hq))
    print(f'  {tag}', flush=True)

print('\n== 3 ms truth control and hits (9 depths) ==')
tau, err, r, rerr = boot9([e3[t] for t in TAGS], mpvfn)
out['truth3'] = [tau, err]
print('  truth control  tau = %.2f +- %.2f' % (tau, err), flush=True)
tau, err, _, _ = boot9([h3[t] for t in TAGS], mpvfn)
out['hits3_raw'] = [tau, err]
print('  hits raw       tau = %.2f +- %.2f' % (tau, err), flush=True)
hcurve = np.array([mpvfn(np.concatenate(raw1['hits'][t]))
                   / mpvfn(np.concatenate(raw1['effq'][t])) for t in TAGS])
tau, err, _, _ = boot9([h3[t] for t in TAGS], mpvfn, div=hcurve)
out['hits3_cal'] = [tau, err]
print('  hits calibr.   tau = %.2f +- %.2f' % (tau, err), flush=True)

# ---------- joint bootstraps in rate space -------------------------------
def collect_sym(suffix, find_dec):
    pool = {'effq': {}, 'dec': {}}
    for tag in TAGS:
        f = np.load(f'{A50.NFS}/{tag}_tred_nb1{suffix}.npz',
                    allow_pickle=True)
        pool['effq'][tag] = []
        pool['dec'][tag] = []
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            pool['effq'][tag].append(
                segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
            fp = find_dec(tag, ev)
            pool['dec'][tag].append(
                segment_dqdx(*smear(*A50.deconv_pix(fp), SIG))
                if fp else np.array([]))
        print(f'  {tag}', flush=True)
    return pool


print('\njoint bootstraps (rate space)', flush=True)
for arm, find1, find3d in [
        ('C', lambda t, e: A50.find_solved('C', t, e), find_3ms),
        ('B', lambda t, e: A50.find_solved('B', t, e), find_b3)]:
    c1 = collect_sym('', find1)
    c3 = collect_sym('_3ms', find3d)
    rng = np.random.default_rng(1)
    rates = []
    d3 = [c3['dec'][t] for t in TAGS]
    for _ in range(200):
        dv = []
        for tag in TAGS:
            cd, ce = c1['dec'][tag], c1['effq'][tag]
            pick = rng.integers(0, 50, 50)
            dv.append(mpvfn(np.concatenate([cd[i] for i in pick]))
                      / mpvfn(np.concatenate([ce[i] for i in pick])))
        m = []
        for i, segs in enumerate(d3):
            pick = rng.integers(0, len(segs), len(segs))
            m.append(mpvfn(np.concatenate([segs[j] for j in pick]))
                     / dv[i])
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        rates.append(-sl*1000.0)
    rates = np.array(rates)
    rmed, rstd = float(np.median(rates)), float(rates.std())
    out[f'joint_{arm}'] = {'rate': [rmed, rstd],
                           'tau_from_rate': [1.0/rmed,
                                             rstd/rmed**2]}
    print('  %s joint: rate %.4f +- %.4f /ms -> tau %.2f +- %.2f'
          % (arm, rmed, rstd, 1.0/rmed, rstd/rmed**2), flush=True)

json.dump(out, open(f'{A50.AO}/iso50_3ms_report/record9.json', 'w'),
          indent=1, default=float)
print('-> record9.json', flush=True)
