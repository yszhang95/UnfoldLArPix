"""Everything refit with the shallowest depth (d = 1.5 cm) EXCLUDED.

Rationale: the acquisition-edge response truncation (the in-simulation
realisation of the Ramo prompt deficit) is 1.3% at d = 1.5 cm and both
arms' capture curves show the point off-trend; the raw hits additionally
dip to 0.91 there. Fits run over the nine depths t = 28-179 us.

Outputs: per-estimator 1 ms tau (3 cm MPV), B/C capture slides, and the
3 ms closures for arms C and B (fixed curve, joint bootstrap, and the
hybrid variant with the remaining 500-event anchor at d = 16.5 cm).
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear
from iso3ms_calib import SIG, find_3ms
from hybrid_calib import find_500
from iso3ms_B_calib import find_b3

TAGS = A50.TAGS[1:]                     # drop d01p5
t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]


def boot_tau9(per_event, statfn=mpvfn, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(nboot):
        m = []
        for evs in per_event:
            pick = rng.integers(0, len(evs), len(evs))
            m.append(statfn(np.concatenate([evs[i] for i in pick])))
        sl, _ = np.polyfit(t_us, np.log(m), 1)
        taus.append(-1.0/sl/1000.0)
    taus = np.array(taus)
    return float(np.median(taus)), float(taus.std())


# ---------- 1 ms sample: raw pools (record statistics) ------------------
print('collect 1 ms raw (all four estimators)', flush=True)
raw1 = {k: {} for k in ['effq', 'hits', 'decB', 'decC']}
sums1 = {k: {} for k in ['effq', 'decB', 'decC']}
for tag in TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for k in raw1:
        raw1[k][tag] = []
    for k in sums1:
        sums1[k][tag] = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
        hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
        raw1['effq'][tag].append(segment_dqdx(el[:, 0], el[:, 1], eq))
        raw1['hits'][tag].append(segment_dqdx(hl[:, 0], hl[:, 1], hq))
        sums1['effq'][tag].append(eq.sum())
        for arm, kk in [('B', 'decB'), ('C', 'decC')]:
            p = A50.find_solved(arm, tag, ev)
            pa, pb, q = A50.deconv_pix(p)
            raw1[kk][tag].append(segment_dqdx(pa, pb, q))
            sums1[kk][tag].append(q.sum())
    print(f'  {tag}', flush=True)

print('\n== 1 ms tau (MPV, 3 cm, 9 depths, d=1.5 excluded) ==')
tau1 = {}
for k in raw1:
    tau1[k] = boot_tau9([raw1[k][t] for t in TAGS])
    print('  %-5s tau = %.3f +- %.3f' % (k, *tau1[k]), flush=True)

print('\n== capture slides, 9 depths ==')
slides = {}
for kk in ['decB', 'decC']:
    ysum = np.log([np.mean(np.array(sums1[kk][t]) / np.array(sums1['effq'][t]))
                   for t in TAGS])
    ympv = np.log([mpvfn(np.concatenate(raw1[kk][t]))
                   / mpvfn(np.concatenate(raw1['effq'][t])) for t in TAGS])
    bs, _ = np.polyfit(t_us, ysum, 1)
    bm, _ = np.polyfit(t_us, ympv, 1)
    slides[kk] = (100*bs*170, 100*bm*170)
    print('  %s: sum %+.2f%%, mpv %+.2f%% / 170 us'
          % (kk, *slides[kk]), flush=True)

# ---------- sym-smeared pools for the closures ---------------------------
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


def closure(arm, cal1, test3, anchor=None):
    """9-depth closure: raw, fixed-curve, joint; optional 500-ev anchor."""
    ratio = np.array([mpvfn(np.concatenate(cal1['dec'][t]))
                      / mpvfn(np.concatenate(cal1['effq'][t]))
                      for t in TAGS])
    if anchor:
        i = TAGS.index(anchor['tag'])
        ratio_h = ratio.copy()
        ratio_h[i] = anchor['ratio']
    res = {}
    mraw = [np.concatenate(test3['dec'][t]) for t in TAGS]
    rng = np.random.default_rng(1)

    def boot(div, joint_cal=None, nboot=200):
        taus = []
        for _ in range(nboot):
            if joint_cal is not None:
                dv = []
                for tag in TAGS:
                    cd, ce = joint_cal['dec'][tag], joint_cal['effq'][tag]
                    n = len(cd)
                    pick = rng.integers(0, n, n)
                    dv.append(mpvfn(np.concatenate([cd[i] for i in pick]))
                              / mpvfn(np.concatenate([ce[i] for i in pick])))
            else:
                dv = div
            m = []
            for i, tag in enumerate(TAGS):
                segs = test3['dec'][tag]
                pick = rng.integers(0, len(segs), len(segs))
                m.append(mpvfn(np.concatenate([segs[j] for j in pick]))
                         / dv[i])
            sl, _ = np.polyfit(t_us, np.log(m), 1)
            taus.append(-1.0/sl/1000.0)
        taus = np.array(taus)
        return float(np.median(taus)), float(taus.std())

    res['raw'] = boot(np.ones(len(TAGS)))
    res['fixed'] = boot(ratio)
    res['joint'] = boot(None, joint_cal=cal1)
    if anchor:
        res['hybrid_fixed'] = boot(ratio_h)
    for k, v in res.items():
        print('  %s %-12s tau = %6.2f +- %5.2f' % (arm, k, *v), flush=True)
    return res


out = {'tau1': tau1, 'slides': slides}

print('\ncollect sym pools: 1 ms C', flush=True)
c1 = collect_sym('', lambda tag, ev: A50.find_solved('C', tag, ev))
print('collect sym pools: 3 ms C', flush=True)
c3 = collect_sym('_3ms', find_3ms)
# 500-event anchor at d16p5 (iso500)
t500 = 'pgun_mu_3gev_iso500_d16p5'
f = np.load(f'{A50.NFS}/{t500}_tred_nb1.npz', allow_pickle=True)
ee, dd = [], []
for ev in range(500):
    el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
    eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
    ee.append(segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
    fp = find_500(t500, ev)
    dd.append(segment_dqdx(*smear(*A50.deconv_pix(fp), SIG))
              if fp else np.array([]))
anch = {'tag': 'pgun_mu_3gev_iso50_d16p5',
        'ratio': mpvfn(np.concatenate(dd)) / mpvfn(np.concatenate(ee))}
print(f'anchor d16p5 (500 ev): ratio {anch["ratio"]:.4f}', flush=True)

print('\n== 3 ms closure, arm C, 9 depths ==')
out['C'] = closure('C', c1, c3, anchor=anch)

print('\ncollect sym pools: 1 ms B', flush=True)
b1 = collect_sym('', lambda tag, ev: A50.find_solved('B', tag, ev))
print('collect sym pools: 3 ms B', flush=True)
b3 = collect_sym('_3ms', find_b3)
print('\n== 3 ms closure, arm B (no refit), 9 depths ==')
out['B'] = closure('B', b1, b3)

json.dump(out, open(f'{A50.AO}/iso50_3ms_report/exclude_first.json', 'w'),
          indent=1, default=float)
print('-> exclude_first.json', flush=True)
