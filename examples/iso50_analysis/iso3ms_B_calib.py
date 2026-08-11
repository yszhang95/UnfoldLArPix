"""The 3 ms closure WITHOUT the refit: arm B end-to-end.

Identical machinery to iso3ms_calib.py, arm B throughout: the
calibration is B's own 1 ms sym-smeared MPV capture curve, divided out
of the 3 ms arm-B dQ/dx.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from iso3ms_calib import SIG
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear

DIRS_B3 = [f'{A50.AO}/iso50_3ms/B', '/home/yousen/iso50_staging_3ms/B']
OUT = f'{A50.AO}/iso50_3ms_report'


def find_b3(tag, ev):
    for base in DIRS_B3:
        p = f'{base}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            return p
    return None


def collect_arm(suffix, find_dec):
    pool = {'effq': {}, 'dec': {}}
    for tag in A50.TAGS:
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
        print(f'  {tag} done', flush=True)
    return pool


t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

print('collect 1 ms arm B (calibration)', flush=True)
p1 = collect_arm('', lambda tag, ev: A50.find_solved('B', tag, ev))
print('collect 3 ms arm B (test)', flush=True)
p3 = collect_arm('_3ms', find_b3)

ratio = np.array([mpvfn(np.concatenate(p1['dec'][t]))
                  / mpvfn(np.concatenate(p1['effq'][t]))
                  for t in A50.TAGS])
print('arm-B calibration curve:', np.round(ratio, 4).tolist())


def boot(joint, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    taus = []
    d3 = [p3['dec'][t] for t in A50.TAGS]
    for _ in range(nboot):
        if joint:
            div = []
            for tag in A50.TAGS:
                cd, ce = p1['dec'][tag], p1['effq'][tag]
                pick = rng.integers(0, 50, 50)
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


mraw = [mpvfn(np.concatenate(p3['dec'][t])) for t in A50.TAGS]
sl, _ = np.polyfit(t_us, np.log(mraw), 1)
print(f'arm-B 3ms raw tau (point fit): {-1.0/sl/1000.0:.2f}')
res = {}
for name, joint in [('fixed', False), ('joint', True)]:
    tau, err = boot(joint)
    res[name] = {'tau': tau, 'err': err}
    print('armB %-6s  tau = %6.2f +- %5.2f' % (name, tau, err), flush=True)
json.dump({'ratio_B': ratio.tolist(), 'tau': res},
          open(f'{OUT}/iso3ms_B_calib.json', 'w'), indent=1)
print(f'-> {OUT}/iso3ms_B_calib.json', flush=True)
