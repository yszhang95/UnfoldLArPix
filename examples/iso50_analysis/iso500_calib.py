"""The 3 ms closure with the 10x calibration sample (iso500).

Calibration: per-depth sym-smeared MPV ratio decC/effq from the iso500
sample (500 independent events per depth, seed family disjoint from the
test samples). Applied to the 3 ms test sample; errors quoted both with
the curve held fixed and from a joint bootstrap over BOTH samples.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear
from iso3ms_calib import collect, find_3ms, SIG

TAGS500 = [l.strip() for l in
           open('/home/yousen/Documents/NDLAr2x2/MuonLArSim/iso500_list.txt')
           if l.strip()]
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

print('collect iso500 (calibration, 500 ev/depth)', flush=True)
cal = {'effq': {}, 'decC': {}}
for tag in TAGS500:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    cal['effq'][tag] = []
    cal['decC'][tag] = []
    for ev in range(500):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        cal['effq'][tag].append(
            segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
        fp = find_500(tag, ev)
        cal['decC'][tag].append(
            segment_dqdx(*smear(*A50.deconv_pix(fp), SIG))
            if fp else np.array([]))
    print(f'  {tag} done', flush=True)

print('collect 3 ms (test sample)', flush=True)
p3 = collect('_3ms', find_3ms)

ratio = np.array([mpvfn(np.concatenate(cal['decC'][t]))
                  / mpvfn(np.concatenate(cal['effq'][t]))
                  for t in TAGS500])
print('iso500 calibration curve:', np.round(ratio, 4).tolist())


def boot(joint, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    taus = []
    d3 = [p3['decC'][t] for t in A50.TAGS]
    for _ in range(nboot):
        if joint:
            div = []
            for tag in TAGS500:
                cd, ce = cal['decC'][tag], cal['effq'][tag]
                pick = rng.integers(0, 500, 500)
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
    print('cal(iso500) %-6s  tau = %6.2f +- %5.2f  (sigma_rate %.2e /us)'
          % (name, tau, err, err / tau**2 * 1e-3), flush=True)
json.dump({'ratio_iso500': ratio.tolist(), 'tau': res},
          open(f'{OUT}/iso500_calib.json', 'w'), indent=1)
print(f'-> {OUT}/iso500_calib.json', flush=True)
