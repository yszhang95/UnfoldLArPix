"""The 3 ms calibration cross-check.

Sample: same seeded primaries as the 1 ms study, lifetime 3.0 ms, arm C.
Estimators (per the study design): fig-11-symmetric smeared truth
(effq * 0.318 px), symmetric smeared deconv (decC * 0.318 px), raw hits.

Test: derive the capture calibration at 1 ms (per-depth MPV ratio
reco/truth under the identical treatment), divide it out of the 3 ms
dQ/dx, and check whether tau returns to 3.0. The deconv slide was shown
amplitude-independent (20 ms null), so its 1 ms curve should transfer;
the hits bias is amplitude-driven, so its curve should NOT transfer.

Outputs: analysis_output/iso50_3ms_report/{calib_test.pdf,.png,
iso3ms_calib.json}
"""
import numpy as np, sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx
from smear_scan import pixmap, smear

SIG = 1.0 / (2.0 * np.pi * 0.5)          # fig-11 pixel kernel, 0.318 px
DIRS_3MS = [f'{A50.AO}/iso50_3ms/C', '/home/yousen/iso50_staging_3ms/C']
OUT = f'{A50.AO}/iso50_3ms_report'
os.makedirs(OUT, exist_ok=True)
COL = {'effq': '#2e8b57', 'hits': '#a6304a', 'decC': '#d1701a'}
LBL = {'effq': 'sym-smeared truth', 'hits': 'raw hits',
       'decC': 'sym-smeared deconv (C)'}


def find_3ms(tag, ev):
    for base in DIRS_3MS:
        p = f'{base}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            return p
    return None


def collect(suffix, find_dec):
    """Per-depth, per-event 3-cm segment arrays for the three estimators."""
    pool = {k: {} for k in COL}
    for tag in A50.TAGS:
        f = np.load(f'{A50.NFS}/{tag}_tred_nb1{suffix}.npz',
                    allow_pickle=True)
        for k in COL:
            pool[k][tag] = []
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            pool['effq'][tag].append(
                segment_dqdx(*smear(*pixmap(el[:, 0], el[:, 1], eq), SIG)))
            hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
            hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
            pool['hits'][tag].append(segment_dqdx(hl[:, 0], hl[:, 1], hq))
            fp = find_dec(tag, ev)
            pool['decC'][tag].append(
                segment_dqdx(*smear(*A50.deconv_pix(fp), SIG))
                if fp else np.array([]))
        print(f'  {tag} done', flush=True)
    return pool


t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

print('collect 1 ms (calibration source)', flush=True)
p1 = collect('', lambda tag, ev: A50.find_solved('C', tag, ev))
print('collect 3 ms (test sample)', flush=True)
p3 = collect('_3ms', find_3ms)

# ---- calibration curves from the 1 ms sample --------------------------
mpv1 = {k: np.array([mpvfn(np.concatenate(p1[k][tag])) for tag in A50.TAGS])
        for k in COL}
CAL = {k: mpv1[k] / mpv1['effq'] for k in ['decC', 'hits']}

# ---- raw and calibrated tau on the 3 ms sample -------------------------
res = {}
for k in COL:
    res[k] = dict(zip(('tau_raw', 'err_raw'),
                      A50.boot_tau(t_us, [p3[k][t] for t in A50.TAGS], mpvfn)))
for k in ['decC', 'hits']:
    calseg = [[s / CAL[k][i] for s in p3[k][tag]]
              for i, tag in enumerate(A50.TAGS)]
    res[k].update(zip(('tau_cal', 'err_cal'),
                      A50.boot_tau(t_us, calseg, mpvfn)))

mpv3 = {k: np.array([mpvfn(np.concatenate(p3[k][tag])) for tag in A50.TAGS])
        for k in COL}
json.dump({'tau': res,
           'cal_curve': {k: CAL[k].tolist() for k in CAL},
           'mpv_1ms': {k: v.tolist() for k, v in mpv1.items()},
           'mpv_3ms': {k: v.tolist() for k, v in mpv3.items()},
           't_us': t_us.tolist()},
          open(f'{OUT}/iso3ms_calib.json', 'w'), indent=1)

print('\n== tau on the 3 ms sample (truth: 3.0) ==')
for k in COL:
    line = f"{LBL[k]:26s} raw {res[k]['tau_raw']:6.2f} +- {res[k]['err_raw']:.2f}"
    if 'tau_cal' in res[k]:
        line += f"   calibrated(1ms curve) {res[k]['tau_cal']:6.2f} +- {res[k]['err_cal']:.2f}"
    print(line, flush=True)

# ---- figure ------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
tt = np.linspace(0, 185, 50)
for k in COL:
    b, a = np.polyfit(t_us, np.log(mpv3[k]), 1)
    a1.plot(t_us, mpv3[k], 'o', color=COL[k], ms=5)
    a1.plot(tt, np.exp(a + b*tt), '-', color=COL[k], lw=1.2,
            label=rf"{LBL[k]}: $\tau = {res[k]['tau_raw']:.2f} \pm {res[k]['err_raw']:.2f}$ ms")
a1.plot(tt, np.exp(np.log(mpv3['effq'][0]) + t_us[0]/3000. - tt/3000.),
        'k--', lw=1, label=r'true $\tau = 3$ ms (slope)')
a1.set_title('uncalibrated')
for k in ['decC', 'hits']:
    m = mpv3[k] / CAL[k]
    b, a = np.polyfit(t_us, np.log(m), 1)
    a2.plot(t_us, m, 'o', color=COL[k], ms=5)
    a2.plot(tt, np.exp(a + b*tt), '-', color=COL[k], lw=1.2,
            label=rf"{LBL[k]} / 1ms curve: $\tau = {res[k]['tau_cal']:.2f} \pm {res[k]['err_cal']:.2f}$ ms")
a2.plot(t_us, mpv3['effq'], 'o', color=COL['effq'], ms=4, alpha=0.6)
b, a = np.polyfit(t_us, np.log(mpv3['effq']), 1)
a2.plot(tt, np.exp(a + b*tt), '-', color=COL['effq'], lw=1.0, alpha=0.6,
        label=rf"truth control: $\tau = {res['effq']['tau_raw']:.2f} \pm {res['effq']['err_raw']:.2f}$ ms")
a2.set_title('calibrated with the 1 ms capture curve')
for ax in (a1, a2):
    ax.set_yscale('log')
    ax.set_xlabel(r'drift time [$\mu$s]')
    ax.set_ylabel('dQ/dx MPV [ke/cm] (3-cm segments)')
    yt = [46, 50, 54, 58, 62]
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f'{OUT}/calib_test.pdf')
fig.savefig(f'{OUT}/calib_test.png', dpi=130)
print(f'-> {OUT}/calib_test.pdf .png iso3ms_calib.json', flush=True)
