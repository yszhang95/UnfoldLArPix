"""Depth bias with and without the refit: arm B vs arm C, 1 ms sample.

Per depth: per-event integral capture (mean +- SEM) and the 3-cm MPV
capture, for both arms; log-slope of each curve over the drift.
"""
import numpy as np, sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx import segment_dqdx

t_us = np.array([L.drift_time_us(float(t.split('_d')[1].replace('p', '.')))
                 for t in A50.TAGS])
mpvfn = lambda v: L.mpv_of(v)[0]

rows = []
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    sums = {'B': [], 'C': []}
    segs = {'B': [], 'C': [], 'effq': []}
    esum = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        esum.append(eq.sum())
        segs['effq'].append(segment_dqdx(el[:, 0], el[:, 1], eq))
        for arm in ('B', 'C'):
            p = A50.find_solved(arm, tag, ev)
            pa, pb, q = A50.deconv_pix(p)
            sums[arm].append(q.sum())
            segs[arm].append(segment_dqdx(pa, pb, q))
    r = {'t': L.drift_time_us(float(tag.split('_d')[1].replace('p', '.')))}
    me = mpvfn(np.concatenate(segs['effq']))
    for arm in ('B', 'C'):
        rat = np.array(sums[arm]) / np.array(esum)
        r[f'sum{arm}'] = rat.mean()
        r[f'sem{arm}'] = rat.std() / np.sqrt(len(rat))
        r[f'mpv{arm}'] = mpvfn(np.concatenate(segs[arm])) / me
    rows.append(r)
    print('%-6.0fus  B: sum %.4f+-%.4f mpv %.4f   C: sum %.4f+-%.4f mpv %.4f'
          % (r['t'], r['sumB'], r['semB'], r['mpvB'],
             r['sumC'], r['semC'], r['mpvC']), flush=True)

print()
FIT = slice(1, None)          # d = 1.5 cm excluded (record convention)
fits = {}
for arm in ('B', 'C'):
    for key in ('sum', 'mpv'):
        y = np.log([r[f'{key}{arm}'] for r in rows])
        b, a = np.polyfit(t_us[FIT], y[FIT], 1)
        fits[(arm, key)] = (a, b)
        print('%s %s capture (9pt): intercept %+.3f%%, slide %+.2f%% / 170 us'
              % (arm, key, 100*(np.exp(a)-1), 100*b*170), flush=True)

OUTD = f'{A50.AO}/iso50_report'
json.dump(rows, open(f'{OUTD}/bvc_depth.json', 'w'), indent=1)
COLB, COLC = '#4062bb', '#d1701a'
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
tt = np.linspace(0, 185, 50)
for ax, key, ttl in [(a1, 'sum', 'integral capture'),
                     (a2, 'mpv', 'MPV capture (3-cm segments)')]:
    for arm, col in [('B', COLB), ('C', COLC)]:
        yv = np.array([r[f'{key}{arm}'] for r in rows])
        if key == 'sum':
            ax.errorbar(t_us[FIT], yv[FIT],
                        yerr=[r[f'sem{arm}'] for r in rows[1:]],
                        fmt='o', color=col, ms=4, capsize=2)
        else:
            ax.plot(t_us[FIT], yv[FIT], 'o', color=col, ms=4)
        ax.plot(t_us[:1], yv[:1], 'o', mfc='none', color=col, ms=5)
        aa, bb = fits[(arm, key)]
        ax.plot(tt, np.exp(aa + bb*tt), '-', color=col, lw=1.3,
                label=f"{'B (no refit)' if arm=='B' else 'C (refit)'}: "
                      f"{100*bb*170:+.1f}%/170 µs")
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_title(ttl)
    ax.set_xlabel('drift time [µs]')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
a1.set_ylabel(r'reco / effq')
fig.suptitle('depth bias with and without the refit (1 ms, 50 events/depth)',
             y=1.0)
fig.tight_layout()
fig.savefig(f'{OUTD}/bvc_depth.pdf')
fig.savefig(f'{OUTD}/bvc_depth.png', dpi=130)
print(f'-> {OUTD}/bvc_depth.pdf .png bvc_depth.json', flush=True)
