"""Depth bias with and without the refit: arm B vs arm C, 1 ms sample.

Per depth: per-event integral capture (mean +- SEM) and the 3-cm MPV
capture, for both arms; log-slope of each curve over the drift.
"""
import numpy as np, sys, os
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
for arm in ('B', 'C'):
    for key in ('sum', 'mpv'):
        y = np.log([r[f'{key}{arm}'] for r in rows])
        b, a = np.polyfit(t_us, y, 1)
        print('%s %s capture: intercept %+.3f%%, slide %+.2f%% / 170 us'
              % (arm, key, 100*(np.exp(a)-1), 100*b*170), flush=True)
