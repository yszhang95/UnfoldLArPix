"""FINAL isochronous-track lifetime analysis on the 50-copy ensemble.

Method of record (user-specified):
  track PCA fit per event -> project ALL charge onto the axis -> 3-cm
  histogram segments -> drop first/last non-empty bins -> pool the ~950
  segments per depth -> Landau(Moyal) MPV, with truncated mean and median
  as shape cross-checks -> ln(stat) vs drift time, over 10 independent
  depths -> tau.  Errors: bootstrap over EVENTS (50 per depth), propagated
  through the weighted fit.
"""
import numpy as np, sys, os, json
sys.path.insert(0, '.')
import dqdx_lib as L
from track_dqdx import segment_dqdx

NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield'
AO = '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output'
TAGS = [l.strip() for l in open('/home/yousen/Documents/NDLAr2x2/MuonLArSim/iso50_list.txt') if l.strip()]
DIRS = {'B': [f'{AO}/iso50/B', '/home/yousen/iso50_staging/B'],
        'C': [f'{AO}/iso50/C', '/home/yousen/iso50_staging/C']}

def tmean(v, keep=0.7):
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    return float(v[:max(int(len(v)*keep), 3)].mean())

def find_solved(arm, tag, ev):
    for base in DIRS[arm]:
        p = f'{base}/{tag}/{tag}_event_0_{ev}.npz'
        if os.path.exists(p):
            return p
    return None

def deconv_pix(path, key='deconv_q_sharp'):
    z = np.load(path, allow_pickle=True)
    per = np.asarray(z[key], float).sum(axis=2)
    off = np.asarray(z['boffset'], float)
    ax_, bx_ = np.nonzero(per > 0)
    return ax_+int(off[0]), bx_+int(off[1]), per[ax_, bx_]

def stats_of(pool_by_event):
    """pool_by_event: list (per event) of segment arrays."""
    allv = np.concatenate([v for v in pool_by_event if len(v)])
    return {'mpv': L.mpv_of(allv)[0], 'tmean': tmean(allv),
            'median': float(np.median(allv)), 'n': int(allv.size)}

def boot_tau(t_us, per_event, statfn, nboot=200, seed=1):
    """Bootstrap over events at each depth."""
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(nboot):
        m = []
        for evs in per_event:
            pick = rng.integers(0, len(evs), len(evs))
            allv = np.concatenate([evs[i] for i in pick if len(evs[i])])
            m.append(statfn(allv))
        y = np.log(m)
        A = np.vstack([t_us, np.ones_like(t_us)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        taus.append(-1.0/sol[0]/1000.0)
    taus = np.array(taus)
    return float(np.median(taus)), float(taus.std())

if __name__ == '__main__':
    per = {k: {} for k in ['effq', 'hits', 'decB', 'decC']}
    t_us = []
    for tag in TAGS:
        d = float(tag.split('_d')[1].replace('p', '.'))
        t_us.append(L.drift_time_us(d))
        f = np.load(f'{NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        for k in per:
            per[k][tag] = []
        n_solved = {'B': 0, 'C': 0}
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
            hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
            per['effq'][tag].append(segment_dqdx(el[:,0], el[:,1], eq))
            per['hits'][tag].append(segment_dqdx(hl[:,0], hl[:,1], hq))
            for arm, kk in [('B','decB'), ('C','decC')]:
                p = find_solved(arm, tag, ev)
                if p is None:
                    per[kk][tag].append(np.array([]))
                    continue
                n_solved[arm] += 1
                per[kk][tag].append(segment_dqdx(*deconv_pix(p)))
        s = {k: stats_of(per[k][tag]) for k in per}
        print(f"{tag} solved B/C {n_solved['B']}/{n_solved['C']}  " +
              '  '.join(f"{k}:{s[k]['mpv']:6.1f}({s[k]['n']})" for k in per),
              flush=True)
    t_us = np.array(t_us)
    print()
    print('%-6s %-8s %10s %10s' % ('est', 'stat', 'tau [ms]', '+- (boot)'))
    out = {}
    for k in per:
        pe = [per[k][tag] for tag in TAGS]
        for stat, fn in [('mpv', lambda v: L.mpv_of(v)[0]),
                         ('tmean', tmean), ('median', np.median)]:
            tau, err = boot_tau(t_us, pe, fn)
            out[f'{k}_{stat}'] = {'tau': tau, 'err': err}
            print('%-6s %-8s %10.3f %10.3f' % (k, stat, tau, err), flush=True)
    json.dump(out, open(f'{AO}/iso50_lifetime_eval.json', 'w'), indent=1)
    print('-> iso50_lifetime_eval.json')
