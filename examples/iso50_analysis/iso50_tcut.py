"""Ensemble analysis WITH the isochronous time cut (delta rejection)."""
import numpy as np, sys, json
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from track_dqdx3d import segment_dqdx_t

def effq_pts(f, ev):
    l = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
    q = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
    return l[:,0], l[:,1], l[:,2].astype(float), q

def hits_pts(f, ev):
    l = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
    q = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
    return l[:,0], l[:,1], l[:,2].astype(float), q

def dec_pts(path):
    z = np.load(path, allow_pickle=True)
    dq = np.asarray(z['deconv_q_sharp'], float)
    off = np.asarray(z['boffset'], float)
    ax_, bx_, kt = np.nonzero(dq > 0.05)
    t = off[2] + (kt + 0.5) * 30.0
    return ax_+int(off[0]), bx_+int(off[1]), t, dq[ax_, bx_, kt]

per = {k: {} for k in ['effq','hits','decC']}
ret = {k: [] for k in per}
t_us = []
for tag in A50.TAGS:
    d = float(tag.split('_d')[1].replace('p','.'))
    t_us.append(L.drift_time_us(d))
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for k in per: per[k][tag] = []
    kept = {k: [0.0, 0.0] for k in per}
    for ev in range(50):
        for k, args in [('effq', effq_pts(f, ev)), ('hits', hits_pts(f, ev)),
                        ('decC', dec_pts(A50.find_solved('C', tag, ev)))]:
            pa, pb, t, q = args
            t0 = np.median(np.repeat(t, np.maximum((q*10).astype(int),1)))
            kept[k][0] += float(q[np.abs(t-t0) < 60.0].sum())
            kept[k][1] += float(q.sum())
            per[k][tag].append(segment_dqdx_t(pa, pb, t, q))
    for k in per:
        ret[k].append(kept[k][0]/kept[k][1])
    allv = {k: np.concatenate([v for v in per[k][tag] if len(v)]) for k in per}
    print(f'{tag}  ' + '  '.join(
        f'{k}: MPV {L.mpv_of(allv[k])[0]:5.1f} ret {ret[k][-1]:.3f}' for k in per),
        flush=True)
t_us = np.array(t_us)
print()
rng = np.random.default_rng(3)
out = {}
for k in per:
    pe = [per[k][tag] for tag in A50.TAGS]
    tau, err = A50.boot_tau(t_us, pe, lambda v: L.mpv_of(v)[0], nboot=150)
    out[k] = {'tau': tau, 'err': err, 'retention': ret[k]}
    print(f'{k:6s} tau(MPV) = {tau:6.3f} +- {err:5.3f} ms   '
          f'retention {min(ret[k]):.3f}..{max(ret[k]):.3f}', flush=True)
json.dump(out, open(f'{A50.AO}/iso50_tcut_eval.json','w'), indent=1)
print('-> iso50_tcut_eval.json')
