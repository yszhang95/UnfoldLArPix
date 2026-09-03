"""Segment-level affine closure: is the dQ/dx MPV slide amplitude, not position?

Pairs each 3 cm dQ/dx segment of the reconstruction with the SAME physical
segment of the truth (truth PCA axis + truth bin edges, so the segmentation
is identical and cannot contribute), fits Q_dec = a*Q_true + c per depth,
and asks whether a*MPV_true + c reproduces the measured MPV_dec.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

rows = []
for tag in PS.TAGS:
    d_cm = float(tag.split('_d')[1].replace('p', '.'))
    f = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    pair = {k: [[], []] for k in ['hits', 'decC', 'decB']}
    for ev in range(50):
        ta, tb, tq = PS.truth_pix(f, ev)
        ax = PS.truth_axis_edges(ta, tb, tq)
        if ax is None:
            continue
        c, dvec, edges, lo, hi = ax
        ht, _ = PS.seg_common(ta, tb, tq, c, dvec, edges)
        if not ht.size:
            continue
        ht = ht[lo:hi] / T.BIN_CM
        est = {'hits': PS.hits_pix(f, ev)}
        for arm in ['C', 'B']:
            p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
            if os.path.exists(p):
                est[f'dec{arm}'] = PS.deconv_pix(p)
        for k, (pa, pb, q) in est.items():
            hr, _ = PS.seg_common(pa, pb, q, c, dvec, edges)
            if not hr.size:
                continue
            hr = hr[lo:hi] / T.BIN_CM
            m = ht > 0
            pair[k][0].append(ht[m]); pair[k][1].append(hr[m])
    out = {'depth_cm': d_cm, 't_us': L.drift_time_us(d_cm)}
    for k in pair:
        if not pair[k][0]:
            continue
        x = np.concatenate(pair[k][0]); y = np.concatenate(pair[k][1])
        a, cc = np.polyfit(x, y, 1)
        mt = L.mpv_of(x)[0]; mr = L.mpv_of(y)[0]
        out[k] = {'a': float(a), 'c': float(cc), 'n': int(x.size),
                  'mpv_t': mt, 'mpv_r': mr, 'mpv_pred': float(a * mt + cc),
                  'sum_ratio': float(y.sum() / x.sum())}
    rows.append(out)
    print(tag, ' '.join(
        f"{k}: a {out[k]['a']:.4f} c {out[k]['c']:+7.2f} "
        f"MPV meas {out[k]['mpv_r']:6.2f} pred {out[k]['mpv_pred']:6.2f} "
        f"({100*(out[k]['mpv_pred']/out[k]['mpv_r']-1):+5.2f}%)"
        for k in ['decC'] if k in out), flush=True)

print()
print('%-6s %8s %8s %8s %8s %8s %8s' % ('depth', 'a', 'c', 'MPVr/MPVt',
                                        'pred/MPVt', 'sum ratio', 'c-term'))
for k in ['decC', 'decB', 'hits']:
    print(f'--- {k}')
    for r in rows:
        if k not in r:
            continue
        o = r[k]
        print('%-6.1f %8.4f %+8.2f %8.4f %8.4f %8.4f %8.4f'
              % (r['depth_cm'], o['a'], o['c'], o['mpv_r'] / o['mpv_t'],
                 o['mpv_pred'] / o['mpv_t'], o['sum_ratio'],
                 o['c'] / o['mpv_t']))
json.dump(rows, open(f'{PS.OUT}/segment_affine.json', 'w'), indent=1)
print('->', f'{PS.OUT}/segment_affine.json')
