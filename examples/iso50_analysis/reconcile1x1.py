"""Reconcile the 1x1 OLS slope (rises with depth) with paired_reg's
through-origin slope (falls with depth): same points, three slope
conventions side by side, vs UNSMEARED truth."""
import numpy as np, sys
sys.path.insert(0, '.')
import iso50_analyse as A50
from smear_scan import pixmap

QFLOOR = 1.0

print('%-6s %8s %9s %9s %9s %9s' %
      ('d[cm]', 'r', 'ols', 'icpt', 'origin', 'ratio'))
for tag in A50.TAGS:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    xs, ys = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        pa, pb, q = pixmap(el[:, 0], el[:, 1], eq)
        gt = dict(zip((pa.astype(np.int64)*100000+pb).tolist(), q.tolist()))
        fp = A50.find_solved('C', tag, ev)
        if fp is None:
            continue
        da, db, dq = A50.deconv_pix(fp)
        gd = dict(zip((da.astype(np.int64)*100000+db).tolist(), dq.tolist()))
        for k in set(gt) | set(gd):
            a, b = gt.get(k, 0.0), gd.get(k, 0.0)
            if max(a, b) > QFLOOR:
                xs.append(a); ys.append(b)
    x = np.array(xs); y = np.array(ys)
    ols, icpt = np.polyfit(x, y, 1)
    d = float(tag.split('_d')[1].replace('p', '.'))
    print('%-6.1f %8.4f %9.3f %9.3f %9.4f %9.4f' %
          (d, np.corrcoef(x, y)[0, 1], ols, icpt,
           (x*y).sum()/(x*x).sum(), y.sum()/x.sum()), flush=True)
