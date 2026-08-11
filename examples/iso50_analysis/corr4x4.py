"""2D correlation, per depth: smeared truth ({SM:g} px) vs decC, aggregated to
4x4 pixel groups on the absolute pixel grid, time-summed, pooled over the
50 events. Annotates Pearson r, OLS slope, and sum ratio."""
import numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, '.')
import dqdx_lib as L
import iso50_analyse as A50
from smear_scan import pixmap, smear

G = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # group size in pixels
SM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # truth smear sigma [px]
SYM = len(sys.argv) > 3 and sys.argv[3] == 'sym'   # fig-11: smear BOTH sides
QFLOOR = 1.0   # ke: keep groups where either side exceeds this


def groups(pa, pb, q):
    key = (np.asarray(pa, np.int64) // G) * 100000 + (np.asarray(pb, np.int64) // G)
    uk, inv = np.unique(key, return_inverse=True)
    qs = np.zeros(uk.size)
    np.add.at(qs, inv, np.asarray(q, float))
    return dict(zip(uk.tolist(), qs.tolist()))


fig, axes = plt.subplots(2, 5, figsize=(20, 8.2))
for ax, tag in zip(axes.ravel(), A50.TAGS):
    d = float(tag.split('_d')[1].replace('p', '.'))
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    xs, ys = [], []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        pa, pb, q = pixmap(el[:, 0], el[:, 1], eq)
        gt = groups(*smear(pa, pb, q, SM))
        fp = A50.find_solved('C', tag, ev)
        if fp is None:
            continue
        dpix = A50.deconv_pix(fp)
        gd = groups(*(smear(*dpix, SM) if SYM else dpix))
        for k in set(gt) | set(gd):
            a, b = gt.get(k, 0.0), gd.get(k, 0.0)
            if max(a, b) > QFLOOR:
                xs.append(a); ys.append(b)
    xs = np.array(xs); ys = np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]
    slope, icpt = np.polyfit(xs, ys, 1)
    hi = np.percentile(np.concatenate([xs, ys]), 99.5)
    ax.hist2d(xs, ys, bins=60, range=[[0, hi], [0, hi]], cmap='viridis',
              norm=LogNorm())
    ax.plot([0, hi], [0, hi], 'r--', lw=1)
    ax.set_title(f'd = {d:g} cm  (t = {L.drift_time_us(d):.0f} µs)',
                 fontsize=11)
    ax.text(0.03, 0.97,
            f'r = {r:.4f}\nslope = {slope:.3f}\n'
            f'Σdec/Σsm = {ys.sum()/xs.sum():.3f}\nN = {len(xs)}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(fc='w', alpha=0.75, ec='none'))
    ax.set_xlabel(f'smeared truth ({SM:g} px), {G}x{G} group [ke]')
    if ax is axes.ravel()[0] or ax is axes.ravel()[5]:
        ax.set_ylabel(f'decC{" (same smear)" if SYM else ""}, {G}x{G} group [ke]')
fig.suptitle(f'{G}x{G}-pixel-group charge: decC vs smeared truth'
             + (f', SYMMETRIC {SM:g} px smear on both sides' if SYM else '')
             + ', 50 events pooled per depth', y=1.0)
fig.tight_layout()
out = (f'./'
       f'corr{G}x{G}' + (f'_sm{SM:g}' if SM != 1.0 else '')
       + ('_sym' if SYM else '') + '.png')
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'wrote corr{G}x{G}.png')
