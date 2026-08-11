"""Render the 9 x 50 per-track voxel-correlation panels from the
pertrack_losses.json / pertrack_pairs.npz produced by pertrack_losses.py.

One page per depth, 50 panels (10 x 5): per-pixel (time-summed) deconv
vs truth scatter, y = x diagonal, annotated with the track's OLS slope,
through-origin slope, integral ratio, data fidelity and censor value.
"""
import numpy as np, sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import iso50_analyse as A50

TAGS = A50.TAGS[1:]
OUTD = f'{A50.AO}/iso50_3ms_report'
losses = json.load(open(f'{OUTD}/pertrack_losses.json'))
pairs = np.load(f'{OUTD}/pertrack_pairs.npz')

pdf = PdfPages(f'{OUTD}/pertrack_panels.pdf')
for tag in TAGS:
    d = float(tag.split('_d')[1].replace('p', '.'))
    fig, axes = plt.subplots(5, 10, figsize=(22, 11.5))
    fig.suptitle(f'deconv (C) vs truth, per pixel (time-summed) --- '
                 f'd = {d:g} cm (t = {L.drift_time_us(d):.0f} us), '
                 f'50 tracks', fontsize=13)
    for ev, ax in enumerate(axes.ravel()):
        x = pairs[f'{tag}_e{ev:02d}_x']
        y = pairs[f'{tag}_e{ev:02d}_y']
        rec = losses[tag][ev]
        hi = float(np.percentile(np.concatenate([x, y]), 99.7)) * 1.05
        ax.plot([0, hi], [0, hi], 'r--', lw=0.6)
        ax.plot(x, y, '.', ms=1.5, color='#30507a', alpha=0.6)
        tt = np.linspace(0, hi, 5)
        ax.plot(tt, rec['slope']*tt + rec['intercept'], '-',
                color='#d1701a', lw=0.8)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.97,
                f"e{ev:02d}  s={rec['slope']:.3f}\n"
                f"s0={rec['slope0']:.3f} r={rec['ratio']:.3f}\n"
                f"D={rec['data_fid']:.0f} C={rec['censor_raw']:.2f}",
                transform=ax.transAxes, va='top', fontsize=4.6,
                family='monospace')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig); plt.close(fig)
    print(f'{tag} page done', flush=True)
pdf.close()
print(f'-> {OUTD}/pertrack_panels.pdf', flush=True)
