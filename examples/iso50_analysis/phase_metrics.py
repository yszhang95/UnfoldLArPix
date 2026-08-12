"""Full metric set for the phase-operator arms (ghost / killed truth).

The phase solves store only deconv_q_sharp on the fit grid, so the
standard evaluator (which needs the charges table and fitted offsets)
does not apply. Here truth and reco are compared on the SAME fit grid
with the same symmetric smearing used everywhere else in section 7
(0.318 px transverse; time integrated per pixel is not used -- the 3D
grid is kept so ghost adjacency is 3D as in metrics_from_blocks).

Arms: stock C, phase alpha x {1.0, 0.5, 0.25}. 20 events x 3 depths.
"""
import numpy as np, sys, os, json
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/'
                   'UnfoldLArPix/src')
import iso50_analyse as A50
from unfoldlarpix.eval.universal import metrics_from_blocks

SIGP = 1.0 / (2.0 * np.pi * 0.5)      # 0.318 px, the note's convention
SIGT = 1.0 / (2.0 * np.pi * 0.005) / 30.0   # 1.6 us in units of 1.5-us bins
CUT = 0.5
TAGS3 = ['pgun_mu_3gev_iso50_d04p5', 'pgun_mu_3gev_iso50_d16p5',
         'pgun_mu_3gev_iso50_d28p5']
ARMS = {'stdC': None,
        'ph1.00': f'{A50.AO}/iso50_phase/C',
        'ph0.50': f'{A50.AO}/iso50_phase_a50/C',
        'ph0.25': f'{A50.AO}/iso50_phase_a25/C'}
NEV = 20


def truth_grid(f, ev, boff, shape, B):
    el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
    eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
    nx, ny, nt = shape
    ix = el[:, 0].astype(int) - int(boff[0])
    iy = el[:, 1].astype(int) - int(boff[1])
    it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (it >= 0) & (it < nt))
    g = np.zeros(shape)
    np.add.at(g, (ix[ok], iy[ok], it[ok]), eq[ok])
    return g


acc = {k: [] for k in ARMS}
for tag in TAGS3:
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    for ev in range(NEV):
        for k, base in ARMS.items():
            p = (A50.find_solved('C', tag, ev) if base is None
                 else f'{base}/{tag}/{tag}_event_0_{ev}.npz')
            z = np.load(p, allow_pickle=True)
            q = np.asarray(z['deconv_q_sharp'], float)
            boff = np.asarray(z['boffset'], float)
            B = 30.0
            t = truth_grid(f, ev, boff, q.shape, B)
            ts = gaussian_filter(t, (SIGP, SIGP, SIGT), mode='constant')
            qs = gaussian_filter(q, (SIGP, SIGP, SIGT), mode='constant')
            m = metrics_from_blocks(ts, qs, corr_threshold=CUT)
            acc[k].append((tag, m))
    print(f'{tag} scored', flush=True)

print('\n%-8s %8s %8s %8s %10s %10s %9s' %
      ('arm', 'r', 'slope', 'integ%', 'ghost_adj', 'ghost_iso', 'killed'))
out = {}
for k in ARMS:
    ms = [m for _, m in acc[k]]
    row = {n: float(np.mean([m[n] for m in ms])) for n in
           ('pearson_r', 'slope', 'integral_pct', 'ghost_adj_frac',
            'ghost_iso_frac', 'ghost_iso_charge', 'true_killed')}
    out[k] = row
    print('%-8s %8.4f %8.3f %8.2f %9.3f%% %9.3f%% %9.1f'
          % (k, row['pearson_r'], row['slope'], row['integral_pct'],
             100*row['ghost_adj_frac'], 100*row['ghost_iso_frac'],
             row['true_killed']), flush=True)
print('\niso-ghost charge [ke]: ' + '  '.join(
    f"{k} {out[k]['ghost_iso_charge']:.2f}" for k in ARMS))
json.dump(out, open(f'{A50.AO}/iso50_3ms_report/phase_metrics.json', 'w'),
          indent=1)
print('-> phase_metrics.json', flush=True)
