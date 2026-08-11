"""NO histograms, NO fits, NO MPV -- raw charge sums only.
Per depth (ensemble mean over 50 events): whole-event totals and
trunk-tube totals (same tube/z-window as the dQ/dx), truth vs deconv_C,
plus the outside-trunk remainder.  Reconciles 'trunk over-credit' with
'whole-event deficit'."""
import numpy as np, sys
sys.path.insert(0, '.')
import iso50_analyse as A50
from track_dqdx import px_to_cm, fit_direction, TUBE_CM
import dqdx_lib as L

def trunk_sum(pa, pb, q, c, d):
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1)
    rel = yz - c; proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    keep = (perp < TUBE_CM) & (z >= L.Z_LO) & (z <= L.Z_HI)
    return float(np.asarray(q, float)[keep].sum())

for dd in ['d04p5', 'd28p5']:
    tag = f'pgun_mu_3gev_iso50_{dd}'
    f = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    T = {'tot_e': 0, 'tot_d': 0, 'trk_e': 0, 'trk_d': 0}
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        y, z = px_to_cm(el[:,0].astype(float), el[:,1].astype(float))
        c, d = fit_direction(np.stack([y, z], axis=1), eq)
        pa, pb, qd = A50.deconv_pix(A50.find_solved('C', tag, ev))
        T['tot_e'] += eq.sum(); T['tot_d'] += qd.sum()
        T['trk_e'] += trunk_sum(el[:,0], el[:,1], eq, c, d)
        T['trk_d'] += trunk_sum(pa, pb, qd, c, d)
    oe = T['tot_e'] - T['trk_e']; od = T['tot_d'] - T['trk_d']
    print(f"\n=== {dd} (50-event sums, ke) ===")
    print(f"  whole event : truth {T['tot_e']:9.0f}  deconv {T['tot_d']:9.0f}"
          f"   ratio {T['tot_d']/T['tot_e']:.4f}")
    print(f"  trunk tube  : truth {T['trk_e']:9.0f}  deconv {T['trk_d']:9.0f}"
          f"   ratio {T['trk_d']/T['trk_e']:.4f}")
    print(f"  outside     : truth {oe:9.0f}  deconv {od:9.0f}"
          f"   ratio {od/oe:.4f}", flush=True)
