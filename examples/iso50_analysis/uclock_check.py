"""Clock-phase (u) control check on the iso50 lifetime fit.

Question (technote app:future, raised by sec:isores:phase): the readout
bins time in fixed 1.5-us bins anchored at t=0.  An isochronous track
deposits all charge at one drift time, so each depth d has a CLOCK PHASE

    u_d = charge-weighted mean of (arrival fine tick mod 30),

i.e. where inside its 1.5-us bin the charge arrives, in 50-ns ticks,
u in [0, 30).  u advances by one full bin per 2.4 mm of depth, so the
3-cm ladder samples it pseudo-randomly, and readout/unfolding
quantities are known to depend on it (noiseless ladder: operator row
bias +0.7..+5.8% tracks u, not depth).  If the per-depth dQ/dx MPVs
carry a u-dependent term, the fitted decay rate lambda of tab:iso-tau
is contaminated.

This script measures, per depth:
  u_d      from the fine-tick effq of the 3-ms rerun files
           (pgun_mu_3gev_iso50_d*_tred_nb1_3ms.npz; the 1-ms files
           store effq pre-summed into 30-tick chunks, so the phase is
           not readable there.  u is geometric -- depth over drift
           velocity plus t0, and the event t0 spacing of 1.2 ms is an
           exact multiple of the 1.5-us bin -- so it is identical
           between the 1-ms and 3-ms simulations; the within-depth
           spread across the 50 events is printed as a check).
  e_d      = min(u_d, 30 - u_d): distance from the arrival to the
           nearest bin boundary, in ticks.
  MPV_d    per-depth pooled dQ/dx Landau-Moyal MPV, 3-cm segments,
           method of record (reuses iso50_analyse machinery verbatim),
           for effq (truth, the null control), deconv B and deconv C,
           on the 1-ms sample the note quotes.

Then:
  1. lambda from the unweighted straight-line fit ln MPV_d vs drift
     time t_d (as iso50_analyse), residuals rho_d = ln MPV_d - fit.
  2. Pearson correlation of rho_d with u_d and with e_d.
  3. Refit with the edge distance as a second regressor,
     ln MPV_d = a - lambda t_d + c e_d, and report the change in
     lambda.  Errors on both fits: bootstrap over the 50 events per
     depth (the tab:iso-tau convention).
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
from track_dqdx import segment_dqdx
from iso50_analyse import find_solved, deconv_pix, TAGS, NFS, AO

B = 30

def u_of_depth(tag):
    """Per-event clock phase from the 3-ms fine-tick effq."""
    f = np.load(f'{NFS}/{tag}_tred_nb1_3ms.npz', allow_pickle=True)
    us = []
    for ev in range(50):
        el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
        us.append(float((el[:, 2] * eq).sum() / eq.sum()) % B)
    us = np.array(us)
    # circular mean/SD (phase wraps at 30)
    ang = us / B * 2 * np.pi
    mean = (np.angle(np.exp(1j * ang).mean()) / (2 * np.pi) * B) % B
    R = np.abs(np.exp(1j * ang).mean())
    sd = np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0.0)) / (2 * np.pi) * B
    return mean, sd

def fits(t_us, e, per_event, nboot=200, seed=1):
    """lambda without and with the edge-distance regressor, bootstrapped."""
    rng = np.random.default_rng(seed)
    lam0, lam1, cs = [], [], []
    for _ in range(nboot):
        y = []
        for evs in per_event:
            pick = rng.integers(0, len(evs), len(evs))
            allv = np.concatenate([evs[i] for i in pick if len(evs[i])])
            y.append(np.log(L.mpv_of(allv)[0]))
        y = np.array(y)
        A0 = np.vstack([t_us, np.ones_like(t_us)]).T
        s0, *_ = np.linalg.lstsq(A0, y, rcond=None)
        lam0.append(-s0[0] * 1000.0)          # ms^-1
        A1 = np.vstack([t_us, np.ones_like(t_us), e]).T
        s1, *_ = np.linalg.lstsq(A1, y, rcond=None)
        lam1.append(-s1[0] * 1000.0)
        cs.append(s1[2])
    return (float(np.median(lam0)), float(np.std(lam0)),
            float(np.median(lam1)), float(np.std(lam1)),
            float(np.median(cs)), float(np.std(cs)))

if __name__ == '__main__':
    t_us, u, usd = [], [], []
    per = {'effq': [], 'decB': [], 'decC': []}
    for tag in TAGS:
        d = float(tag.split('_d')[1].replace('p', '.'))
        t_us.append(L.drift_time_us(d))
        um, us_ = u_of_depth(tag)
        u.append(um); usd.append(us_)
        f = np.load(f'{NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        ev_effq, ev_B, ev_C = [], [], []
        for ev in range(50):
            el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
            eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
            ev_effq.append(segment_dqdx(el[:, 0], el[:, 1], eq))
            for arm, acc in (('B', ev_B), ('C', ev_C)):
                p = find_solved(arm, tag, ev)
                acc.append(segment_dqdx(*deconv_pix(p)) if p else np.array([]))
        per['effq'].append(ev_effq)
        per['decB'].append(ev_B)
        per['decC'].append(ev_C)
        print(f'{tag}: u = {um:5.2f} +- {us_:4.2f} ticks', flush=True)
    t_us = np.array(t_us); u = np.array(u)
    e = np.minimum(u, B - u)
    out = {'u': u.tolist(), 'u_sd': list(map(float, usd)),
           'edge': e.tolist(), 't_us': t_us.tolist()}
    print(f'\nedge distance e_d [ticks]: ' +
          ' '.join(f'{x:.1f}' for x in e))
    for k in per:
        # central values and residuals
        y = np.array([np.log(L.mpv_of(np.concatenate(
            [v for v in evs if len(v)]))[0]) for evs in per[k]])
        A0 = np.vstack([t_us, np.ones_like(t_us)]).T
        s0, *_ = np.linalg.lstsq(A0, y, rcond=None)
        rho = y - A0 @ s0
        cu = float(np.corrcoef(rho, u)[0, 1])
        ce = float(np.corrcoef(rho, e)[0, 1])
        l0, e0, l1, e1, c, cerr = fits(t_us, e, per[k])
        out[k] = dict(lambda0=l0, lambda0_err=e0, lambda_u=l1,
                      lambda_u_err=e1, c_edge=c, c_edge_err=cerr,
                      corr_rho_u=cu, corr_rho_edge=ce,
                      rho=rho.tolist())
        print(f'{k:5s} lambda {l0:.3f}+-{e0:.3f} -> u-controlled '
              f'{l1:.3f}+-{e1:.3f}  c_edge {c:+.4f}+-{cerr:.4f} '
              f'corr(rho,u) {cu:+.2f}  corr(rho,e) {ce:+.2f}',
              flush=True)
    json.dump(out, open(f'{AO}/iso50_uclock.json', 'w'), indent=1)
    print(f'-> {AO}/iso50_uclock.json')
