"""Window-sampling weight schemes, tested against the exact waveform.

The operator's dominant error is the first-order window/bin overlap
(current assumed flat inside a fit bin). Using the noiseless waveform
(fine-tick current per pixel) the sampling error can be isolated from
the response convolution entirely:

  exact  : integral of the fine current over the window
  box    : bin the current to B ticks, window integral = full bins +
           overlap-fraction x edge-bin value          (current scheme)
  linear : same bins, but the density is reconstructed piecewise-
           linearly between bin centres before integrating the edges

Windows: per fired pixel, (previous reset, latch] of each sequence --
the same partition wfbudget.py uses.
"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/src')
from unfoldlarpix.data_loader import DataLoader
NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield/'


def box_integral(C, B, a, b):
    """Window integral (a,b] in fine ticks from bin integrals C, box rule."""
    lo, hi = a / B, b / B
    i0, i1 = int(np.floor(lo)), int(np.floor(hi))
    n = len(C)
    def frac(i, f0, f1):
        return C[i] * (f1 - f0) if 0 <= i < n else 0.0
    if i0 == i1:
        return frac(i0, lo - i0, hi - i1)
    s = frac(i0, lo - i0, 1.0) + frac(i1, 0.0, hi - i1)
    s += C[max(i0+1, 0):max(min(i1, n), 0)].sum()
    return s


def lin_integral(C, B, a, b):
    """Same, but density piecewise-linear between bin centres."""
    n = len(C)
    dens = C / B                      # mean density per bin
    centers = (np.arange(n) + 0.5) * B
    def cum(t):                       # integral of pw-linear density, 0..t
        t = np.clip(t, 0.0, n * B)
        # segment index between centers
        if t <= centers[0]:
            return dens[0] * t
        if t >= centers[-1]:
            return C[:n].sum() - dens[-1] * (n * B - t)
        j = int(np.searchsorted(centers, t)) - 1
        t0 = centers[j]
        f = (t - t0) / B
        d0, d1 = dens[j], dens[j+1]
        # integral up to center j: full bins 0..j-1 halves...
        base = C[:j].sum() + 0.5 * C[j]
        return base + B * (d0 * f + 0.5 * (d1 - d0) * f * f)
    return cum(b) - cum(a)


for ang in ['00', '50']:
    WF = f'{NFS}pgun_mu_3gev_ang{ang}_tred_nb1_wf.npz'
    loader = DataLoader(WF)
    rc = loader.get_readout_config()
    ev = [e for e in loader.iter_events() if e.hits and e.tpc_id == 0][0]
    B = int(rc.adc_hold_delay)
    CSA = float(rc.csa_reset_time)
    z = np.load(WF, allow_pickle=True)
    cur = np.asarray(z['current_tpc0_batch0'])
    cur = cur.reshape(-1, cur.shape[-1])
    cl = np.asarray(z['current_tpc0_batch0_location'])
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]
    hloc = np.asarray(ev.hits.location)
    seqs = {}
    for i in range(len(hloc)):
        k = (int(hloc[i, 0]), int(hloc[i, 1]))
        seqs.setdefault(k, []).append((float(hloc[i, 2]),
                                       float(hloc[i, 3])))
    res = {'box': [], 'lin': []}
    exact_all = []
    for k, v in seqs.items():
        if k not in idx:
            continue
        c = cur[idx[k]]
        cs = np.concatenate([[0.0], np.cumsum(c)])
        nb = Nt // B
        C = c[:nb*B].reshape(nb, B).sum(1)
        prev = 0.0
        for (tr, lat) in sorted(v):
            a, b = prev, min(lat, Nt)
            prev = lat + CSA
            if b <= a:
                continue
            exact = float(cs[int(b)] - cs[int(a)])
            res['box'].append(box_integral(C, B, a, b) - exact)
            res['lin'].append(lin_integral(C, B, a, b) - exact)
            exact_all.append(exact)
    e = np.array(exact_all)
    print(f'mu_a{ang}_nb1: {len(e)} windows, total exact {e.sum():.1f} ke')
    for s in ('box', 'lin'):
        r = np.array(res[s])
        print('  %-6s rms/row %6.3f ke   net bias %+7.2f ke (%+.3f%%)   '
              'max|err| %6.3f'
              % (s, r.std(), r.sum(), 100*r.sum()/e.sum(), np.abs(r).max()),
              flush=True)
