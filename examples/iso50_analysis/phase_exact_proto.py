"""Prototype: phase-exact edge kernels vs box/linear window weights.

Everything lives in temp; the repository is untouched.

Model under test (the solver's): charge uniform within each 1.5-us bin.
Model fine current on pixel p:
    I_model(p, t) = sum_{p', b} q_b(p') * S[p-p', t - bin_start(b)]
with S = response box-averaged over B fine ticks (the exact current of a
uniform-density bin). Rows are then computed three ways:
    phase-exact : exact integral of I_model over the window  (proposed)
    box         : I_model binned to B, overlap-fraction edges (current)
    linear      : same bins, piecewise-linear density edges
and each is compared to the TRUE window integral from the waveform.
(phase-exact - truth) is the irreducible floor of the q-binning model;
(box - phase-exact) is the sampling error the scheme change removes.
"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/src')
from unfoldlarpix.data_loader import DataLoader
from unfoldlarpix.field_response import FieldResponseProcessor

NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield/'
RESP = ('/srv/storage1/yousen/tred_workspace/'
        'response_44_v2a_full_25x25pixel_tred.npz')
NEI = 6            # response neighbourhood used (|dx|,|dy| <= NEI)

print('loading response...', flush=True)
R = FieldResponseProcessor(RESP).process_response()   # (25,25,T) fine
print('R shape', R.shape, flush=True)
cR = R.shape[0] // 2
Tr = R.shape[2]


def box_int(C, B, a, b, t0):
    lo, hi = (a - t0) / B, (b - t0) / B
    i0, i1 = int(np.floor(lo)), int(np.floor(hi))
    n = len(C)
    def frac(i, f0, f1):
        return C[i] * (f1 - f0) if 0 <= i < n else 0.0
    if i0 == i1:
        return frac(i0, lo - i0, hi - i1)
    s = frac(i0, lo - i0, 1.0) + frac(i1, 0.0, hi - i1)
    s += C[max(i0+1, 0):max(min(i1, n), 0)].sum()
    return s


def lin_int(C, B, a, b, t0):
    n = len(C)
    dens = C / B
    centers = t0 + (np.arange(n) + 0.5) * B
    def cum(t):
        t = np.clip(t, t0, t0 + n * B)
        if t <= centers[0]:
            return dens[0] * (t - t0)
        if t >= centers[-1]:
            return C.sum() - dens[-1] * (t0 + n * B - t)
        j = int(np.searchsorted(centers, t)) - 1
        f = (t - centers[j]) / B
        return (C[:j].sum() + 0.5 * C[j]
                + B * (dens[j] * f + 0.5 * (dens[j+1] - dens[j]) * f * f))
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
    cidx = {(int(a), int(b)): i for i, (a, b) in enumerate(cl[:, :2])}
    Nt = cur.shape[1]

    el = np.asarray(z['effq_tpc0_batch0_location'])
    eq = np.asarray(z['effq_tpc0_batch0'], float)[:, 3]
    # coarse-bin the truth per pixel, bins anchored at absolute tick 0
    binidx = np.floor(el[:, 2] / B).astype(int)
    keys = {}
    for (px, py), bi, q in zip(el[:, :2].astype(int), binidx, eq):
        keys[(px, py, bi)] = keys.get((px, py, bi), 0.0) + q

    # frame: model array covers [T0, Nt) fine ticks
    bmin = min(k[2] for k in keys)
    T0 = min(0, bmin * B)
    T = Nt - T0
    pxs = np.array([k[0] for k in keys]); pys = np.array([k[1] for k in keys])
    x0, x1 = pxs.min() - NEI, pxs.max() + NEI
    y0, y1 = pys.min() - NEI, pys.max() + NEI
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    I = np.zeros((nx, ny, T))
    # S = response box-averaged over B (current of a uniform bin), causal
    ker = np.ones(B) / B
    S = np.apply_along_axis(lambda v: np.convolve(v, ker), 2,
                            R[cR-NEI:cR+NEI+1, cR-NEI:cR+NEI+1, :])
    Ls = S.shape[2]
    print(f'a{ang}: {len(keys)} (pixel,bin) charges, grid {nx}x{ny}x{T}',
          flush=True)
    for (px, py, bi), q in keys.items():
        ts = bi * B - T0
        ix, iy = px - x0, py - y0
        lo = max(0, -ts); hi = min(Ls, T - ts)
        if hi <= lo:
            continue
        I[ix-NEI:ix+NEI+1, iy-NEI:iy+NEI+1, ts+lo:ts+hi] += \
            q * S[:, :, lo:hi]

    # windows on fired pixels
    hloc = np.asarray(ev.hits.location)
    seqs = {}
    for i in range(len(hloc)):
        k = (int(hloc[i, 0]), int(hloc[i, 1]))
        seqs.setdefault(k, []).append((float(hloc[i, 2]),
                                       float(hloc[i, 3])))
    res = {'phase': [], 'box': [], 'lin': []}
    exact_all = []
    for k, v in seqs.items():
        if k not in cidx:
            continue
        ix, iy = k[0] - x0, k[1] - y0
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        Im = I[ix, iy]
        csm = np.concatenate([[0.0], np.cumsum(Im)])
        ctrue = cur[cidx[k]]
        cst = np.concatenate([[0.0], np.cumsum(ctrue)])
        nb = (T // B)
        C = Im[:nb*B].reshape(nb, B).sum(1)
        prev = 0.0
        for (tr, lat) in sorted(v):
            a, b = prev, min(lat, Nt)
            prev = lat + CSA
            if b <= a:
                continue
            truth = float(cst[int(b)] - cst[int(a)])
            ph = float(csm[int(b) - T0] - csm[int(a) - T0])
            res['phase'].append(ph - truth)
            res['box'].append(box_int(C, B, a, b, T0) - truth)
            res['lin'].append(lin_int(C, B, a, b, T0) - truth)
            exact_all.append(truth)
    e = np.array(exact_all)
    print(f'--- mu_a{ang}_nb1: {len(e)} windows, total true {e.sum():.1f} ke')
    for s in ('phase', 'box', 'lin'):
        r = np.array(res[s])
        print('  %-6s rms/row %6.3f ke   net bias %+8.2f ke (%+.3f%%)'
              % (s, r.std(), r.sum(), 100*r.sum()/e.sum()), flush=True)
