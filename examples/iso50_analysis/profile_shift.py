"""Is the dQ/dx MPV deficit a POSITIONAL effect?

Question (user, 2026-08-29): the iso50 dQ/dx MPV sits ~10% low and slides
with depth.  Is that because the reconstructed charge profile is
systematically DISPLACED along the pixel axis (the along-track axis b = z
index, the very axis the 3 cm dQ/dx segments are cut on), and does that
displacement depend on drift depth?

The isochronous geometry makes this clean: the track runs along +z at
fixed y and fixed drift time, so the pixel plane index b IS the dQ/dx
projection axis and a is purely transverse.

Per event we measure, on a common interior window (both estimators, the
same window, so amplitude scale cancels):

  d_cen   centroid(reco) - centroid(truth) along b            [pixels]
  d_shift amplitude-free registration shift: argmin_delta over
          || P_reco(b) - alpha * P_truth(b - delta) ||^2      [pixels]
  d_wid   profile rms width ratio reco/truth along b
  grad    OLS slope of the per-row ratio P_reco/P_truth vs b   [/pixel]
  d_cen_a transverse centroid shift                            [pixels]
  cap     window capture sum(P_reco)/sum(P_truth)

then the LEVERAGE of any such shift on the statistic that is actually
wrong:

  A  rigid sub-pixel shift applied to the TRUTH pixel charges, pushed
     through the segment_dqdx pipeline -> MPV(delta)/MPV(0)
  B  reco scored on the TRUTH PCA axis and the TRUTH bin edges (common
     segmentation) vs its own -> how much MPV is segmentation phase

Outputs -> analysis_output/iso_profile_shift/profile_shift.json
"""
import numpy as np, sys, os, json
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqdx_lib as L
import track_dqdx as T

NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield'
AO = '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output'
OUT = f'{AO}/iso_profile_shift'
TAGS = [l.strip() for l in open('/home/yousen/Documents/NDLAr2x2/MuonLArSim/iso50_list.txt') if l.strip()]
if os.environ.get('TAGSEL'):
    TAGS = [TAGS[int(i)] for i in os.environ['TAGSEL'].split(',')]
DIRS = {'B': [f'{AO}/iso50/B'], 'C': [f'{AO}/iso50/C']}

TUBE = 3        # +-3 pixels transversally (dqdx_lib convention, ~1.3 cm)
TRIM = 5        # pixels trimmed off each lit end -> interior window
NEV = int(os.environ.get('NEV', 50))


# ---------------------------------------------------------------- loading
def truth_pix(f, ev):
    el = np.asarray(f[f'effq_tpc0_batch{ev}_location'])
    eq = np.asarray(f[f'effq_tpc0_batch{ev}'], float)[:, 3]
    return el[:, 0], el[:, 1], eq


def hits_pix(f, ev):
    hl = np.asarray(f[f'hits_tpc0_batch{ev}_location'])
    hq = np.asarray(f[f'hits_tpc0_batch{ev}'], float)[:, -1]
    return hl[:, 0], hl[:, 1], hq


def deconv_pix(path, key='deconv_q_sharp'):
    z = np.load(path, allow_pickle=True)
    per = np.asarray(z[key], float).sum(axis=2)
    off = np.asarray(z['boffset'], float)
    a_, b_ = np.nonzero(per > 0)
    return a_ + int(off[0]), b_ + int(off[1]), per[a_, b_]


def dense(pa, pb, q, a0, a1, b0, b1):
    A = np.zeros((a1 - a0 + 1, b1 - b0 + 1))
    m = (pa >= a0) & (pa <= a1) & (pb >= b0) & (pb <= b1)
    np.add.at(A, (pa[m] - a0, pb[m] - b0), q[m])
    return A


# ------------------------------------------------------------ registration
def reg_shift(Pt, Pr, lo=-3.0, hi=3.0):
    """Amplitude-free shift: argmin_d min_alpha ||Pr - alpha*Pt(.-d)||^2.

    Pt shifted by +d means charge moves toward larger b."""
    n = Pt.size
    x = np.arange(n, dtype=float)

    def cost(d):
        s = np.interp(x - d, x, Pt, left=0.0, right=0.0)
        den = s @ s
        if den <= 0:
            return np.inf
        al = (s @ Pr) / den
        r = Pr - al * s
        return float(r @ r)

    grid = np.linspace(lo, hi, 61)
    c = np.array([cost(d) for d in grid])
    i = int(np.argmin(c))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(cost, bounds=(a, b), method='bounded',
                        options={'xatol': 1e-5})
    return float(r.x) if r.success else float(grid[i])


def moments(P, b):
    s = P.sum()
    if s <= 0:
        return np.nan, np.nan
    c = float((b * P).sum() / s)
    w = float(np.sqrt(max((P * (b - c) ** 2).sum() / s, 0.0)))
    return c, w


# ------------------------------------------------------------- per event
def event_metrics(tag, ev, f, arms):
    ta, tb, tq = truth_pix(f, ev)
    ha, hb, hq = hits_pix(f, ev)
    est = {'hits': (ha, hb, hq)}
    for arm in arms:
        p = f'{DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
        if not os.path.exists(p):
            return None
        est[f'dec{arm}'] = deconv_pix(p)

    a0 = min(ta.min(), *[e[0].min() for e in est.values()])
    a1 = max(ta.max(), *[e[0].max() for e in est.values()])
    b0 = min(tb.min(), *[e[1].min() for e in est.values()])
    b1 = max(tb.max(), *[e[1].max() for e in est.values()])
    Qt = dense(ta, tb, tq, a0, a1, b0, b1)
    ac = int(np.argmax(Qt.sum(axis=1)))                 # trunk row (truth)
    rows = slice(max(ac - TUBE, 0), ac + TUBE + 1)

    Pt_full = Qt[rows].sum(axis=0)
    lit = np.nonzero(Pt_full > 0)[0]
    if lit.size < 2 * TRIM + 12:
        return None
    w0, w1 = lit[0] + TRIM, lit[-1] - TRIM              # interior window
    win = slice(w0, w1 + 1)
    bidx = np.arange(w0, w1 + 1, dtype=float)
    Pt = Pt_full[win]
    if Pt.sum() <= 0:
        return None
    ct, wt = moments(Pt, bidx)
    # transverse (truth), same window
    arow = np.arange(a0, a1 + 1, dtype=float)[rows]
    cat = float((arow * Qt[rows, win].sum(axis=1)).sum() / Qt[rows, win].sum())

    out = {'depth_lit': int(lit.size), 'w0': int(w0 + b0), 'w1': int(w1 + b0),
           'cen_t': ct + b0, 'wid_t': wt, 'sum_t': float(Pt.sum()),
           'cen_a_t': cat}
    for k, (pa, pb, q) in est.items():
        Qr = dense(pa, pb, q, a0, a1, b0, b1)
        Pr = Qr[rows].sum(axis=0)[win]
        if Pr.sum() <= 0:
            continue
        cr, wr = moments(Pr, bidx)
        car = float((arow * Qr[rows, win].sum(axis=1)).sum() / Qr[rows, win].sum())
        m = Pt > 0.05 * np.median(Pt[Pt > 0])
        rr = np.full(bidx.size, np.nan)
        rr[m] = Pr[m] / Pt[m]
        gm = np.isfinite(rr)
        grad = np.polyfit(bidx[gm] - ct, rr[gm], 1)[0] if gm.sum() > 8 else np.nan
        litr = np.nonzero(Qr[rows].sum(axis=0) > 1.0)[0]
        out[k] = {'d_cen': cr - ct, 'd_shift': reg_shift(Pt, Pr),
                  'lit_lo': float(litr[0] - lit[0]) if litr.size else np.nan,
                  'lit_hi': float(litr[-1] - lit[-1]) if litr.size else np.nan,
                  'wid_r': wr, 'd_wid': wr / wt if wt > 0 else np.nan,
                  'grad': float(grad), 'cap': float(Pr.sum() / Pt.sum()),
                  'd_cen_a': car - cat}
    return out


# --------------------------------------------------------------- leverage
def seg_from_pix(pa, pb, q):
    """segment_dqdx accepting FRACTIONAL pixel indices."""
    return T.segment_dqdx(np.asarray(pa, float), np.asarray(pb, float), q)


def seg_common(pa, pb, q, c, d, edges):
    """Score a charge set on a GIVEN axis (c, d) and GIVEN bin edges."""
    y, z = T.px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1)
    q = np.asarray(q, float)
    rel = yz - c
    proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    keep = perp < T.TUBE_CM
    if keep.sum() < T.NVALID:
        return np.array([]), None
    h, _ = np.histogram(proj[keep], bins=edges, weights=q[keep])
    return h, None


def truth_axis_edges(pa, pb, q):
    """Reproduce segment_dqdx's axis + edges for the truth, and return them."""
    y, z = T.px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1)
    q = np.asarray(q, float)
    c, d = T.fit_direction(yz, q)
    if c is None:
        return None
    rel = yz - c
    proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    keep = perp < T.TUBE_CM
    if keep.sum() < T.NVALID:
        return None
    c, d = T.fit_direction(yz[keep], q[keep])
    if c is None:
        return None
    proj = (yz[keep] - c) @ d
    edges = np.arange(proj.min(), proj.max() + T.BIN_CM, T.BIN_CM)
    if len(edges) < 4:
        return None
    h, _ = np.histogram(proj, bins=edges, weights=q[keep])
    ne = np.nonzero(h > 0)[0]
    if len(ne) < 3:
        return None
    return c, d, edges, ne[0] + 1, ne[-1]      # keep slice [lo:hi]


def shift_pixels(pa, pb, q, delta):
    """Rigid sub-pixel translation along the pixel axis b, charge split
    linearly between the two straddled rows (mass and centroid exact)."""
    if delta == 0:
        return np.asarray(pa), np.asarray(pb, float), np.asarray(q, float)
    fb = np.asarray(pb, float) + delta
    lo = np.floor(fb)
    frac = fb - lo
    pa2 = np.concatenate([pa, pa])
    pb2 = np.concatenate([lo, lo + 1])
    q2 = np.concatenate([np.asarray(q, float) * (1 - frac),
                         np.asarray(q, float) * frac])
    return pa2, pb2, q2


if __name__ == '__main__':
    arms = os.environ.get('ARMS', 'C').split(',')
    ests = ['hits'] + [f'dec{a}' for a in arms]
    res = {'meta': {'tube_pix': TUBE, 'trim_pix': TRIM, 'nev': NEV,
                    'arms': arms, 'bin_cm': T.BIN_CM, 'tube_cm': T.TUBE_CM},
           'depths': {}}
    DELTAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    for tag in TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        f = np.load(f'{NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        ev_metrics = []
        pool_shift = {dd: [] for dd in DELTAS}          # leverage A (truth)
        pool_own = {k: [] for k in ests}                # own segmentation
        pool_com = {k: [] for k in ests}                # truth segmentation
        pool_t = []
        for ev in range(NEV):
            m = event_metrics(tag, ev, f, arms)
            if m is not None:
                ev_metrics.append(m)
            ta, tb, tq = truth_pix(f, ev)
            # ---- leverage A: shift the truth, same pipeline
            for dd in DELTAS:
                pa2, pb2, q2 = shift_pixels(ta, tb, tq, dd)
                pool_shift[dd].append(seg_from_pix(pa2, pb2, q2))
            # ---- leverage B: common segmentation
            ax = truth_axis_edges(ta, tb, tq)
            pool_t.append(seg_from_pix(ta, tb, tq))
            est = {'hits': hits_pix(f, ev)}
            for arm in arms:
                p = f'{DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = deconv_pix(p)
            for k, (pa, pb, q) in est.items():
                pool_own[k].append(seg_from_pix(pa, pb, q))
                if ax is not None:
                    c, dvec, edges, lo, hi = ax
                    h, _ = seg_common(pa, pb, q, c, dvec, edges)
                    if h.size:
                        hh = h[lo:hi]
                        pool_com[k].append(hh[hh > 0] / T.BIN_CM)

        def pooled_mpv(pool):
            v = [x for x in pool if len(x)]
            if not v:
                return float('nan')
            return float(L.mpv_of(np.concatenate(v))[0])

        agg = {'depth_cm': d_cm, 't_us': L.drift_time_us(d_cm),
               'n_ev': len(ev_metrics)}
        for k in ests:
            for fld in ['d_cen', 'd_shift', 'd_wid', 'grad', 'cap', 'd_cen_a',
                        'lit_lo', 'lit_hi']:
                v = np.array([m[k][fld] for m in ev_metrics if k in m], float)
                v = v[np.isfinite(v)]
                agg[f'{k}.{fld}'] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)),
                                     float(v.std(ddof=1)), float(np.median(v))] if v.size > 2 else [np.nan] * 4
        agg['mpv_truth_shift'] = {str(dd): pooled_mpv(pool_shift[dd]) for dd in DELTAS}
        agg['mpv_own'] = {k: pooled_mpv(pool_own[k]) for k in ests}
        agg['mpv_own']['effq'] = pooled_mpv(pool_t)
        agg['mpv_common'] = {k: pooled_mpv(pool_com[k]) for k in ests}
        res['depths'][tag] = agg
        for k in ests:
            print(f"{tag} n={agg['n_ev']:2d} {k:5s} "
                  f"dcen {agg[f'{k}.d_cen'][0]:+.3f}+-{agg[f'{k}.d_cen'][1]:.3f} "
                  f"(med {agg[f'{k}.d_cen'][3]:+.3f}) "
                  f"shift {agg[f'{k}.d_shift'][0]:+.4f}+-{agg[f'{k}.d_shift'][1]:.4f} "
                  f"dwid {agg[f'{k}.d_wid'][0]:.4f} "
                  f"grad {agg[f'{k}.grad'][0]:+.5f}+-{agg[f'{k}.grad'][1]:.5f} "
                  f"cap {agg[f'{k}.cap'][0]:.4f} "
                  f"dcenA {agg[f'{k}.d_cen_a'][0]:+.4f}+-{agg[f'{k}.d_cen_a'][1]:.4f} "
                  f"lit {agg[f'{k}.lit_lo'][0]:+.2f}/{agg[f'{k}.lit_hi'][0]:+.2f}",
                  flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/profile_shift.json', 'w'), indent=1)
    print('->', f'{OUT}/profile_shift.json')
