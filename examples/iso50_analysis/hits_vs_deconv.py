"""Is `raw hits` really beating the deconvolution, and on what?

Two questions:
 1. SHARD ARTEFACT?  The solves were sharded over two hosts; depths
    1.5/10.5/19.5/28.5 loaded the response from /srv/storage1, the rest from
    /nfs (same filename).  If those files differed, the every-third-depth
    pattern would inject a sawtooth into the capture curve.  Tested by
    fitting capture = a*t + b + h*HOST and asking whether h != 0.
 2. ON WHAT does the deconv lose?  lambda is a pure DEPTH DERIVATIVE of the
    gain.  Report, per depth, the quantities a reconstruction is actually
    for: charge scale, per-segment accuracy, neighbour fidelity.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import profile_transport as PT
import track_dqdx as T
import dqdx_lib as L

HERACLES = {'pgun_mu_3gev_iso50_d01p5', 'pgun_mu_3gev_iso50_d10p5',
            'pgun_mu_3gev_iso50_d19p5', 'pgun_mu_3gev_iso50_d28p5'}

if __name__ == '__main__':
    J = json.load(open(f'{PS.OUT}/profile_shift.json'))['depths']
    tags = list(J)
    t = np.array([J[g]['t_us'] for g in tags])
    host = np.array([1.0 if g in HERACLES else 0.0 for g in tags])

    print('=' * 78)
    print('1. SHARD TEST: capture = a*t + b + h*HOST   (HOST = 1 for the')
    print('   /srv/storage1 shard: depths 1.5, 10.5, 19.5, 28.5)')
    print('=' * 78)
    for k in ['hits', 'decC', 'decB']:
        y = np.array([J[g][f'{k}.cap'][0] for g in tags])
        e = np.array([J[g][f'{k}.cap'][1] for g in tags])
        m = t > 20
        A = np.vstack([t[m], np.ones(m.sum()), host[m]]).T
        W = np.diag(1 / e[m] ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        p = cov @ (A.T @ W @ y[m])
        print('  %-6s host offset h = %+.5f +- %.5f  (%.1f sigma)   slope %+.3e'
              % (k, p[2], np.sqrt(cov[2, 2]), abs(p[2]) / np.sqrt(cov[2, 2]), p[0]))

    print()
    print('=' * 78)
    print('2. WHAT EACH ESTIMATOR IS GOOD AT (3 cm segments, common truth')
    print('   segmentation; RMS and r are per-segment against the truth)')
    print('=' * 78)
    print('%-7s %-6s %9s %9s %9s %9s %9s' %
          ('depth', 'est', 'sum/true', 'RMS%', 'r', 'MPV/true', 'medbias%'))
    agg = {k: {'rms': [], 'r': []} for k in ['hits', 'decC', 'decB']}
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        pr = {k: [[], []] for k in ['hits', 'decC', 'decB']}
        for ev in range(50):
            ta, tb, tq = PS.truth_pix(fz, ev)
            ax = PS.truth_axis_edges(ta, tb, tq)
            if ax is None:
                continue
            c, dv, edges, lo, hi = ax
            ht, _ = PS.seg_common(ta, tb, tq, c, dv, edges)
            if not ht.size:
                continue
            ht = ht[lo:hi]
            est = {'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for k, (pa, pb, q) in est.items():
                hr, _ = PS.seg_common(pa, pb, q, c, dv, edges)
                if hr.size:
                    m = ht > 0
                    pr[k][0].append(ht[m]); pr[k][1].append(hr[lo:hi][m])
        for k in pr:
            if not pr[k][0]:
                continue
            x = np.concatenate(pr[k][0]); y = np.concatenate(pr[k][1])
            rel = (y - x) / x
            rms = float(np.sqrt((rel ** 2).mean()) * 100)
            r = float(np.corrcoef(x, y)[0, 1])
            agg[k]['rms'].append(rms); agg[k]['r'].append(r)
            print('%-7.1f %-6s %9.4f %9.2f %9.5f %9.4f %+9.2f'
                  % (d_cm, k, y.sum() / x.sum(), rms, r,
                     L.mpv_of(y)[0] / L.mpv_of(x)[0], np.median(rel) * 100))
        print()

    print('mean over the 9 fitted depths:')
    for k in agg:
        print('  %-6s  per-segment RMS %6.2f %%   r %.5f'
              % (k, np.mean(agg[k]['rms'][1:]), np.mean(agg[k]['r'][1:])))

    print()
    print('=' * 78)
    print('3. THE SPLIT: absolute accuracy vs depth-stability of the gain')
    print('=' * 78)
    K = json.load(open(f'{PS.OUT}/transport_kernel.json'))['depths']
    for k in ['hits', 'decC', 'decB']:
        tot = np.array([K[g][k]['total'] for g in K if K[g]['depth_cm'] > 3])
        nb = np.array([K[g][k]['w_pm1'] for g in K if K[g]['depth_cm'] > 3])
        cap = np.array([J[g][f'{k}.cap'][0] for g in tags])
        m = t > 20
        sl = np.polyfit(t[m], cap[m], 1)[0] * (t[m].max() - t[m].min())
        print('  %-6s  |kernel total - 1| = %5.1f %%   neighbour share = %+6.1f %%'
              '   capture slide over the span = %+5.1f %%'
              % (k, abs(tot.mean() - 1) * 100, nb.mean() * 100, sl * 100))
