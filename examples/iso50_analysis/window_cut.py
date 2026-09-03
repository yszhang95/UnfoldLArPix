"""Does restricting the lambda fit to d >= 16.5 cm help?  And how badly is
the linear system underdetermined?

CPU only, on the archived solves; nothing re-run.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_shift as PS
import track_dqdx as T
import dqdx_lib as L

NB = 300
WINDOWS = [(20.0, 'all 9 (28-179 us)'), (60.0, 'd >= 10.5 cm'),
           (80.0, 'd >= 13.5 cm'), (100.0, 'd >= 16.5 cm'),
           (120.0, 'd >= 19.5 cm')]


def seg3(pa, pb, q):
    return T.segment_dqdx(np.asarray(pa, float), np.asarray(pb, float), q)


def boot_lambda(t_us, per_depth, tmin, nb=NB, seed=1):
    rng = np.random.default_rng(seed)
    keep = t_us > tmin
    lam = []
    for _ in range(nb):
        m = []
        for di, evs in enumerate(per_depth):
            if not keep[di]:
                m.append(np.nan); continue
            pk = rng.integers(0, len(evs), len(evs))
            v = [evs[i] for i in pk if len(evs[i])]
            m.append(L.mpv_of(np.concatenate(v))[0] if v else np.nan)
        m = np.array(m); k = np.isfinite(m) & keep
        if k.sum() < 3:
            continue
        A = np.vstack([t_us[k], np.ones(k.sum())]).T
        lam.append(-np.linalg.lstsq(A, np.log(m[k]), rcond=None)[0][0] * 1000)
    lam = np.array(lam)
    return float(np.median(lam)), float(lam.std(ddof=1))


if __name__ == '__main__':
    t_us, store, sysinfo = [], {k: [] for k in ['effq', 'hits', 'decC', 'decB']}, []
    for tag in PS.TAGS:
        d_cm = float(tag.split('_d')[1].replace('p', '.'))
        t_us.append(L.drift_time_us(d_cm))
        fz = np.load(f'{PS.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
        per = {k: [] for k in store}
        nh, nv, nb_ = [], [], []
        for ev in range(50):
            ta, tb, tq = PS.truth_pix(fz, ev)
            est = {'effq': (ta, tb, tq), 'hits': PS.hits_pix(fz, ev)}
            for arm in ['C', 'B']:
                p = f'{PS.DIRS[arm][0]}/{tag}/{tag}_event_0_{ev}.npz'
                if os.path.exists(p):
                    est[f'dec{arm}'] = PS.deconv_pix(p)
            for k, v in est.items():
                per[k].append(seg3(*v))
            if ev < 10:
                p = f'{PS.DIRS["C"][0]}/{tag}/{tag}_event_0_{ev}.npz'
                z = np.load(p, allow_pickle=True)
                qs = np.asarray(z['deconv_q_sharp'])
                nh.append(len(est['hits'][2])); nv.append(int((qs > 0).sum()))
                nb_.append(int(np.prod(qs.shape)))
        for k in store:
            store[k].append(per[k])
        sysinfo.append((d_cm, np.mean(nh), np.mean(nv), np.mean(nb_)))
        print('loaded', tag, flush=True)
    t_us = np.array(t_us)

    print()
    print('=' * 78)
    print('HOW UNDERDETERMINED IS THE SOLVE?  (arm C, 10 events per depth)')
    print('=' * 78)
    print('%8s %10s %12s %14s %12s' % ('d [cm]', 'hits', 'block voxels',
                                       'nonzero out', 'rows/unknown'))
    for d, h, v, b in sysinfo:
        print('%8.1f %10.0f %12.0f %14.0f %12.4f' % (d, h, b, v, h / v))

    print()
    print('=' * 78)
    print('lambda [1/ms] vs FIT WINDOW  (3 cm segments, bootstrap over events)')
    print('=' * 78)
    print('%-22s %5s %15s %15s %15s %15s'
          % ('fit window', 'N', 'effq (truth)', 'raw hits', 'deconv C', 'deconv B'))
    res = {}
    for tmin, lab in WINDOWS:
        n = int((t_us > tmin).sum())
        row = []
        for k in ['effq', 'hits', 'decC', 'decB']:
            m, e = boot_lambda(t_us, store[k], tmin)
            row.append(f'{m:.3f} +- {e:.3f}')
            res[(lab, k)] = (m, e)
        print('%-22s %5d %15s %15s %15s %15s' % (lab, n, *row))

    print()
    print('WHY: MPV capture MPV_dec/MPV_true and its slope inside each window')
    mv = {k: np.array([L.mpv_of(np.concatenate([x for x in store[k][d] if len(x)]))[0]
                       for d in range(len(PS.TAGS))]) for k in store}
    cap = mv['decC'] / mv['effq']
    caph = mv['hits'] / mv['effq']
    print('%-22s %14s %14s' % ('window', 'decC slope/span', 'hits slope/span'))
    for tmin, lab in WINDOWS:
        k = t_us > tmin
        span = t_us[k].max() - t_us[k].min()
        s1 = np.polyfit(t_us[k], cap[k], 1)[0] * span
        s2 = np.polyfit(t_us[k], caph[k], 1)[0] * span
        print('%-22s %13.2f%% %13.2f%%' % (lab, s1 * 100, s2 * 100))
    print()
    print('per-depth MPV capture (decC / effq):')
    for i, tag in enumerate(PS.TAGS):
        print('   d=%4.1f cm  t=%6.1f us   %.4f' % (
            float(tag.split('_d')[1].replace('p', '.')), t_us[i], cap[i]))
