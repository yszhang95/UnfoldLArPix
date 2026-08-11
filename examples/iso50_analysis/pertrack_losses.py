"""Per-track forensics for the near-anode over-credit: for every event
(9 depths x 50 tracks, arm C) evaluate the objective components at the
stored deconv_q_sharp -- data fidelity, censor (raw, beta-independent),
l1 norm (the refit runs at alpha = 0, so the l1 LOSS at the solution is
zero by construction; the norm is recorded) -- plus per-pixel
(time-summed) truth/deconv pairs and per-track regression numbers.

Outputs: analysis_output/iso50_3ms_report/pertrack_losses.json and
pertrack_pairs.npz (pair arrays per event, for the panel plots).
"""
import numpy as np, sys, os, json, gc, warnings, yaml, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DRV = '/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output/_drivers'
sys.path.insert(0, DRV)
import dqdx_lib as L
import iso50_analyse as A50
from smear_scan import pixmap
from eval_alpha_beta import build_job, EventStore, CensorRunningMax, IterCtx
from unfoldlarpix.fwk.component import ALGORITHMS

TAGS = A50.TAGS[1:]
AO = A50.AO


RESP_LOCAL = ('/srv/storage1/yousen/tred_workspace/'
              'response_44_v2a_full_25x25pixel_tred.npz')


def job_yaml(tag):
    for base in [f'{AO}/iso50/C', '/home/yousen/iso50_staging/C']:
        p = f'{base}/job_{tag}.yaml'
        if os.path.exists(p):
            cfg = yaml.safe_load(open(p))
            cfg['services']['detector']['response'] = RESP_LOCAL
            return cfg
    raise FileNotFoundError(tag)


_SVC = {}
out = {}
pairs = {}
for tag in TAGS:
    cfg = job_yaml(tag)
    keep = [e for e in cfg['sequence']
            if list(e)[0] in ('LoadEvent', 'FFTWarmStart',
                              'BuildMeasurement')]
    skey = json.dumps(cfg['services'], sort_keys=True)
    if skey not in _SVC:
        _SVC[skey], _ = build_job({'services': cfg['services'],
                                   'sequence': keep})
    services = _SVC[skey]
    algs = []
    for entry in keep:
        (name, props), = entry.items()
        a = ALGORITHMS[name](**(props or {}))
        a.initialize(services)
        algs.append(a)
    tcfg = [t for e in cfg['sequence'] if list(e)[0] == 'Solve'
            for t in (e['Solve'].get('terms') or [])
            if t['type'] == 'censor'][0]
    f_truth = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    out[tag] = []
    for ev in range(50):
        store = EventStore()
        store.put('job.config', cfg, 'runner')
        for a in algs:
            a.execute(store)          # LoadEvent advances its own cursor
        op = store.get('op')
        p = A50.find_solved('C', tag, ev)
        z = np.load(p, allow_pickle=True)
        q = torch.as_tensor(np.asarray(z['deconv_q_sharp'], np.float64),
                            dtype=op.dtype, device=op.device)
        r = op.forward(q) - op.d
        rc = store.get('readout_config')
        term = CensorRunningMax.from_hits(
            op, store.get('hits_view'), store.get('block_offset'),
            csa_reset_time=float(rc.csa_reset_time or 0),
            threshold=float(rc.threshold), npad_bins=50,
            beta=float(tcfg['beta']), margin=float(tcfg['margin']),
            norm=tcfg.get('norm', 'l2'),
            bin_ticks=int(rc.adc_hold_delay))
        rec = {'ev': ev,
               'data_fid': 0.5 * float((r * r).sum()),
               'censor_raw': float(term.value(IterCtx(q, op)))
                             / float(tcfg['beta']),
               'sum_q': float(q.sum()),
               'sum_d': float(op.d.sum()),
               'n_rows': int(op.d.numel())}
        # per-pixel pairs (time-summed), absolute grid
        el = np.asarray(f_truth[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f_truth[f'effq_tpc0_batch{ev}'], float)[:, 3]
        ta, tb, tq = pixmap(el[:, 0], el[:, 1], eq)
        da, db, dq = A50.deconv_pix(p)
        gt = dict(zip((ta.astype(np.int64)*100000+tb).tolist(), tq))
        gd = dict(zip((da.astype(np.int64)*100000+db).tolist(), dq))
        ks = sorted(set(gt) | set(gd))
        x = np.array([gt.get(k, 0.0) for k in ks])
        y = np.array([gd.get(k, 0.0) for k in ks])
        m = np.maximum(x, y) > 0.5
        x, y = x[m], y[m]
        sl, ic = np.polyfit(x, y, 1)
        rec.update(slope=float(sl), intercept=float(ic),
                   slope0=float((x*y).sum()/(x*x).sum()),
                   ratio=float(y.sum()/x.sum()),
                   sum_truth=float(tq.sum()), npix=int(len(x)))
        out[tag].append(rec)
        pairs[f'{tag}_e{ev:02d}_x'] = x.astype(np.float32)
        pairs[f'{tag}_e{ev:02d}_y'] = y.astype(np.float32)
        del op, q, r, term, store
        gc.collect()
        torch.cuda.empty_cache()
    dsum = np.array([e['data_fid'] for e in out[tag]])
    csum = np.array([e['censor_raw'] for e in out[tag]])
    ssum = np.array([e['slope'] for e in out[tag]])
    print(f'{tag}: data_fid med {np.median(dsum):.1f} '
          f'censor med {np.median(csum):.3f} slope med {np.median(ssum):.3f}',
          flush=True)

OUTD = f'{AO}/iso50_3ms_report'
json.dump(out, open(f'{OUTD}/pertrack_losses.json', 'w'), indent=1)
np.savez_compressed(f'{OUTD}/pertrack_pairs.npz', **pairs)
print('-> pertrack_losses.json, pertrack_pairs.npz', flush=True)
