"""The gain audit: does the booked charge sit at lower measurement gain?

Per event (9 depths x 50, arm C): c = A^T 1 on the fit grid;
  <c>_qhat  = sum(c * qhat) / sum(qhat)
  <c>_truth = sum(c * q_truth_on_grid) / sum(q_truth_on_grid)
plus c-binned charge sums for qhat and truth. The accounting identity
predicts <c>_qhat / <c>_truth ~ Sigma_d-closure / capture-ratio
(~0.91 at d = 4.5 cm, ~1.01 at 28.5).

Truth mapping: effq lives at the response reference plane, which IS the
charge grid's time frame, so bin = (t_effq - boffset_raw[2]) / B.
"""
import numpy as np, sys, os, json, gc, warnings, yaml, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DRV = ('/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/'
       'examples/analysis_output/_drivers')
sys.path.insert(0, DRV)
import iso50_analyse as A50
from eval_alpha_beta import build_job, EventStore
from unfoldlarpix.fwk.component import ALGORITHMS

TAGS = A50.TAGS[1:]
AO = A50.AO
RESP_LOCAL = ('/srv/storage1/yousen/tred_workspace/'
              'response_44_v2a_full_25x25pixel_tred.npz')
CBINS = [0.0, 0.7, 0.9, 0.97, 1.03, 10.0]


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
    f_truth = np.load(f'{A50.NFS}/{tag}_tred_nb1.npz', allow_pickle=True)
    out[tag] = []
    for ev in range(50):
        store = EventStore()
        store.put('job.config', cfg, 'runner')
        for a in algs:
            a.execute(store)
        op = store.get('op')
        boff = np.asarray(store.get('block_offset'), dtype=float)
        c = op.measurement_gain().cpu().numpy()      # q-grid (nx, ny, qt)
        nx, ny, qt = c.shape
        B = float(store.get('readout_config').adc_hold_delay)
        p = A50.find_solved('C', tag, ev)
        z = np.load(p, allow_pickle=True)
        qh = np.asarray(z['deconv_q_sharp'], np.float64)
        # truth onto the q grid
        el = np.asarray(f_truth[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f_truth[f'effq_tpc0_batch{ev}'], float)[:, 3]
        ix = el[:, 0].astype(int) - int(boff[0])
        iy = el[:, 1].astype(int) - int(boff[1])
        it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
        ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
              & (it >= 0) & (it < qt))
        qt_grid = np.zeros_like(c)
        np.add.at(qt_grid, (ix[ok], iy[ok], it[ok]), eq[ok])
        th = torch.as_tensor(qh, dtype=op.dtype, device=op.device)
        tt = torch.as_tensor(qt_grid, dtype=op.dtype, device=op.device)
        rec = {'ev': ev,
               'truth_in_grid': float(eq[ok].sum() / eq.sum()),
               'c_qhat': float((c*qh).sum() / max(qh.sum(), 1e-9)),
               'c_truth': float((c*qt_grid).sum()
                                / max(qt_grid.sum(), 1e-9)),
               'sum_d': float(op.d.sum()),
               'sum_Aqhat': float(op.forward(th).sum()),
               'sum_Aqtruth': float(op.forward(tt).sum()),
               'sum_qhat': float(qh.sum()),
               'sum_qtruth': float(qt_grid.sum())}
        del th, tt
        hb_q, _ = np.histogram(c.ravel(), bins=CBINS, weights=qh.ravel())
        hb_t, _ = np.histogram(c.ravel(), bins=CBINS,
                               weights=qt_grid.ravel())
        rec['cbin_qhat'] = hb_q.tolist()
        rec['cbin_truth'] = hb_t.tolist()
        out[tag].append(rec)
        del op, store, c, qh, qt_grid
        gc.collect(); torch.cuda.empty_cache()
    cq = np.array([e['c_qhat'] for e in out[tag]])
    ct = np.array([e['c_truth'] for e in out[tag]])
    tg = np.array([e['truth_in_grid'] for e in out[tag]])
    sd = np.array([e['sum_d'] for e in out[tag]])
    sah = np.array([e['sum_Aqhat'] for e in out[tag]])
    sat = np.array([e['sum_Aqtruth'] for e in out[tag]])
    print('%s: <c>q %.4f <c>t %.4f | Aqhat/d %.4f  Aqtruth/d %.4f  '
          'Aqhat/Aqtruth %.4f'
          % (tag.split('_d')[1], cq.mean(), ct.mean(),
             (sah/sd).mean(), (sat/sd).mean(), (sah/sat).mean()),
          flush=True)

json.dump(out, open(f'{AO}/iso50_3ms_report/gain_audit.json', 'w'),
          indent=1)
print('-> gain_audit.json', flush=True)
