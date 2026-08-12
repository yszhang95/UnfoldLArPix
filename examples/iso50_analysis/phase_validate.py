"""Validation 1 for ZSOperatorPhase: forward closure on real events.

For iso50 events, apply BOTH operators to the truth mapped onto the fit
grid and compare against the recorded data: Sigma(A q_truth)/Sigma(d).
The phase-exact operator should sit closer to 1 (waveform prototype:
box loses ~1-2 points at the edges).
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
from unfoldlarpix.constrained_solver import build_latch_rows
from unfoldlarpix.model.phase_operator import ZSOperatorPhase

RESP_LOCAL = ('/srv/storage1/yousen/tred_workspace/'
              'response_44_v2a_full_25x25pixel_tred.npz')
NEV = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def job_yaml(tag):
    for base in [f'{A50.AO}/iso50/C', '/home/yousen/iso50_staging/C']:
        p = f'{base}/job_{tag}.yaml'
        if os.path.exists(p):
            cfg = yaml.safe_load(open(p))
            cfg['services']['detector']['response'] = RESP_LOCAL
            return cfg
    raise FileNotFoundError(tag)


_SVC = {}
for tag in ['pgun_mu_3gev_iso50_d04p5', 'pgun_mu_3gev_iso50_d16p5',
            'pgun_mu_3gev_iso50_d28p5']:
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
    rb, rp = [], []
    for ev in range(NEV):
        store = EventStore()
        store.put('job.config', cfg, 'runner')
        for a in algs:
            a.execute(store)
        op = store.get('op')                       # stock box operator
        rc = store.get('readout_config')
        evd = store.get('event')
        boff = np.asarray(store.get('block_offset'), dtype=float)
        B = int(rc.adc_hold_delay)
        prepared = services['detector'].prepared(B)
        windows, _ = build_latch_rows(
            evd.hits.location, evd.hits.data, B, boff,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=float(rc.threshold),
            acq_start=getattr(evd, 'acq_start', None),
            burst_tau=None)
        # NB: replicate BuildMeasurement's exact props for burst_tau
        bm = [e for e in cfg['sequence'] if 'BuildMeasurement' in e][0]
        from unfoldlarpix.model.conventions import resolve_burst_tau
        bt = bm['BuildMeasurement'].get('burst_tau')
        bt = (None if bt is None else resolve_burst_tau(
            rc, None if bt == 'auto' else int(bt)))
        windows, _ = build_latch_rows(
            evd.hits.location, evd.hits.data, B, boff,
            csa_reset_time=rc.csa_reset_time,
            split_threshold=(float(rc.threshold)
                             if bm['BuildMeasurement'].get(
                                 'split_trigger', True) else None),
            acq_start=getattr(evd, 'acq_start', None), burst_tau=bt)
        opp = ZSOperatorPhase(prepared.integrated_response,
                              prepared.full_response,
                              op.block_shape, windows, B,
                              device=op.device, dtype=torch.float64)
        # truth onto the q grid (reference-plane frame, gain_audit mapping)
        nx, ny, qt = op.q_shape
        el = np.asarray(f_truth[f'effq_tpc0_batch{ev}_location'])
        eq = np.asarray(f_truth[f'effq_tpc0_batch{ev}'], float)[:, 3]
        ix = el[:, 0].astype(int) - int(boff[0])
        iy = el[:, 1].astype(int) - int(boff[1])
        it = np.floor((el[:, 2] - boff[2]) / B).astype(int)
        ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
              & (it >= 0) & (it < qt))
        qg = np.zeros((nx, ny, qt))
        np.add.at(qg, (ix[ok], iy[ok], it[ok]), eq[ok])
        qtn = torch.as_tensor(qg, dtype=torch.float64, device=op.device)
        qtn32 = qtn.to(op.dtype)
        sd = float(op.d.sum())
        rb.append(float(op.forward(qtn32).sum()) / sd)
        rp.append(float(opp.forward(qtn).sum()) / sd)
        del op, opp, store
        gc.collect(); torch.cuda.empty_cache()
    print('%s (%d ev):  box A*qtruth/d = %.4f   phase = %.4f'
          % (tag.split('_d')[1], NEV, np.mean(rb), np.mean(rp)),
          flush=True)
