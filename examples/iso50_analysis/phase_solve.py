"""Validation 2 for ZSOperatorPhase: full arm-C-style solves.

Replicates the arm-C solve (ladder + censor + refit) with the
phase-exact operator and PhaseDataFidelity, on three iso50 depths x 50
events. The refit is inlined (FinalRefit swaps only stock DataFidelity
instances). Outputs NPZ files compatible with the capture analysis.

Usage: phase_solve.py [resp_override] [outdir]
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
from unfoldlarpix.model.conventions import resolve_burst_tau
from unfoldlarpix.model.phase_operator import (PhaseDataFidelity,
                                               ZSOperatorPhase)
from unfoldlarpix.solve.engine import Fista
from unfoldlarpix.solve.strategy import Ladder, SolveState
from unfoldlarpix.terms.base import CoordProx
from unfoldlarpix.terms.censor import CensorRunningMax

RESP = (sys.argv[1] if len(sys.argv) > 1 else
        '/srv/storage1/yousen/tred_workspace/'
        'response_44_v2a_full_25x25pixel_tred.npz')
OUTD = (sys.argv[2] if len(sys.argv) > 2 else
        f'{A50.AO}/iso50_phase')
TAGS3 = ['pgun_mu_3gev_iso50_d04p5', 'pgun_mu_3gev_iso50_d16p5',
         'pgun_mu_3gev_iso50_d28p5']


def job_yaml(tag):
    for base in [f'{A50.AO}/iso50/C', '/home/yousen/iso50_staging/C']:
        p = f'{base}/job_{tag}.yaml'
        if os.path.exists(p):
            cfg = yaml.safe_load(open(p))
            cfg['services']['detector']['response'] = RESP
            return cfg
    raise FileNotFoundError(tag)


_SVC = {}
for tag in TAGS3:
    cfg = job_yaml(tag)
    keep = [e for e in cfg['sequence']
            if list(e)[0] in ('LoadEvent', 'FFTWarmStart',
                              'BuildMeasurement', 'BuildSupport')]
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
    bm = [e for e in cfg['sequence'] if 'BuildMeasurement' in e][0]
    scfg = [e for e in cfg['sequence'] if 'Solve' in e][0]['Solve']
    tcfg = [t for t in scfg['terms'] if t['type'] == 'censor'][0]
    os.makedirs(f'{OUTD}/C/{tag}', exist_ok=True)
    for ev in range(50):
        outp = f'{OUTD}/C/{tag}/{tag}_event_0_{ev}.npz'
        store = EventStore()
        store.put('job.config', cfg, 'runner')
        for a in algs:
            a.execute(store)
        if os.path.exists(outp):
            continue
        op = store.get('op')
        rc = store.get('readout_config')
        evd = store.get('event')
        boff = np.asarray(store.get('block_offset'), dtype=float)
        B = int(rc.adc_hold_delay)
        prepared = services['detector'].prepared(B)
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
                              device=op.device, dtype=op.dtype)
        terms = [PhaseDataFidelity(opp),
                 CensorRunningMax.from_hits(
                     opp, store.get('hits_view'),
                     store.get('block_offset'),
                     csa_reset_time=float(rc.csa_reset_time or 0),
                     threshold=float(rc.threshold), npad_bins=50,
                     beta=float(tcfg['beta']),
                     margin=float(tcfg['margin']),
                     norm=tcfg.get('norm', 'l2'), bin_ticks=B)]
        support = opp.to_tensor(
            store.get('support').astype(np.float64))
        q0 = opp.to_tensor(np.clip(store.get('warm.deconv_q'), 0.0, None)
                           [:, :, :opp.q_shape[2]])
        engine = Fista(n_iter=int(scfg['engine']['iters']))
        lad = dict(scfg['strategy']); lad.pop('type')
        state = Ladder(n_iter=engine.n_iter, **lad).run(
            engine, opp, terms, support, SolveState(q=q0))
        # inline refit (phase version of FinalRefit)
        rcfg = scfg['refit']
        strong = state.q > float(rcfg.get('eps', 0.5))
        q_faint = torch.where(strong, torch.zeros_like(state.q), state.q)
        target = opp.d - opp.forward(q_faint)
        terms_r = [PhaseDataFidelity(opp, target=target), terms[1]]
        prox = CoordProx(float(rcfg.get('alpha', 0.0)),
                         strong.to(opp.dtype))
        q_strong = Fista(n_iter=engine.n_iter,
                         safety=engine.safety).minimize(
            opp, terms_r, prox,
            q0=torch.where(strong, state.q, torch.zeros_like(state.q)))
        q = (q_strong + q_faint).cpu().numpy().astype(np.float64)
        np.savez_compressed(outp, deconv_q_sharp=q.astype(np.float32),
                            boffset=boff)
        del op, opp, store, state, q_strong, q_faint, terms, terms_r
        gc.collect(); torch.cuda.empty_cache()
        if ev % 10 == 0:
            print(f'{tag} ev {ev} done', flush=True)
    print(f'{tag} COMPLETE', flush=True)
print('PHASE SOLVES DONE', flush=True)
