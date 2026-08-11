"""Forward closure per depth: does A.q_truth / d drift with depth?
If yes, the solver (which fits d) is FORCED to under-recover at depth --
none of the regularisation knobs can matter, exactly as measured."""
import numpy as np, torch, sys, gc
sys.path.insert(0, '.')
import glsfit as G
import loss_probe as LP
import iso50_analyse as A50
from unfoldlarpix.data_loader import DataLoader
from unfoldlarpix.constrained_solver import build_latch_windows
from unfoldlarpix.model.operator import ZSOperator
from unfoldlarpix.model.warm_start import fft_warm_start, resolve_burst_tau

print('%-8s %12s %12s %12s' % ('d','sum(Aq_t)','sum(d)','Aq_t/d'))
for dd in ['d04p5', 'd16p5', 'd28p5']:
    tag = f'pgun_mu_3gev_iso50_{dd}'
    loader = DataLoader(f'{A50.NFS}/{tag}_tred20ms_nb1.npz')
    rc = loader.get_readout_config()
    B = int(rc.adc_hold_delay)
    tau = resolve_burst_tau(rc, None)
    sA = sD = 0.0
    nev = 0
    for ev in loader.iter_events():
        if not ev.hits or ev.tpc_id != 0:
            continue
        nev += 1
        if nev > 15:            # 15 events per depth is plenty for a ratio
            break
        ws = fft_warm_start(ev.hits, rc, LP.prepared(B), sigma_time=0.005,
                            sigma_pixel=0.2, pad_pixels=12, tau=None,
                            device=LP.DEV, dtype=LP.DT)
        boff = np.asarray(ws.block_offset)
        win = build_latch_windows(ev.hits.location, ev.hits.data, B, boff,
                                  csa_reset_time=rc.csa_reset_time,
                                  split_threshold=float(rc.threshold),
                                  acq_start=getattr(ev, 'acq_start', None),
                                  burst_tau=tau)
        op = ZSOperator(LP.prepared(B).integrated_response, ws.block.shape,
                        win, B, device=LP.DEV, dtype=LP.DT)
        el = np.asarray(ev.effq.location, float)
        eq = np.asarray(ev.effq.data, float)[:, 3]
        qx, qy, qt = op.q_shape
        ix = np.rint(el[:,0] - boff[0]).astype(int)
        iy = np.rint(el[:,1] - boff[1]).astype(int)
        it = np.floor((el[:,2] - boff[2]) / B).astype(int)
        m = (ix>=0)&(ix<qx)&(iy>=0)&(iy<qy)&(it>=0)&(it<qt)
        qtr = np.zeros(op.q_shape)
        np.add.at(qtr, (ix[m], iy[m], it[m]), eq[m])
        pred = op.forward(op.to_tensor(qtr))
        sA += float(pred.sum()); sD += float(op.d.sum())
        del op, ws; gc.collect(); torch.cuda.empty_cache()
    print('%-8s %12.1f %12.1f %12.4f' % (dd, sA, sD, sA/sD), flush=True)
