"""Dynamic-support test: arm C + hits-source support + gain_alpha.

Identical solver settings to run_iso50_solve.py arm C; input files carry
the _3ms suffix. Shard by depth: argv W NW [response_override]
[outdir_override].
"""
import os, sys, subprocess, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_alpha_beta import PY, OUT, RESP, ROOT

NFS = '/home/yousen/Documents/NDLAr2x2/tred_worktree/pgun_far_field/tests/pgun_farfield'
TAGS = [l.strip() for l in open('/home/yousen/Documents/NDLAr2x2/MuonLArSim/iso50_list.txt') if l.strip()]
W, NW = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    RESP = sys.argv[3]
OD = sys.argv[4] if len(sys.argv) > 4 else f'{OUT}/iso50_dyn'
CEN = [{'type': 'censor', 'beta': 1.0, 'margin': 3.0, 'norm': 'l2'}]
jobs = []
for i, t in enumerate(TAGS):
    if i % NW != W:
        continue
    solve = {'engine': {'iters': 600},
             'strategy': {'type': 'ladder', 'alphas': [1.0, 0.5, 0.3],
                          'seed_cut': 0.5, 'soft_len': 2.0,
                          'gain_alpha': {}},
             'terms': CEN,
             'refit': {'eps': 0.5, 'alpha': 0.0}}
    seq = [
        {'LoadEvent': {'input': f'{NFS}/{t}_tred_nb1.npz', 'tpc': 0,
                       'max_events': 50}},
        {'FFTWarmStart': {'sigma_time': 0.005, 'sigma_pixel': 0.2,
                          'pad_pixels': 12}},
        {'BuildMeasurement': {'split_trigger': True, 'acq_start': 'event',
                              'burst_tau': 'auto'}},
        {'BuildSupport': {'source': 'hits', 'hits_dilate': 2}},
        {'Solve': solve},
        {'CentroidPositions': {'window': 2}},
        {'WriteCharges': {'out_dir': f'{OD}/C/{t}', 'prefix': t}},
    ]
    jobs.append((t, {'services': {'compute': {'device': 'cuda',
                                              'dtype': 'float32'},
                                  'detector': {'response': RESP},
                                  'rng': {'seed': 0}},
                     'sequence': seq}))
print(f'worker {W}/{NW}: {len(jobs)} jobs', flush=True)
for i, (t, c) in enumerate(jobs, 1):
    os.makedirs(f'{OD}/C/{t}', exist_ok=True)
    marker = f'{OD}/C/{t}/{t}_event_0_49.npz'
    if os.path.exists(marker):
        print(f'[{i}] skip C/{t}', flush=True); continue
    if not os.path.exists(f'{NFS}/{t}_tred_nb1.npz'):
        print(f'[{i}] INPUT MISSING {t}', flush=True)
        continue
    cp = f'{OD}/C/job_{t}.yaml'
    with open(cp, 'w') as fh:
        yaml.safe_dump(c, fh, sort_keys=False)
    print(f'[{i}/{len(jobs)}] C/{t}', flush=True)
    with open(f'{OD}/C/{t}.log', 'w') as lg:
        r = subprocess.run([PY, '-m', 'unfoldlarpix.fwk.runner', cp], cwd=ROOT,
                           stdout=lg, stderr=subprocess.STDOUT,
                           env={**os.environ, 'PYTHONPATH': f'{ROOT}/src'})
    print(f'    rc={r.returncode}', flush=True)
print(f'WORKER{W} DONE', flush=True)
