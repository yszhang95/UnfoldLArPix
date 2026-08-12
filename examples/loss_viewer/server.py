"""Local web viewer for solve outputs: optimization history + config.

No dependencies beyond numpy and the standard library; nothing leaves
the machine. Serves index.html and a small JSON API over the solved
NPZ files under analysis_output (and any extra roots given on the
command line).

    python examples/loss_viewer/server.py [--port 8765] [extra_root ...]
    -> http://127.0.0.1:8765

API
    GET /api/runs                 -> [{name, path, n}]  (directories of solves)
    GET /api/events?run=<name>    -> [{event, file}]
    GET /api/event?path=<file>    -> {trace, loss, config, summary}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
AO = os.path.join(ROOT, 'examples', 'analysis_output')
EXTRA: list[str] = []


def roots() -> list[str]:
    return [AO] + EXTRA


def _safe(path: str) -> str:
    p = os.path.abspath(path)
    if not any(p.startswith(os.path.abspath(r)) for r in roots()):
        raise PermissionError(path)
    return p


def scan_runs() -> list[dict]:
    """Every directory that directly contains solved *_event_*.npz files."""
    out = {}
    for r in roots():
        if not os.path.isdir(r):
            continue
        for dirpath, _dirs, files in os.walk(r):
            n = sum(1 for f in files if f.endswith('.npz')
                    and '_event_' in f)
            if n:
                out[dirpath] = {'name': os.path.relpath(dirpath, r),
                                'path': dirpath, 'n': n}
    return sorted(out.values(), key=lambda d: d['name'])


def list_events(path: str) -> list[dict]:
    p = _safe(path)
    ev = []
    for f in sorted(os.listdir(p)):
        if f.endswith('.npz') and '_event_' in f:
            m = re.search(r'_event_(\d+)_(\d+)\.npz$', f)
            ev.append({'event': f'{m.group(1)}/{m.group(2)}' if m else f,
                       'file': os.path.join(p, f)})
    return sorted(ev, key=lambda d: [int(x) for x in
                                     re.findall(r'\d+', d['event'])] or [0])


def read_event(path: str) -> dict:
    p = _safe(path)
    z = np.load(p, allow_pickle=True)

    def js(key):
        if key not in z.files:
            return None
        try:
            return json.loads(str(z[key]))
        except Exception:
            return None

    summary = {'file': os.path.basename(p)}
    for k in ('adc_hold_delay', 'readout_nburst', 'readout_threshold'):
        if k in z.files:
            summary[k] = float(np.asarray(z[k]).ravel()[0])
    if 'deconv_q_sharp' in z.files:
        q = np.asarray(z['deconv_q_sharp'], float)
        summary['sum_q'] = float(q.sum())
        summary['nnz'] = int((q > 0.01).sum())
        summary['grid'] = list(q.shape)
    return {'trace': js('loss_trace'), 'loss': js('loss_components'),
            'config': js('job_config'), 'summary': summary}


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype='application/json'):
        body = (obj if isinstance(obj, bytes)
                else json.dumps(obj).encode())
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):        # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ('/', '/index.html'):
                with open(os.path.join(HERE, 'index.html'), 'rb') as fh:
                    return self._send(fh.read(), ctype='text/html')
            if u.path == '/api/runs':
                return self._send(scan_runs())
            if u.path == '/api/events':
                return self._send(list_events(q['path'][0]))
            if u.path == '/api/event':
                return self._send(read_event(q['path'][0]))
            self._send({'error': 'not found'}, 404)
        except Exception as exc:                     # keep the server up
            self._send({'error': f'{type(exc).__name__}: {exc}'}, 500)

    def log_message(self, *a):                        # quiet
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('roots', nargs='*', help='extra directories to scan')
    args = ap.parse_args()
    EXTRA.extend(os.path.abspath(r) for r in args.roots)
    print(f'loss viewer: http://{args.host}:{args.port}')
    print(f'  scanning: {", ".join(roots())}')
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
