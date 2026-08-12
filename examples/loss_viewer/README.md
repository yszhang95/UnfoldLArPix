# Solve viewer — optimization history, losses, hyper-parameters

A local, dependency-free web tool for inspecting solves: the FISTA
objective history stage by stage, the final loss components, and the
full `job_config` that produced them.

## Record a trace

Tracing is opt-in and sampled, so ordinary solves are unaffected. Add
`trace_every` to the `Solve` step (every N iterations, per stage):

```yaml
- Solve:
    engine: {iters: 600}
    strategy: {type: ladder, alphas: [1.0, 0.5, 0.3], seed_cut: 0.5, soft_len: 2.0}
    terms: [{type: censor, beta: 1.0, margin: 3.0, norm: l2}]
    refit: {eps: 0.5, alpha: 0.0}
    trace_every: 20          # <- 0 (default) = off
```

Each output NPZ then carries

| field | contents |
|---|---|
| `loss_trace` | JSON rows: `stage`, `iter`, one entry per smooth term (`DataFidelity`, `CensorRunningMax`, …), `l1`, `objective`, `q_sum`, `nnz` |
| `loss_components` | the same components evaluated at the final solution |
| `job_config` | every hyper-parameter of the run (always stored) |

Cost: one extra `value()` per term per sampled iteration.

## Run the viewer

```bash
cd <repo root>
python examples/loss_viewer/server.py                 # http://127.0.0.1:8765
python examples/loss_viewer/server.py --port 8791 /extra/dir …
```

It scans `examples/analysis_output` (plus any extra roots given) for
directories containing `*_event_*.npz`, and serves everything locally —
no network access, no third-party JavaScript.

* **configuration / run** — every solve directory found (`?run=<substring>`
  deep-links one).
* **events** — ctrl/shift-select up to five to overlay them.
* **series** — click to toggle any traced quantity; **y axis** log/linear.
* Dashed verticals mark the ladder/refit stage boundaries.
* Lower panels: final losses, run summary, and the full `job_config`.

Files without a trace still show their losses, summary and config.
