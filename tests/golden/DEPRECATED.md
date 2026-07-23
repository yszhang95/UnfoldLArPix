# Golden reference DEPRECATED — computed on buggy-readout data

`golden_metrics.json` (`nb4_adopted_centroid_w1`, `nb1_censorL2_600_centroid_w2`)
was computed on positron datasets produced by the tred readout **before** the
memoryless-CSA-reset fix. That readout accumulated a fresh per-reset kTC
baseline into the running accumulator without ever removing the previous one,
so on bright pixels the piled-up baseline stayed above threshold long after the
real charge was collected and faked sustained late re-triggers (e.g. pixel
(139,80): 18 latches recording 223 ke vs a true 116.8 ke). Those fake latches
inflated the measurement and biased every metric.

Evidence the reference is stale — nb4, identical config, on FIXED-readout data:

| metric            | golden (buggy data) | fixed-readout data |
|-------------------|---------------------|--------------------|
| integral_pct      | -2.094              | ~-2.13             |
| ghost_iso_charge  | 3.54                | ~0.6               |
| ghost_frac        | 0.0449              | ~0.038             |

**Status:** invalid. Do NOT treat `golden_gate.py` pass/fail as authoritative
until the reference is regenerated.

**Action:** regenerate `golden_metrics.json` on datasets produced with the fixed
readout (tred `pgun_farfield` @ 8545637), then re-enable the gate. This should be
committed together with the UnfoldLArPix burst-`tau` patch (currently uncommitted
working-tree changes to `model/conventions.py`, `model/warm_start.py`,
`algs/reco_algs.py` + `tests/test_burst_tau.py`), since the adopted pipeline runs
with the physical `tau` floor.
