# Deprecated positron datasets — buggy readout (do not use)

These `noises` positron datasets were produced by the tred readout **before**
the memoryless-CSA-reset fix (tred `pgun_farfield` @ 8545637). That readout
accumulated per-reset kTC baselines and faked sustained late re-triggers on
bright pixels, inflating the measurement (e.g. pixel (139,80): 18 latches,
223 ke- vs a true 116.8). They are kept only for provenance.

The canonical names in `examples/data/` now symlink to the regenerated
`*_resetfix.npz` datasets (fixed readout, 25×25 field response). Archived here
on 2026-07-23.
