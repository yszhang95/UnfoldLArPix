# Untested code in this repository

Written 2026-09-16, at the commit that first brings the modules below under
version control.  Everything listed here is committed **without tests**, so
that the campaigns already produced from it have a real commit hash instead of
a dirty working tree.  The repository's convention elsewhere is a `test_*.py`
beside every module (`zsbasis_algs.py` / `test_zsbasis.py`, 26 tests;
`exactrows_algs.py` / `test_exactrows.py`), and the study notes quote test
counts as provenance.  These do not have that yet.

The full suite (392 tests) passes at this commit.  It passes because every new
solver option defaults to off, so the default code path is bit-identical to
`6fd577a` and the archived results reproduce.  A passing suite is therefore
**not** evidence that any of the code below is correct.

## Library code

| symbol | file | archived results that depend on it |
|---|---|---|
| `GroupFloorProx` | `terms/base.py` | **yes** — `calib_zs/floor_c{1,5}`, `calib_zs/floorladder_*`, `norm_uncertainty` arm `m2_floor020` |
| `universal_group_phase` | `terms/base.py` | **yes** — same jobs (`group_universal: true`) |
| `SimplexProx` | `terms/base.py` | no job sets it |
| `Fista(restart=...)` | `solve/engine.py` | no job sets `fista_restart` |
| `Fista(rel_tol=, patience=)` | `solve/engine.py` | no job sets `fista_rel_tol` |
| `Fista(backtrack=, bt_L0_div=, bt_up=, bt_down=)` | `solve/engine.py` | no job sets it; this one changes the step rule |
| `Fista(trace_every=)` | `solve/engine.py` | no job sets it |

`GroupFloorProx` is the one that matters most: it is shared library code, it
has already produced archived numbers, and it projects onto a non-trivial set
(each cell `>= -floor`, each group of `group` consecutive cells summing
`>= 0`).  Nothing currently checks that the projection is the projection.

## Campaign code

| module | lines | archived results |
|---|---|---|
| `algs/zsgradflow_algs.py` | 1145 | `gradflow_zs/`, `norm_uncertainty/{angscan,alphascan,alphascan_D}`, `bestfit_isoline/` |
| `algs/frvariant_algs.py` | 583 | `frvariant_isoline/` |
| `algs/zscalib_algs.py` | 455 | `calib_zs/` |
| `algs/zsgradflow_figs.py` | 366 | figures of the above |

## Tests owed

1. `GroupFloorProx`: closed form on a small input worked out by hand;
   idempotence `P(P(x)) == P(x)`; every group sum `>= 0`; every cell
   `>= -floor`; the `group = 1` case reducing to `CoordProx`.
2. `universal_group_phase`: the phase against a hand-computed block offset,
   and the misalignment it reports when `cell_ticks` does not divide the
   group.
3. `Fista(rel_tol=)`: stops at the iteration the criterion implies, and
   `stopped_at` equals `n_iter` when it never fires.
4. `Fista(restart=)`: same fixed point as plain FISTA on a small convex
   problem, within tolerance.
5. `Fista(backtrack=)`: the accepted step satisfies the sufficient-decrease
   condition; `L_k` never exceeds the proven bound `L`; same fixed point.
6. `SimplexProx`: `sum(P(x)) == target`; `P` idempotent; the negative-`tau`
   branch (input summing below the target) actually adds charge.
7. One `test_*.py` per campaign module, in the style of `test_zsbasis.py`.

Delete the corresponding row when the test lands.
