# Legacy study scripts

These scripts drove the pre-refactor studies (seed-source, sub-bin,
censor/IRL1 scans) via the removed `deconv_positron_solver.py` driver.
They no longer run; they are kept as provenance for the FINDINGS ledger
(`/srv/.../analysis_20260716_zs_fixes/report/FINDINGS.md`).  The current
pipeline is config-driven: `python -m unfoldlarpix.fwk.runner
configs/<config>.yaml`.
