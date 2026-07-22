# Archived documentation

Historical design notes and analysis-session reports, moved here
2026-07-22.  They were accurate when written but describe superseded
states of the code or completed/closed studies.  Kept for provenance;
**do not use them as a guide to the current code.**

Current entry points:
- `examples/PIPELINE.md` — how to run the current (solver-based) pipeline.
- `/srv/storage1/yousen/analysis/charge_unfolading_ndlar/analysis_20260716_zs_fixes/report/FINDINGS.md`
  — the findings ledger for the constrained-solver era (the current method).
- `src/unfoldlarpix/README_burst_processor.md` — burst processor module
  doc (still current; the processor feeds the solver's warm start).

| file | date | what it was | superseded by |
|---|---|---|---|
| ANALYSIS_SUMMARY.md | 2026-03 | session log: dash display, v2 runs, output conventions | dated dirs + FINDINGS ledger |
| ANALYSIS_HISTORY.md | 2026-03 | March TPC-0 analysis run log | same |
| V3_ANALYSIS_SUMMARY.md | 2026-03 | v3 processor analysis session | v3 is now the solver warm start |
| POWER_SPECTRA_REPORT.md | 2026-03 | truth power-spectra study | tier_spectra_report.py + FINDINGS |
| report_deconv_analysis.md | 2026-03 | FFT-deconv parameter study | solver supersedes FFT-deconv as estimator (FINDINGS) |
| DYNAMIC_REPORT_DESIGN.md | 2026-03 | design of report_dynamic.html viewer | viewer artifacts remain in examples/ |
| BURST_PROCESSOR_SUMMARY.md | 2026-02 | implementation-completion checklist | src/unfoldlarpix/README_burst_processor.md |
| CHANGELOGS.md | 2026-04 | change log since 2026-03-31 | git history |
| INSTRUCTIONS.md | 2026-04 | run/record conventions around ANALYSIS_SUMMARY | PIPELINE.md + /srv archive convention |
| IMPLEMENTATION_CHECKLIST.md | 2026-05 | Wiener-filter build checklist | implemented & closed (FINDINGS: stationary priors ruled out) |
| WIENER_FILTER_ANALYSIS.md | 2026-05 | study notes on arXiv 1802.08709 | FINDINGS §7 (ruled out) + memory |
| WIENER_ROI_IMPLEMENTATION_REPORT.md | 2026-05 | Wiener-ROI branch report | merged; ROI closed by FINDINGS (does not remove voxel-level ghosts) |
| SMOOTHED_TEMPLATE_PROPOSAL.md | 2026-07 | Gaussian-smoothed template proposal | tested as Tier 1b: marginal after phase repair, NOT adopted (FINDINGS) |
| SCRIPTS_GUIDE.md | 2026-04 | examples/ script inventory | predates the solver-era scripts; PIPELINE.md covers the current flow |
