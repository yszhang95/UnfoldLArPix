#!/bin/bash
# Evaluate + plot the seed-source comparison (universal grid, Gaussian deposit).
set -e
cd "$(dirname "$0")"

D=analysis_output/analysis_20260716_zs_fixes
OUT=$D/seed_source
REP=$D/report
PY=../.venv/bin/python

TRUTH_NB4=$D/tier3_solver/deconv_positron_solver_nb4_ladder_split_event_0_0.npz
TRUTH_NB1=$D/nb1/deconv_positron_solver_nb1_ladder_event_0_0.npz

$PY eval_deconv_metrics.py \
  "$OUT"/deconv_positron_solver_nb4_seedDECONV_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb4_seedHITS_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb4_seedHITS_cold_event_0_0.npz \
  --labels nb4_seedDECONV nb4_seedHITS nb4_seedHITS_cold \
  --truth-npz "$TRUTH_NB4" --universal-grid --deposit-shape gaussian \
  --json "$REP"/metrics_seed_source_nb4.json

$PY eval_deconv_metrics.py \
  "$OUT"/deconv_positron_solver_nb1_seedDECONV_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb1_seedHITS_event_0_0.npz \
  --labels nb1_seedDECONV nb1_seedHITS \
  --truth-npz "$TRUTH_NB1" --universal-grid --deposit-shape gaussian \
  --json "$REP"/metrics_seed_source_nb1.json

$PY corr2d_report.py \
  "$OUT"/deconv_positron_solver_nb4_seedDECONV_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb4_seedHITS_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb4_seedHITS_cold_event_0_0.npz \
  --labels "seed: FFT-deconv" "seed: hits" "seed: hits (cold start)" \
  --truth-npz "$TRUTH_NB4" --ncols 3 \
  --out "$REP"/corr2d_seed_source_nb4.png

$PY corr2d_report.py \
  "$OUT"/deconv_positron_solver_nb1_seedDECONV_event_0_0.npz \
  "$OUT"/deconv_positron_solver_nb1_seedHITS_event_0_0.npz \
  --labels "seed: FFT-deconv" "seed: hits" \
  --truth-npz "$TRUTH_NB1" --ncols 2 \
  --out "$REP"/corr2d_seed_source_nb1.png

echo eval complete
