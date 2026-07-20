#!/bin/bash
# =====================================================================
# ZS charge-unfolding pipeline — end to end, one event.
#
#   input NPZ  ->  constrained-solver unfold  ->  truth/reco metrics
#              ->  2D correlation plot  ->  3-view event-display PNG
#
# The solver output is SELF-CONTAINED (smeared truth + reco in one
# file), so the metrics/plot steps need no external truth reference.
#
# Usage:
#   ./run_pipeline.sh INPUT.npz [TPC] [EVENT] [TAG]
#
# Example:
#   ./run_pipeline.sh \
#     data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz 0 0 nb4
#
# Requirements: the tred venv (numpy/scipy/matplotlib + torch/CUDA).
# For the interactive 3D HTML also: pip install plotly (optional; the
# PNG projection is produced regardless).
# =====================================================================
set -e
cd "$(dirname "$0")"

# ---- configuration ---------------------------------------------------
INPUT=${1:?"usage: run_pipeline.sh INPUT.npz [TPC] [EVENT] [TAG]"}
TPC=${2:-0}
EVENT=${3:-0}
TAG=${4:-run}

PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
OUT=analysis_output/pipeline_${TAG}
mkdir -p "$OUT"

SOLVED="$OUT/deconv_positron_solver_${TAG}_event_${TPC}_${EVENT}.npz"

# ADOPTED DEFAULT CONFIG (see report/FINDINGS.md).  Fixed knobs:
#   soft-ladder homotopy (alpha 1.0 -> 0.5 -> 0.3), strong-q seeding,
#   trigger split, 12-pixel spatial pad, deconv-support ROI, per-bin
#   quiet-window penalty, reco-centroid sub-bin positions (window 1).
SOLVER_ARGS="\
  --alpha-ladder 1.0 0.5 0.3 --seed-cut 0.5 --soft-seed-len 2 \
  --split-trigger --pad-pixels 12 --support-eps 0.3 --support-dilate 1 \
  --beta-quiet 1.0 --ladder-iters 150 --centroid-window 1 \
  --backend torch --device cuda"

echo "############ 1/4  UNFOLD  ###############################"
PYTHONPATH=../src $PY -u deconv_positron_solver.py \
  --input-file "$INPUT" --field-response "$FR" \
  --tpc-id "$TPC" $SOLVER_ARGS \
  --output-dir "$OUT" --output-suffix "$TAG"

echo "############ 2/4  METRICS  ##############################"
# universal grid (edges at global multiples of adc_hold_delay),
# Gaussian-shape deposit, sub-bin centroid positions applied.
PYTHONPATH=../src $PY eval_deconv_metrics.py "$SOLVED" \
  --labels "$TAG" \
  --universal-grid --deposit-shape gaussian --use-fitted-offsets \
  --json "$OUT/metrics_${TAG}.json"

echo "############ 3/4  CORRELATION PLOT  #####################"
PYTHONPATH=../src $PY corr2d_report.py "$SOLVED" \
  --labels "$TAG" --out "$OUT/corr2d_${TAG}.png"

echo "############ 4/4  EVENT DISPLAY  ########################"
PYTHONPATH=../src $PY event_display_3d.py "$SOLVED" \
  --labels "$TAG" --out "$OUT/event_${TAG}.html"

echo
echo "DONE.  Outputs in $OUT/:"
echo "  metrics_${TAG}.json     scalar metrics (also printed above)"
echo "  corr2d_${TAG}.png       truth vs reco 2D correlation"
echo "  event_${TAG}.png        3-view projections (truth + reco + ghosts)"
echo "  event_${TAG}.html       interactive 3D (if plotly installed)"
echo "  $(basename "$SOLVED")  self-contained solver output (~1.9 GB)"
