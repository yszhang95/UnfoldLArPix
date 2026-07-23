#!/bin/bash
# =====================================================================
# ZS charge-unfolding pipeline — end to end, one event, config-driven.
#
#   input NPZ -> framework unfold -> metrics -> correlation plot
#             -> 3-view event display
#
# Usage:   ./run_pipeline.sh INPUT.npz [TPC] [TAG]
# Example: ./run_pipeline.sh \
#            data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz 0 nb4
#
# The unfold step instantiates configs/adopted_nb4.yaml with your input
# (a copy of the resolved config is embedded in the output NPZ).  The
# output is SELF-CONTAINED (smeared truth included), so the analysis
# steps need no external truth reference.
# Requirements: tred venv (torch + CUDA).  Optional: plotly for the
# interactive 3D HTML.
# =====================================================================
set -e
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)

INPUT=${1:?"usage: run_pipeline.sh INPUT.npz [TPC] [TAG]"}
TPC=${2:-0}
TAG=${3:-run}

PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
OUT=analysis_output/pipeline_${TAG}
mkdir -p "$OUT"

echo "############ 1/4  UNFOLD (framework) ####################"
CFG=$OUT/job_${TAG}.yaml
sed -e "s|input: .*npz|input: examples/$INPUT|" \
    -e "s|tpc: 0|tpc: $TPC|" \
    -e "s|out_dir: .*|out_dir: examples/$OUT|" \
    -e "s|prefix: .*|prefix: $TAG|" \
    -e "s|WriteCharges:|WriteCharges:\n      embed_truth: true|" \
    "$REPO/configs/adopted_nb4.yaml" > "$CFG"
(cd "$REPO" && PYTHONPATH=src $PY -m unfoldlarpix.fwk.runner "examples/$CFG")

SOLVED=$(ls -t $OUT/${TAG}_event_*.npz | head -1)

echo "############ 2/4  METRICS  ##############################"
PYTHONPATH=../src $PY eval_deconv_metrics.py "$SOLVED" --labels "$TAG" \
  --universal-grid --deposit-shape gaussian --use-fitted-offsets \
  --json "$OUT/metrics_${TAG}.json"

echo "############ 3/4  CORRELATION PLOT  #####################"
PYTHONPATH=../src $PY corr2d_report.py "$SOLVED" --labels "$TAG" \
  --out "$OUT/corr2d_${TAG}.png"

echo "############ 4/4  EVENT DISPLAY  ########################"
PYTHONPATH=../src $PY event_display_3d.py "$SOLVED" --labels "$TAG" \
  --out "$OUT/event_${TAG}.html"

echo
echo "DONE.  Outputs in $OUT/:  metrics_${TAG}.json  corr2d_${TAG}.png"
echo "       event_${TAG}.png(.html)  $(basename "$SOLVED")  job_${TAG}.yaml"
