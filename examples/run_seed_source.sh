#!/bin/bash
# Seeding-source comparison: FFT-deconv support vs raw-hits (latch-window)
# support, adopted soft-ladder config, nb4 + nb1 (selftrigger, non-shield).
set -e
cd "$(dirname "$0")"

OUT=analysis_output/analysis_20260716_zs_fixes/seed_source
mkdir -p "$OUT"
TRED_PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
NB4=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz
NB1=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger.npz

COMMON="--tpc-id 0 --alpha-ladder 1.0 0.5 0.3 --seed-cut 0.5 --soft-seed-len 2 \
  --split-trigger --pad-pixels 12 --support-dilate 1 --beta-quiet 1.0 \
  --ladder-iters 150 --backend torch --device cuda --lean-output \
  --output-dir $OUT"

run() {  # dataset tag extra-flags...
  local input=$1 tag=$2; shift 2
  echo "=== $tag"
  PYTHONPATH=../src $TRED_PY -u deconv_positron_solver.py \
    --input-file "$input" --field-response "$FR" $COMMON \
    --output-suffix "$tag" "$@" 2>&1 | grep -E "support|residual|stage|Saved|Error" || true
}

run "$NB4" nb4_seedDECONV --support-eps 0.3
run "$NB4" nb4_seedHITS   --support-source hits
run "$NB4" nb4_seedHITS_cold --support-source hits --cold-start
run "$NB1" nb1_seedDECONV --support-eps 0.3
run "$NB1" nb1_seedHITS   --support-source hits

echo "all runs complete: $OUT"
