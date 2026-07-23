#!/bin/bash
# Sub-bin position stage on top of the adopted soft-ladder config (nb4 + nb1).
set -e
cd "$(dirname "$0")"

OUT=analysis_output/analysis_20260716_zs_fixes/subbin
mkdir -p "$OUT"
TRED_PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
NB4=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz
NB1=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger.npz

COMMON="--tpc-id 0 --alpha-ladder 1.0 0.5 0.3 --seed-cut 0.5 --soft-seed-len 2 \
  --split-trigger --pad-pixels 12 --support-eps 0.3 --support-dilate 1 \
  --beta-quiet 1.0 --ladder-iters 150 --backend torch --device cuda \
  --lean-output --subbin-rounds 3 --output-dir $OUT"

for CFG in "$NB4:nb4_subbin" "$NB1:nb1_subbin"; do
  IFS=: read -r INPUT TAG <<< "$CFG"
  echo "=== $TAG"
  PYTHONPATH=../src $TRED_PY -u deconv_positron_solver.py \
    --input-file "$INPUT" --field-response "$FR" $COMMON \
    --output-suffix "$TAG" 2>&1 | grep -E "support|subbin|residual|Saved|Error|Traceback" || true
done
echo "done: $OUT"
