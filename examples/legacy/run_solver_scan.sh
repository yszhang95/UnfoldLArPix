#!/bin/bash
# Seed/regularization scan of the constrained solver (GPU, lean outputs).
# Usage: ./run_solver_scan.sh <output_dir>
set -e
cd "$(dirname "$0")"

OUT=${1:-analysis_output/analysis_20260716_zs_fixes/scan_seed_reg}
mkdir -p "$OUT"
D=analysis_output/analysis_20260716_zs_fixes
TRED_PY=/home/yousen/Documents/NDLAr2x2/tred/.venv/bin/python
INPUT=data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz
FR=/srv/storage1/yousen/tred_workspace/response_44_v2a_full_25x25pixel_tred.npz
FILT=$D/tier1c_xspec/muon_time_filter_truth_lin.npz

# seeding configs: warm-sigma : support-eps : tag
SEEDS=(
  "0.002:0.5:ws002e05"
  "0.005:0.8:ws005e08"
  "0.005:0.3:ws005e03"
)
# regularizations: lam_tv : lam_l2 : tag
REGS=(
  "0:0:tv0l0"
  "0.1:0.01:tv01l001"
  "0.3:0.01:tv03l001"
  "0.3:0:tv03l0"
)

for S in "${SEEDS[@]}"; do
  IFS=: read -r WS EPS STAG <<< "$S"
  for R in "${REGS[@]}"; do
    IFS=: read -r TV L2 RTAG <<< "$R"
    TAG="${STAG}_${RTAG}"
    echo "=== $TAG (warm-sigma $WS, eps $EPS, tv $TV, l2 $L2)"
    PYTHONPATH=../src $TRED_PY -u deconv_positron_solver.py \
      --input-file "$INPUT" --field-response "$FR" --tpc-id 0 \
      --beta-quiet 1.0 --alpha-ladder 1.0 0.5 0.3 \
      --seed-cut 0.5 --seed-dilate 2 --ladder-iters 150 \
      --support-eps "$EPS" --support-dilate 1 --split-trigger \
      --warm-sigma "$WS" --lam-tv "$TV" --lam-l2 "$L2" \
      --backend torch --device cuda --lean-output \
      --time-filter-npz "$FILT" \
      --output-suffix "$TAG" \
      --output-dir "$OUT" 2>&1 | grep -E "residual|Saved"
  done
done
echo "scan complete: $OUT"
