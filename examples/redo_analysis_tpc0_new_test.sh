#!/bin/bash
# LArPix Deconvolution Analysis - new test files and sigma scan

set -e

cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples

echo "=========================================="
echo "LArPix Deconvolution Analysis - New Test"
echo "=========================================="
echo ""

# Input files
INPUT_FILES=(
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz"
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz"
)

# Common parameters
THRESHOLD=0.5
VERSION=v2
DEST_DIR="analysis_20260319_tpc0"
PLOT_DIR="analysis_20260319_tpc0"
SIGMAS=(0.002 0.001)
SIGMA_PIXEL=0.2

echo "Output Directory: ${DEST_DIR}/"
echo "Configuration: ${#SIGMAS[@]} parameter combinations x ${#INPUT_FILES[@]} datasets"
echo ""
echo "Input Files:"
for f in "${INPUT_FILES[@]}"; do
  echo "  $f"
done
echo ""

run_idx=1
for sigma in "${SIGMAS[@]}"; do
  echo "[${run_idx}/${#SIGMAS[@]}] sigma_temporal=${sigma}, sigma_pixel=${SIGMA_PIXEL}"
  python run_analysis.py \
    --sigmas "${sigma}" \
    --sigma-pxls "${SIGMA_PIXEL}" \
    --thresholds "${THRESHOLD}" \
    --versions "${VERSION}" \
    --input-files "${INPUT_FILES[@]}" \
    --dest-dir "${DEST_DIR}" \
    --plot-dir "${PLOT_DIR}" \
    --steps 1 2 3 4
  echo "Completed run ${run_idx}"
  echo ""
  run_idx=$((run_idx + 1))
done

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""

if [ -d "${DEST_DIR}" ]; then
  json_count=$(find "${DEST_DIR}" -name "*.json" 2>/dev/null | wc -l)
  png_count=$(find "${DEST_DIR}" -name "*.png" 2>/dev/null | wc -l)
  total_count=$(find "${DEST_DIR}" -type f 2>/dev/null | wc -l)
  size=$(du -sh "${DEST_DIR}" 2>/dev/null | cut -f1)

  echo "Directory: ${DEST_DIR}/"
  echo "JSON files: ${json_count}"
  echo "PNG plots: ${png_count}"
  echo "Total files: ${total_count}"
  echo "Size: ${size}"
else
  echo "Output directory not found: ${DEST_DIR}/"
fi
echo ""
echo "Parameter Summary:"
echo "  sigma_temporal=0.002, sigma_pixel=0.2"
echo "  sigma_temporal=0.001, sigma_pixel=0.2"
