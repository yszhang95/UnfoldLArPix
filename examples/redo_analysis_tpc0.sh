#!/bin/bash
# LArPix Deconvolution Analysis - TPC 0 Only Re-run
# Re-executes 7 parameter configurations
# with --tpc-id 0 filtering to process only TPC 0 data

set -e

cd /home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples

echo "=========================================="
echo "LArPix Deconvolution Analysis - TPC 0 Only"
echo "=========================================="
echo ""
echo "Output Directory: analysis_20260318_tpc0/"
echo "Configuration: 7 parameter combinations × 4 datasets"
echo ""

# Input files (all 4 datasets)
INPUT_FILES=(
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres1k_nburst256.npz"
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst256.npz"
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_nburst4.npz"
  "data/pgun_positron_3gev_tred_noises_effq_nt1_thres5k_selftrigger.npz"
)

# Common parameters
THRESHOLD=0.5
VERSION=v2
DEST_DIR="analysis_20260318_tpc0"
PLOT_DIR="analysis_20260318_tpc0"

echo "Input Files:"
for f in "${INPUT_FILES[@]}"; do
  echo "  ✓ $f"
done
echo ""

# ============================================================================
# Run 1: σ_temporal=0.005, σ_pixel=0.1
# ============================================================================
echo "[1/7] σ_temporal=0.005, σ_pixel=0.1"
python run_analysis.py \
  --sigmas 0.005 \
  --sigma-pxls 0.1 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 1 complete"
echo ""

# ============================================================================
# Run 2: σ_temporal=0.005, σ_pixel=0.2
# ============================================================================
echo "[2/7] σ_temporal=0.005, σ_pixel=0.2"
python run_analysis.py \
  --sigmas 0.005 \
  --sigma-pxls 0.2 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 2 complete"
echo ""

# ============================================================================
# Run 3: σ_temporal=0.004, σ_pixel=0.2
# ============================================================================
echo "[3/7] σ_temporal=0.004, σ_pixel=0.2"
python run_analysis.py \
  --sigmas 0.004 \
  --sigma-pxls 0.2 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 3 complete"
echo ""

# ============================================================================
# Run 4: σ_temporal=0.003, σ_pixel=0.2
# ============================================================================
echo "[4/7] σ_temporal=0.003, σ_pixel=0.2"
python run_analysis.py \
  --sigmas 0.003 \
  --sigma-pxls 0.2 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 4 complete"
echo ""

# ============================================================================
# Run 5: σ_temporal=0.003, σ_pixel=0.15
# ============================================================================
echo "[5/7] σ_temporal=0.003, σ_pixel=0.15"
python run_analysis.py \
  --sigmas 0.003 \
  --sigma-pxls 0.15 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 5 complete"
echo ""

echo "[6/7] σ_temporal=0.004, σ_pixel=0.1"
python run_analysis.py \
  --sigmas 0.004 \
  --sigma-pxls 0.1 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 6 complete"
echo ""

echo "[7/7] σ_temporal=0.002, σ_pixel=0.2"
python run_analysis.py \
  --sigmas 0.002 \
  --sigma-pxls 0.2 \
  --thresholds "$THRESHOLD" \
  --versions "$VERSION" \
  --input-files "${INPUT_FILES[@]}" \
  --dest-dir "$DEST_DIR" \
  --plot-dir "$PLOT_DIR" \
  --steps 1 2 3 4
echo "✓ Run 7 complete"
echo ""


# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Output Summary:"
if [ -d "$DEST_DIR" ]; then
  json_count=$(find "$DEST_DIR" -name "*.json" 2>/dev/null | wc -l)
  png_count=$(find "$DEST_DIR" -name "*.png" 2>/dev/null | wc -l)
  total_count=$(ls -1 "$DEST_DIR" 2>/dev/null | wc -l)
  size=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1)

  echo "  Directory: $DEST_DIR/"
  echo "  JSON files: $json_count"
  echo "  PNG plots:  $png_count"
  echo "  Total files: $total_count"
  echo "  Size: $size"
  echo ""

  echo "✅ Analysis complete! All files generated successfully (TPC 0 only)"
else
  echo "  ❌ Output directory not found"
fi

echo ""
echo "Parameter Summary:"
echo "  Config 1: σ_temporal=0.005, σ_pixel=0.1   → JSONs + plots"
echo "  Config 2: σ_temporal=0.005, σ_pixel=0.2   → JSONs + plots"
echo "  Config 3: σ_temporal=0.004, σ_pixel=0.2   → JSONs + plots"
echo "  Config 4: σ_temporal=0.003, σ_pixel=0.2   → JSONs + plots"
echo "  Config 5: σ_temporal=0.003, σ_pixel=0.15  → JSONs + plots"
echo "  Config 6: σ_temporal=0.004, σ_pixel=0.1   → JSONs + plots"
echo "  Config 7: σ_temporal=0.002, σ_pixel=0.2   → JSONs + plots"
echo ""
