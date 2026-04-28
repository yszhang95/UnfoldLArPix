#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Default sigma values; override with environment variables or arguments
SIGMA=${SIGMA:-0.005}
SIGMA_PXL=${SIGMA_PXL:-0.2}
RESPONSE_TEMPLATE=${RESPONSE_TEMPLATE:-center}

echo "=== Running deconv_positron_v1.py (sigma=$SIGMA, sigma_pxl=$SIGMA_PXL, response_template=$RESPONSE_TEMPLATE) ==="
python deconv_positron_v1.py --sigma "$SIGMA" --sigma-pxl "$SIGMA_PXL" \
    --response-template "$RESPONSE_TEMPLATE"

echo "=== Running deconv_positron_v2.py (sigma=$SIGMA, sigma_pxl=$SIGMA_PXL, response_template=$RESPONSE_TEMPLATE) ==="
python deconv_positron_v2.py --sigma "$SIGMA" --sigma-pxl "$SIGMA_PXL" \
    --response-template "$RESPONSE_TEMPLATE"

echo "=== Running deconv_xyz.py (V1, TPC 0, event 0) ==="
python deconv_xyz.py deconv_positron_event_0_0.npz \
    --threshold 1   \
    --tpc-id 0 --event-id 0 --prefix v1 --output-dir output_v1

echo "=== Running deconv_xyz.py (V2, TPC 0, event 0) ==="
python deconv_xyz.py deconv_positron_v2_event_0_0.npz \
    --threshold 1   \
    --tpc-id 0 --event-id 0 --prefix v2 --output-dir output_v2

echo "=== Copying json to target directory"
cd raw_positron/
cp ../output_v?/data/0/0-v?.json data/0/
zip -r mydata.zip data/
echo "=== Done ==="
