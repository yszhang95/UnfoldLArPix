#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Create output directory
mkdir -p raw_positron/data/0
RESPONSE_TEMPLATE=${RESPONSE_TEMPLATE:-center}

# Define sigma and sigma_pxl values
sigmas=(0.005 0.01)
sigma_pxls=(0.1 0.15 0.2)

echo "=== Running deconv for all sigma/sigma_pxl combinations ==="

for sigma in "${sigmas[@]}"; do
    for sigma_pxl in "${sigma_pxls[@]}"; do
        # Format parameter strings for filenames (remove leading 0 from decimals)
        sigma_str=$(printf "%.3f" "$sigma" | sed 's/^0\.//')
        sigma_pxl_str=$(printf "%.2f" "$sigma_pxl" | sed 's/^0\.//')

        echo ""
        echo "=== sigma=$sigma, sigma_pxl=$sigma_pxl ==="

        # Run V1
        echo "Running deconv_positron_v1.py..."
        python deconv_positron_v1.py --sigma "$sigma" --sigma-pxl "$sigma_pxl" \
            --response-template "$RESPONSE_TEMPLATE"

        # Run V2
        echo "Running deconv_positron_v2.py..."
        python deconv_positron_v2.py --sigma "$sigma" --sigma-pxl "$sigma_pxl" \
            --response-template "$RESPONSE_TEMPLATE"

        # Generate JSON from V1 output
        echo "Generating V1 JSON..."
        python deconv_xyz.py deconv_positron_event_0_0.npz \
            --tpc-id 0 --event-id 0 \
            --prefix "v1_s${sigma_str}_sp${sigma_pxl_str}" \
            --output-dir "output_matrix" 2>&1 | tail -1

        # Copy V1 JSON to raw_positron
        cp "output_matrix/data/0/0-v1_s${sigma_str}_sp${sigma_pxl_str}.json" \
           "raw_positron/data/0/v1_s${sigma_str}_sp${sigma_pxl_str}.json"

        # Generate JSON from V2 output
        echo "Generating V2 JSON..."
        python deconv_xyz.py deconv_positron_v2_event_0_0.npz \
            --tpc-id 0 --event-id 0 \
            --prefix "v2_s${sigma_str}_sp${sigma_pxl_str}" \
            --output-dir "output_matrix" 2>&1 | tail -1

        # Copy V2 JSON to raw_positron
        cp "output_matrix/data/0/0-v2_s${sigma_str}_sp${sigma_pxl_str}.json" \
           "raw_positron/data/0/v2_s${sigma_str}_sp${sigma_pxl_str}.json"
    done
done

echo ""
echo "=== All combinations complete ==="
echo "JSON files saved to: raw_positron/data/0/"
ls -lh raw_positron/data/0/*.json
