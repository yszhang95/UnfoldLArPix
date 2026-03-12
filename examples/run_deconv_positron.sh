#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Running deconv_positron_v1.py ==="
python deconv_positron_v1.py

echo "=== Running deconv_positron_v2.py ==="
python deconv_positron_v2.py

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
