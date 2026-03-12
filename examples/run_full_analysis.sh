#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

sigmas=(0.005 0.01)
sigma_pxls=(0.1 0.15 0.2)
thresholds=(0.5 1.0 1.5 2.0 2.5 3.0)

mkdir -p raw_positron/data/0
mkdir -p output_matrix/data/0
mkdir -p plots

echo "=== Step 1+2+3+4: Loop over sigma/sigma_pxl combinations ==="

for sigma in "${sigmas[@]}"; do
    for sigma_pxl in "${sigma_pxls[@]}"; do
        sigma_str=$(printf "%.3f" "$sigma" | sed 's/^0\.//')
        sigma_pxl_str=$(printf "%.2f" "$sigma_pxl" | sed 's/^0\.//')

        echo ""
        echo "--- sigma=$sigma  sigma_pxl=$sigma_pxl ---"

        # Step 1: Run deconvolution (v1 and v2)
        echo "[1] deconv_positron_v1.py ..."
        python deconv_positron_v1.py --sigma "$sigma" --sigma-pxl "$sigma_pxl"

        echo "[1] deconv_positron_v2.py ..."
        python deconv_positron_v2.py --sigma "$sigma" --sigma-pxl "$sigma_pxl"

        # Step 4: Make histogram plots from plot_proj.py
        echo "[4] plot_proj.py (v1) ..."
        python plot_proj.py deconv_positron_event_0_0.npz \
            --prefix "plots/v1_s${sigma_str}_sp${sigma_pxl_str}"

        echo "[4] plot_proj.py (v2) ..."
        python plot_proj.py deconv_positron_v2_event_0_0.npz \
            --prefix "plots/v2_s${sigma_str}_sp${sigma_pxl_str}"

        # Step 2+3: Export JSONs for each threshold, copy to raw_positron/data/0/
        for threshold in "${thresholds[@]}"; do
            threshold_str=$(printf "%s" "$threshold" | sed 's/\./p/')

            v1_prefix="v1_s${sigma_str}_sp${sigma_pxl_str}_t${threshold_str}"
            v2_prefix="v2_s${sigma_str}_sp${sigma_pxl_str}_t${threshold_str}"

            echo "[2] deconv_xyz.py v1 threshold=$threshold ..."
            python deconv_xyz.py deconv_positron_event_0_0.npz \
                --tpc-id 0 --event-id 0 \
                --threshold "$threshold" \
                --prefix         "${v1_prefix}" \
                --smeared-prefix "${v1_prefix}_smeared" \
                --output-dir output_matrix 2>&1 | tail -2

            echo "[2] deconv_xyz.py v2 threshold=$threshold ..."
            python deconv_xyz.py deconv_positron_v2_event_0_0.npz \
                --tpc-id 0 --event-id 0 \
                --threshold "$threshold" \
                --prefix         "${v2_prefix}" \
                --smeared-prefix "${v2_prefix}_smeared" \
                --output-dir output_matrix 2>&1 | tail -2

            # Step 3: Move to raw_positron/data/0/
            cp "output_matrix/data/0/0-${v1_prefix}.json"         "raw_positron/data/0/"
            cp "output_matrix/data/0/0-${v1_prefix}_smeared.json" "raw_positron/data/0/"
            cp "output_matrix/data/0/0-${v2_prefix}.json"         "raw_positron/data/0/"
            cp "output_matrix/data/0/0-${v2_prefix}_smeared.json" "raw_positron/data/0/"
        done
    done
done

echo ""
echo "=== All done ==="
echo "JSON files in raw_positron/data/0/: $(ls raw_positron/data/0/*.json | wc -l)"
echo "Plots in plots/: $(ls plots/*.png 2>/dev/null | wc -l)"
